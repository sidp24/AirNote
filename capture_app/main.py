import argparse, time, uuid
import cv2
import numpy as np

from hand_state import HandsDetector, parse_hands, fingertip
from plane_select import draw_reticle, compute_H0
from tracker import PlanarTracker
from board_canvas import BoardCanvas, warp_point
from utils import alpha_blend

# Optional Firebase uploader
try:
    from uploader import FirebaseUploader
except Exception:
    FirebaseUploader = None

# ---------- Config ----------
BOARD_W, BOARD_H = 1200, 800
DEFAULT_WIDTH = 5
ERASER_DIAM = 34  # bigger eraser for usability
COLORS = [(0,255,0),(255,0,0),(0,140,255),(255,255,255)]  # BGR

# Hotbar + gestures
HOTBAR = ["DRAW", "ERASE", "COLOR", "WIDTH", "UNDO", "REDO", "CLEAR", "NEW", "SAVE"]
TOGGLE_TAP_MS = 300
SAVE_LONG_MS = 700

# Tracker thresholds
INLIER_THRESH = 35
FREEZE_FRAMES = 6
RESEED_LOW_STREAK = 8
HYSTERESIS_FRAMES = 5

# Temporal smoothing for quad corners (tracking stability)
QUAD_SMOOTH_A = 0.18  # smaller = smoother

# ---------- Runtime state ----------
color_idx = 0
DRAW_WIDTH = DEFAULT_WIDTH
state = "IDLE"         # IDLE, DRAW, ERASE
_last_t = None
fps = 0.0

# Hands UX
tool_mode = "DRAW"     # "DRAW" or "ERASE"
hotbar_idx = 0
# Gate logic: whichever hand is pinched and HELD becomes the gate; the other hand taps to nav/activate
gate_hand = None       # "Left" or "Right" or None
gate_down = False
# tap tracking for both hands (for bi-directional nav)
last_pinch = {"Left": False, "Right": False}
pinch_down_ms = {"Left": 0, "Right": 0}
pinch_progress = 0.0   # render ring for whichever hand is tapping/long-pressing

# Feedback HUD
last_action_text = ""
last_action_until = 0
save_flash_until = 0

# Mouse (optional)
mouse_draw = False
mouse_down = False
mouse_pos = (0,0)

# Firebase runtime
firebase_uploader = None
FB_PROJECT = ""
FB_BUCKET = ""
FB_PUBLIC = False

# Autosave
AUTOSAVE_MS = 5000
last_autosave_ms = 0
dirty_since_save = False

# Save prefixes
PREFIX_MANUAL = "save"
PREFIX_AUTO   = "auto"

# Smoothed quad
smoothed_quad = None

def now_ms(): return int(time.time()*1000)

def set_action(msg, ms=1000):
    global last_action_text, last_action_until
    last_action_text = msg
    last_action_until = now_ms() + ms

def tick_fps():
    global _last_t, fps
    t = time.time()
    if _last_t is None:
        _last_t = t; return 0.0
    dt = t - _last_t; _last_t = t
    if dt > 0:
        fps = (0.9*fps + 0.1*(1.0/dt)) if fps > 0 else (1.0/dt)
    return fps

# ----- Pretty UI primitives (modern hotbar) -----
def _aa_roundrect(img, pt1, pt2, radius, color, thickness=-1):
    x1,y1 = pt1; x2,y2 = pt2
    r = int(max(1, radius))
    if thickness < 0:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1+r, y1+r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2-r, y1+r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x1+r, y2-r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x2-r, y2-r), r, color, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, thickness, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1+r, y1+r), (r,r), 0, 180, 270, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2-r, y1+r), (r,r), 0, 270, 360, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x1+r, y2-r), (r,r), 0,  90, 180, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x2-r, y2-r), (r,r), 0,   0,  90, color, thickness, cv2.LINE_AA)

def _drop_shadow(img, pt1, pt2, radius, base, alpha=0.45, offset=(0,3)):
    x1,y1 = pt1; x2,y2 = pt2
    ox,oy = offset
    shadow = (int(base[0]*0.2), int(base[1]*0.2), int(base[2]*0.2))
    overlay = img.copy()
    _aa_roundrect(overlay, (x1+ox, y1+oy), (x2+ox, y2+oy), radius, shadow, -1)
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

def _icon(img, cx, cy, name, color=(230,230,230)):
    if name == "DRAW":
        cv2.line(img, (cx-10,cy+6), (cx+12,cy-6), color, 2, cv2.LINE_AA)
        cv2.circle(img, (cx+12,cy-6), 3, color, -1, cv2.LINE_AA)
    elif name == "ERASE":
        cv2.rectangle(img, (cx-12,cy-8), (cx+6,cy+8), color, 2, cv2.LINE_AA)
        cv2.line(img, (cx-6,cy+8), (cx+12,cy+8), color, 2, cv2.LINE_AA)
    elif name == "COLOR":
        cv2.circle(img, (cx,cy), 9, color, 2, cv2.LINE_AA)
    elif name == "WIDTH":
        cv2.line(img, (cx-12,cy-6), (cx-2,cy-6), color, 1, cv2.LINE_AA)
        cv2.line(img, (cx-12,cy),   (cx,cy),     color, 2, cv2.LINE_AA)
        cv2.line(img, (cx-12,cy+6), (cx+4,cy+6), color, 3, cv2.LINE_AA)
    elif name == "UNDO":
        cv2.arrowedLine(img, (cx+10,cy-6), (cx-8,cy-6), color, 2, tipLength=0.5, line_type=cv2.LINE_AA)
        cv2.ellipse(img, (cx-4,cy), (10,8), 0, 90, 200, color, 2, cv2.LINE_AA)
    elif name == "REDO":
        cv2.arrowedLine(img, (cx-10,cy-6), (cx+8,cy-6), color, 2, tipLength=0.5, line_type=cv2.LINE_AA)
        cv2.ellipse(img, (cx+4,cy), (10,8), 0, -20, 90, color, 2, cv2.LINE_AA)
    elif name == "CLEAR":
        cv2.rectangle(img, (cx-10,cy-7), (cx+10,cy+7), color, 2, cv2.LINE_AA)
        cv2.line(img, (cx-10,cy-7), (cx+10,cy+7), color, 2, cv2.LINE_AA)
    elif name == "NEW":
        cv2.rectangle(img, (cx-10,cy-7), (cx+10,cy+7), color, 2, cv2.LINE_AA)
        cv2.line(img, (cx,cy-10), (cx,cy+10), color, 2, cv2.LINE_AA)
        cv2.line(img, (cx-10,cy), (cx+10,cy), color, 2, cv2.LINE_AA)
    elif name == "SAVE":
        cv2.rectangle(img, (cx-10,cy-8), (cx+10,cy+8), color, 2, cv2.LINE_AA)
        cv2.circle(img, (cx,cy+2), 4, color, 2, cv2.LINE_AA)

def draw_hotbar(img, w, h, selected_idx, draw_width, color_bgr):
    margin = 16
    bar_h = 64
    y0 = h - bar_h - margin
    x0 = margin
    x1 = w - margin
    radius = 18
    _drop_shadow(img, (x0, y0), (x1, h - margin), radius, (40,40,40), alpha=0.45)
    overlay = img.copy()
    _aa_roundrect(overlay, (x0, y0), (x1, h - margin), radius, (32,32,32), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    n = len(HOTBAR)
    gap = 8
    inner_w = (x1 - x0) - (n+1)*gap
    cell_w = int(inner_w / n)
    y_pad = 6

    for i, name in enumerate(HOTBAR):
        cx0 = x0 + gap + i*(cell_w + gap)
        cx1 = cx0 + cell_w
        btn_radius = 12

        active = (i == selected_idx)
        base = (60,60,60) if not active else (70,110,70)
        _aa_roundrect(img, (cx0, y0+y_pad), (cx1, h - margin - y_pad), btn_radius, base, -1)

        if active:
            glow = img.copy()
            _aa_roundrect(glow, (cx0, y0+y_pad), (cx1, h - margin - y_pad), btn_radius, (80,170,80), -1)
            cv2.addWeighted(glow, 0.15, img, 0.85, 0, img)

        cx = (cx0 + cx1)//2
        cy = y0 + y_pad + (bar_h - 2*y_pad)//2 - 6
        _icon(img, cx, cy, name)

        label = name
        if name == "COLOR":
            cv2.rectangle(img, (cx-12, cy+10), (cx+12, cy+22), color_bgr, -1, cv2.LINE_AA)
            cv2.rectangle(img, (cx-12, cy+10), (cx+12, cy+22), (20,20,20), 1, cv2.LINE_AA)
        elif name == "WIDTH":
            label = f"W:{draw_width}"

        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        tx = cx - tw//2
        ty = y0 + bar_h - 14
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (225,225,225), 1, cv2.LINE_AA)

def draw_progress_ring(img, center, radius, t01, color=(0,255,255)):
    t01 = max(0.0, min(1.0, float(t01)))
    if t01 <= 0: return
    start_angle = -90
    end_angle = start_angle + int(360 * t01)
    cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, 2, lineType=cv2.LINE_AA)

# ---------- Save helper (composite-only) ----------
def do_save(board, session_id, H0, curr_quad, color_idx, DRAW_WIDTH, hotbar_idx,
            firebase_uploader, frame_bgr, H_curr, auto=False):
    """
    Save 1 file for the image: camera rectified + strokes overlaid (composite),
    plus strokes.json and meta.json. Prunes older for same (prefix,page).
    """
    prefix = "auto" if auto else "save"

    # Rectify camera into board space
    composite = None
    try:
        cam_rect = cv2.warpPerspective(frame_bgr, H_curr, (board.W, board.H))
        strokes = board._render_to(include_grid=False)  # BGRA
        # Blend strokes onto the rectified camera
        sb, sg, sr, sa = cv2.split(strokes)
        sa_f = sa.astype(np.float32) / 255.0
        inv = 1.0 - sa_f
        comp_b = (sb.astype(np.float32)*sa_f + cam_rect[:,:,0].astype(np.float32)*inv).astype(np.uint8)
        comp_g = (sg.astype(np.float32)*sa_f + cam_rect[:,:,1].astype(np.float32)*inv).astype(np.uint8)
        comp_r = (sr.astype(np.float32)*sa_f + cam_rect[:,:,2].astype(np.float32)*inv).astype(np.uint8)
        composite = cv2.merge([comp_b, comp_g, comp_r])
    except Exception:
        # fallback to strokes only
        composite = cv2.cvtColor(board._render_to(include_grid=False), cv2.COLOR_BGRA2BGR)

    try:
        comp_path, json_path, meta_path = board.save(
            out_root="out",
            session_id=session_id,
            page_idx=board.page_idx,
            H0=H0,
            curr_quad=curr_quad,
            color_idx=color_idx,
            draw_width=DRAW_WIDTH,
            hotbar_idx=hotbar_idx,
            page_count=board.page_count(),
            composite_bgr=composite,
            prefix=prefix,
            prune_previous=True
        )

        if firebase_uploader is not None:
            files = {
                "composite_jpg": comp_path,
                "strokes_json": json_path,
                "meta_json": meta_path,
            }
            meta = {
                "W": board.W, "H": board.H,
                "page_idx": board.page_idx,
                "page_count": board.page_count(),
                "color_idx": color_idx,
                "draw_width": DRAW_WIDTH,
                "hotbar_idx": hotbar_idx,
                "H0": H0.tolist() if H0 is not None else None,
                "curr_quad": curr_quad.tolist() if curr_quad is not None else None,
                "session_id": session_id,
            }
            save_ts = int(time.time())
            extra = {"source": "AirNote", "auto": bool(auto), "prefix": prefix}
            firebase_uploader.upload_save(
                session_id=session_id,
                save_ts=save_ts,
                files=files,
                meta=meta,
                make_public=FB_PUBLIC,
                extra_fields=extra
            )
        return True
    except Exception as e:
        print("[Save] failed:", e)
        return False

def main():
    global state, color_idx, DRAW_WIDTH, tool_mode, hotbar_idx
    global gate_hand, gate_down, last_pinch, pinch_down_ms, pinch_progress
    global last_action_text, last_action_until, save_flash_until
    global mouse_draw, mouse_down, mouse_pos
    global firebase_uploader, FB_PROJECT, FB_BUCKET, FB_PUBLIC
    global AUTOSAVE_MS, last_autosave_ms, dirty_since_save, smoothed_quad

    # CLI
    ap = argparse.ArgumentParser(description="AirNote Capture App")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fb_project", type=str, default="")
    ap.add_argument("--fb_bucket", type=str, default="")
    ap.add_argument("--fb_public", action="store_true")
    ap.add_argument("--autosave_sec", type=float, default=5.0)
    args = ap.parse_args()
    MIRROR_DISPLAY = bool(args.mirror)
    FB_PROJECT = args.fb_project
    FB_BUCKET = args.fb_bucket
    FB_PUBLIC = bool(args.fb_public)
    AUTOSAVE_MS = max(0, int(args.autosave_sec * 1000))

    # Camera
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Session id
    session_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]

    # Modules
    hd = HandsDetector()
    tracker = None
    H0 = None
    H_curr = None
    board = None
    corners = []

    # Firebase (lazy) init
    if FB_PROJECT and FB_BUCKET and FirebaseUploader is not None:
        try:
            firebase_uploader = FirebaseUploader(project_id=FB_PROJECT, bucket_name=FB_BUCKET)
            print("[Firebase] Initialized")
        except Exception as e:
            print("[Firebase] init failed:", e)
            firebase_uploader = None

    # Mouse fallback
    def on_mouse(event, x, y, flags, param):
        nonlocal tracker, H0, H_curr, board, corners
        global state, mouse_draw, mouse_down, mouse_pos, dirty_since_save
        mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_down = True
            if H0 is None and len(corners) < 4:
                corners.append((x, y))
            elif H_curr is not None and mouse_draw:
                if state != "DRAW":
                    state = "DRAW"; board.begin(COLORS[color_idx], DRAW_WIDTH, mode="draw")
                bx, by = warp_point(H_curr, x, y); board.add_point(bx, by); dirty_since_save = True
        elif event == cv2.EVENT_MOUSEMOVE and mouse_down and mouse_draw and H_curr is not None:
            bx, by = warp_point(H_curr, x, y); board.add_point(bx, by); dirty_since_save = True
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_down = False
            if mouse_draw and state == "DRAW":
                board.end(); state = "IDLE"

    cv2.namedWindow("AirNote - Capture"); cv2.setMouseCallback("AirNote - Capture", on_mouse)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # Helper for bi-directional hotbar nav
    def cycle_hotbar(direction):
        global hotbar_idx, tool_mode, state  # <-- FIXED: use global, not nonlocal
        n = len(HOTBAR)
        hotbar_idx = (hotbar_idx + direction) % n
        sel = HOTBAR[hotbar_idx]
        if sel in ("DRAW","ERASE"):
            if state in ("DRAW","ERASE"):
                board.end(); state = "IDLE"
            tool_mode = sel
            set_action(f"Tool → {tool_mode}")
        else:
            set_action(f"Select → {sel}")

    while True:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Hands
        res = hd.process(rgb)
        hands = parse_hands(res, w, h)
        left = hands.get("Left")
        right = hands.get("Right")

        # Per-hand pinch booleans
        pinch = {
            "Left": bool(left and left["pinch"]),
            "Right": bool(right and right["pinch"]),
        }
        point_left = bool(left and left["point"])
        tip_left = fingertip(left["landmarks"]) if left else None
        tip_right = fingertip(right["landmarks"]) if right else None

        # Keys (fallbacks only)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: break  # ESC
        if key == ord('m'): mouse_draw = not mouse_draw
        if key == ord('g') and board is not None: board.show_grid = not board.show_grid
        if key == ord('b') and board is not None: board.dark_bg = not board.dark_bg
        if key == ord('r'):
            H0 = None; H_curr = None; corners = []; tracker = None; board = None; state = "IDLE"
            gate_hand = None; gate_down = False
            last_pinch = {"Left": False, "Right": False}
            pinch_down_ms = {"Left": 0, "Right": 0}
            smoothed_quad = None
            dirty_since_save = False

        # --- Plane selection ---
        if H0 is None:
            # Use whichever hand is visible to place corners; pinch to drop
            ret = (w//2, h//2)
            # prefer right index tip if available else left
            if right and right.get("landmarks"):
                ret = (right["landmarks"][8][0], right["landmarks"][8][1])
            elif left and left.get("landmarks"):
                ret = (left["landmarks"][8][0], left["landmarks"][8][1])

            draw_reticle(frame, ret, (0,255,255))
            cv2.putText(frame, f"Place corner {len(corners)+1}/4 (pinch either hand)",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            if (pinch["Left"] and not last_pinch["Left"] and len(corners) < 4) or \
               (pinch["Right"] and not last_pinch["Right"] and len(corners) < 4):
                corners.append(ret)

            last_pinch["Left"]  = pinch["Left"]
            last_pinch["Right"] = pinch["Right"]

            for i,p in enumerate(corners):
                cv2.circle(frame, p, 6, (0,200,0), -1)
                cv2.putText(frame, str(i+1), (p[0]+6,p[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)

            if len(corners) == 4:
                H0, _, _ = compute_H0(corners, BOARD_W, BOARD_H)
                gray0 = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                tracker = PlanarTracker(
                    corners, gray0,
                    inlier_thresh=INLIER_THRESH,
                    freeze_frames=FREEZE_FRAMES,
                    reseed_low_streak_thresh=RESEED_LOW_STREAK,
                    hysteresis_frames=HYSTERESIS_FRAMES
                )
                board = BoardCanvas(BOARD_W, BOARD_H)
                H_curr = H0
                smoothed_quad = np.array(corners, dtype=np.float32)
                dirty_since_save = False
                last_autosave_ms = now_ms()

            show = cv2.flip(frame, 1) if MIRROR_DISPLAY else frame
            cv2.imshow("AirNote - Capture", show); continue

        # --- Tracking update ---
        gray = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        Ht, inl = tracker.update(gray)

        if tracker.need_reseed:
            tracker.reseed(gray, quad=None)
        if tracker.just_reseeded:
            tracker.just_reseeded = False
            set_action("Reseeded features")

        ok_track = tracker.ok and Ht is not None
        if ok_track:
            cv2.putText(frame, f"Inliers: {inl}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 1)
            # Compute current quad, then EMA it for stability
            orig = np.float32([corners]).reshape(-1,1,2)
            curr = cv2.perspectiveTransform(orig, tracker.H_t).reshape(4,2)
            if smoothed_quad is None:
                smoothed_quad = curr.copy()
            else:
                smoothed_quad = (1-QUAD_SMOOTH_A)*smoothed_quad + QUAD_SMOOTH_A*curr
            # Recompute H_curr from smoothed corners
            H_curr, _, _ = compute_H0(smoothed_quad, BOARD_W, BOARD_H)
        else:
            cv2.putText(frame, "Re-lock (press r) - low track", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # ---------- Gate + bi-directional nav ----------
        # Gate becomes whichever hand is currently held pinched.
        if gate_hand is None:
            if pinch["Left"] ^ pinch["Right"]:
                gate_hand = "Left" if pinch["Left"] else "Right"
                gate_down = True
        else:
            if not pinch[gate_hand]:
                # gate released
                gate_hand = None
                gate_down = False

        # Tap/long-press on the OTHER hand (non-gate) controls nav/activation
        other = None
        if gate_hand == "Left" and right is not None:
            other = "Right"
        elif gate_hand == "Right" and left is not None:
            other = "Left"

        if other is not None and gate_down:
            # rising edge for other hand pinch
            if pinch[other] and not last_pinch[other]:
                pinch_down_ms[other] = now_ms()
            # falling edge (decide action)
            if (not pinch[other]) and last_pinch[other] and pinch_down_ms[other] > 0:
                dur = now_ms() - pinch_down_ms[other]
                pinch_progress = 0.0
                if dur < TOGGLE_TAP_MS:
                    # direction: Left tap => forward; Right tap => backward
                    direction = +1 if other == "Left" else -1
                    cycle_hotbar(direction)
                elif dur >= SAVE_LONG_MS:
                    sel = HOTBAR[hotbar_idx]
                    if sel == "DRAW":
                        tool_mode = "DRAW"; set_action("Tool → DRAW")
                    elif sel == "ERASE":
                        tool_mode = "ERASE"; set_action("Tool → ERASE")
                    elif sel == "COLOR":
                        color_idx = (color_idx + 1) % len(COLORS); set_action("Color changed"); dirty_since_save = True
                    elif sel == "WIDTH":
                        DRAW_WIDTH = min(28, DRAW_WIDTH + 2) if DRAW_WIDTH < 28 else 4
                        set_action(f"Width → {DRAW_WIDTH}"); dirty_since_save = True
                    elif sel == "UNDO" and board is not None:
                        board.undo(); set_action("Undo"); dirty_since_save = True
                    elif sel == "REDO" and board is not None:
                        board.redo(); set_action("Redo"); dirty_since_save = True
                    elif sel == "CLEAR" and board is not None:
                        board.clear_page(); set_action("Clear page"); dirty_since_save = True
                    elif sel == "NEW" and board is not None:
                        board.new_page(); set_action("New page"); dirty_since_save = True
                    elif sel == "SAVE" and board is not None:
                        curr_quad = smoothed_quad if smoothed_quad is not None else None
                        ok_save = do_save(board, session_id, H0, curr_quad, color_idx, DRAW_WIDTH, hotbar_idx,
                                          firebase_uploader, frame, H_curr, auto=False)
                        if ok_save:
                            set_action("Saved")
                            save_flash_until = now_ms() + 250
                            dirty_since_save = False
                            last_autosave_ms = now_ms()
                pinch_down_ms[other] = 0

            # progress ring for long-press
            if pinch[other] and pinch_down_ms[other] > 0:
                pinch_progress = min(1.0, (now_ms() - pinch_down_ms[other]) / float(SAVE_LONG_MS))
            else:
                pinch_progress = 0.0

        # Update last pinch states
        last_pinch["Left"]  = pinch["Left"]
        last_pinch["Right"] = pinch["Right"]

        # --- Tool execution (gate must be held) ---
        if board is not None and H_curr is not None and gate_down and gate_hand is not None:
            # draw/erase with LEFT hand fingertip for consistency (more precise)
            tip = tip_left if tip_left is not None else tip_right
            active_pose = (tool_mode == "ERASE") or (tool_mode == "DRAW" and point_left)
            if active_pose:
                if state != tool_mode:
                    if state in ("DRAW","ERASE"): board.end()
                    state = tool_mode
                    width = (ERASER_DIAM if tool_mode == "ERASE" else DRAW_WIDTH)
                    board.begin(COLORS[color_idx], width, mode=("draw" if tool_mode=="DRAW" else "erase"))
                if tip is not None:
                    xb, yb = warp_point(H_curr, tip[0], tip[1]); board.add_point(xb, yb)
                    dirty_since_save = True
            else:
                if state in ("DRAW","ERASE"):
                    board.end(); state = "IDLE"
        else:
            if state in ("DRAW","ERASE"):
                board.end(); state = "IDLE"

        # --- Autosave (if enabled) ---
        if board is not None and AUTOSAVE_MS > 0 and dirty_since_save:
            if now_ms() - last_autosave_ms >= AUTOSAVE_MS:
                curr_quad = smoothed_quad if smoothed_quad is not None else None
                ok_save = do_save(board, session_id, H0, curr_quad, color_idx, DRAW_WIDTH, hotbar_idx,
                                  firebase_uploader, frame, H_curr, auto=True)
                if ok_save:
                    set_action("Autosaved", ms=750)
                    dirty_since_save = False
                    last_autosave_ms = now_ms()

        # --- Overlay compose ---
        if board is None or H_curr is None:
            show = cv2.flip(frame, 1) if MIRROR_DISPLAY else frame
            cv2.imshow("AirNote - Capture", show); continue

        board_img = board.render()
        inv = np.linalg.inv(H_curr) if np.linalg.det(H_curr) != 0 else np.linalg.pinv(H_curr)
        overlay = cv2.warpPerspective(board_img, inv, (w, h))
        out = alpha_blend(frame, overlay)

        # Quad outline
        if smoothed_quad is not None:
            q = smoothed_quad.astype(int)
            cv2.polylines(out, [q], True, (0,255,0), 2)

        # HUD
        curr_fps = tick_fps()
        cv2.putText(out, f"State: {state}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0) if state!="IDLE" else (200,200,200), 2)
        cv2.putText(out, f"FPS: {curr_fps:.1f}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # Action toast
        if now_ms() < last_action_until and last_action_text:
            cv2.putText(out, last_action_text, (20, h-80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # Save flash (manual saves only)
        if now_ms() < save_flash_until:
            cv2.rectangle(out, (0,0), (w-1,h-1), (255,255,255), 10)

        # Progress ring over the OTHER hand fingertip (if any)
        tip_for_ring = tip_left if (gate_hand=="Right") else (tip_right if gate_hand=="Left" else None)
        if tip_for_ring is not None and pinch_progress > 0:
            draw_progress_ring(out, (tip_for_ring[0], tip_for_ring[1]), 16, pinch_progress, color=(0,200,255))

        # Modern bottom hotbar
        draw_hotbar(out, w, h, hotbar_idx, DRAW_WIDTH, COLORS[color_idx])

        # Small rectified preview (top-left)
        try:
            cam_preview = cv2.warpPerspective(frame, H_curr, (BOARD_W, BOARD_H))
            th = 150; tw = int(cam_preview.shape[1] * (th / cam_preview.shape[0]))
            cam_preview = cv2.resize(cam_preview, (tw, th))
            out[12:12+th, 12:12+tw] = cam_preview
        except Exception:
            pass

        show = cv2.flip(out, 1) if MIRROR_DISPLAY else out
        cv2.imshow("AirNote - Capture", show)

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
