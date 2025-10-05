import argparse, time, uuid
import threading  # for non-blocking voice → AI flow
import cv2
import numpy as np

from hand_state import HandsDetector, parse_hands, fingertip
from plane_select import draw_reticle, compute_H0
from tracker import PlanarTracker
from board_canvas import BoardCanvas, warp_point
from utils import alpha_blend, roi_gaussian_frosted  # <- frosted HUD panel
from ai_client import ask_gemini  # <- Gemini client
from voice_io import record_push_to_talk, transcribe_pcm16, speak


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
HOTBAR = ["DRAW", "ERASE", "COLOR", "WIDTH", "UNDO", "REDO", "CLEAR", "NEW", "SAVE", "ASK"]
TOGGLE_TAP_MS = 300
SAVE_LONG_MS = 700
PINCH_DEBOUNCE_MS = 200  # minimum ms between recognized pinch rising edges per hand
TIP_SMOOTH_A = 0.35      # fingertip exponential smoothing (0-1); higher = more responsive
TEMPORAL_SMOOTH_A = 0.0  # frame-to-frame camera smoothing (0 disables)
PROGRESS_SMOOTH_A = 0.3  # smoothing for long-press progress ring

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


ingest_timestamp_mode="epoch"

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
last_pinch_rise_ms = {"Left":0, "Right":0}  # for debounce
pinch_progress_smooth = 0.0

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
FB_PREFIX = "snapshots"
FB_FS_ROOT = "sessions"
FB_FS_SUBCOL = "saves"   # <--- add this


# Autosave
AUTOSAVE_MS = 5000
last_autosave_ms = 0
dirty_since_save = False

# Save prefixes
PREFIX_MANUAL = "save"
PREFIX_AUTO   = "auto"

# Smoothed quad
smoothed_quad = None
smoothed_tip = None
prev_frame_smooth = None
stroke_primed = False  # ensures first frame after resume doesn't interpolate from old position

# AI HUD state
ai_text = ""
ai_text_until = 0
_frost = roi_gaussian_frosted

# --- Cross-platform key state for 'V' (Windows uses GetAsyncKeyState; else falls back to waitKey) ---
def _make_is_v_down():
    try:
        import sys
        if sys.platform.startswith("win"):
            import ctypes
            user32 = ctypes.windll.user32
            VK_V = 0x56
            def is_down():
                return (user32.GetAsyncKeyState(VK_V) & 0x8000) != 0
            return is_down
    except Exception:
        pass
    # Fallback: requires OpenCV window focus
    def is_down():
        k = cv2.waitKey(1) & 0xFF
        return k in (ord('v'), ord('V'))
    return is_down

IS_V_DOWN = _make_is_v_down()

def now_ms(): return int(time.time()*1000)

# --- small helper: build composite once (reuse your do_save logic style) ---
def build_composite_for_ai(board, frame_bgr, H_curr):
    try:
        cam_rect = cv2.warpPerspective(frame_bgr, H_curr, (board.W, board.H))
        strokes = board._render_to(include_grid=False)  # BGRA
        sb, sg, sr, sa = cv2.split(strokes)
        sa_f = sa.astype("float32")/255.0; inv = 1.0 - sa_f
        comp_b = (sb.astype("float32")*sa_f + cam_rect[:,:,0].astype("float32")*inv).astype("uint8")
        comp_g = (sg.astype("float32")*sa_f + cam_rect[:,:,1].astype("float32")*inv).astype("uint8")
        comp_r = (sr.astype("float32")*sa_f + cam_rect[:,:,2].astype("float32")*inv).astype("uint8")
        return cv2.merge([comp_b, comp_g, comp_r])
    except Exception:
        return cv2.cvtColor(board._render_to(include_grid=False), cv2.COLOR_BGRA2BGR)
    
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
    elif name == "ASK":
        cv2.circle(img, (cx-6,cy-2), 5, color, 2, cv2.LINE_AA)
        cv2.line(img, (cx-1,cy-2), (cx+10,cy-2), color, 2, cv2.LINE_AA)
        cv2.circle(img, (cx+13,cy-2), 3, color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx-6,cy+9), 2, color, -1, cv2.LINE_AA)

def draw_hotbar(img, w, h, selected_idx, draw_width, color_bgr, t_sec=0.0):
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

    pulse = 0.5 + 0.5*np.sin(t_sec*3.5)  # 0..1
    for i, name in enumerate(HOTBAR):
        cx0 = x0 + gap + i*(cell_w + gap)
        cx1 = cx0 + cell_w
        btn_radius = 12

        active = (i == selected_idx)
        if active:
            g = int(60 + 70*pulse)
            base = (60, g, 60)
        else:
            base = (55,55,55)
        _aa_roundrect(img, (cx0, y0+y_pad), (cx1, h - margin - y_pad), btn_radius, base, -1)

        if active:
            glow = img.copy()
            glow_col = (90, 190, 90)
            _aa_roundrect(glow, (cx0, y0+y_pad), (cx1, h - margin - y_pad), btn_radius, glow_col, -1)
            cv2.addWeighted(glow, 0.18 + 0.07*pulse, img, 1-(0.18+0.07*pulse), 0, img)

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

def draw_wrapped_text(img, text, x, y, max_w, line_h=22, color=(255,255,255), scale=0.6, thickness=1):
    if not text: return y
    words = text.split()
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        size,_ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if size[0] > max_w and line:
            cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
            y += line_h
            line = w
        else:
            line = test
    if line:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += line_h
    return y

# ---------- Geometry helpers ----------
def auto_square_corners(corners_list):
    """Return best-fit rectangle corners (tl,tr,br,bl) using minAreaRect if 4 points.
    Keeps orientation by reordering via sum/diff heuristic."""
    if len(corners_list) != 4:
        return corners_list
    pts = np.array(corners_list, dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    # order corners
    tl = min(box, key=lambda p: p[0]+p[1])
    br = max(box, key=lambda p: p[0]+p[1])
    tr = min(box, key=lambda p: -p[0]+p[1])
    bl = max(box, key=lambda p: -p[0]+p[1])
    ordered = [tuple(map(int, tl)), tuple(map(int, tr)), tuple(map(int, br)), tuple(map(int, bl))]
    return ordered

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
# after board.save(), where you already have comp_path, json_path, meta_path
            firebase_uploader.upload_save(
                session_id=session_id,
                save_ts=int(time.time()),
                files={
                    "composite_jpg": comp_path,   # whatever board.save wrote; we’ll re-encode to PNG for POST
                    "strokes_json": json_path,
                    "meta_json": meta_path,
                },
                meta={
                    "W": board.W, "H": board.H,
                    "page_idx": board.page_idx,
                    "page_count": board.page_count(),
                    "color_idx": color_idx,
                    "draw_width": DRAW_WIDTH,
                    "hotbar_idx": hotbar_idx,
                    "H0": H0.tolist() if H0 is not None else None,
                    "curr_quad": curr_quad.tolist() if curr_quad is not None else None,
                    "session_id": session_id,
                    
                },
                make_public=FB_PUBLIC,
                extra_fields={"source": "AirNote", "auto": bool(auto), "prefix": prefix},

                # local ingest EXACTLY like curl
                ingest_url="http://127.0.0.1:5050/ingest_note",
                use_gemini=False,           # not needed for strict curl parity
                gemini_host="127.0.0.1",
                gemini_port=8000,
                ingest_minimal=True,        # <-- IMPORTANT
                ingest_force_png=True,       # <-- IMPORTANT
                ingest_timestamp_mode="bucket10"  # default; mirrors your curl's %TS:~0,10%
            )

            




        return True
    except Exception as e:
        print("[Save] failed:", e)
        return False

# (Removed legacy text prompt + ask helper; replaced by voice-first flow inside gesture handler.)

def main():
    global state, color_idx, DRAW_WIDTH, tool_mode, hotbar_idx
    global gate_hand, gate_down, last_pinch, pinch_down_ms, pinch_progress, pinch_progress_smooth
    global last_action_text, last_action_until, save_flash_until
    global mouse_draw, mouse_down, mouse_pos
    global firebase_uploader, FB_PROJECT, FB_BUCKET, FB_PUBLIC, FB_PREFIX, FB_FS_ROOT, FB_FS_SUBCOL
    global AUTOSAVE_MS, last_autosave_ms, dirty_since_save, smoothed_quad, TIP_SMOOTH_A, TEMPORAL_SMOOTH_A, smoothed_tip, stroke_primed
    global ai_text, ai_text_until

    # CLI
    # ap = argparse.ArgumentParser(description="AirNote Capture App")
    # ap.add_argument("--cam", type=int, default=0)
    # ap.add_argument("--mirror", action="store_true")
    # ap.add_argument("--width", type=int, default=1280)
    # ap.add_argument("--height", type=int, default=720)
    # ap.add_argument("--fb_project", type=str, default="")
    # ap.add_argument("--fb_bucket", type=str, default="")
    # ap.add_argument("--fb_public", action="store_true")
    # ap.add_argument("--fb_prefix", type=str, default="snapshots",
    #                 help="Storage object prefix (folder-like), e.g., 'Airnote'")
    # ap.add_argument("--fb_fs_root", type=str, default="sessions",
    #                 help="Firestore root collection, e.g., 'Airnote'")
    # ap.add_argument("--autosave_sec", type=float, default=5.0)
    # ap.add_argument("--no_auto_square", action="store_true", help="Disable auto rectangle correction after 4 corners")
    # ap.add_argument("--tip_smooth", type=float, default=TIP_SMOOTH_A, help="Override fingertip smoothing alpha (0-1)")
    # ap.add_argument("--temporal_smooth", type=float, default=TEMPORAL_SMOOTH_A, help="Temporal frame smoothing alpha (0-1, small blur)")
    # ap.add_argument("--fb_fs_subcol", type=str, default="saves",
    #             help="Firestore subcollection under each session doc, e.g., 'notes'")

    ap = argparse.ArgumentParser(description="AirNote Capture App")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)

    # Firebase flags
    ap.add_argument("--fb_project", type=str, default="")
    ap.add_argument("--fb_bucket", type=str, default="")
    ap.add_argument("--fb_public", action="store_true")

    # Where to store in GCS and Firestore
    ap.add_argument("--fb_prefix",   type=str, default="snapshots",
                    help="Cloud Storage prefix (folder-like), e.g., 'notes'")
    ap.add_argument("--fb_fs_root",  type=str, default="sessions",
                    help="Firestore root collection, e.g., 'Airnote'")
    ap.add_argument("--fb_fs_subcol", type=str, default="saves",
                    help="Firestore subcollection under each session doc, e.g., 'notes'")

    ap.add_argument("--autosave_sec", type=float, default=5.0)
    ap.add_argument("--no_auto_square", action="store_true",
                    help="Disable auto rectangle correction after 4 corners")
    ap.add_argument("--tip_smooth", type=float, default=TIP_SMOOTH_A,
                    help="Override fingertip smoothing alpha (0-1)")
    ap.add_argument("--temporal_smooth", type=float, default=TEMPORAL_SMOOTH_A,
                    help="Temporal frame smoothing alpha (0-1, small blur)")

    args = ap.parse_args()

    MIRROR_DISPLAY = bool(args.mirror)
    FB_PROJECT     = args.fb_project
    FB_BUCKET      = args.fb_bucket
    FB_PUBLIC      = bool(args.fb_public)
    FB_PREFIX      = args.fb_prefix
    FB_FS_ROOT     = args.fb_fs_root
    FB_FS_SUBCOL   = args.fb_fs_subcol
    AUTOSAVE_MS    = max(0, int(args.autosave_sec * 1000))
    AUTO_SQUARE    = (not args.no_auto_square)
    # clamp alphas
    TIP_SMOOTH_A      = max(0.0, min(1.0, float(args.tip_smooth)))
    TEMPORAL_SMOOTH_A = max(0.0, min(0.95, float(args.temporal_smooth)))

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
# main.py — Firebase init
            firebase_uploader = FirebaseUploader(
                project_id=FB_PROJECT,
                bucket_name=FB_BUCKET,
                storage_prefix=FB_PREFIX,  # e.g., "notes"
                fs_root=FB_FS_ROOT,        # e.g., "Airnote"
                fs_subcol=FB_FS_SUBCOL     # e.g., "notes"
            )

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
        global hotbar_idx, tool_mode, state
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
        # Optional temporal smoothing (simple EMA to mitigate flicker & noise)
        global prev_frame_smooth
        if TEMPORAL_SMOOTH_A > 0.0 and prev_frame_smooth is not None:
            frame = cv2.addWeighted(frame, 1-TEMPORAL_SMOOTH_A, prev_frame_smooth, TEMPORAL_SMOOTH_A, 0)
        prev_frame_smooth = frame.copy()
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
            ai_text = ""; ai_text_until = 0

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

            now = now_ms()
            # Debounced corner placement: only accept rising edge if sufficient time since last
            if len(corners) < 4:
                for hand in ("Left","Right"):
                    if pinch[hand] and not last_pinch[hand]:
                        if now - last_pinch_rise_ms[hand] >= PINCH_DEBOUNCE_MS:
                            corners.append(ret)
                            last_pinch_rise_ms[hand] = now
                            break

            last_pinch["Left"]  = pinch["Left"]
            last_pinch["Right"] = pinch["Right"]

            for i,p in enumerate(corners):
                cv2.circle(frame, p, 6, (0,200,0), -1)
                cv2.putText(frame, str(i+1), (p[0]+6,p[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)

            if len(corners) == 4:
                if AUTO_SQUARE:
                    sq = auto_square_corners(corners)
                    if sq and len(sq) == 4:
                        corners = sq
                        set_action("Auto-squared")
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
                # rising edge with debounce
                tnow = now_ms()
                if tnow - last_pinch_rise_ms[other] >= PINCH_DEBOUNCE_MS:
                    pinch_down_ms[other] = tnow
                    last_pinch_rise_ms[other] = tnow
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
                        ok_save = do_save(
                                board, session_id, H0, curr_quad, color_idx, DRAW_WIDTH, hotbar_idx,
                                firebase_uploader, frame, H_curr, auto=False
                    )

                        if ok_save:
                            set_action("Saved")
                            save_flash_until = now_ms() + 250
                            dirty_since_save = False
                            last_autosave_ms = now_ms()
                    elif sel == "ASK" and board is not None and H_curr is not None:
                        # Non-blocking voice ask now handled by global V key listener each frame.
                        # Long-press merely arms ASK mode and gives user instruction.
                        set_action("ASK armed – hold V and speak", ms=1400)
                pinch_down_ms[other] = 0

            # progress ring for long-press
            if pinch[other] and pinch_down_ms[other] > 0:
                pinch_progress = min(1.0, (now_ms() - pinch_down_ms[other]) / float(SAVE_LONG_MS))
                # smooth for rendering
                pinch_progress_smooth = (1-PROGRESS_SMOOTH_A)*pinch_progress_smooth + PROGRESS_SMOOTH_A*pinch_progress
            else:
                pinch_progress = 0.0
                pinch_progress_smooth = (1-PROGRESS_SMOOTH_A)*pinch_progress_smooth

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
                    # Reset smoothing so we don't drag a line from old stroke location
                    smoothed_tip = None
                    stroke_primed = False
                if tip is not None:
                    # fingertip smoothing for steadier strokes
                    if smoothed_tip is None:
                        smoothed_tip = np.array(tip, dtype=np.float32)
                        if not stroke_primed:
                            xb0, yb0 = warp_point(H_curr, tip[0], tip[1]); board.add_point(xb0, yb0)
                            stroke_primed = True
                    else:
                        smoothed_tip = (1-TIP_SMOOTH_A)*smoothed_tip + TIP_SMOOTH_A*np.array(tip, dtype=np.float32)
                        tip_use = (int(smoothed_tip[0]), int(smoothed_tip[1]))
                        xb, yb = warp_point(H_curr, tip_use[0], tip_use[1]); board.add_point(xb, yb)
                    dirty_since_save = True
            else:
                if state in ("DRAW","ERASE"):
                    board.end(); state = "IDLE"; smoothed_tip = None; stroke_primed = False
        else:
            if state in ("DRAW","ERASE"):
                board.end(); state = "IDLE"; smoothed_tip = None; stroke_primed = False

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

        # --- Voice / AI Flow (non-blocking push-to-talk) ---
        # Trigger only when ASK hotbar item is selected.
        if HOTBAR[hotbar_idx] == "ASK" and board is not None and H_curr is not None:
            is_v_down = IS_V_DOWN()
            # Rising edge: start listening thread
            if is_v_down and state != "ASK_VOICE_LISTENING":
                print("[Voice] PTT down. Starting listen loop…")
                state = "ASK_VOICE_LISTENING"
                # Capture composite & H snapshot for thread (avoid mutation issues)
                composite_for_ai = build_composite_for_ai(board, frame, H_curr)

                def _ask_thread_target(comp_img, H_snapshot):
                    global ai_text, ai_text_until, state
                    try:
                        print("[Voice Thread] Starting recording…")
                        pcm_bytes, sr_local = record_push_to_talk(IS_V_DOWN, start_timeout_sec=0.1)
                        if not pcm_bytes:
                            ai_text = "(AI) No audio captured."; ai_text_until = now_ms() + 2500
                            speak(ai_text)
                            return
                        ai_text = "(AI) Transcribing…"; ai_text_until = now_ms() + 10000
                        question = transcribe_pcm16(pcm_bytes, sr_local).strip()
                        print(f"[Voice Thread] Transcription: '{question}'")
                        if not question:
                            ai_text = "(AI) Didn't catch that."; ai_text_until = now_ms() + 2200
                            speak(ai_text)
                            return
                        ai_text = f"(AI) Thinking: {question[:60]}"; ai_text_until = now_ms() + 12000
                        system_hint = (
                            "You are a vision-based assistant answering the user's question about the image. "
                            "Answer directly; be concise unless elaboration is requested."
                        )
                        answer = ask_gemini(
                            comp_img,
                            question,
                            system_hint=system_hint,
                            max_chars=600,
                        )

                        ai_text = answer or "(AI) No answer."; ai_text_until = now_ms() + 10000
                        if answer:
                            try: speak(answer)
                            except Exception: pass
                        print(f"[AI Thread] Answer: {answer}")
                    except Exception as e:
                        err = f"(AI Thread Error) {str(e)[:100]}"; print(err)
                        ai_text = err; ai_text_until = now_ms() + 3500
                    finally:
                        state = "IDLE"

                threading.Thread(target=_ask_thread_target, args=(composite_for_ai, H_curr), daemon=True).start()

            # Falling edge: show processing toast (thread will reset state when done)
            if (not is_v_down) and state == "ASK_VOICE_LISTENING":
                set_action("AI: Processing voice & image…", ms=1500)

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
            draw_progress_ring(out, (tip_for_ring[0], tip_for_ring[1]), 16, pinch_progress_smooth, color=(0,200,255))

        # Modern bottom hotbar
        draw_hotbar(out, w, h, hotbar_idx, DRAW_WIDTH, COLORS[color_idx], t_sec=time.time())

        # --- AI pill (top-right) ---
        if now_ms() < ai_text_until and ai_text:
            pad = 14
            box_w = min(520, w - 2*pad)
            x0, y0 = w - box_w - pad, pad
            try:
                _frost(out, x0, y0, x0+box_w, y0+72, alpha=0.78, ksize=19)
            except Exception:
                cv2.rectangle(out, (x0,y0), (x0+box_w,y0+72), (32,32,32), -1)
            cv2.rectangle(out, (x0,y0), (x0+box_w,y0+72), (60,60,60), 1, cv2.LINE_AA)
            cv2.putText(out, "Gemini", (x0+14, y0+26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2, cv2.LINE_AA)
            txt = ai_text.strip().replace("\n", " ")
            (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            maxw = box_w - 28
            while tw > maxw and len(txt) > 4:
                txt = txt[:-2] + "…"
                (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(out, txt, (x0+14, y0+48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240,240,240), 2, cv2.LINE_AA)

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

# ------------------------------
# Future improvements (TODO):
# 1. Replace fingertip EMA with per-axis 1€ filter (better dynamic smoothing).
# 2. Implement Kalman filter over homography for predictive tracking under occlusion.
# 3. Adaptive reseed based on spatial feature dispersion + motion magnitude.
# 4. Multi-scale ORB/FAST mix to stabilize at varying distances.
# 5. Optional GPU path for warps & blending for higher FPS on large boards.
# 6. Gesture classifier (e.g., circle for undo) to reduce hotbar dependency on small screens.
# 7. Push-to-talk: stream mic → /ask with STT, auto-fill question.
# ------------------------------