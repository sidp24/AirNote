import argparse, time
import cv2
import numpy as np

from hand_state import HandsDetector, parse_hands, fingertip
from plane_select import draw_reticle, compute_H0
from tracker import PlanarTracker
from board_canvas import BoardCanvas, warp_point
from utils import alpha_blend

# ------ Board Defaults ------
BOARD_W, BOARD_H = 1200, 800
DRAW_WIDTH = 5
COLORS = [(0,255,0),(255,0,0),(0,140,255),(255,255,255)]  # BGR

# ------ Runtime state ------
color_idx = 0
state = "IDLE"       # IDLE, DRAW, ERASE
debounce_until = 0
DEBUG_DRAW = False
_last_t = None
fps = 0.0

# Hands-first UX state
tool_mode = "DRAW"          # active drawing tool: "DRAW" | "ERASE"
last_gate_pinch = False     # right-hand pinch (gate)
last_tool_pinch = False     # left-hand pinch (cycles/activates)
tool_pinch_down_ms = 0
tool_pinch_progress = 0.0   # 0..1 for long-press ring
SAVE_LONG_MS = 700
TOGGLE_TAP_MS = 300

# Hotbar items & selection
HOTBAR = ["DRAW", "ERASE", "COLOR", "WIDTH", "UNDO", "REDO", "CLEAR", "NEW", "SAVE"]
hotbar_idx = 0

# Mouse helpers (optional fallback)
mouse_draw = False
mouse_down = False
mouse_pos = (0,0)

def now_ms(): return int(time.time()*1000)

def can_transition(): return now_ms() > debounce_until

def set_debounce(ms=160):
    global debounce_until
    debounce_until = now_ms() + ms

def tick_fps():
    global _last_t, fps
    t = time.time()
    if _last_t is None:
        _last_t = t
        return 0.0
    dt = t - _last_t
    _last_t = t
    if dt > 0:
        fps = (0.9*fps + 0.1*(1.0/dt)) if fps > 0 else (1.0/dt)
    return fps

def draw_progress_ring(img, center, radius, t01, color=(0,255,255)):
    t01 = max(0.0, min(1.0, float(t01)))
    if t01 <= 0: return
    start_angle = -90
    end_angle = start_angle + int(360 * t01)
    cv2.ellipse(img, center, (radius, radius), 0, start_angle, end_angle, color, 2, lineType=cv2.LINE_AA)

def draw_hotbar(img, w, h, selected_idx, draw_width, color_bgr, font_scale=0.5):
    """Render a bottom hotbar with the current selection highlighted."""
    margin = 10
    bar_h = 58
    y0 = h - bar_h - margin
    cv2.rectangle(img, (margin, y0), (w - margin, h - margin), (30,30,30), thickness=2)

    items = HOTBAR
    n = len(items)
    cell_w = int((w - 2*margin) / n)

    for i, name in enumerate(items):
        x0 = margin + i*cell_w
        x1 = margin + (i+1)*cell_w
        # cell background
        if i == selected_idx:
            cv2.rectangle(img, (x0+1, y0+1), (x1-1, h - margin - 1), (50,100,50), thickness=-1)
        else:
            cv2.rectangle(img, (x0+1, y0+1), (x1-1, h - margin - 1), (25,25,25), thickness=-1)
        cv2.rectangle(img, (x0+1, y0+1), (x1-1, h - margin - 1), (100,100,100), thickness=1)

        label = name
        # augment label or draw icon for COLOR/WIDTH
        if name == "COLOR":
            # show current color swatch
            box_w, box_h = 20, 14
            bx = x0 + (cell_w - box_w)//2
            by = y0 + 6
            cv2.rectangle(img, (bx, by), (bx+box_w, by+box_h), color_bgr, thickness=-1)
            cv2.rectangle(img, (bx, by), (bx+box_w, by+box_h), (0,0,0), thickness=1)
            label = "COLOR"
        elif name == "WIDTH":
            label = f"W:{draw_width}"

        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        tx = x0 + (cell_w - tw)//2
        ty = y0 + bar_h - 16
        cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (220,220,220), 1)

def main():
    global state, color_idx, DRAW_WIDTH, DEBUG_DRAW
    global mouse_draw, mouse_down, mouse_pos
    global tool_mode, last_gate_pinch, last_tool_pinch, tool_pinch_down_ms, tool_pinch_progress
    global hotbar_idx

    # ---- CLI ----
    parser = argparse.ArgumentParser(description="AirNote Capture App")
    parser.add_argument("--cam", type=int, default=0, help="Camera index: 0=built-in, 1/2=external")
    parser.add_argument("--mirror", action="store_true", help="Mirror PREVIEW only (processing stays non-mirrored)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    args = parser.parse_args()
    MIRROR_DISPLAY = bool(args.mirror)

    # ---- Camera ----
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # ---- Modules ----
    hd = HandsDetector()
    tracker = None
    H0 = None
    H_curr = None
    board = None
    corners = []

    # ---- Mouse (optional fallback) ----
    def _on_mouse(event, x, y, flags, param):
        nonlocal tracker, H0, H_curr, board, corners
        global state, color_idx, DRAW_WIDTH, mouse_draw, mouse_down, mouse_pos

        mouse_pos = (x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_down = True
            if H0 is None and len(corners) < 4:
                corners.append((x, y))
            elif H_curr is not None and mouse_draw:
                if state != "DRAW":
                    state = "DRAW"
                    board.begin(COLORS[color_idx], DRAW_WIDTH, mode="draw")
                bx, by = warp_point(H_curr, x, y)
                board.add_point(bx, by)

        elif event == cv2.EVENT_MOUSEMOVE and mouse_down and mouse_draw and H_curr is not None:
            bx, by = warp_point(H_curr, x, y)
            board.add_point(bx, by)

        elif event == cv2.EVENT_LBUTTONUP:
            mouse_down = False
            if mouse_draw and state == "DRAW":
                board.end()
                state = "IDLE"

    cv2.namedWindow("AirNote - Capture")
    cv2.setMouseCallback("AirNote - Capture", _on_mouse)

    # ---- CLAHE for tracking ----
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Hands
        res = hd.process(rgb)
        hands = parse_hands(res, w, h)
        left = hands.get("Left")     # tool hand (pinch cycles/activates; point draws when DRAW)
        right = hands.get("Right")   # gate hand (pinch holds to arm)

        gate_pinch = bool(right and right["pinch"])            # right = gate
        tool_point = bool(left and left["point"])              # left index extend
        tool_pinch = bool(left and left["pinch"])              # left pinch
        tip = fingertip(left["landmarks"]) if left else None   # left fingertip

        # Minimal keys remain as backups only
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == ord('t'): DEBUG_DRAW = not DEBUG_DRAW
        if key == ord('m'): mouse_draw = not mouse_draw
        if key == ord('g') and board is not None: board.show_grid = not board.show_grid
        if key == ord('b') and board is not None: board.dark_bg = not board.dark_bg
        if key == ord('R') and tracker is not None:
            gray_now = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            tracker.reseed(gray_now, quad=None)
        if key == ord('r'):
            H0 = None; H_curr = None; corners = []; tracker = None; board = None
            state = "IDLE"
            last_gate_pinch = last_tool_pinch = False
            tool_pinch_down_ms = 0; tool_pinch_progress = 0.0
            tool_mode = "DRAW"; hotbar_idx = 0

        # ---------- Plane selection ----------
        if H0 is None:
            ret = (w//2, h//2)
            if right and right.get("landmarks"):
                ret = (right["landmarks"][8][0], right["landmarks"][8][1])
            draw_reticle(frame, ret, (0,255,255))
            cv2.putText(frame, f"Place corner {len(corners)+1}/4 (right-pinch or mouse click)",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            if gate_pinch and not last_gate_pinch and len(corners) < 4:
                corners.append(ret)
            last_gate_pinch = gate_pinch

            for i,p in enumerate(corners):
                cv2.circle(frame, p, 6, (0,200,0), -1)
                cv2.putText(frame, str(i+1), (p[0]+6,p[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)

            if len(corners) == 4:
                H0, _, _ = compute_H0(corners, BOARD_W, BOARD_H)
                gray0 = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                tracker = PlanarTracker(corners, gray0)
                board = BoardCanvas(BOARD_W, BOARD_H)
                H_curr = H0
                last_gate_pinch = False

            show = cv2.flip(frame, 1) if MIRROR_DISPLAY else frame
            cv2.imshow("AirNote - Capture", show)
            continue

        # ---------- Tracking update ----------
        gray = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        Ht, inl = tracker.update(gray)
        ok_track = tracker.ok and Ht is not None
        if ok_track:
            cv2.putText(frame, f"Track inliers: {inl}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 1)
            corners_np = np.float32([corners]).reshape(-1,1,2)
            curr = cv2.perspectiveTransform(corners_np, tracker.H_t).reshape(4,2)
            H_curr, _, _ = compute_H0(curr, BOARD_W, BOARD_H)
        else:
            cv2.putText(frame, "Re-lock (press R) - low track",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # ---------- Hotbar gestures ----------
        # Cycle/activate happen whenever gate is held (to avoid accidental pinches when resting).
        if gate_pinch:
            if tool_pinch and not last_tool_pinch:
                tool_pinch_down_ms = now_ms()
            if (not tool_pinch) and last_tool_pinch:
                dur = now_ms() - tool_pinch_down_ms
                tool_pinch_progress = 0.0
                tool_pinch_down_ms = 0
                if dur < TOGGLE_TAP_MS:
                    # quick tap -> select next item
                    hotbar_idx = (hotbar_idx + 1) % len(HOTBAR)
                    # NEW: if the newly selected item is a tool, make it active immediately
                    sel = HOTBAR[hotbar_idx]
                    if sel in ("DRAW", "ERASE"):
                        # end any current stroke, switch tool
                        if state in ("DRAW","ERASE"):
                            board.end()
                            state = "IDLE"
                        tool_mode = sel
                elif dur >= SAVE_LONG_MS:
                    # long press -> activate selected item
                    sel = HOTBAR[hotbar_idx]
                    if sel == "DRAW":
                        tool_mode = "DRAW"
                    elif sel == "ERASE":
                        tool_mode = "ERASE"
                    elif sel == "COLOR":
                        color_idx = (color_idx + 1) % len(COLORS)
                    elif sel == "WIDTH":
                        DRAW_WIDTH = min(24, DRAW_WIDTH + 2) if DRAW_WIDTH < 24 else 4
                    elif sel == "UNDO" and board is not None:
                        board.undo()
                    elif sel == "REDO" and board is not None:
                        board.redo()
                    elif sel == "CLEAR" and board is not None:
                        board.clear_page()
                    elif sel == "NEW" and board is not None:
                        board.new_page()
                    elif sel == "SAVE" and board is not None:
                        curr_quad = cv2.perspectiveTransform(
                            np.float32([corners]).reshape(-1,1,2), tracker.H_t
                        ).reshape(4,2)
                        board.save(H0=H0, curr_quad=curr_quad)

            if tool_pinch and tool_pinch_down_ms > 0:
                tool_pinch_progress = min(1.0, (now_ms() - tool_pinch_down_ms) / float(SAVE_LONG_MS))
            else:
                tool_pinch_progress = 0.0

        last_tool_pinch = tool_pinch
        last_gate_pinch = gate_pinch

        # ---------- Tool execution (drawing/erasing) ----------
        if board is not None and H_curr is not None and gate_pinch:
            # enter/maintain DRAW/ERASE when left hand pose matches selected tool
            active_pose = False
            if tool_mode == "DRAW":
                active_pose = tool_point
            elif tool_mode == "ERASE":
                active_pose = True  # eraser does not require pointing; any movement with gate on

            if active_pose:
                if can_transition() and state != tool_mode:
                    if state in ("DRAW","ERASE"):
                        board.end()
                    state = tool_mode
                    board.begin(COLORS[color_idx], DRAW_WIDTH, mode=("draw" if tool_mode=="DRAW" else "erase"))
                if tip is not None and state in ("DRAW","ERASE"):
                    xb, yb = warp_point(H_curr, tip[0], tip[1])
                    board.add_point(xb, yb)
            else:
                if state in ("DRAW","ERASE"):
                    board.end()
                    state = "IDLE"
        else:
            if state in ("DRAW","ERASE"):
                board.end()
                state = "IDLE"

        # ---------- Overlay ----------
        if board is None or H_curr is None:
            show = cv2.flip(frame, 1) if MIRROR_DISPLAY else frame
            cv2.imshow("AirNote - Capture", show)
            continue

        board_img = board.render()
        inv = np.linalg.inv(H_curr) if np.linalg.det(H_curr) != 0 else np.linalg.pinv(H_curr)
        overlay = cv2.warpPerspective(board_img, inv, (w, h))
        out = alpha_blend(frame, overlay)

        # Quad outline
        if tracker is not None and tracker.H_t is not None and len(corners) == 4:
            orig = np.float32([corners]).reshape(-1,1,2)
            curr = cv2.perspectiveTransform(orig, tracker.H_t).reshape(4,2).astype(int)
            cv2.polylines(out, [curr], isClosed=True, color=(0,255,0), thickness=2)

        # DEBUG points
        if DEBUG_DRAW and tracker is not None and tracker.last_good1 is not None:
            pts = tracker.last_good1.astype(int)
            mask = tracker.last_inliers_mask
            for i, p in enumerate(pts):
                col = (0,255,0) if (mask is not None and mask[i]) else (0,0,255)
                cv2.circle(out, (p[0], p[1]), 2, col, -1)
            if tracker.last_good0 is not None:
                prev = tracker.last_good0.astype(int)
                for p0, p1 in zip(prev, pts):
                    cv2.line(out, (p0[0], p0[1]), (p1[0], p1[1]), (180,180,180), 1)

        # HUD
        curr_fps = tick_fps()
        cv2.putText(out, f"State: {state}", (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0) if state!="IDLE" else (200,200,200), 2)
        cv2.putText(out, f"FPS: {curr_fps:.1f}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # Tool label near fingertip + long-press ring
        if tip is not None:
            label = tool_mode
            cv2.putText(out, label, (tip[0]+14, tip[1]-14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            if tool_pinch_progress > 0:
                draw_progress_ring(out, (tip[0], tip[1]), 14, tool_pinch_progress, color=(0,200,255))

        # Bottom hotbar
        draw_hotbar(out, w, h, hotbar_idx, DRAW_WIDTH, COLORS[color_idx])

        # Thumbnail (top-left)
        thumb = cv2.cvtColor(board_img, cv2.COLOR_BGRA2BGR)
        th = 150
        tw = int(thumb.shape[1] * (th / thumb.shape[0]))
        thumb = cv2.resize(thumb, (tw, th))
        out[12:12+th, 12:12+tw] = thumb

        show = cv2.flip(out, 1) if MIRROR_DISPLAY else out
        cv2.imshow("AirNote - Capture", show)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    