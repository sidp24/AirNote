import argparse, time
import cv2
import numpy as np

from hand_state import HandsDetector, parse_hands, fingertip
from plane_select import draw_reticle, compute_H0
from tracker import PlanarTracker
from board_canvas import BoardCanvas, warp_point
from utils import alpha_blend

# ------ Defaults ------
BOARD_W, BOARD_H = 1200, 800
DRAW_WIDTH = 5
COLORS = [(0,255,0),(255,0,0),(0,140,255),(255,255,255)]

# ------ Runtime state ------
color_idx = 0
state = "IDLE"       # IDLE, DRAW, ERASE
last_left_pinch = False
debounce_until = 0
DEBUG_DRAW = False
_last_t = None
fps = 0.0

def now_ms():
    return int(time.time()*1000)

def can_transition():
    return now_ms() > debounce_until


def set_debounce(ms=160):
    global debounce_until
    debounce_until = now_ms() + ms

def tick_fps():
    """Exponential-smoothed FPS for display."""
    debug_mode = False
    global _last_t, fps
    t = time.time()
    if _last_t is None:
        _last_t = t
        return 0.0
    dt = t - _last_t
    _last_t = t
    if dt > 0:
        if fps <= 0:
            fps = 1.0/dt
        else:
            fps = 0.9*fps + 0.1*(1.0/dt)
    return fps

def main():
    global state, color_idx, DRAW_WIDTH, last_left_pinch, DEBUG_DRAW

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
    mouse_draw = False
    mouse_down = False
    last_mouse = None

    # mouse callback: maps clicks to image coords and then to board coords when H_curr exists
    def _on_mouse(event, x, y, flags, param):
        nonlocal mouse_down, last_mouse
        if not mouse_draw:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_down = True
            last_mouse = (x,y)
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_down = False
            last_mouse = None
        elif event == cv2.EVENT_MOUSEMOVE and mouse_down and H_curr is not None:
            # warp image point (x,y) to board space
            try:
                bx, by = warp_point(H_curr, x, y)
                board.add_point(bx, by)
            except Exception:
                pass

    cv2.namedWindow("AirNote - Capture")
    cv2.setMouseCallback("AirNote - Capture", _on_mouse)

    # ---- Contrast improver for tracking (helps low texture / lighting) ----
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # IMPORTANT: do NOT flip for processing (glasses POV)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Hands
        res = hd.process(rgb)
        hands = parse_hands(res, w, h)
        left = hands.get("Left"); right = hands.get("Right")
        left_pinch = bool(left and left["pinch"])
        right_point = bool(right and right["point"])
        right_pinch = bool(right and right["pinch"])
        tip = fingertip(right["landmarks"]) if right else None

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == ord('c'):
            color_idx = (color_idx + 1) % len(COLORS)
        if key == ord('e'):
            state = "ERASE" if state != "ERASE" else "IDLE"
        if key == ord('['):
            DRAW_WIDTH = max(1, DRAW_WIDTH - 1)
        if key == ord(']'):
            DRAW_WIDTH += 1
        if key == ord('t'):
            DEBUG_DRAW = not DEBUG_DRAW

        # ---------- Plane selection ----------
        if H0 is None:
            # Reticle follows left index tip if available
            ret = (w//2, h//2)
            if left and left.get("landmarks"):
                ret = (left["landmarks"][8][0], left["landmarks"][8][1])
            draw_reticle(frame, ret, (0,255,255))
            cv2.putText(frame, f"Place corner {len(corners)+1}/4 with LEFT PINCH",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            if left_pinch and not last_left_pinch:
                corners.append(ret)
            last_left_pinch = left_pinch

            for i,p in enumerate(corners):
                cv2.circle(frame, p, 6, (0,200,0), -1)
                cv2.putText(frame, str(i+1), (p[0]+6,p[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)

            if len(corners) == 4:
                H0, _, _ = compute_H0(corners, BOARD_W, BOARD_H)
                gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray0 = clahe.apply(gray0)
                tracker = PlanarTracker(corners, gray0)
                board = BoardCanvas(BOARD_W, BOARD_H)
                H_curr = H0
                last_left_pinch = False

            show = cv2.flip(frame, 1) if MIRROR_DISPLAY else frame
            cv2.imshow("AirNote - Capture", show)
            continue

        # ---------- Tracking update ----------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        Ht, inl = tracker.update(gray)
        ok_track = tracker.ok and Ht is not None
        if ok_track:
            cv2.putText(frame, f"Track inliers: {inl}",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 1)
            # Warp original corners by current motion, then recompute image->board homography
            corners_np = np.float32([corners]).reshape(-1,1,2)
            curr = cv2.perspectiveTransform(corners_np, tracker.H_t).reshape(4,2)
            H_curr, _, _ = compute_H0(curr, BOARD_W, BOARD_H)
        else:
            cv2.putText(frame, "Re-lock (press R) - low track",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        if key == ord('r'):
            H0 = None

        # ---------- Tool FSM (left pinch gates actions) ----------
        if left_pinch:
            if right_point and state != "DRAW" and can_transition():
                state = "DRAW"; set_debounce()
                board.begin(COLORS[color_idx], DRAW_WIDTH)
            elif right_pinch and state != "ERASE" and can_transition():
                state = "ERASE"; set_debounce()
            elif not right_point and not right_pinch and state != "IDLE" and can_transition():
                if state == "DRAW":
                    board.end()
                state = "IDLE"; set_debounce()
        else:
            if state == "DRAW":
                board.end()
            state = "IDLE"

        # ---------- Draw / erase in board space ----------
        if tip is not None and H_curr is not None:
            x_b, y_b = warp_point(H_curr, tip[0], tip[1])
            if state == "DRAW":
                board.add_point(x_b, y_b)
            elif state == "ERASE":
                board.erase_at(x_b, y_b, r=18)

        # ---------- Compose AR overlay ----------
        board_img = board.render()
        inv = np.linalg.pinv(H_curr) if H_curr is not None else np.eye(3)
        overlay = cv2.warpPerspective(board_img, inv, (w, h))
        out = alpha_blend(frame, overlay)

        # ---------- DEBUG OVERLAY ----------
        if DEBUG_DRAW and tracker is not None:
            # Show current quad estimate (green)
            if tracker.H_t is not None and len(corners) == 4:
                orig = np.float32([corners]).reshape(-1,1,2)
                curr = cv2.perspectiveTransform(orig, tracker.H_t).reshape(4,2).astype(int)
                cv2.polylines(out, [curr], isClosed=True, color=(0,255,0), thickness=2)

            # Tracked points (green=inliers, red=outliers)
            if tracker.last_good1 is not None:
                pts = tracker.last_good1.astype(int)
                mask = tracker.last_inliers_mask
                if mask is None:
                    for p in pts:
                        cv2.circle(out, (p[0], p[1]), 2, (0,255,255), -1)
                else:
                    for i, p in enumerate(pts):
                        col = (0,255,0) if mask[i] else (0,0,255)
                        cv2.circle(out, (p[0], p[1]), 2, col, -1)
                # optional flow vectors
                if tracker.last_good0 is not None:
                    prev = tracker.last_good0.astype(int)
                    for p0, p1 in zip(prev, pts):
                        cv2.line(out, (p0[0], p0[1]), (p1[0], p1[1]), (180,180,180), 1)

        # ---------- HUD ----------
        curr_fps = tick_fps()
        cv2.putText(out, f"State: {state}", (20,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0) if state!="IDLE" else (200,200,200), 2)
        cv2.putText(out, f"FPS: {curr_fps:.1f}", (20,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(out, "T: debug  C: color  [: thinner  ]: thicker  E: eraser  R: re-lock  S: save",
                    (20,h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)
        cv2.rectangle(out, (w-180,20), (w-20,80), COLORS[color_idx], 2)
        cv2.putText(out, f"W:{DRAW_WIDTH}", (w-170,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # Save board
        if key == ord('s'):
            img_path, json_path = board.save()
            cv2.putText(out, f"Saved: {img_path}", (20,h-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

        # Mirror PREVIEW only (optional)
        show = cv2.flip(out, 1) if MIRROR_DISPLAY else out
        cv2.imshow("AirNote - Capture", show)

        # Thumbnail rectified board in top-left
        try:
            thumb = cv2.resize(board.render()[:,:,:3], (200, 133))
            tb = np.copy(show)
            tb[10:10+thumb.shape[0], 10:10+thumb.shape[1]] = thumb
            cv2.imshow("AirNote - Capture", tb)
        except Exception:
            pass

        # Optional debug tiled view (bottom-right key 't' toggles)
        if DEBUG_DRAW:
            try:
                b_h, b_w = board_img.shape[:2]
                alpha = board_img[:, :, 3]
                alpha_vis = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
                overlay_bgr = cv2.cvtColor(board_img, cv2.COLOR_BGRA2BGR)
            except Exception:
                alpha_vis = np.zeros((h//3, w//3, 3), dtype=np.uint8)
                overlay_bgr = np.zeros((h//3, w//3, 3), dtype=np.uint8)

            base_vis = cv2.resize(frame, (w//3, h//3))
            overlay_vis = cv2.resize(overlay_bgr, (w//3, h//3))
            alpha_vis = cv2.resize(alpha_vis, (w//3, h//3))
            comp_vis = cv2.resize(show, (w//3, h//3))

            top = np.hstack([base_vis, overlay_vis, alpha_vis])
            bot = np.hstack([comp_vis, np.zeros_like(comp_vis), np.zeros_like(comp_vis)])
            dbg = np.vstack([top, bot])
            cv2.imshow("DEBUG - base | overlay | alpha -- comp", dbg)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
