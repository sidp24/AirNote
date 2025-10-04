import argparse, time
import cv2
import numpy as np

from hand_state import HandsDetector, parse_hands, fingertip
from plane_select import draw_reticle, compute_H0  # ordering enforced inside compute_H0
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

# Mouse helpers
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

def main():
    global state, color_idx, DRAW_WIDTH, last_left_pinch, DEBUG_DRAW
    global mouse_draw, mouse_down, mouse_pos

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

    # ---- Mouse callback ----
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

    # ---- Contrast improver for tracking ----
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
        left = hands.get("Left"); right = hands.get("Right")
        left_pinch = bool(left and left["pinch"])
        right_point = bool(right and right["point"])
        right_pinch = bool(right and right["pinch"])
        tip = fingertip(right["landmarks"]) if right else None

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == ord('c'): color_idx = (color_idx + 1) % len(COLORS)
        if key == ord('e'): state = "ERASE" if state != "ERASE" else "IDLE"  # manual toggle (still have gestures)
        if key == ord('['): DRAW_WIDTH = max(1, DRAW_WIDTH - 1)
        if key == ord(']'): DRAW_WIDTH += 1
        if key == ord('t'): DEBUG_DRAW = not DEBUG_DRAW
        if key == ord('m'): mouse_draw = not mouse_draw
        if key == ord('g') and board is not None: board.show_grid = not board.show_grid
        if key == ord('b') and board is not None: board.dark_bg = not board.dark_bg  # contrast bg
        if key == ord('u') and board is not None: board.undo()
        if key == ord('y') and board is not None: board.redo()
        if key == ord('x') and board is not None: board.clear_page()
        if key == ord('n') and board is not None: board.new_page()
        if key == ord(',') and board is not None: board.prev_page()
        if key == ord('.') and board is not None: board.next_page()
        # Shift+R -> reseed features without redefining corners
        if key == ord('R'):  # uppercase R is 82 with cv2.waitKey? safer to catch both:
            pass
        if key == ord('r'):
            if tracker is not None and H0 is not None and H_curr is not None and board is not None and len(corners)==4:
                # SHIFT held?
                import ctypes
                if bool(ctypes.windll.user32.GetKeyState(0x10) & 0x8000):  # VK_SHIFT on Windows; noop on non-Windows
                    gray_now = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    tracker.reseed(gray_now, quad=None)
                else:
                    # full re-lock
                    H0 = None
                    H_curr = None
                    corners = []
                    tracker = None
                    board = None
                    state = "IDLE"
                    last_left_pinch = False

        # ---------- Plane selection ----------
        if H0 is None:
            ret = (w//2, h//2)
            if left and left.get("landmarks"):
                ret = (left["landmarks"][8][0], left["landmarks"][8][1])
            draw_reticle(frame, ret, (0,255,255))
            cv2.putText(frame, f"Place corner {len(corners)+1}/4 (left-pinch or mouse click)",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            if left_pinch and not last_left_pinch and len(corners) < 4:
                corners.append(ret)
            last_left_pinch = left_pinch

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
                last_left_pinch = False

            show = cv2.flip(frame, 1) if MIRROR_DISPLAY else frame
            cv2.imshow("AirNote - Capture", show)
            continue

        # ---------- Tracking update ----------
        gray = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        Ht, inl = tracker.update(gray)
        ok_track = tracker.ok and Ht is not None
        if ok_track:
            text = f"Track inliers: {inl}"
            cv2.putText(frame, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 1)
            corners_np = np.float32([corners]).reshape(-1,1,2)
            curr = cv2.perspectiveTransform(corners_np, tracker.H_t).reshape(4,2)
            H_curr, _, _ = compute_H0(curr, BOARD_W, BOARD_H)
        else:
            cv2.putText(frame, "Re-lock (press R) - low track",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # ---------- Gesture FSM (left pinch = gate) ----------
        if board is not None and H_curr is not None:
            if left_pinch:
                if can_transition():
                    if right_point and state != "DRAW":
                        if state in ("DRAW","ERASE"):
                            board.end()
                        state = "DRAW"
                        board.begin(COLORS[color_idx], DRAW_WIDTH, mode="draw")
                        set_debounce()
                    elif right_pinch and state != "ERASE":
                        if state in ("DRAW","ERASE"):
                            board.end()
                        state = "ERASE"
                        board.begin(COLORS[color_idx], DRAW_WIDTH, mode="erase")
                        set_debounce()
                if tip is not None and state in ("DRAW","ERASE"):
                    xb, yb = warp_point(H_curr, tip[0], tip[1])
                    board.add_point(xb, yb)
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
        help_text = ("M: mouse-draw [{0}]   T: debug   G: grid   B: bg "
                     "  C: color   [: thinner   ]: thicker   "
                     "U: undo   Y: redo   X: clear   N: new   <: prev   >: next   "
                     "R: re-lock (Shift+R reseed)   S: save").format('ON' if mouse_draw else 'OFF')
        cv2.putText(out, help_text, (20,h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)
        cv2.rectangle(out, (w-180,20), (w-20,80), COLORS[color_idx], 2)
        cv2.putText(out, f"W:{DRAW_WIDTH}", (w-170,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # Tool HUD near fingertip while gated
        if left_pinch and tip is not None:
            label = "DRAW" if state=="DRAW" else ("ERASE" if state=="ERASE" else "")
            if label:
                cv2.putText(out, label, (tip[0]+12, tip[1]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        # Save (also write meta)
        if key == ord('s'):
            # Current quad in image space (ints for readability)
            curr_quad = cv2.perspectiveTransform(np.float32([corners]).reshape(-1,1,2), tracker.H_t).reshape(4,2)
            img_path, json_path, preview_path, meta_path = board.save(H0=H0, curr_quad=curr_quad)
            cv2.putText(out, f"Saved: {img_path}", (20,h-15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

        # Thumbnail
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
