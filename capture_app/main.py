import cv2, time, argparse
import numpy as np
from hand_state import HandsDetector, parse_hands, fingertip
from plane_select import draw_reticle, compute_H0
from tracker import PlanarTracker
from board_canvas import BoardCanvas, warp_point
from utils import alpha_blend

# --- Defaults ---
BOARD_W, BOARD_H = 1200, 800
DRAW_WIDTH = 5
COLORS = [(0,255,0),(255,0,0),(0,140,255),(255,255,255)]

# Runtime state
color_idx = 0
state = "IDLE"  # IDLE, DRAW, ERASE
last_left_pinch = False
debounce_until = 0

def now_ms(): 
    return int(time.time()*1000)

def can_transition():
    return now_ms() > debounce_until

def set_debounce(ms=160):
    global debounce_until
    debounce_until = now_ms() + ms

def main():
    global state, color_idx, DRAW_WIDTH

    # --- CLI args ---
    parser = argparse.ArgumentParser(description="AirNote Capture App")
    parser.add_argument("--cam", type=int, default=0, help="Camera index: 0=built-in, 1/2=external")
    parser.add_argument("--mirror", action="store_true", help="Mirror PREVIEW window only (processing stays non-mirrored)")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    args = parser.parse_args()

    MIRROR_DISPLAY = bool(args.mirror)

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    hd = HandsDetector()
    tracker = None
    H0 = None
    H_curr = None
    board = None
    corners = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # IMPORTANT: do NOT flip frame for processing (glasses POV).
        # All computations use the raw camera orientation.

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
            color_idx = (color_idx+1) % len(COLORS)
        if key == ord('e'):
            state = "ERASE" if state != "ERASE" else "IDLE"
        if key == ord('['):
            DRAW_WIDTH = max(1, DRAW_WIDTH-1)
        if key == ord(']'):
            DRAW_WIDTH += 1

        # --- Plane selection phase ---
        if H0 is None:
            # Reticle follows left index tip if present, else screen center
            ret = (w//2, h//2)
            if left and left.get("landmarks"):
                ret = (left["landmarks"][8][0], left["landmarks"][8][1])
            draw_reticle(frame, ret, (0,255,255))
            cv2.putText(frame, f"Place corner {len(corners)+1}/4 with LEFT PINCH",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # On rising edge of left pinch, record corner
            global last_left_pinch
            if left_pinch and not last_left_pinch:
                corners.append(ret)
            last_left_pinch = left_pinch

            # Draw placed corners
            for i,p in enumerate(corners):
                cv2.circle(frame, p, 6, (0,200,0), -1)
                cv2.putText(frame, str(i+1), (p[0]+6,p[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)

            if len(corners) == 4:
                H0, Wb, Hb = compute_H0(corners, BOARD_W, BOARD_H)
                gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                tracker = PlanarTracker(corners, gray0)
                board = BoardCanvas(BOARD_W, BOARD_H)
                H_curr = H0
                last_left_pinch = False

            show = cv2.flip(frame, 1) if MIRROR_DISPLAY else frame
            cv2.imshow("AirNote - Capture", show)
            continue

        # --- Tracking update ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Ht, inl = tracker.update(gray)
        ok_track = tracker.ok and Ht is not None
        if ok_track:
            cv2.putText(frame, f"Track inliers: {inl}",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 1)
            # Recompute current homography from original corners warped by Ht
            corners_np = np.float32([corners]).reshape(-1,1,2)
            curr = cv2.perspectiveTransform(corners_np, tracker.H_t).reshape(4,2)
            H_curr, _, _ = compute_H0(curr, BOARD_W, BOARD_H)
        else:
            cv2.putText(frame, "Re-lock (press R) - low track",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        if key == ord('r'):
            # Re-lock: reset plane definition
            H0 = None
            H_curr = None
            corners = []
            tracker = None
            board = None
            show = cv2.flip(frame, 1) if MIRROR_DISPLAY else frame
            cv2.imshow("AirNote - Capture", show)
            continue

        # --- FSM: left pinch gates tools ---
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

        # --- Draw / Erase in board space ---
        if tip is not None and H_curr is not None:
            x_b, y_b = warp_point(H_curr, tip[0], tip[1])
            if state == "DRAW":
                board.add_point(x_b, y_b)
            elif state == "ERASE":
                board.erase_at(x_b, y_b, r=18)

        # Render overlay
        board_img = board.render()
        inv = np.linalg.pinv(H_curr) if H_curr is not None else np.eye(3)
        overlay = cv2.warpPerspective(board_img, inv, (w,h))
        out = alpha_blend(frame, overlay)

        # HUD
        cv2.putText(out, f"State: {state}", (20,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0) if state!="IDLE" else (200,200,200), 2)
        cv2.rectangle(out, (w-180,20), (w-20,80), COLORS[color_idx], 2)
        cv2.putText(out, f"W:{DRAW_WIDTH}", (w-170,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # Save
        if key == ord('s'):
            img_path, json_path = board.save()
            cv2.putText(out, f"Saved: {img_path}", (20,h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Mirror only for display if requested
        show = cv2.flip(out, 1) if MIRROR_DISPLAY else out
        cv2.imshow("AirNote - Capture", show)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
