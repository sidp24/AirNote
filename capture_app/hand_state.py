import math
from typing import Optional, Tuple
import mediapipe as mp

mp_hands = mp.solutions.hands

class HandsDetector:
    def __init__(self, max_hands=2):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def process(self, rgb_frame):
        return self.hands.process(rgb_frame)

def _dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def handedness_to_label(handedness):
    # 'Left'/'Right' from MediaPipe perspective (mirror aware)
    return handedness.classification[0].label

def parse_hands(results, frame_w, frame_h):
    """Return dict: {'Left': {...}, 'Right': {...}} each with landmarks in pixel coords."""
    hands = {}
    if not results.multi_hand_landmarks:
        return hands
    for lm, hnd in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = handedness_to_label(hnd)
        pts = []
        for p in lm.landmark:
            pts.append((int(p.x*frame_w), int(p.y*frame_h), p.z))
        hands[label] = {
            "landmarks": pts,
            "pinch": is_pinch(pts),
            "point": is_pointing(pts)
        }
    return hands

def is_pinch(pts, thresh_px: int = 40) -> bool:
    # Thumb tip (4) to index tip (8)
    if not pts: return False
    t = pts[4]; i = pts[8]
    return _dist(t, i) < thresh_px

def finger_extended(pts, tip_idx, pip_idx, mcp_idx, ang_thresh_deg=35) -> bool:
    # Angle at PIP: mcp->pip and pip->tip
    import numpy as np
    p_tip = np.array(pts[tip_idx][:2], float)
    p_pip = np.array(pts[pip_idx][:2], float)
    p_mcp = np.array(pts[mcp_idx][:2], float)
    v1 = p_mcp - p_pip
    v2 = p_tip - p_pip
    cosang = (v1@v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-6)
    ang = math.degrees(math.acos(max(-1,min(1,cosang))))
    return ang > (180 - ang_thresh_deg)  # near straight line

def is_pointing(pts) -> bool:
    # Index extended, middle/ring/pinky curled
    if not pts: return False
    idx_ext = finger_extended(pts, 8, 6, 5)
    mid_curled = not finger_extended(pts, 12, 10, 9)
    ring_curled = not finger_extended(pts, 16, 14, 13)
    pinky_curled = not finger_extended(pts, 20, 18, 17)
    return idx_ext and (mid_curled and ring_curled and pinky_curled)

def fingertip(pts) -> Optional[Tuple[int,int]]:
    return (pts[8][0], pts[8][1]) if pts else None
