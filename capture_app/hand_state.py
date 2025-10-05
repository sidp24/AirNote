# hand_state.py
"""
Lightweight wrapper around MediaPipe Hands with pinch/point detection.

Exports:
    - HandsDetector: .process(rgb_frame) -> mediapipe results
    - parse_hands(results, frame_w, frame_h) -> dict keyed by "Left"/"Right"
    - fingertip(pts) -> (x, y) or None
    - is_pinch(pts, thresh_px=40) -> bool
    - is_pointing(pts) -> bool

Notes:
    - Coordinates returned by parse_hands are integer pixel tuples.
    - 'Left'/'Right' correspond to MediaPipe's handedness classification
      (which accounts for mirroring).
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError(
        "mediapipe is required. Install with: pip install mediapipe"
    ) from e


mp_hands = mp.solutions.hands


class HandsDetector:
    """
    Thin wrapper over mp.solutions.hands.Hands.
    """

    def __init__(
        self,
        max_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        model_complexity: int = 1,
        static_image_mode: bool = False,
    ):
        self.hands = mp_hands.Hands(
            static_image_mode=bool(static_image_mode),
            max_num_hands=int(max_hands),
            model_complexity=int(model_complexity),
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )

    def process(self, rgb_frame: np.ndarray):
        """
        Args:
            rgb_frame: HxWx3 RGB ndarray (uint8)

        Returns:
            MediaPipe Hands result.
        """
        return self.hands.process(rgb_frame)

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass


# -----------------------
# Geometry / finger utils
# -----------------------

def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def handedness_to_label(handedness) -> str:
    """
    Returns 'Left' or 'Right' per MediaPipe's classification.
    """
    return handedness.classification[0].label


def finger_extended(
    pts: List[Tuple[int, int, float]],
    tip_idx: int,
    pip_idx: int,
    mcp_idx: int,
    ang_thresh_deg: float = 35.0,
) -> bool:
    """
    Heuristic: consider the finger extended if the angle at PIP is near 180 degrees.

    Args:
        pts: hand landmarks as (x,y,z) integer pixel coords with z retained from MP (ignored here)
        tip_idx, pip_idx, mcp_idx: landmark indices for that finger
        ang_thresh_deg: tolerance from straight line

    Returns:
        True if extended, False otherwise.
    """
    p_tip = np.array(pts[tip_idx][:2], dtype=float)
    p_pip = np.array(pts[pip_idx][:2], dtype=float)
    p_mcp = np.array(pts[mcp_idx][:2], dtype=float)

    v1 = p_mcp - p_pip
    v2 = p_tip - p_pip
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
    cosang = float(np.clip((v1 @ v2) / denom, -1.0, 1.0))
    ang = math.degrees(math.acos(cosang))
    return ang > (180.0 - float(ang_thresh_deg))


def is_pinch(pts: List[Tuple[int, int, float]], thresh_px: int = 40) -> bool:
    """
    Thumb tip (4) close to index tip (8).
    """
    if not pts or len(pts) < 9:
        return False
    t = (pts[4][0], pts[4][1])
    i = (pts[8][0], pts[8][1])
    return _dist(t, i) < float(thresh_px)


def is_pointing(pts: List[Tuple[int, int, float]]) -> bool:
    """
    Index extended while middle/ring/pinky curled (rough heuristic).
    """
    if not pts or len(pts) < 21:
        return False

    idx_ext = finger_extended(pts, 8, 6, 5)
    mid_curled = not finger_extended(pts, 12, 10, 9)
    ring_curled = not finger_extended(pts, 16, 14, 13)
    pinky_curled = not finger_extended(pts, 20, 18, 17)
    return bool(idx_ext and mid_curled and ring_curled and pinky_curled)


def fingertip(pts: Optional[List[Tuple[int, int, float]]]) -> Optional[Tuple[int, int]]:
    """
    Returns pixel coords for the index fingertip (landmark 8) or None.
    """
    return (int(pts[8][0]), int(pts[8][1])) if pts and len(pts) > 8 else None


# -----------------------
# Parsing
# -----------------------

def parse_hands(results, frame_w: int, frame_h: int) -> Dict[str, Dict[str, Any]]:
    """
    Convert MediaPipe results into a dict keyed by 'Left'/'Right':
        {
            'Left': {
                'landmarks': [(x,y,z), ...],   # pixel coordinates (z from MP, not scaled)
                'pinch': bool,
                'point': bool,
            },
            'Right': { ... }
        }

    Args:
        results: output of HandsDetector.process(rgb_frame)
        frame_w, frame_h: source frame dimensions for pixel scaling

    Returns:
        Dict with zero, one, or two entries ('Left', 'Right').
    """
    hands: Dict[str, Dict[str, Any]] = {}
    if not results or not getattr(results, "multi_hand_landmarks", None):
        return hands

    for lm, hnd in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = handedness_to_label(hnd)  # 'Left' or 'Right'
        pts: List[Tuple[int, int, float]] = []
        for p in lm.landmark:
            x = int(p.x * frame_w)
            y = int(p.y * frame_h)
            pts.append((x, y, p.z))

        hands[label] = {
            "landmarks": pts,
            "pinch": is_pinch(pts),
            "point": is_pointing(pts),
        }

    return hands
