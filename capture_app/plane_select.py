# plane_select.py
"""
Plane selection and homography utilities for AirNote.

Exports:
    - draw_reticle(frame, pos, color=(255,255,255))
    - order_corners_tl_tr_br_bl(pts4)
    - auto_square_corners(corners_list)
    - compute_H0(corners, W=1200, H=800)
    - compute_H0_square(corners, side=1024)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


# -----------------------------
# Simple UI helper
# -----------------------------
def draw_reticle(frame: np.ndarray, pos: Tuple[int, int], color=(255, 255, 255)) -> None:
    """
    Draws a minimal reticle (circle + crosshair) at pos on the given BGR frame.
    """
    x, y = int(pos[0]), int(pos[1])
    cv2.circle(frame, (x, y), 8, color, 2, lineType=cv2.LINE_AA)
    cv2.line(frame, (x - 12, y), (x + 12, y), color, 1, lineType=cv2.LINE_AA)
    cv2.line(frame, (x, y - 12), (x, y + 12), color, 1, lineType=cv2.LINE_AA)


# -----------------------------
# Corner ordering
# -----------------------------
def order_corners_tl_tr_br_bl(pts4: Sequence[Tuple[int, int]] | np.ndarray) -> np.ndarray:
    """
    Given 4 corner points in any order, return them ordered as:
        TL, TR, BR, BL  (counter-clockwise winding, starting at top-left)

    Args:
        pts4: 4 (x, y) points (list/tuple or Nx2 array)

    Returns:
        4x2 float32 array ordered [TL, TR, BR, BL]
    """
    pts = np.array(pts4, dtype=np.float32).reshape(4, 2)

    # Sum/diff heuristic
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    ordered = np.stack([tl, tr, br, bl], axis=0).astype(np.float32)
    return ordered


def auto_square_corners(corners_list: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    If 4 points are given, fit a minAreaRect and return its box corners ordered TL,TR,BR,BL.
    This is useful after manual corner picking to enforce a perfect rectangle.

    Returns the input unchanged if not exactly 4 points.
    """
    if len(corners_list) != 4:
        return list(corners_list)

    pts = np.array(corners_list, dtype=np.float32)
    rect = cv2.minAreaRect(pts)           # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect)             # 4x2 float array

    # Order using the same heuristic as above
    s = box.sum(axis=1)
    diff = box[:, 0] - box[:, 1]
    tl = box[np.argmin(s)]
    br = box[np.argmax(s)]
    tr = box[np.argmin(diff)]
    bl = box[np.argmax(diff)]

    ordered = np.stack([tl, tr, br, bl], axis=0).astype(np.int32)
    return [(int(x), int(y)) for (x, y) in ordered]


# -----------------------------
# Homography builders
# -----------------------------
def compute_H0(
    corners: Sequence[Tuple[int, int]] | np.ndarray,
    W: int = 1200,
    H: int = 800,
) -> Tuple[np.ndarray, int, int]:
    """
    Compute a rectifying homography from an arbitrary 4-corner quadrilateral to a W x H rectangle.

    Args:
        corners: 4 points (x,y) of the selected plane in source image coordinates
        W, H: destination rectangle size

    Returns:
        (H0, W, H) where H0 maps src->dst (cv2.warpPerspective(src, H0, (W,H)))
    """
    corners = np.array(corners, dtype=np.float32).reshape(4, 2)
    src = order_corners_tl_tr_br_bl(corners)
    dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    H0 = cv2.getPerspectiveTransform(src, dst)
    return H0, int(W), int(H)


def compute_H0_square(
    corners: Sequence[Tuple[int, int]] | np.ndarray,
    side: int = 1024,
) -> Tuple[np.ndarray, int, int]:
    """
    Like compute_H0, but destination is a perfect square (side x side).
    Useful to remove anisotropic stretching when you do not care about original aspect ratio.

    Args:
        corners: 4 points (x,y) in source image coordinates
        side: output square side

    Returns:
        (H0, side, side)
    """
    corners = np.array(corners, dtype=np.float32).reshape(4, 2)
    src = order_corners_tl_tr_br_bl(corners)
    dst = np.float32([[0, 0], [side, 0], [side, side], [0, side]])
    H0 = cv2.getPerspectiveTransform(src, dst)
    return H0, int(side), int(side)
