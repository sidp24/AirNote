# utils.py
"""
Utility functions for AirNote.

Exports:
    - alpha_blend(base_bgr, overlay_bgra) -> BGR
    - roi_gaussian_frosted(img_bgr, x0, y0, x1, y1, alpha=0.75, ksize=21) -> None (in-place)
    - SmoothBool(k_up=2, k_down=2).update(raw_bool) -> bool
    - sanitize_for_opencv(text: str) -> str
"""

from __future__ import annotations

import re
import unicodedata

import cv2
import numpy as np


def alpha_blend(base_bgr: np.ndarray, overlay_bgra: np.ndarray) -> np.ndarray:
    """
    Alpha-blend a BGRA overlay onto a BGR base, returning a new BGR image.

    Args:
        base_bgr: HxWx3 uint8
        overlay_bgra: HxWx4 uint8

    Returns:
        HxWx3 uint8 blended image
    """
    if base_bgr is None or overlay_bgra is None:
        raise ValueError("alpha_blend: inputs must be non-None numpy arrays")

    if base_bgr.shape[:2] != overlay_bgra.shape[:2]:
        raise ValueError("alpha_blend: base and overlay must have same HxW")

    if base_bgr.dtype != np.uint8 or overlay_bgra.dtype != np.uint8:
        raise TypeError("alpha_blend: images must be uint8")

    b, g, r = cv2.split(base_bgr)
    ob, og, or_, oa = cv2.split(overlay_bgra)

    oa_f = oa.astype(np.float32) / 255.0
    inv = 1.0 - oa_f

    out_b = (ob.astype(np.float32) * oa_f + b.astype(np.float32) * inv).astype(np.uint8)
    out_g = (og.astype(np.float32) * oa_f + g.astype(np.float32) * inv).astype(np.uint8)
    out_r = (or_.astype(np.float32) * oa_f + r.astype(np.float32) * inv).astype(np.uint8)

    return cv2.merge([out_b, out_g, out_r])


def roi_gaussian_frosted(
    img_bgr: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    alpha: float = 0.75,
    ksize: int = 21,
) -> None:
    """
    Apply a frosted glass effect to a rectangular ROI in-place:
      - blur the ROI with a Gaussian kernel
      - blend it back with weight alpha

    Args:
        img_bgr: HxWx3 uint8 image (modified in place)
        x0, y0, x1, y1: rectangle bounds (inclusive-exclusive is not required; we clamp)
        alpha: blend factor for the blurred ROI
        ksize: Gaussian kernel size (odd)
    """
    if img_bgr is None:
        return

    h, w = img_bgr.shape[:2]
    x0 = max(0, min(w - 1, int(x0)))
    y0 = max(0, min(h - 1, int(y0)))
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))

    if x1 <= x0 or y1 <= y0:
        return

    roi = img_bgr[y0:y1, x0:x1].copy()

    k = max(3, int(ksize) | 1)  # ensure odd and >=3
    blur = cv2.GaussianBlur(roi, (k, k), 0)

    cv2.addWeighted(blur, float(alpha), roi, 1.0 - float(alpha), 0, roi)
    img_bgr[y0:y1, x0:x1] = roi


class SmoothBool:
    """
    Debounced boolean with Schmitt-style behavior:
      - requires k_up consecutive True to switch ON
      - requires k_down consecutive False to switch OFF

    Example:
        sb = SmoothBool(k_up=2, k_down=3)
        out = sb.update(raw)
    """

    def __init__(self, k_up: int = 2, k_down: int = 2):
        self.k_up = int(max(1, k_up))
        self.k_down = int(max(1, k_down))
        self.on = False
        self.cnt = 0

    def update(self, raw: bool) -> bool:
        if self.on:
            if raw:
                self.cnt = 0
            else:
                self.cnt += 1
                if self.cnt >= self.k_down:
                    self.on = False
                    self.cnt = 0
        else:
            if raw:
                self.cnt += 1
                if self.cnt >= self.k_up:
                    self.on = True
                    self.cnt = 0
            else:
                self.cnt = 0
        return self.on


def sanitize_for_opencv(s: str) -> str:
    """
    Make a string safe for cv2.putText by:
      - replacing common Unicode punctuation with ASCII equivalents
      - removing any remaining non-ASCII/control characters

    Args:
        s: input string

    Returns:
        ASCII-only, single-line friendly string
    """
    if not s:
        return ""
    s = (
        s.replace("•", "-")
        .replace("‣", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("…", "...")
    )
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\x20-\x7E]", "", s)
    return s
