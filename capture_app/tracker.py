# tracker.py
"""
PlanarTracker: feature-tracking homography estimator with robustness features.

Implements:
    - Sparse optical flow (Lucas–Kanade pyramids) forward pass
    - Forward–backward flow consistency check to cull unstable tracks
    - Adaptive RANSAC inlier threshold based on motion magnitude
    - Freeze fallback and hysteresis to mask brief tracking drops
    - Auto-reseed of features after sustained low-inlier streaks

Usage:
    import cv2
    import numpy as np
    from tracker import PlanarTracker

    # gray0: initial grayscale frame (uint8)
    # init_quad: 4x2 list/array of (x,y) corners of the tracked plane in gray0
    tracker = PlanarTracker(init_quad, gray0)

    # per-frame:
    Ht, inliers = tracker.update(gray1)
    if tracker.need_reseed:
        tracker.reseed(gray1)  # optionally with an updated quad
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


def _fb_check(
    gray0: np.ndarray,
    gray1: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    back_thresh: float = 1.6,
) -> np.ndarray:
    """
    Forward–backward optical flow check:
        Track 0->1 to get pts1, then 1->0 to get pts0r.
        Reject points whose back-tracking error is large.

    Args:
        gray0, gray1: consecutive grayscale frames (uint8)
        pts0: Nx2 float32 points in gray0 that produced pts1
        pts1: Nx2 float32 points in gray1 tracked from pts0
        back_thresh: maximum allowed back-tracking error (pixels)

    Returns:
        boolean mask (N,) indicating which points pass the check
    """
    pts0f = pts0.reshape(-1, 1, 2).astype(np.float32)
    pts1f = pts1.reshape(-1, 1, 2).astype(np.float32)

    pts0r, st, err = cv2.calcOpticalFlowPyrLK(
        gray1,
        gray0,
        pts1f,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    d = np.linalg.norm(pts0f - pts0r, axis=2).reshape(-1)
    ok = d < float(back_thresh)
    return ok


@dataclass
class _State:
    inlier_thresh: int = 35         # min inliers to accept fresh H
    freeze_frames: int = 6          # frames to hold last good H during drops
    reseed_low_streak: int = 8      # consecutive low-inlier frames before reseed flag
    hysteresis_frames: int = 5      # frames to continue using last good H after recovery
    ransac_px: float = 3.0          # base RANSAC pixel threshold (adaptive around this)


class PlanarTracker:
    """
    LK + RANSAC homography tracker with:
      - forward/backward flow culling
      - adaptive RANSAC threshold
      - freeze fallback + hysteresis
      - auto-reseed after sustained low inliers
    """

    def __init__(
        self,
        init_quad,
        gray0: np.ndarray,
        max_pts: int = 900,
        inlier_thresh: int = 40,
        freeze_frames: int = 8,
        reseed_low_streak_thresh: int = 8,
        hysteresis_frames: int = 6,
        ransac_px: float = 3.0,
    ):
        """
        Args:
            init_quad: 4x2 iterable of initial plane corners (x,y) in gray0
            gray0: initial grayscale frame (uint8)
            max_pts: max number of goodFeaturesToTrack points to seed
            inlier_thresh: minimum RANSAC inliers to accept a new H
            freeze_frames: how long to keep last good H when inliers drop
            reseed_low_streak_thresh: consecutive low-inlier frames before triggering reseed
            hysteresis_frames: keep last good H for a short time after recovery
            ransac_px: base RANSAC reprojection threshold (adaptive)
        """
        self.quad0 = np.array(init_quad, dtype=np.float32).reshape(4, 2)
        self.quad = self.quad0.copy()

        self.cfg = _State(
            inlier_thresh=int(inlier_thresh),
            freeze_frames=int(freeze_frames),
            reseed_low_streak=int(reseed_low_streak_thresh),
            hysteresis_frames=int(hysteresis_frames),
            ransac_px=float(ransac_px),
        )

        # Seed features inside initial quad
        self.mask = np.zeros_like(gray0, dtype=np.uint8)
        cv2.fillConvexPoly(self.mask, self.quad0.astype(int), 255)

        self.pts0 = cv2.goodFeaturesToTrack(
            gray0,
            mask=self.mask,
            maxCorners=int(max_pts),
            qualityLevel=0.0045,
            minDistance=5,
            blockSize=5,
        )

        self.gray_prev = gray0
        self.H_t: Optional[np.ndarray] = None
        self._last_good_H: Optional[np.ndarray] = None

        self._low_streak = 0
        self._freeze_left = 0
        self._hysteresis_left = 0

        self.ok: bool = False
        self.need_reseed: bool = False
        self.just_reseeded: bool = False

        # Debug bookkeeping (optional consumers can read these)
        self.last_good0: Optional[np.ndarray] = None
        self.last_good1: Optional[np.ndarray] = None
        self.last_inliers_mask: Optional[np.ndarray] = None

    # -------------------------------------------------------

    def update(self, gray1: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        """
        Advance the tracker to the next frame.

        Returns:
            (H_eff, inliers)
                H_eff: the homography to use this frame:
                       - a fresh estimate (if enough inliers), or
                       - the last good H during freeze/hysteresis windows, or
                       - None if no estimate available
                inliers: number of inliers reported by RANSAC for the fresh estimate
        """
        if self.pts0 is None or len(self.pts0) < 8:
            return self._bad_return(gray1)

        # LK forward pass
        pts1, st, err = cv2.calcOpticalFlowPyrLK(
            self.gray_prev,
            gray1,
            self.pts0,
            None,
            winSize=(23, 23),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 32, 0.01),
        )

        good0 = self.pts0[st == 1].reshape(-1, 2) if st is not None else np.empty((0, 2), np.float32)
        good1 = pts1[st == 1].reshape(-1, 2) if st is not None else np.empty((0, 2), np.float32)
        if len(good0) < 12:
            return self._maybe_freeze_and_mark_low(gray1, good1, inl=0)

        # Forward–backward culling
        ok_fb = _fb_check(self.gray_prev, gray1, good0, good1, back_thresh=1.6)
        good0 = good0[ok_fb]
        good1 = good1[ok_fb]
        if len(good0) < 12:
            return self._maybe_freeze_and_mark_low(gray1, good1, inl=0)

        # Adaptive RANSAC threshold based on median displacement
        disp = np.linalg.norm(good1 - good0, axis=1)
        med = float(np.median(disp)) if len(disp) else 0.0
        thr = max(2.0, min(5.0, self.cfg.ransac_px + 0.25 * (med - 2.0)))

        H, inliers = cv2.findHomography(good0, good1, cv2.RANSAC, thr)
        inl = int(inliers.sum()) if inliers is not None else 0

        # Debug bookkeeping
        self.last_good0 = good0.copy()
        self.last_good1 = good1.copy()
        self.last_inliers_mask = inliers.reshape(-1).astype(bool) if inliers is not None else None

        # Advance LK state for next iteration
        self.gray_prev = gray1
        self.pts0 = good1.reshape(-1, 1, 2)

        # Decision
        if H is not None and inl >= self.cfg.inlier_thresh:
            self._last_good_H = H
            self._low_streak = 0
            self._freeze_left = 0
            self._hysteresis_left = self.cfg.hysteresis_frames
            self.ok = True
            self.H_t = H
            self.quad = cv2.perspectiveTransform(self.quad0.reshape(-1, 1, 2), H).reshape(4, 2)
            return self.H_t, inl

        # Low inliers path
        self._low_streak = min(self._low_streak + 1, 1_000_000)
        if self._low_streak >= self.cfg.reseed_low_streak:
            self.need_reseed = True

        return self._maybe_freeze(gray1, inl)

    # -------------------------------------------------------

    def _bad_return(self, gray1: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        self.ok = False
        self.last_good0 = None
        self.last_good1 = None
        self.last_inliers_mask = None
        self.gray_prev = gray1
        return None, 0

    def _maybe_freeze_and_mark_low(self, gray1: np.ndarray, good1: np.ndarray, inl: int):
        self._low_streak = min(self._low_streak + 1, 1_000_000)
        if self._low_streak >= self.cfg.reseed_low_streak:
            self.need_reseed = True

        # Keep advancing pts0 if we have any tracks at all
        self.gray_prev = gray1
        if good1 is not None and len(good1):
            self.pts0 = good1.reshape(-1, 1, 2)

        return self._maybe_freeze(gray1, inl)

    def _maybe_freeze(self, gray1: np.ndarray, inl: int):
        """
        Hold last good H during brief dips or within hysteresis window.
        """
        if self._last_good_H is not None and (self._freeze_left < self.cfg.freeze_frames or self._hysteresis_left > 0):
            self.ok = True
            self.H_t = self._last_good_H
            self._freeze_left = min(self.cfg.freeze_frames, self._freeze_left + 1)
            self._hysteresis_left = max(0, self._hysteresis_left - 1)
            self.quad = cv2.perspectiveTransform(self.quad0.reshape(-1, 1, 2), self.H_t).reshape(4, 2)
            return self.H_t, inl

        self.ok = False
        self.H_t = None
        return None, inl

    # -------------------------------------------------------

    def reseed(self, gray: np.ndarray, quad: Optional[np.ndarray] = None, max_pts: int = 900) -> None:
        """
        Reseed new features inside the given quad (or last-estimated one).

        Args:
            gray: current grayscale frame
            quad: 4x2 polygon to seed within; defaults to last estimated quad
            max_pts: number of corners to seed
        """
        if quad is None:
            quad = self.quad if self.quad is not None else self.quad0

        quad = np.array(quad, dtype=np.float32).reshape(4, 2)

        self.mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillConvexPoly(self.mask, quad.astype(int), 255)

        self.pts0 = cv2.goodFeaturesToTrack(
            gray,
            mask=self.mask,
            maxCorners=int(max_pts),
            qualityLevel=0.0045,
            minDistance=5,
            blockSize=5,
        )

        self.gray_prev = gray
        self.need_reseed = False
        self.just_reseeded = True
        # Preserve self._last_good_H so freeze/hysteresis can still operate
