import cv2
import numpy as np

class PlanarTracker:
    """
    LK + RANSAC homography tracker with:
      - freeze fallback for brief dropouts
      - auto-reseed after sustained low inliers
      - simple hysteresis to avoid flip/flop after recovery
    """
    def __init__(
        self,
        init_quad,
        gray0,
        max_pts=600,
        inlier_thresh=35,
        freeze_frames=6,
        reseed_low_streak_thresh=8,
        hysteresis_frames=5,
    ):
        """
        init_quad: (4,2) float32 image-space corners (clockwise)
        gray0: initial grayscale (already CLAHE'd)
        """
        self.quad0 = np.array(init_quad, dtype=np.float32)
        self.quad = np.array(init_quad, dtype=np.float32)

        self.inlier_thresh = int(inlier_thresh)
        self.freeze_frames = int(freeze_frames)
        self.reseed_low_streak_thresh = int(reseed_low_streak_thresh)
        self.hysteresis_frames = int(hysteresis_frames)

        # region mask
        self.mask = np.zeros_like(gray0, dtype=np.uint8)
        cv2.fillConvexPoly(self.mask, self.quad.astype(int), 255)

        # seed features
        self.pts0 = cv2.goodFeaturesToTrack(
            gray0, mask=self.mask,
            maxCorners=max_pts,
            qualityLevel=0.005,
            minDistance=5,
            blockSize=5
        )
        self.gray_prev = gray0

        # state
        self.H_t = None
        self._last_good_H = None
        self._low_streak = 0
        self._freeze_left = 0
        self._hysteresis_left = 0
        self.ok = False

        # events/flags (main reads & clears)
        self.need_reseed = False
        self.just_reseeded = False

        # debug
        self.last_good0 = None
        self.last_good1 = None
        self.last_inliers_mask = None

    def update(self, gray1):
        """Returns (H_eff, inliers). H_eff may be frozen/held by hysteresis."""
        if self.pts0 is None or len(self.pts0) < 8:
            return self._bad_return(gray1)

        pts1, st, err = cv2.calcOpticalFlowPyrLK(
            self.gray_prev, gray1, self.pts0, None,
            winSize=(21,21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        good0 = self.pts0[st==1]
        good1 = pts1[st==1]
        if len(good0) < 8:
            return self._maybe_freeze_and_mark_low(gray1, good1, 0)

        H, inliers = cv2.findHomography(good0, good1, cv2.RANSAC, 3.0)
        inl = int(inliers.sum()) if inliers is not None else 0

        # debug bookkeeping
        self.last_good0 = good0.reshape(-1,2)
        self.last_good1 = good1.reshape(-1,2)
        self.last_inliers_mask = inliers.reshape(-1).astype(bool) if inliers is not None else None

        # advance LK state
        self.gray_prev = gray1
        self.pts0 = good1.reshape(-1,1,2)

        # decision
        if H is not None and inl >= self.inlier_thresh:
            # fresh good estimate
            self._last_good_H = H
            self._low_streak = 0
            self._freeze_left = 0
            # kick hysteresis so we don't immediately collapse if next frame is borderline
            self._hysteresis_left = self.hysteresis_frames
            self.ok = True
            self.H_t = H
            self.quad = cv2.perspectiveTransform(self.quad0.reshape(-1,1,2), H).reshape(4,2)
            return self.H_t, inl

        # low inliers
        self._low_streak = min(self._low_streak + 1, 1_000_000)
        if self._low_streak >= self.reseed_low_streak_thresh:
            # suggest reseed; main will call reseed(gray) this frame
            self.need_reseed = True
        return self._maybe_freeze(gray1, inl)

    # ---- helpers ----
    def _bad_return(self, gray1):
        self.ok = False
        self.last_good0 = self.last_good1 = None
        self.last_inliers_mask = None
        self.gray_prev = gray1
        return None, 0

    def _maybe_freeze_and_mark_low(self, gray1, good1, inl):
        self._low_streak = min(self._low_streak + 1, 1_000_000)
        if self._low_streak >= self.reseed_low_streak_thresh:
            self.need_reseed = True
        self.gray_prev = gray1
        if good1 is not None and len(good1):
            self.pts0 = good1.reshape(-1,1,2)
        return self._maybe_freeze(gray1, inl)

    def _maybe_freeze(self, gray1, inl):
        # Use last good H while we wait out brief dips, or hysteresis window
        if self._last_good_H is not None and (self._freeze_left < self.freeze_frames or self._hysteresis_left > 0):
            self.ok = True
            self.H_t = self._last_good_H
            self._freeze_left = min(self.freeze_frames, self._freeze_left + 1)
            self._hysteresis_left = max(0, self._hysteresis_left - 1)
            self.quad = cv2.perspectiveTransform(self.quad0.reshape(-1,1,2), self.H_t).reshape(4,2)
            return self.H_t, inl

        # out of freeze/hysteresis and no good H
        self.ok = False
        self.H_t = None
        return None, inl

    def reseed(self, gray, quad=None, max_pts=600):
        """Reseed features inside the given quad (or last-estimated)."""
        if quad is None:
            quad = self.quad if self.quad is not None else self.quad0
        quad = np.array(quad, dtype=np.float32)

        self.mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.fillConvexPoly(self.mask, quad.astype(int), 255)
        self.pts0 = cv2.goodFeaturesToTrack(
            gray, mask=self.mask,
            maxCorners=max_pts,
            qualityLevel=0.005,
            minDistance=5,
            blockSize=5
        )
        self.gray_prev = gray
        self.need_reseed = False
        self.just_reseeded = True
        # keep last_good_H so freeze continues to operate
