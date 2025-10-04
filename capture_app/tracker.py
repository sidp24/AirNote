import cv2
import numpy as np

class PlanarTracker:
    def __init__(self, init_quad, gray0, max_pts=600, inlier_thresh=35, freeze_frames=6):
        """
        init_quad: 4x2 float32 array of image-space corner points (clockwise)
        gray0: first grayscale frame (already preprocessed, e.g., CLAHE)
        """
        self.quad0 = np.array(init_quad, dtype=np.float32)  # original picked quad (image space)
        self.quad = np.array(init_quad, dtype=np.float32)   # last estimated quad

        self.inlier_thresh = int(inlier_thresh)
        self.freeze_frames = int(freeze_frames)

        # Mask only the selected quad
        self.mask = np.zeros_like(gray0, dtype=np.uint8)
        cv2.fillConvexPoly(self.mask, self.quad.astype(int), 255)

        # Track more points and be a bit more permissive to survive low texture
        self.pts0 = cv2.goodFeaturesToTrack(
            gray0, mask=self.mask,
            maxCorners=max_pts,
            qualityLevel=0.005,
            minDistance=5,
            blockSize=5
        )
        self.gray_prev = gray0

        # Homography bookkeeping
        self.H_t = None              # last *effective* H used externally (may be frozen)
        self._last_good_H = None     # last *freshly-estimated* good H
        self._low_streak = 0         # consecutive low-inlier frames
        self.ok = False

        # --- Debug visualization state (set every update) ---
        self.last_good0 = None         # (N,2) prev-frame points
        self.last_good1 = None         # (N,2) curr-frame points
        self.last_inliers_mask = None  # (N,) bool

    def update(self, gray1):
        """
        Update tracking with new grayscale frame.
        Applies freeze fallback for brief inlier dropouts.
        Returns (H_eff, inlier_count) where H_eff may be frozen.
        """
        if self.pts0 is None or len(self.pts0) < 8:
            self.ok = False
            self.last_good0 = self.last_good1 = None
            self.last_inliers_mask = None
            return None, 0

        # Pyramidal Lucas–Kanade optical flow
        pts1, st, err = cv2.calcOpticalFlowPyrLK(
            self.gray_prev, gray1, self.pts0, None,
            winSize=(21,21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        good0 = self.pts0[st==1]
        good1 = pts1[st==1]
        if len(good0) < 8:
            # No estimate; maybe freeze?
            H_eff = self._maybe_freeze(None)
            self._set_debug(None, None, None)
            self.gray_prev = gray1
            self.pts0 = good1.reshape(-1,1,2) if good1 is not None and len(good1) else self.pts0
            return H_eff, 0

        # Robust homography with RANSAC
        H, inliers = cv2.findHomography(good0, good1, cv2.RANSAC, 3.0)
        inl = int(inliers.sum()) if inliers is not None else 0

        # Keep debug info
        self._set_debug(good0.reshape(-1, 2), good1.reshape(-1, 2),
                        inliers.reshape(-1).astype(bool) if inliers is not None else None)

        # Update internal state for next round
        self.gray_prev = gray1
        self.pts0 = good1.reshape(-1,1,2)

        if H is not None and inl >= self.inlier_thresh:
            # Fresh, good estimate
            self._last_good_H = H
            self._low_streak = 0
            self.ok = True
            self.H_t = H
            # Update quad estimate from fresh H
            self.quad = cv2.perspectiveTransform(self.quad0.reshape(-1,1,2), H).reshape(4,2)
            return self.H_t, inl

        # Otherwise try freezing to the last good H for a few frames
        H_eff = self._maybe_freeze(H)
        return H_eff, inl

    def _maybe_freeze(self, H_candidate):
        """
        If current H is bad, but we have a last good H and haven't exceeded freeze window,
        reuse last good H to keep overlay stable.
        """
        if H_candidate is not None:
            # Caller can still pass a candidate; if it's bad, we fall through to freeze logic.
            pass

        if self._last_good_H is not None and self._low_streak < self.freeze_frames:
            self._low_streak += 1
            self.ok = True
            self.H_t = self._last_good_H
            # Keep quad based on last good H
            self.quad = cv2.perspectiveTransform(self.quad0.reshape(-1,1,2), self.H_t).reshape(4,2)
            return self.H_t

        # No freeze available; tracking is considered bad
        self.ok = False
        self.H_t = None
        return None

    def _set_debug(self, good0, good1, inliers_mask):
        self.last_good0 = good0
        self.last_good1 = good1
        self.last_inliers_mask = inliers_mask

    def reseed(self, gray, quad=None, max_pts=600):
        """
        Reseed features inside the given quad (or current quad) without redefining corners.
        Use when tracking starts to feel sparse or after pressing Shift+R.
        """
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
        # Keep last good H so freeze logic can continue
