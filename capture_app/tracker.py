import cv2
import numpy as np

class PlanarTracker:
    def __init__(self, init_quad, gray0, max_pts=600):
        """
        init_quad: 4x2 float32 array of image-space corner points (clockwise)
        gray0: first grayscale frame (already preprocessed, e.g., CLAHE)
        """
        self.quad = np.array(init_quad, dtype=np.float32)

        # Mask only the selected quad
        self.mask = np.zeros_like(gray0, dtype=np.uint8)
        cv2.fillConvexPoly(self.mask, self.quad.astype(int), 255)

        # Track more points and be a bit more permissive to survive low texture
        self.pts0 = cv2.goodFeaturesToTrack(
            gray0, mask=self.mask,
            maxCorners=max_pts,
            qualityLevel=0.005,   # was 0.01
            minDistance=5,        # was 7
            blockSize=5           # was 7
        )
        self.gray_prev = gray0
        self.H_t = None
        self.ok = False

        # --- Debug visualization state (set every update) ---
        self.last_good0 = None         # (N,2) prev-frame points
        self.last_good1 = None         # (N,2) curr-frame points
        self.last_inliers_mask = None  # (N,) bool

    def update(self, gray1):
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
            self.ok = False
            self.last_good0 = self.last_good1 = None
            self.last_inliers_mask = None
            return None, 0

        # Robust homography with RANSAC
        H, inliers = cv2.findHomography(good0, good1, cv2.RANSAC, 3.0)
        inl = int(inliers.sum()) if inliers is not None else 0

        # Keep debug info
        self.last_good0 = good0.reshape(-1, 2)
        self.last_good1 = good1.reshape(-1, 2)
        self.last_inliers_mask = inliers.reshape(-1).astype(bool) if inliers is not None else None

        # Update internal state for next round
        self.gray_prev = gray1
        self.pts0 = good1.reshape(-1,1,2)
        self.H_t = H

        # Be a bit more forgiving than 50 to keep the experience smooth
        self.ok = H is not None and inl >= 35
        return H, inl
