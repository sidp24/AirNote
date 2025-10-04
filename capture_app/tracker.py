import cv2
import numpy as np

class PlanarTracker:
    def __init__(self, init_quad, gray0, max_pts=300):
        self.quad = np.array(init_quad, dtype=np.float32)
        self.mask = np.zeros_like(gray0, dtype=np.uint8)
        cv2.fillConvexPoly(self.mask, self.quad.astype(int), 255)
        self.pts0 = cv2.goodFeaturesToTrack(
            gray0, mask=self.mask, maxCorners=max_pts,
            qualityLevel=0.01, minDistance=7, blockSize=7)
        self.gray_prev = gray0
        self.H_t = None
        self.ok = False

    def update(self, gray1):
        if self.pts0 is None or len(self.pts0) < 8:
            self.ok = False
            return None, 0
        pts1, st, err = cv2.calcOpticalFlowPyrLK(self.gray_prev, gray1, self.pts0, None,
                                                 winSize=(21,21), maxLevel=3,
                                                 criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        good0 = self.pts0[st==1]
        good1 = pts1[st==1]
        if len(good0) < 8:
            self.ok = False
            return None, 0
        H, inliers = cv2.findHomography(good0, good1, cv2.RANSAC, 3.0)
        inl = int(inliers.sum()) if inliers is not None else 0
        self.gray_prev = gray1
        self.pts0 = good1.reshape(-1,1,2)
        self.H_t = H
        self.ok = H is not None and inl >= 50
        return H, inl
