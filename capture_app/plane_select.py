import cv2
import numpy as np

def draw_reticle(frame, pos, color=(255,255,255)):
    x,y = pos
    cv2.circle(frame, (x,y), 8, color, 2)
    cv2.line(frame, (x-12,y), (x+12,y), color, 1)
    cv2.line(frame, (x,y-12), (x,y+12), color, 1)

def compute_H0(corners, W=1200, H=800):
    # Ensure corners are in TL,TR,BR,BL order for a stable homography
    pts = np.array(corners, dtype=np.float32)
    if pts.shape[0] != 4:
        raise ValueError("compute_H0 requires 4 corner points")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    src = np.float32([tl, tr, br, bl])
    dst = np.float32([[0,0],[W,0],[W,H],[0,H]])
    H0 = cv2.getPerspectiveTransform(src, dst)
    return H0, W, H
