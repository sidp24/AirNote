import cv2
import numpy as np

def draw_reticle(frame, pos, color=(255,255,255)):
    x,y = pos
    cv2.circle(frame, (x,y), 8, color, 2)
    cv2.line(frame, (x-12,y), (x+12,y), color, 1)
    cv2.line(frame, (x,y-12), (x,y+12), color, 1)

def order_corners_tl_tr_br_bl(pts):
    """Return corners ordered TL, TR, BR, BL from 4 arbitrary points."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    # IMPORTANT: use (x - y), not diff() which gives (y - x)
    diff = pts[:,0] - pts[:,1]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def compute_H0(corners, W=1200, H=800):
    if len(corners) != 4:
        raise ValueError("compute_H0 requires exactly 4 corners")
    src = order_corners_tl_tr_br_bl(corners)
    dst = np.float32([[0,0],[W,0],[W,H],[0,H]])
    H0 = cv2.getPerspectiveTransform(src, dst)
    return H0, W, H
