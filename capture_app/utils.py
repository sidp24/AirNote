import cv2
import numpy as np

def alpha_blend(base_bgr, overlay_bgra):
    b,g,r = cv2.split(base_bgr)
    ob,og,or_,oa = cv2.split(overlay_bgra)
    oa_f = oa.astype(float)/255.0
    inv = 1.0 - oa_f
    out_b = (ob*oa_f + b*inv).astype('uint8')
    out_g = (og*oa_f + g*inv).astype('uint8')
    out_r = (or_*oa_f + r*inv).astype('uint8')
    return cv2.merge([out_b,out_g,out_r])
