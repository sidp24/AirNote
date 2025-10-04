import json, time
from dataclasses import dataclass, field
from typing import List, Tuple
import cv2
import numpy as np

def warp_point(H, x, y):
    p = np.array([x,y,1.0], dtype=np.float32)
    q = H @ p
    return (q[0]/q[2], q[1]/q[2])

def ema(prev, new, a=0.5):
    if prev is None: return new
    return (a*new[0]+(1-a)*prev[0], a*new[1]+(1-a)*prev[1])

@dataclass
class Stroke:
    color: Tuple[int,int,int]=(0,255,0)
    width: int=3
    points: List[Tuple[float,float]] = field(default_factory=list)

class BoardCanvas:
    def __init__(self, W=1200, H=800):
        self.W, self.H = W, H
        self.canvas = np.zeros((H, W, 4), dtype=np.uint8)  # RGBA
        self.strokes: List[Stroke] = []
        self.current: Stroke|None = None
        self._smooth_pt = None

    def begin(self, color, width):
        self.current = Stroke(color=color, width=width)
        self._smooth_pt = None

    def add_point(self, x, y):
        p = (float(x), float(y))
        self._smooth_pt = ema(self._smooth_pt, p, a=0.35)
        self.current.points.append(self._smooth_pt)

    def end(self):
        if self.current and len(self.current.points)>1:
            self.strokes.append(self.current)
        self.current = None

    def erase_at(self, x, y, r=18):
        cv2.circle(self.canvas, (int(x),int(y)), int(r), (0,0,0,0), thickness=-1)

    def render(self):
        # redraw from strokes to keep canvas clean after erases
        self.canvas[:] = 0
        for s in self.strokes + ([self.current] if self.current else []):
            if not s or len(s.points)<2: continue
            for i in range(1, len(s.points)):
                p0 = (int(s.points[i-1][0]), int(s.points[i-1][1]))
                p1 = (int(s.points[i][0]), int(s.points[i][1]))
                cv2.line(self.canvas, p0, p1, (*s.color, 255), s.width)
        return self.canvas

    def save(self, out_dir="out"):
        import os
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time())
        img_path = f"{out_dir}/board_{ts}.png"
        json_path = f"{out_dir}/strokes_{ts}.json"
        bgr = cv2.cvtColor(self.canvas, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(img_path, bgr)
        with open(json_path, "w") as f:
            json.dump([{"color": s.color, "width": s.width, "points": s.points} for s in self.strokes], f)
        return img_path, json_path
