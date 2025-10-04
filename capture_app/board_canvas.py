import json, time, os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import cv2
import numpy as np

def warp_point(H, x, y):
    p = np.array([x, y, 1.0], dtype=np.float32)
    q = H @ p
    w = q[2] if q[2] != 0 else 1e-6
    return (q[0]/w, q[1]/w)

def ema(prev, new, a=0.5):
    if prev is None: return new
    return (a*new[0]+(1-a)*prev[0], a*new[1]+(1-a)*prev[1])

@dataclass
class Stroke:
    mode: str = "draw"                        # "draw" or "erase"
    color: Tuple[int,int,int]=(0,255,0)       # BGR (used only for draw)
    width: int=3
    points: List[Tuple[float,float]] = field(default_factory=list)

class BoardCanvas:
    """
    Adds notetaking affordances: undo/redo, clear, multipage.
    """
    def __init__(self, W=1200, H=800):
        self.W, self.H = W, H
        self.canvas = np.zeros((H, W, 4), dtype=np.uint8)

        # Pages are lists of strokes
        self.pages: List[List[Stroke]] = [[]]
        self.page_idx = 0
        self.current: Optional[Stroke] = None
        self._smooth_pt = None

        # Per-page undo stacks (redo history)
        self._redos: List[List[Stroke]] = [[]]  # mirrors pages by index

        self.show_grid = True   # grid toggle
        self.dark_bg = False    # dark background toggle for contrast

    # ---------- Page helpers ----------
    def _page(self) -> List[Stroke]:
        return self.pages[self.page_idx]

    def _redo_stack(self) -> List[Stroke]:
        return self._redos[self.page_idx]

    def new_page(self):
        self.end()
        self.pages.append([])
        self._redos.append([])
        self.page_idx = len(self.pages) - 1

    def next_page(self):
        self.end()
        if self.page_idx < len(self.pages) - 1:
            self.page_idx += 1

    def prev_page(self):
        self.end()
        if self.page_idx > 0:
            self.page_idx -= 1

    def clear_page(self):
        self.end()
        self.pages[self.page_idx] = []
        self._redos[self.page_idx] = []

    # ---------- Editing ----------
    def undo(self):
        self.end()
        if self._page():
            s = self._page().pop()
            self._redo_stack().append(s)

    def redo(self):
        self.end()
        if self._redo_stack():
            s = self._redo_stack().pop()
            self._page().append(s)

    # ---------- Drawing ----------
    def begin(self, color, width, mode="draw"):
        self.current = Stroke(mode=str(mode), color=tuple(color), width=int(width))
        self._smooth_pt = None

    def add_point(self, x, y):
        if self.current is None:
            self.begin((0,255,0), 3, mode="draw")
        p = (float(x), float(y))
        self._smooth_pt = ema(self._smooth_pt, p, a=0.35)
        self.current.points.append(self._smooth_pt)

    def end(self):
        if self.current and len(self.current.points) > 1:
            self._page().append(self.current)
            self._redo_stack().clear()
        self.current = None

    # ---------- Rendering ----------
    def _draw_grid(self, target):
        grid_color = (0,255,0,40)  # BGRA low alpha
        step_x = max(60, self.W//12)
        step_y = max(60, self.H//12)
        for x in range(0, self.W, step_x):
            cv2.line(target, (x,0), (x,self.H), grid_color, 1)
        for y in range(0, self.H, step_y):
            cv2.line(target, (0,y), (self.W,y), grid_color, 1)

    def _render_to(self, include_grid: bool):
        out = np.zeros((self.H, self.W, 4), dtype=np.uint8)

        # optional faint background for contrast
        if self.dark_bg:
            out[:,:,3] = 20  # tiny alpha backdrop

        erase_mask = np.full((self.H, self.W), 255, dtype=np.uint8)
        seq = self._page() + ([self.current] if self.current else [])

        for s in seq:
            if not s or len(s.points) < 2:
                continue
            if s.mode == "draw":
                col = (int(s.color[0]), int(s.color[1]), int(s.color[2]), 255)
                for i in range(1, len(s.points)):
                    p0 = (int(s.points[i-1][0]), int(s.points[i-1][1]))
                    p1 = (int(s.points[i][0]),   int(s.points[i][1]))
                    # stroke with light shadow to pop
                    cv2.line(out, (p0[0]+1,p0[1]+1), (p1[0]+1,p1[1]+1), (0,0,0,100), s.width+1, lineType=cv2.LINE_AA)
                    cv2.line(out, p0, p1, col, s.width, lineType=cv2.LINE_AA)
            else:
                for i in range(1, len(s.points)):
                    p0 = (int(s.points[i-1][0]), int(s.points[i-1][1]))
                    p1 = (int(s.points[i][0]),   int(s.points[i][1]))
                    cv2.line(erase_mask, p0, p1, 0, s.width, lineType=cv2.LINE_AA)

        # apply eraser to alpha channel
        alpha = out[:,:,3]
        alpha = cv2.bitwise_and(alpha, erase_mask)
        out[:,:,3] = alpha

        if include_grid:
            grid_layer = np.zeros_like(out)
            self._draw_grid(grid_layer)
            gb,gg,gr,ga = cv2.split(grid_layer)
            ob,og,or_,oa = cv2.split(out)
            ga_f = ga.astype(np.float32)/255.0
            inv = 1.0 - ga_f
            b = (gb.astype(np.float32)* (ga.astype(np.float32)/255.0) + ob.astype(np.float32)*inv).astype(np.uint8)
            g = (gg.astype(np.float32)* (ga.astype(np.float32)/255.0) + og.astype(np.float32)*inv).astype(np.uint8)
            r = (gr.astype(np.float32)* (ga.astype(np.float32)/255.0) + or_.astype(np.float32)*inv).astype(np.uint8)
            a = np.clip(ga.astype(np.int32) + oa.astype(np.int32), 0, 255).astype(np.uint8)
            out = cv2.merge([b,g,r,a])

        return out

    def render(self):
        self.canvas = self._render_to(include_grid=self.show_grid)
        return self.canvas

    def save(self, out_dir="out", H0=None, curr_quad=None):
        """
        Saves:
          - Clean rectified board (no grid): board_<ts>.png
          - Preview (with grid): board_preview_<ts>.png
          - strokes_<ts>.json (current page only, with modes)
          - meta_<ts>.json (W,H,H0,curr_quad,page_idx)
        """
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time())
        clean_path = f"{out_dir}/board_{ts}.png"
        preview_path = f"{out_dir}/board_preview_{ts}.png"
        json_path = f"{out_dir}/strokes_{ts}.json"
        meta_path = f"{out_dir}/meta_{ts}.json"

        clean = self._render_to(include_grid=False)
        prevw = self._render_to(include_grid=True)

        cv2.imwrite(clean_path, cv2.cvtColor(clean, cv2.COLOR_BGRA2BGR))
        cv2.imwrite(preview_path, cv2.cvtColor(prevw, cv2.COLOR_BGRA2BGR))

        with open(json_path, "w") as f:
            json.dump(
                [{"mode": s.mode, "color": s.color, "width": s.width, "points": s.points}
                 for s in self._page()],
                f
            )

        meta = {
            "W": self.W, "H": self.H,
            "page_idx": self.page_idx,
            "H0": (np.asarray(H0).tolist() if H0 is not None else None),
            "curr_quad": (np.asarray(curr_quad).tolist() if curr_quad is not None else None)
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        return clean_path, json_path, preview_path, meta_path
