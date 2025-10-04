import json, time, os, glob, math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import cv2
import numpy as np

def warp_point(H, x, y):
    p = np.array([x, y, 1.0], dtype=np.float32)
    q = H @ p
    w = q[2] if q[2] != 0 else 1e-6
    return (q[0]/w, q[1]/w)

def ema(prev, new, a=0.25):
    # heavier smoothing -> fewer jitters
    if prev is None: return new
    return (a*new[0]+(1-a)*prev[0], a*new[1]+(1-a)*prev[1])

@dataclass
class Stroke:
    mode: str = "draw"                        # "draw" or "erase"
    color: Tuple[int,int,int]=(0,255,0)       # BGR
    width: int=3
    points: List[Tuple[float,float]] = field(default_factory=list)

class BoardCanvas:
    """
    Notetaking canvas with pages, undo/redo, grid/dark bg, and sessionized saves.
    """
    def __init__(self, W=1200, H=800):
        self.W, self.H = W, H
        self.canvas = np.zeros((H, W, 4), dtype=np.uint8)

        self.pages: List[List[Stroke]] = [[]]
        self.page_idx = 0
        self._redos: List[List[Stroke]] = [[]]

        self.current: Optional[Stroke] = None
        self._smooth_pt = None

        self.show_grid = True
        self.dark_bg = False

    # ---------- page helpers ----------
    def page_count(self) -> int:
        return len(self.pages)

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

    # ---------- edit ----------
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

    # ---------- draw ----------
    def begin(self, color, width, mode="draw"):
        self.current = Stroke(mode=str(mode), color=tuple(color), width=int(width))
        self._smooth_pt = None

    def _densify(self, p_last, p_new, spacing=2.5):
        """Insert intermediate points if gap too large to prevent 'holes'."""
        (x0, y0), (x1, y1) = p_last, p_new
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist <= spacing: 
            return [p_new]
        n = int(dist // spacing)
        pts = [(x0 + dx*(i/(n+1)), y0 + dy*(i/(n+1))) for i in range(1, n+1)]
        pts.append(p_new)
        return pts

    def add_point(self, x, y):
        if self.current is None:
            self.begin((0,255,0), 3, mode="draw")
        p_raw = (float(x), float(y))
        p_sm = ema(self._smooth_pt, p_raw, a=0.25)
        if self.current.points:
            self.current.points.extend(self._densify(self.current.points[-1], p_sm))
        else:
            self.current.points.append(p_sm)
        self._smooth_pt = p_sm

    def end(self):
        if self.current and len(self.current.points) > 1:
            self._page().append(self.current)
            self._redo_stack().clear()
        self.current = None

    # ---------- render ----------
    def _draw_grid(self, target):
        grid_color = (0,255,0,40)
        step_x = max(60, self.W//12)
        step_y = max(60, self.H//12)
        for x in range(0, self.W, step_x):
            cv2.line(target, (x,0), (x,self.H), grid_color, 1)
        for y in range(0, self.H, step_y):
            cv2.line(target, (0,y), (self.W,y), grid_color, 1)

    def _render_to(self, include_grid: bool):
        out = np.zeros((self.H, self.W, 4), dtype=np.uint8)
        if self.dark_bg:
            out[:,:,3] = 20  # subtle alpha plate for contrast

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
                    # soft shadow + ink
                    cv2.line(out, (p0[0]+1,p0[1]+1), (p1[0]+1,p1[1]+1), (0,0,0,100), s.width+1, lineType=cv2.LINE_AA)
                    cv2.line(out, p0, p1, col, s.width, lineType=cv2.LINE_AA)
            else:
                for i in range(1, len(s.points)):
                    p0 = (int(s.points[i-1][0]), int(s.points[i-1][1]))
                    p1 = (int(s.points[i][0]),   int(s.points[i][1]))
                    cv2.line(erase_mask, p0, p1, 0, s.width, lineType=cv2.LINE_AA)

        alpha = out[:,:,3]
        alpha = cv2.bitwise_and(alpha, erase_mask)
        out[:,:,3] = alpha

        if include_grid:
            grid = np.zeros_like(out)
            self._draw_grid(grid)
            gb,gg,gr,ga = cv2.split(grid)
            ob,og,or_,oa = cv2.split(out)
            ga_f = ga.astype(np.float32)/255.0
            inv = 1.0 - ga_f
            b = (gb.astype(np.float32)*ga_f + ob.astype(np.float32)*inv).astype(np.uint8)
            g = (gg.astype(np.float32)*ga_f + og.astype(np.float32)*inv).astype(np.uint8)
            r = (gr.astype(np.float32)*ga_f + or_.astype(np.float32)*inv).astype(np.uint8)
            a = np.clip(ga.astype(np.int32) + oa.astype(np.int32), 0, 255).astype(np.uint8)
            out = cv2.merge([b,g,r,a])

        return out

    def render(self):
        self.canvas = self._render_to(include_grid=self.show_grid)
        return self.canvas

    # ---------- persistence ----------
    def save(
        self,
        out_root="out",
        session_id: Optional[str]=None,
        page_idx: Optional[int]=None,
        H0=None,
        curr_quad=None,
        color_idx=None,
        draw_width=None,
        hotbar_idx=None,
        page_count: Optional[int]=None,
        composite_bgr: Optional[np.ndarray]=None,
        prefix: str = "save",
        prune_previous: bool = True
    ):
        """
        Saves to out/<session_id>/:
          - <prefix>_composite_<page>_<ts>.jpg   (camera rectified + strokes overlay)
          - <prefix>_strokes_<page>_<ts>.json    (current page strokes)
          - <prefix>_meta_<ts>.json              (session/page/tool metadata)
        Any legacy board/preview files are pruned; only the composite is kept.
        """
        ts = int(time.time())
        sid = session_id or str(ts)
        page = self.page_idx if page_idx is None else int(page_idx)
        out_dir = os.path.join(out_root, sid)
        os.makedirs(out_dir, exist_ok=True)

        # Compose composite if not given (fallback to strokes over transparent bg)
        if composite_bgr is None:
            strokes = self._render_to(include_grid=False)
            composite_bgr = cv2.cvtColor(strokes, cv2.COLOR_BGRA2BGR)

        comp_path = os.path.join(out_dir, f"{prefix}_composite_{page}_{ts}.jpg")
        json_path = os.path.join(out_dir, f"{prefix}_strokes_{page}_{ts}.json")
        meta_path = os.path.join(out_dir, f"{prefix}_meta_{ts}.json")

        cv2.imwrite(comp_path, composite_bgr)

        with open(json_path, "w") as f:
            json.dump(
                [{"mode": s.mode, "color": s.color, "width": s.width, "points": s.points}
                 for s in self._page()],
                f
            )

        meta = {
            "W": self.W, "H": self.H,
            "page_idx": page,
            "page_count": page_count if page_count is not None else self.page_count(),
            "color_idx": int(color_idx) if color_idx is not None else None,
            "draw_width": int(draw_width) if draw_width is not None else None,
            "hotbar_idx": int(hotbar_idx) if hotbar_idx is not None else None,
            "H0": (np.asarray(H0).tolist() if H0 is not None else None),
            "curr_quad": (np.asarray(curr_quad).tolist() if curr_quad is not None else None),
            "session_id": sid,
            "timestamp": ts,
            "prefix": prefix
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        # Prune: keep only newest composite/strokes/meta for this (prefix,page)
        if prune_previous:
            patterns = [
                os.path.join(out_dir, f"{prefix}_composite_{page}_*.jpg"),
                os.path.join(out_dir, f"{prefix}_strokes_{page}_*.json"),
            ]
            for pat in patterns:
                files = sorted(glob.glob(pat))
                if len(files) > 1:
                    for old in files[:-1]:
                        try: os.remove(old)
                        except Exception: pass

            metas = sorted(glob.glob(os.path.join(out_dir, f"{prefix}_meta_*.json")))
            if len(metas) > 1:
                for old in metas[:-1]:
                    try: os.remove(old)
                    except Exception: pass

            # Also nuke any legacy board/preview leftovers
            for pat in [
                os.path.join(out_dir, f"{prefix}_board_*"),
                os.path.join(out_dir, f"{prefix}_board_preview_*"),
                os.path.join(out_dir, f"{prefix}_camera_*"),
            ]:
                for old in glob.glob(pat):
                    try: os.remove(old)
                    except Exception: pass

        return comp_path, json_path, meta_path
