# board_canvas.py
"""
BoardCanvas: stroke-based notetaking canvas (multi-page, undo/redo, grid/dark modes)
with compact JSON persistence and composite save helpers.

Usage:
    from board_canvas import BoardCanvas, warp_point

    board = BoardCanvas(W=1200, H=800)
    board.begin((0, 255, 0), 4, mode="draw")
    board.add_point(100, 120)
    board.add_point(120, 140)
    board.end()

    img_bgra = board.render()  # BGRA with transparency
    comp_path, json_path, meta_path = board.save(
        out_root="out",
        session_id="20250101_abc123",
        composite_bgr=None,  # or supply a rectified camera image to blend with strokes
        prefix="save",
        prune_previous=True
    )
"""

from __future__ import annotations

import os
import io
import glob
import json
import time
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import cv2
import numpy as np


def warp_point(H: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    """
    Apply homography H to a point (x, y) in image coordinates.

    Args:
        H: 3x3 homography matrix
        x, y: point coordinates

    Returns:
        (x', y') warped coordinates (float)
    """
    p = np.array([x, y, 1.0], dtype=np.float32)
    q = H @ p
    w = q[2] if q[2] != 0 else 1e-6
    return float(q[0] / w), float(q[1] / w)


def _ema(prev: Optional[Tuple[float, float]], new: Tuple[float, float], a: float = 0.25) -> Tuple[float, float]:
    """
    Exponential moving average for 2D points. Heavier smoothing (smaller 'a') reduces jitter.

    Args:
        prev: previous smoothed point or None
        new: new raw point
        a: alpha in [0,1]; larger -> more responsive, smaller -> smoother

    Returns:
        Smoothed point (x, y)
    """
    if prev is None:
        return new
    return (a * new[0] + (1 - a) * prev[0], a * new[1] + (1 - a) * prev[1])


@dataclass
class Stroke:
    """
    A single stroke: sequence of points with width, color, and mode (draw or erase).
    Color is BGR tuple to match OpenCV convention.
    """
    mode: str = "draw"  # "draw" or "erase"
    color: Tuple[int, int, int] = (0, 255, 0)  # BGR
    width: int = 3
    points: List[Tuple[float, float]] = field(default_factory=list)


class BoardCanvas:
    """
    Stroke-based canvas that renders to BGRA (so it can be alpha-blended onto camera frames).
    Supports multiple pages, undo/redo, grid toggling, dark background plate, and JSON saves.
    """

    def __init__(self, W: int = 1200, H: int = 800):
        self.W: int = int(W)
        self.H: int = int(H)

        # Last full render (BGRA)
        self.canvas: np.ndarray = np.zeros((self.H, self.W, 4), dtype=np.uint8)

        # Page state
        self.pages: List[List[Stroke]] = [[]]     # list of stroke lists
        self._redos: List[List[Stroke]] = [[]]    # redo stacks per page
        self.page_idx: int = 0

        # Active stroke
        self.current: Optional[Stroke] = None
        self._smooth_pt: Optional[Tuple[float, float]] = None

        # UI toggles
        self.show_grid: bool = True
        self.dark_bg: bool = False

        # Densify spacing (px) to avoid gaps when moving fast
        self._densify_spacing: float = 2.5

        # Smoothing alpha for stroke points
        self._smooth_alpha: float = 0.25

    # ------------- Page helpers -------------

    def page_count(self) -> int:
        return len(self.pages)

    def _page(self) -> List[Stroke]:
        return self.pages[self.page_idx]

    def _redo_stack(self) -> List[Stroke]:
        return self._redos[self.page_idx]

    def new_page(self) -> None:
        self.end()
        self.pages.append([])
        self._redos.append([])
        self.page_idx = len(self.pages) - 1

    def next_page(self) -> None:
        self.end()
        if self.page_idx < len(self.pages) - 1:
            self.page_idx += 1

    def prev_page(self) -> None:
        self.end()
        if self.page_idx > 0:
            self.page_idx -= 1

    def clear_page(self) -> None:
        self.end()
        self.pages[self.page_idx] = []
        self._redos[self.page_idx] = []

    # ------------- Edit ops -------------

    def undo(self) -> None:
        self.end()
        if self._page():
            s = self._page().pop()
            self._redo_stack().append(s)

    def redo(self) -> None:
        self.end()
        if self._redo_stack():
            s = self._redo_stack().pop()
            self._page().append(s)

    # ------------- Drawing -------------

    def begin(self, color: Tuple[int, int, int], width: int, mode: str = "draw") -> None:
        """
        Begin a new stroke. If a stroke is already active, it will be replaced.
        """
        mode = "erase" if str(mode).lower().startswith("e") else "draw"
        self.current = Stroke(mode=mode, color=(int(color[0]), int(color[1]), int(color[2])), width=int(width))
        self._smooth_pt = None

    def _densify(self, p_last: Tuple[float, float], p_new: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Insert intermediate points between p_last and p_new when the gap is large,
        to avoid visible "holes" in lines when drawing quickly.
        """
        (x0, y0), (x1, y1) = p_last, p_new
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        spacing = float(self._densify_spacing)
        if dist <= spacing:
            return [p_new]
        n = int(dist // spacing)
        pts = [(x0 + dx * (i / (n + 1)), y0 + dy * (i / (n + 1))) for i in range(1, n + 1)]
        pts.append(p_new)
        return pts

    def add_point(self, x: float, y: float) -> None:
        """
        Add a point to the current stroke; creates a default draw stroke if none is active.
        Applies exponential smoothing and densification.
        """
        if self.current is None:
            self.begin((0, 255, 0), 3, mode="draw")

        p_raw = (float(x), float(y))
        p_sm = _ema(self._smooth_pt, p_raw, a=self._smooth_alpha)

        if self.current.points:
            self.current.points.extend(self._densify(self.current.points[-1], p_sm))
        else:
            self.current.points.append(p_sm)

        self._smooth_pt = p_sm

    def end(self) -> None:
        """
        Finish the current stroke and push it onto the page if it contains at least two points.
        Clears the redo stack for this page (standard editor semantics).
        """
        if self.current and len(self.current.points) > 1:
            self._page().append(self.current)
            self._redo_stack().clear()
        self.current = None
        self._smooth_pt = None

    # ------------- Rendering -------------

    def _draw_grid(self, target: np.ndarray) -> None:
        """
        Draw a subtle grid on the target BGRA buffer. Grid alpha is baked into the color's A channel.
        """
        grid_color = (0, 255, 0, 40)  # subtle translucent green
        step_x = max(60, self.W // 12)
        step_y = max(60, self.H // 12)
        for x in range(0, self.W, step_x):
            cv2.line(target, (x, 0), (x, self.H), grid_color, 1, cv2.LINE_AA)
        for y in range(0, self.H, step_y):
            cv2.line(target, (0, y), (self.W, y), grid_color, 1, cv2.LINE_AA)

    def _render_to(self, include_grid: bool) -> np.ndarray:
        """
        Render all strokes (and current active stroke, if any) into a BGRA image with premultiplied-like alpha logic
        for erasing. The base is transparent (alpha=0), with optional dark plate for contrast.
        """
        out = np.zeros((self.H, self.W, 4), dtype=np.uint8)

        if self.dark_bg:
            # Subtle plate to make strokes pop against busy camera textures
            out[:, :, 3] = 20  # small alpha so background camera still shows through

        # Prepare an erase mask (white = keep, black = erase)
        erase_mask = np.full((self.H, self.W), 255, dtype=np.uint8)

        seq = list(self._page())
        if self.current is not None:
            seq.append(self.current)

        for s in seq:
            if not s or len(s.points) < 2:
                continue

            if s.mode == "draw":
                col = (int(s.color[0]), int(s.color[1]), int(s.color[2]), 255)

                # Paint with a soft drop-shadow then the ink line
                for i in range(1, len(s.points)):
                    p0 = (int(s.points[i - 1][0]), int(s.points[i - 1][1]))
                    p1 = (int(s.points[i][0]), int(s.points[i][1]))

                    # shadow (slightly offset)
                    cv2.line(
                        out,
                        (p0[0] + 1, p0[1] + 1),
                        (p1[0] + 1, p1[1] + 1),
                        (0, 0, 0, 100),
                        s.width + 1,
                        lineType=cv2.LINE_AA,
                    )
                    # ink
                    cv2.line(out, p0, p1, col, s.width, lineType=cv2.LINE_AA)

            else:
                # Eraser: carve alpha by drawing into erase_mask
                for i in range(1, len(s.points)):
                    p0 = (int(s.points[i - 1][0]), int(s.points[i - 1][1]))
                    p1 = (int(s.points[i][0]), int(s.points[i][1]))
                    cv2.line(erase_mask, p0, p1, 0, s.width, lineType=cv2.LINE_AA)

        # Apply erase mask to the alpha channel
        alpha = out[:, :, 3]
        alpha = cv2.bitwise_and(alpha, erase_mask)
        out[:, :, 3] = alpha

        if include_grid:
            grid = np.zeros_like(out)
            self._draw_grid(grid)
            gb, gg, gr, ga = cv2.split(grid)
            ob, og, or_, oa = cv2.split(out)
            ga_f = ga.astype(np.float32) / 255.0
            inv = 1.0 - ga_f
            b = (gb.astype(np.float32) * ga_f + ob.astype(np.float32) * inv).astype(np.uint8)
            g = (gg.astype(np.float32) * ga_f + og.astype(np.float32) * inv).astype(np.uint8)
            r = (gr.astype(np.float32) * ga_f + or_.astype(np.float32) * inv).astype(np.uint8)
            a = np.clip(ga.astype(np.int32) + oa.astype(np.int32), 0, 255).astype(np.uint8)
            out = cv2.merge([b, g, r, a])

        return out

    def render(self) -> np.ndarray:
        """
        Public render: writes and returns the internal BGRA canvas.
        """
        self.canvas = self._render_to(include_grid=self.show_grid)
        return self.canvas

    # ------------- Persistence -------------

    def save(
        self,
        out_root: str = "out",
        session_id: Optional[str] = None,
        page_idx: Optional[int] = None,
        H0: Optional[np.ndarray] = None,
        curr_quad: Optional[np.ndarray] = None,
        color_idx: Optional[int] = None,
        draw_width: Optional[int] = None,
        hotbar_idx: Optional[int] = None,
        page_count: Optional[int] = None,
        composite_bgr: Optional[np.ndarray] = None,
        prefix: str = "save",
        prune_previous: bool = True,
    ) -> Tuple[str, str, str]:
        """
        Persist the current page strokes and small metadata, plus a composite JPG.

        If composite_bgr is provided, it is saved as the composite image. Otherwise the strokes
        are rendered on a transparent BGRA and converted to BGR (no camera background).

        Files written to out/<session_id>/:
            <prefix>_composite_<page>_<ts>.jpg
            <prefix>_strokes_<page>_<ts>.json
            <prefix>_meta_<ts>.json

        When prune_previous=True, older versions for the same (prefix, page) are removed.
        Returns: (composite_path, strokes_json_path, meta_json_path)
        """
        ts = int(time.time())
        sid = session_id or str(ts)
        page = self.page_idx if page_idx is None else int(page_idx)

        out_dir = os.path.join(out_root, sid)
        os.makedirs(out_dir, exist_ok=True)

        # Prepare composite if not provided
        if composite_bgr is None:
            strokes = self._render_to(include_grid=False)  # BGRA
            composite_bgr = cv2.cvtColor(strokes, cv2.COLOR_BGRA2BGR)

        comp_path = os.path.join(out_dir, f"{prefix}_composite_{page}_{ts}.jpg")
        json_path = os.path.join(out_dir, f"{prefix}_strokes_{page}_{ts}.json")
        meta_path = os.path.join(out_dir, f"{prefix}_meta_{ts}.json")

        # Save composite
        cv2.imwrite(comp_path, composite_bgr)

        # Save strokes (compact JSON)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "mode": s.mode,
                        "color": [int(s.color[0]), int(s.color[1]), int(s.color[2])],
                        "width": int(s.width),
                        "points": [[float(px), float(py)] for (px, py) in s.points],
                    }
                    for s in self._page()
                ],
                f,
                separators=(",", ":"),
            )

        # Metadata (keep it small but useful)
        meta = {
            "W": self.W,
            "H": self.H,
            "page_idx": page,
            "page_count": int(page_count) if page_count is not None else self.page_count(),
            "color_idx": int(color_idx) if color_idx is not None else None,
            "draw_width": int(draw_width) if draw_width is not None else None,
            "hotbar_idx": int(hotbar_idx) if hotbar_idx is not None else None,
            "H0": (np.asarray(H0).tolist() if H0 is not None else None),
            "curr_quad": (np.asarray(curr_quad).tolist() if curr_quad is not None else None),
            "session_id": sid,
            "timestamp": ts,
            "prefix": str(prefix),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, separators=(",", ":"))

        # Prune older files for this (prefix, page)
        if prune_previous:
            self._prune_older_versions(out_dir, prefix, page, keep_latest=1)

        # Also clean up any legacy naming patterns that may linger
        self._prune_legacy(out_dir, prefix)

        return comp_path, json_path, meta_path

    # ------------- Pruning -------------

    @staticmethod
    def _prune_older_versions(out_dir: str, prefix: str, page: int, keep_latest: int = 1) -> None:
        """
        Keep only the newest N composite/strokes/meta files for a given (prefix, page).
        """
        patterns = [
            os.path.join(out_dir, f"{prefix}_composite_{page}_*.jpg"),
            os.path.join(out_dir, f"{prefix}_strokes_{page}_*.json"),
        ]
        for pat in patterns:
            files = sorted(glob.glob(pat))
            if len(files) > keep_latest:
                for old in files[:-keep_latest]:
                    try:
                        os.remove(old)
                    except Exception:
                        pass

        # Meta files are not page-specific; still keep the newest N
        metas = sorted(glob.glob(os.path.join(out_dir, f"{prefix}_meta_*.json")))
        if len(metas) > keep_latest:
            for old in metas[:-keep_latest]:
                try:
                    os.remove(old)
                except Exception:
                    pass

    @staticmethod
    def _prune_legacy(out_dir: str, prefix: str) -> None:
        """
        Remove stale files from older naming schemes (board, preview, camera) if present.
        """
        legacy = [
            os.path.join(out_dir, f"{prefix}_board_*"),
            os.path.join(out_dir, f"{prefix}_board_preview_*"),
            os.path.join(out_dir, f"{prefix}_camera_*"),
        ]
        for pat in legacy:
            for old in glob.glob(pat):
                try:
                    os.remove(old)
                except Exception:
                    pass
