# one_euro_filter.py

from __future__ import annotations
import math
import time
from typing import Optional

class OneEuroFilter:
    """
    The 1€ filter (One Euro Filter) for smoother data.
    Implements a simple, yet powerful, low-pass filter with adaptive
    smoothing based on the rate of change of the input signal.

    Use one filter instance per coordinate (x, y, z).
    """

    def __init__(self, freq: float, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        """
        Args:
            freq (float): Sampling frequency of the signal (e.g., FPS).
            min_cutoff (float): Minimum cutoff frequency (f_c_min). Lower -> more smooth.
            beta (float): Beta value (beta). Higher -> more aggressive smoothing for fast movements.
            d_cutoff (float): Cutoff frequency for the derivation filter (f_d).
        """
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        # State
        self.x_prev: Optional[float] = None
        self.dx_prev: Optional[float] = None
        self.t_prev: Optional[float] = None

    def _alpha(self, cutoff: float) -> float:
        """Computes the smoothing factor 'alpha' for a given cutoff frequency."""
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def _smooth_data(self, alpha: float, x_curr: float, x_prev: Optional[float]) -> float:
        if x_prev is None:
            return x_curr
        return alpha * x_curr + (1.0 - alpha) * x_prev

    def filter(self, x_curr: float, t_curr: Optional[float] = None) -> float:
        """Applies the 1€ filter to the current data point."""
        if t_curr is None:
            t_curr = time.time()

        # 1. Compute dt (time elapsed)
        if self.t_prev is None:
            dt = 1.0 / self.freq
        else:
            dt = t_curr - self.t_prev
        self.t_prev = t_curr

        # Adjust frequency based on actual dt
        if dt > 0:
            self.freq = 1.0 / dt

        # 2. Estimate the rate of change (derivation)
        dx_curr = (x_curr - self.x_prev) / dt if self.x_prev is not None else 0.0

        # 3. Smooth the rate of change (low-pass filter)
        alpha_d = self._alpha(self.d_cutoff)
        dx_filt = self._smooth_data(alpha_d, dx_curr, self.dx_prev)
        self.dx_prev = dx_filt

        # 4. Compute adaptive cutoff frequency f_c
        f_c = self.min_cutoff + self.beta * abs(dx_filt)

        # 5. Smooth the signal itself (using the adaptive f_c)
        alpha_x = self._alpha(f_c)
        x_filt = self._smooth_data(alpha_x, x_curr, self.x_prev)
        self.x_prev = x_filt

        return x_filt