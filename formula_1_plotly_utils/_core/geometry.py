import numpy as np
import pandas as pd
from typing import Iterable
from scipy.signal import savgol_filter


def rotate(xy: Iterable[float], *, angle: float) -> np.ndarray:
    """Rotate a 2‑D coordinate array by *angle* (rad)."""
    rot = np.array([[np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot)



def _smooth_series(s: pd.Series, window: int = 15, polyorder: int = 2) -> np.ndarray:
    """Safely applies a Savitzky-Golay filter to smooth discrete telemetry noise: https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter"""
    arr = s.to_numpy(dtype=float)
    n = len(arr)
    if n < 5:
        return arr
    w = min(window, n)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return arr
    p = min(polyorder, w - 1)
    return savgol_filter(arr, window_length=w, polyorder=p)