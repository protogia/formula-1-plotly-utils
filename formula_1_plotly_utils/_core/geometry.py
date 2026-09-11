from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable

def rotate(xy: Iterable[float], *, angle: float) -> np.ndarray:
    """Rotate a 2‑D coordinate array by *angle* (rad)."""
    rot = np.array([[np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot)



