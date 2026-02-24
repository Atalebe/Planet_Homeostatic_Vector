from __future__ import annotations
import numpy as np

def in_window(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return ((x >= lo) & (x <= hi)).astype(np.int8)
