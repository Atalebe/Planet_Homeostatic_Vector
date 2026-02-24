from __future__ import annotations
import numpy as np
import pandas as pd

def mad_z(series: pd.Series) -> pd.Series:
    x = series.astype(float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return (x - med) * 0.0
    return (x - med) / mad
