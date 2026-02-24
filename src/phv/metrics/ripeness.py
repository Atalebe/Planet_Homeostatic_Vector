from __future__ import annotations
import numpy as np
import pandas as pd

def static_ripeness(window_member: pd.Series, st_age_gyr: pd.Series | None) -> pd.Series:
    """
    Static ripeness proxy: 1(window) * stellar age.
    If age missing, returns NaN for that row.
    """
    if st_age_gyr is None:
        return pd.Series(np.nan, index=window_member.index)
    age = st_age_gyr.astype(float)
    return window_member.astype(float) * age
