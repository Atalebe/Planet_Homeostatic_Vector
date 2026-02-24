from __future__ import annotations

import os
import pandas as pd


def read_exoplanet_csv(path: str) -> pd.DataFrame:
    """
    NASA Exoplanet Archive CSV exports may begin with comment lines '# ...'
    then the real header later. comment='#' skips those lines.

    low_memory=False avoids dtype chunk warnings on wide tables.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing exoplanet CSV: {path}")

    df = pd.read_csv(path, comment="#", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df
