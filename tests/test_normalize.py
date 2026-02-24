import pandas as pd
from phv.metrics.normalize import mad_z

def test_mad_z_runs():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    z = mad_z(s)
    assert len(z) == 4
