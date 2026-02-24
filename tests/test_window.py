import numpy as np
from phv.metrics.window import in_window

def test_in_window():
    x = np.array([-1.0, 0.0, 0.5, 2.0])
    w = in_window(x, 0.0, 1.0)
    assert w.tolist() == [0, 1, 1, 0]
