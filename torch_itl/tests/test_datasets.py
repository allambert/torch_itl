import numpy as np

from torch_itl.datasets import import_data_toy_quantile


def test_synthetic():
    x, y, _ = import_data_toy_quantile(100)
    np.testing.assert_equal(x.shape[0], y.shape[0])
