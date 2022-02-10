import numpy as np

from models.our.utils import mse


def percentile_threshold(data, reconstruction, percentile) -> float:
    errors = mse(data, reconstruction)
    return np.percentile(errors, percentile)


def max_threshold(data, reconstruction) -> float:
    errors = mse(data, reconstruction)
    return np.max(errors)
