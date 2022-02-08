import numpy as np


def mse(data, predictions):
    return np.mean(np.power(data - predictions, 2), axis=1)