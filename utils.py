import numpy as np


def remove_anomalies(data, predictions):
    return data[np.where(predictions == 0)]

