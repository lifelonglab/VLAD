from typing import Dict

import numpy as np

from models.model_base import ModelBase


class AlwaysValueModel(ModelBase):
    def __init__(self, value):
        self.value = value

    def name(self):
        return f'Always_{self.value}'

    def learn(self, data) -> None:
        pass

    def predict(self, data, task_name=None) -> (np.ndarray, np.ndarray):
        pred = np.array([self.value] * len(data))
        return pred, pred

    def parameters(self) -> Dict:
        return {}