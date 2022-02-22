import random
from typing import Dict

import numpy as np

from models.model_base import ModelBase


class RandomModel(ModelBase):
    def name(self):
        return 'Random'

    def learn(self, data) -> None:
        pass

    def predict(self, data, task_name=None) -> (np.ndarray, np.ndarray):
        pred = [random.choice([0, 1]) for _ in range(len(data))]
        return pred, pred

    def parameters(self) -> Dict:
        return {}