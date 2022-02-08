import math
from typing import Dict, Callable

from models.model import Model


class IncrementalBatchLearnerWrapper(Model):
    def __init__(self, model_creation_fn: Callable[[], Model]):
        self.model = model_creation_fn()
        self.batch_size = 1024

    def name(self):
        return f'IncrementalBatchLearner-{self.model.name()}'

    def learn(self, data) -> None:
        iterations = math.ceil(data.shape[0] / self.batch_size)
        for i in range(iterations):
            self.model.learn(data[i * self.batch_size: (i+1) * self.batch_size])

    def predict(self, data, task_name=None):
        return self.model.predict(data)

    def parameters(self) -> Dict:
        return {**self.model.parameters(), **{'batch_size': self.batch_size}}
