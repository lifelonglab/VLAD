import math
from typing import Dict, Callable

from models.model_base import ModelBase
from strategies.strategy import Strategy


class IncrementalBatchLearnerWrapper(Strategy):
    def __init__(self, model_creation_fn: Callable[[], ModelBase], execution_no=0):
        self._model = model_creation_fn()
        self.batch_size = 1024
        self.execution_no = execution_no

    def learn(self, data) -> None:
        iterations = math.ceil(data.shape[0] / self.batch_size)
        for i in range(iterations):
            print(f'Going through iteration {i}')
            self._model.learn(data[i * self.batch_size: (i + 1) * self.batch_size])

    def predict(self, data, task_name=None):
        return self._model.predict(data)

    def parameters(self) -> Dict:
        return {**self._model.parameters(), **{'batch_size': self.batch_size}}

    def model(self) -> ModelBase:
        return self._model

    def strategy_name(self):
        return f'IncrementalBatchLearner_{self.execution_no}'

