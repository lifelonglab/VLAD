from typing import Callable, List, Dict

import numpy as np

from models.model import Model
from strategies.strategy import Strategy
from task import Task


class KnowItAllLearnerWrapper(Strategy):
    """
    KnowItAllWrapper is a wrapper for any model. It trains model just once on the whole data from all tasks.
    """
    def __init__(self, model_creation_fn: Callable[[], Model], learning_tasks: List[Task]):
        self._model = model_creation_fn()
        self.learning_tasks = learning_tasks
        self.is_trained = False
        self.results = {}   # we can store results, as we do not learn

    def learn(self, _) -> None:
        if not self.is_trained:
            data = np.concatenate([t.data for t in self.learning_tasks])
            self._model.learn(data)
            self.is_trained = True
            
    def predict(self, data, task_name=None):
        if task_name is None:
            return self._model.predict(data)
        if task_name not in self.results:
            self.results[task_name] = self._model.predict(data)
        return self.results[task_name]

    def model(self) -> Model:
        return self._model

    def strategy_name(self):
        return 'KnowItAll'
