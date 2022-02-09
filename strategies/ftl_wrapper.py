from typing import Callable

from models.model import Model
from strategies.strategy import Strategy


class FirstTaskLearnerWrapper(Strategy):
    """
    FirstTaskLearnerWrapper is a wrapper for any model. It trains model just once on the first task.
    """
    def __init__(self, model_creation_fn: Callable[[], Model]):
        self.model_creation_fn = model_creation_fn
        self._model = self.model_creation_fn()
        self.is_trained = False
        self.results = {}   # we can store results, as we do not learn

    def learn(self, data) -> None:
        if not self.is_trained:
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
        return 'FirstTaskLearner'
