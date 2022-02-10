from typing import Callable

from models.model_base import ModelBase
from strategies.strategy import Strategy


class SingleTaskLearnerWrapper(Strategy):
    """
    SingleTaskLearnerWrapper is a wrapper for any model. It creates a new model every time it is supposed to learn
    new data.
    """
    def __init__(self, model_creation_fn: Callable[[], ModelBase]):
        self.model_creation_fn = model_creation_fn
        self._model = self.model_creation_fn()

    def learn(self, data) -> None:
        self._model = self.model_creation_fn()
        self._model.learn(data)

    def predict(self, data, task_name=None):
        return self._model.predict(data)

    def model(self) -> ModelBase:
        return self._model

    def strategy_name(self):
        return 'SingleTaskLearner'

