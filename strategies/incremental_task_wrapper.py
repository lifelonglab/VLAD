from typing import Dict, Callable

from models.model_base import ModelBase
from strategies.strategy import Strategy


class IncrementalTaskLearnerWrapper(Strategy):
    """
    IncrementalTaskLearnerWrapper is a wrapper for any model. It trains the same model on all incoming tasks, one at a time.
    """
    def __init__(self, model_creation_fn: Callable[[], ModelBase]):
        self._model = model_creation_fn()

    def learn(self, data) -> None:
        self._model.learn(data)

    def predict(self, data, task_name=None):
        return self._model.predict(data)

    def model(self) -> ModelBase:
        return self._model

    def strategy_name(self):
        return 'IncrementalTaskLearner'
