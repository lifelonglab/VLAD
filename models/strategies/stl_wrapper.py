from typing import Callable

from models.model import Model


class SingleTaskLearnerWrapper(Model):
    """
    SingleTaskLearnerWrapper is a wrapper for any model. It creates a new model every time it is supposed to learn
    new data.
    """
    def __init__(self, model_creation_fn: Callable[[], Model]):
        self.model_creation_fn = model_creation_fn
        self.model = None

    def learn(self, data) -> None:
        self.model = self.model_creation_fn()
        self.model.learn(data)

    def predict(self, data, task_name=None):
        return self.model.predict(data)

    def name(self):
        return f'SingleTaskLearner-{self.model.name() if self.model is not None else ""}'

    def parameters(self):
        return self.model.parameters()

