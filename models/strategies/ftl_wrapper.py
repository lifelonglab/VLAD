from typing import Callable

from models.model import Model


class FirstTaskLearnerWrapper(Model):
    """
    FirstTaskLearnerWrapper is a wrapper for any model. It trains model just once on the first task.
    """
    def __init__(self, model_creation_fn: Callable[[], Model]):
        self.model_creation_fn = model_creation_fn
        self.model = None
        self.results = {}   # we can store results, as we do not learn

    def learn(self, data) -> None:
        if self.model is None:
            self.model = self.model_creation_fn()
            self.model.learn(data)

    def predict(self, data, task_name=None):
        if task_name is None:
            return self.model.predict(data)
        if task_name not in self.results:
            self.results[task_name] = self.model.predict(data)
        return self.results[task_name]

    def name(self):
        return f'FirstTaskLearner-{self.model.name() if self.model is not None else ""}'

    def parameters(self):
        return self.model.parameters()
