from typing import Dict, Callable

from models.model import Model


class IncrementalTaskLearnerWrapper(Model):
    """
    IncrementalTaskLearnerWrapper is a wrapper for any model. It trains the same model on all incoming tasks, one at a time.
    """
    def __init__(self, model_creation_fn: Callable[[], Model]):
        self.model = model_creation_fn()

    def name(self):
        return f'IncrementalTaskLearner-{self.model.name()}'

    def learn(self, data) -> None:
        self.model.learn(data)

    def predict(self, data, task_name=None):
        return self.model.predict(data)

    def parameters(self) -> Dict:
        return self.model.parameters()
