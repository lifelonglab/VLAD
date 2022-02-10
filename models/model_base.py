from abc import ABC, abstractmethod
from typing import Dict


class ModelBase(ABC):
    @abstractmethod
    def name(self):
        ...

    @abstractmethod
    def learn(self, data) -> None:
        ...

    @abstractmethod
    def predict(self, data, task_name=None):
        ...

    @abstractmethod
    def parameters(self) -> Dict:
        ...
