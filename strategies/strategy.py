from abc import ABC, abstractmethod
from typing import Dict

from models.model_base import ModelBase


class Strategy(ModelBase, ABC):

    @abstractmethod
    def model(self) -> ModelBase:
        ...

    @abstractmethod
    def strategy_name(self):
        ...

    def model_name(self):
        return self.model().name()

    def name(self):
        return f'{self.strategy_name()}_{self.model_name()}'

    def parameters(self) -> Dict:
        return self.model().parameters()
