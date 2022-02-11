from abc import ABC, abstractmethod
from typing import Dict, Optional


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

    def additional_measurements(self) -> Optional[Dict]:
        return None
