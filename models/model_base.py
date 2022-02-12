from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np


class ModelBase(ABC):
    @abstractmethod
    def name(self):
        ...

    @abstractmethod
    def learn(self, data) -> None:
        ...

    @abstractmethod
    def predict(self, data, task_name=None) -> (np.ndarray, np.ndarray):
        ...

    @abstractmethod
    def parameters(self) -> Dict:
        ...

    def additional_measurements(self) -> Optional[Dict]:
        return None
