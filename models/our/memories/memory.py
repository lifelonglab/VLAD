from abc import ABC, abstractmethod
from typing import Optional, Dict

import numpy as np


class Memory(ABC):
    @abstractmethod
    def name(self):
        ...

    @abstractmethod
    def new_data(self, data, is_new_dist: bool, distribution: Optional[int] = None):
        ...

    @abstractmethod
    def get_replay(self) -> np.ndarray:
        ...

    @abstractmethod
    def params(self) -> Dict:
        ...

    @abstractmethod
    def samples_number(self) -> int:
        ...

    def organize(self):
        pass

    def should_summarize(self) -> bool:
        pass

    def summarize(self):
        pass

    def additional_measurements(self) -> Dict:
        return {}
