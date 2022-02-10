from abc import ABC, abstractmethod
from typing import Optional, Dict


class Memory(ABC):
    @abstractmethod
    def name(self):
        ...

    @abstractmethod
    def new_data(self, data, is_new_dist: bool, distribution: Optional[int] = None):
        ...

    @abstractmethod
    def get_replay(self):
        ...

    @abstractmethod
    def params(self) -> Dict:
        ...
