from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class ChangePoint:
    index: int
    is_new_dist: bool
    distribution: Optional[int] = None


class CPD(ABC):
    @abstractmethod
    def detect_cp(self, data) -> List[ChangePoint]:
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def params(self) -> Dict:
        ...