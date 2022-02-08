from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ChangePoint:
    index: int
    is_new: bool
    distribution: Optional[int] = None


class CPD(ABC):
    @abstractmethod
    def detect_cp(self, data) -> List[ChangePoint]:
        ...
