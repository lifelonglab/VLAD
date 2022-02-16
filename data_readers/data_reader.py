from abc import ABC, abstractmethod
from typing import List, Iterable

from task import Task


class DataReader(ABC):

    @abstractmethod
    def dataset_id(self) -> str:
        ...

    @abstractmethod
    def load_test_tasks(self) -> List[Task]:
        ...

    @abstractmethod
    def iterate_tasks(self) -> Iterable[Task]:
        ...

    @abstractmethod
    def input_features(self) -> int:
        ...

