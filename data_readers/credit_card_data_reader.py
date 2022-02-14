from typing import Iterable, List

import numpy as np

from data_readers.data_reader import DataReader
from task import Task


class CreditCardDataReader(DataReader):
    def __init__(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        self.train_tasks = []
        self.test_tasks = []

        for c in data:
            self.train_tasks.append(Task(name=c['name'], data=c['train_data'], labels=None))
            test_data = c['test_data']
            test_labels = c['test_labels']
            self.test_tasks.append(Task(name=c['name'], data=test_data, labels=test_labels))

    def dataset_id(self) -> str:
        return 'credit_card'

    def load_test_tasks(self) -> List[Task]:
        return self.test_tasks

    def iterate_tasks(self) -> Iterable[Task]:
        return self.train_tasks
