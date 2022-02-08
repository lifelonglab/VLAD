from typing import Iterable, List

import numpy as np

from data_readers.data_reader import DataReader
from task import Task

machines = ['machine-1-1', 'machine-1-2', 'machine-1-3', 'machine-1-4', 'machine-1-5', 'machine-1-6', 'machine-1-7',
            'machine-1-8', 'machine-2-1', 'machine-2-2', 'machine-2-3', 'machine-2-4', 'machine-2-5', 'machine-2-6',
            'machine-2-7', 'machine-2-8', 'machine-2-9', 'machine-3-1', 'machine-3-2', 'machine-3-3', 'machine-3-4',
            'machine-3-5', 'machine-3-6', 'machine-3-7', 'machine-3-8', 'machine-3-9', 'machine-3-10', 'machine-3-11']

path = 'v2/data/smd'


class SmdDataReader(DataReader):
    def __init__(self):
        self.train = np.load(f'{path}/train.npy', allow_pickle=True)
        self.test = np.load(f'{path}/test.npy', allow_pickle=True)
        self.test_labels = np.load(f'{path}/test_label.npy', allow_pickle=True)

    def dataset_id(self) -> str:
        return 'SMD'

    def load_test_tasks(self) -> List[Task]:
        tasks = []
        for i, machine in enumerate(machines):
            tasks.append(Task(name=machine, data=self.test[i], labels=self.test_labels[i]))

        return tasks

    def iterate_tasks(self) -> Iterable[Task]:
        tasks = []
        for i, machine in enumerate(machines):
            tasks.append(Task(name=machine, data=self.train[i], labels=None))

        return tasks


if __name__ == '__main__':
    reader = SmdDataReader()
    for t in reader.iterate_tasks():
        print(t)