from typing import Iterable, List

import pandas as pd
from sklearn import preprocessing

from data_readers.data_reader import DataReader
from task import Task


class AdfaDataReader(DataReader):
    def __init__(self, first_snapshot_file, filepath):
        self.first_snapshot_file = first_snapshot_file
        self.filepath = filepath

        self.tasks = []
        for i in range(5):
            data, labels = self.read_task(i)
            self.tasks.append(Task(name=f'task_{i}', data=data, labels=[int(v) for v in labels.values]))

    def load_test_tasks(self) -> List[Task]:
        return self.tasks

    def iterate_tasks(self) -> Iterable[Task]:
        return self.tasks

    def read_task(self, i):
        if i == 0:
            return self._read(self.first_snapshot_file)
        else:
            return self._read(f'{self.filepath}_iter_{i+1}.csv')

    def _read(self, filepath):
        data = pd.read_csv(filepath, header=None)
        values = preprocessing.MinMaxScaler().fit_transform(data.iloc[:, :-1])
        labels = data.iloc[:, -1]
        return values, labels

    def dataset_id(self) -> str:
        return 'ADFA_5'


if __name__ == '__main__':
    reader = AdfaDataReader('data_with_attacks/Adduser_k_5_rate_10_iter_1.csv', 'data_with_attacks/adfa_ld_attacks/Adduser/k_5/rate_10/Adduser_k_5_rate_10')
    v, l = reader.read_task(2)
    print(v)
    print(l)