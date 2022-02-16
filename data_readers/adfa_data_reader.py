from typing import Iterable, List

import pandas as pd
from sklearn import preprocessing

from data_readers.clustered_data_reader import ClusteredDataReader
from data_readers.data_reader import DataReader
from task import Task


class AdfaDataReader(ClusteredDataReader):
    def __init__(self, data_path, name):
        super().__init__(data_path)
        self.name = name

    def dataset_id(self) -> str:
        return self.name

    def input_features(self) -> int:
        return 6

if __name__ == '__main__':
    reader = AdfaDataReader('data/adfa/adfa.npy')
    v, l = reader.iterate_tasks()
    print(v)
    print(l)
