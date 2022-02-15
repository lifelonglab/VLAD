from typing import Iterable, List

import pandas as pd
from sklearn import preprocessing

from data_readers.clustered_data_reader import ClusteredDataReader
from data_readers.data_reader import DataReader
from task import Task


class AdfaDataReader(ClusteredDataReader):
    def __init__(self, data_path):
        super().__init__(data_path)

    def dataset_id(self) -> str:
        return 'ADFA-LD'


if __name__ == '__main__':
    reader = AdfaDataReader('data/adfa/adfa.npy')
    v, l = reader.iterate_tasks()
    print(v)
    print(l)
