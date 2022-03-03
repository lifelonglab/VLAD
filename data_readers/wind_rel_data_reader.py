from typing import Iterable, List

from data_readers.clustered_data_reader import ClusteredDataReader
from data_readers.data_reader import DataReader
from task import Task


class WindEnergyDataReader(ClusteredDataReader):
    def __init__(self, file):
        super().__init__(file)

    def dataset_id(self) -> str:
        return 'wind_rel_wind'

    def input_features(self) -> int:
        return 10
