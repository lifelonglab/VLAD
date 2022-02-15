from typing import Iterable, List

import numpy as np

from data_readers.clustered_data_reader import ClusteredDataReader
from data_readers.data_reader import DataReader
from task import Task


class CreditCardDataReader(ClusteredDataReader):
    def __init__(self, data_path):
        super().__init__(self, data_path)

    def dataset_id(self) -> str:
        return 'credit_card'
