from data_readers.clustered_data_reader import ClusteredDataReader


class NslDataReader(ClusteredDataReader):
    def __init__(self, file, name='nsl'):
        super().__init__(file)
        self.name = name

    def dataset_id(self) -> str:
        return self.name

    def input_features(self) -> int:
        return 41
