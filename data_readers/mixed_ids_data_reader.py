from data_readers.clustered_data_reader import ClusteredDataReader


class MixedIdsDataReader(ClusteredDataReader):
    def __init__(self, data_path, name):
        super().__init__(data_path)
        self.name = name

    def dataset_id(self) -> str:
        return self.name

    def input_features(self) -> int:
        return 6
