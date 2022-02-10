from itertools import chain
from typing import Optional, Dict

import numpy as np

from models.our.memories.memory import Memory
from models.our.memories.summarization.centroids import k_means_summarization


class FlatMemoryWithSummarization(Memory):
    def __init__(self):
        self.memory = []
        self.store_limit = 1000

    def new_data(self, data, is_new_dist: bool, distribution: Optional[int]=None):
        if is_new_dist or len(self.memory) == 0:
            self.memory.append(data)
        else:
            self.memory[-1] = np.concatenate((self.memory[-1], data))

        self._summarize()

    def get_replay(self) -> np.ndarray:
        return np.array(list(chain(*self.memory)))

    def name(self):
        return 'FlatMemoryWithSummarization'

    def params(self) -> Dict:
        return {
            'store_limit': self.store_limit
        }

    def _summarize(self):
        limit_per_concept = self.store_limit / len(self.memory)
        threshold = limit_per_concept * 1.5

        new_memory = []
        for cluster_data in self.memory:
            if len(cluster_data) > threshold:
                new_memory.append(k_means_summarization(cluster_data, limit_per_concept))
            else:
                new_memory.append(cluster_data)

        self.memory = new_memory

    def samples_number(self) -> int:
        return sum([len(m) for m in self.memory])
