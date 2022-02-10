from itertools import chain
from typing import Optional, Dict

import numpy as np

from models.our.memories.memory import Memory


class SimpleFlatMemory(Memory):
    def __init__(self):
        self.memory = []
        self.limit_per_task = 100

    def new_data(self, data, is_new_dist: bool, distribution: Optional[int]=None):
        if is_new_dist or len(self.memory) == 0:
            self.memory.append(data[:self.limit_per_task])
        else:
            self.memory[-1] = np.concatenate((self.memory[-1], data))[-self.limit_per_task:]

    def get_replay(self) -> np.ndarray:
        return np.array(list(chain(*self.memory)))

    def name(self):
        return 'SimpleFlatMemory'

    def params(self) -> Dict:
        return {
            'limit_per_concept': self.limit_per_task
        }

    def samples_number(self) -> int:
        return sum([len(m) for m in self.memory])
