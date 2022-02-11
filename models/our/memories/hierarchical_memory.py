from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np

from models.our.memories.memory import Memory


@dataclass
class Distribution:
    given_id: int
    data: np.ndarray
    children: List


class HierarchicalMemory(Memory):
    def __init__(self):
        self.distributions = []
        self.new_dists = []

    def name(self):
        return 'HierarchicalMemory'

    def new_data(self, data, is_new_dist: bool, distribution: Optional[int] = None):
        if is_new_dist or len(self.distributions) == 0:
            distribution_id = distribution if distribution is not None else self.max_dist_id() + 1
            self.new_dists.append(Distribution(given_id=distribution_id, data=data, children=[]))
        else:
            dist = self._get_dist_by(distribution)
            dist.data = np.concatenate([dist.data, data])

    def get_replay(self) -> np.ndarray:
        all_dists = self._all_dists()
        return np.concatenate([d.data for d in all_dists]) if len(all_dists) > 0 else []

    def params(self) -> Dict:
        return {}

    def samples_number(self) -> int:
        return sum([len(d.data) for d in self._all_dists()])

    def _organize(self):
        pass
        # dist between distributions and some threshold?
        # dist between batches in distributions?
        # shall we keep data on all levels or just leaves?

    def _get_dist_by(self, given_id) -> Distribution:
        return [d for d in self._all_dists() if d.given_id == given_id][0]

    def _all_dists(self) -> List[Distribution]:
        return _all_dists(self.distributions) + self.new_dists

    def max_dist_id(self):
        return max([d.given_id for d in self._all_dists()] + [-1])


def _all_dists(dists: List[Distribution]) -> List[Distribution]:
    return dists + [c for d in dists for c in _all_dists(d.children)]
