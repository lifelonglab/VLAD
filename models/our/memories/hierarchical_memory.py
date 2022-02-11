from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

import numpy as np

from models.our.cpds.lifewatch.wasserstein import wassertein_distance
from models.our.memories.memory import Memory


@dataclass
class Distribution:
    given_id: int
    data: np.ndarray
    children: List


class HierarchicalMemory(Memory):
    def __init__(self):
        self.hierarchical_distributions = []
        self.new_dists = []

    def name(self):
        return 'HierarchicalMemory'

    def new_data(self, data, is_new_dist: bool, distribution: Optional[int] = None):
        if is_new_dist or len(self.hierarchical_distributions) == 0:
            distribution_id = distribution if distribution is not None else self.max_dist_id() + 1
            self.new_dists.append(Distribution(given_id=distribution_id, data=data, children=[]))
        else:
            dist = self._get_dist_by(distribution)
            dist.data = np.concatenate([dist.data, data])

        self._organize()

    def get_replay(self) -> np.ndarray:
        all_dists = self._all_dists()
        return np.concatenate([d.data for d in all_dists]) if len(all_dists) > 0 else []

    def params(self) -> Dict:
        return {}

    def samples_number(self) -> int:
        return sum([len(d.data) for d in self._all_dists()])

    def _organize(self):
        # find the closest dist
        for new_dist in self.new_dists:
            dist, distance = self._find_closest_dist(new_dist)

            if dist is not None and distance < 100:
                dist.children.append(new_dist)
            else:
                self.hierarchical_distributions.append(new_dist)

        self.new_dists = []
        print(self.hierarchical_distributions)
        # dist between distributions and some threshold?
        # dist between batches in distributions?
        # shall we keep data on all levels or just leaves?

    def _find_closest_dist(self, new_dist: Distribution) -> Tuple[Optional[Distribution], float]:
        distances = [(dist, self._compute_distance(new_dist, dist)) for dist in self._flatten_hierarchical_dists()]
        distances.sort(key=lambda x: x[1])
        return distances[0] if len(distances) > 0 else (None, 0)

    def _compute_distance(self, dist1, dist2):
        return wassertein_distance(dist1.data, dist2.data)

    def _get_dist_by(self, given_id) -> Distribution:
        return [d for d in self._all_dists() if d.given_id == given_id][0]

    def _all_dists(self) -> List[Distribution]:
        return _all_dists(self.hierarchical_distributions) + self.new_dists

    def _flatten_hierarchical_dists(self) -> List[Distribution]:
        return _all_dists(self.hierarchical_distributions)

    def max_dist_id(self):
        return max([d.given_id for d in self._all_dists()] + [-1])


def _all_dists(dists: List[Distribution]) -> List[Distribution]:
    return dists + [c for d in dists for c in _all_dists(d.children)]
