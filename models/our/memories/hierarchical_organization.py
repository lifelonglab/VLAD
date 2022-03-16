from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from models.our.cpds.lifewatch.wasserstein import wassertein_distance


@dataclass
class DistributionInHierarchy:
    dist_id: int
    children: List


class HierarchicalOrganization:
    def __init__(self, subconcept_threshold_ratio=100):
        self.hierarchy: List[DistributionInHierarchy] = []
        self.subconcept_threshold_ratio = subconcept_threshold_ratio

    def organize(self, dists: Dict[int, List], thresholds: Dict[int, float]):
        new_dists_keys = [key for key in dists.keys() if key not in self._all_keys_from_hierarchy()]
        already_existing_dists = {key: values for key, values in dists.items() if key not in new_dists_keys}

        # if first time - first is main concept
        if len(self.hierarchy) == 0:
            self.hierarchy.append(DistributionInHierarchy(dist_id=new_dists_keys[0], children=[]))
            new_dists_keys = new_dists_keys[1:]

        # decide whether subconcept or new dist
        for new_dist_id in new_dists_keys:
            dist_id, distance = self._find_closest_dist(dists[new_dist_id], already_existing_dists)
            print(distance)
            if dist_id is not None and distance < thresholds[dist_id] * self.subconcept_threshold_ratio:
                self._get_dist_by(dist_id).children.append(DistributionInHierarchy(new_dist_id, children=[]))
            else:
                self.hierarchy.append(DistributionInHierarchy(dist_id=new_dist_id, children=[]))

    def _find_closest_dist(self, new_dist: List, all_dists: Dict[int, List]) -> Tuple[int, float]:
        distances = [(dist_id, self._compute_distance(new_dist, dist_data)) for dist_id, dist_data in all_dists.items()]
        distances.sort(key=lambda x: x[1])
        return distances[0] if len(distances) > 0 else (None, 0)

    def _compute_distance(self, dist1, dist2):
        return wassertein_distance(np.array(dist1), np.array(dist2))

    def _all_keys_from_hierarchy(self):
        return [d.dist_id for d in self._flatten_hierarchical_dists()]

    def _flatten_hierarchical_dists(self) -> List[DistributionInHierarchy]:
        return _all_dists(self.hierarchy)

    def _get_dist_by(self, given_id) -> DistributionInHierarchy:
        return [d for d in self._flatten_hierarchical_dists() if d.dist_id == given_id][0]

    def dists_by_layer(self) -> List[List[int]]:
        return _all_dists_by_layer(self.hierarchy)

    def params(self) -> Dict:
        return {'subconcept_threshold_ratio': self.subconcept_threshold_ratio}

    def serializable_hierarchy(self) -> List:
        return _hierarchy_as_dict(self.hierarchy)


def _all_dists(dists: List[DistributionInHierarchy]) -> List[DistributionInHierarchy]:
    return dists + [c for d in dists for c in _all_dists(d.children)]


def _all_dists_by_layer(dists: List[DistributionInHierarchy]) -> List[List[int]]:
    if len(dists) == 0:
        return []
    return [[d.dist_id for d in dists], *_all_dists_by_layer([c for d in dists for c in d.children])]


def _hierarchy_as_dict(dists: List[DistributionInHierarchy]) -> List:
    if len(dists) == 0:
        return []
    return [{'dist_id': d.dist_id, 'children': _hierarchy_as_dict(d.children)} for d in dists]