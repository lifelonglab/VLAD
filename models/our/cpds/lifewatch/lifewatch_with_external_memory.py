from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from models.our.cpds.cpd import CPD, ChangePoint
from models.our.cpds.lifewatch.iterate_batches import iterate_batches
from models.our.cpds.lifewatch.wasserstein import wassertein_distance


@dataclass
class Distribution:
    dist_id: str
    data: np.ndarray


class LIFEWATCHWithExternalMemory(CPD):
    def __init__(self, threshold_ratio=2, sample_size=1, min_dist_size=100):
        self.threshold_ratio = threshold_ratio
        self.sample_size = sample_size
        self.min_dist_size = min_dist_size
        self.distributions: Dict[any, List] = {}
        self.thresholds: Dict[any, float] = {}

        self.is_creating_new_dist = True
        self.creating_dist_id = 0
        self.current_dist = 0

    def detect_cp(self, data) -> List[ChangePoint]:
        cps: List[ChangePoint] = []
        for mini_batch_id, mini_batch in enumerate(iterate_batches(data, self.sample_size)):
            if len(mini_batch) != self.sample_size:
                continue
            if self.is_creating_new_dist:
                self.distributions[self.creating_dist_id].extend(mini_batch)
                if len(self.distributions[self.creating_dist_id]) >= self.min_dist_size:
                    self.update_threshold(self.creating_dist_id)
                    self.is_creating_new_dist = False
                    self.current_dist = self.creating_dist_id
                    self.creating_dist_id = None
            else:
                ratios = {dist_id: wassertein_distance(mini_batch, np.array(dist)) / self.thresholds[dist_id] for
                          dist_id, dist
                          in self.distributions.items()}
                current_ratio = ratios[self.current_dist]
                if current_ratio < 1:  # the same dist:
                    self.distributions[self.current_dist].extend(mini_batch.tolist())
                else:
                    best_dist, best_ratio = sorted(ratios.items(), key=lambda it: it[1])[0]
                    cp_index = mini_batch_id * self.sample_size
                    if best_ratio > 1:  # new dist
                        self.is_creating_new_dist = True
                        self.distributions[len(self.distributions)] = mini_batch.tolist()
                        cps.append(ChangePoint(index=cp_index, is_new_dist=True))
                    else:
                        self.current_dist = best_dist
                        self.distributions[best_dist].extend(mini_batch.tolist())
                        cps.append(ChangePoint(index=cp_index, is_new_dist=False, distribution=best_dist))
        return cps

    def update_threshold(self, dist_id):
        dist = np.array(self.distributions[dist_id])
        values = [wassertein_distance(np.array(s), dist) for s in iterate_batches(dist, self.sample_size)]
        self.thresholds[dist_id] = np.max(values) * self.threshold_ratio

    def set_distributions(self, distributions: Dict[any, List]):
        self.distributions = distributions
        for dist_id in self.distributions.keys():
            self.update_threshold(dist_id)

    def name(self) -> str:
        return 'LIFEWATCH-external'

    def params(self) -> Dict:
        return {'threshold_ratio': self.threshold_ratio, 'sample_size': self.sample_size,
                'min_dist_size': self.min_dist_size}
