from typing import Dict, List

import numpy as np

from models.our.cpds.cpd import CPD, ChangePoint
from models.our.cpds.lifewatch.iterate_batches import iterate_batches
from models.our.cpds.lifewatch.wasserstein import wassertein_distance


class LIFEWATCH(CPD):
    def __init__(self, threshold_ratio=5, sample_size=1, size_limit=30, min_dist_size=50):
        self.threshold_ratio = threshold_ratio
        self.sample_size = sample_size
        self.size_limit = size_limit
        self.min_dist_size = min_dist_size

        self.distributions = {0: []}
        self.thresholds = {}
        self.is_creating_new_dist = True
        self.current_dist = None

    def detect_cp(self, batch: np.array) -> List[ChangePoint]:
        cps: List[ChangePoint] = []
        for mini_batch_id, mini_batch in enumerate(iterate_batches(batch, self.sample_size)):
            if len(mini_batch) != self.sample_size:
                continue
            if self.is_creating_new_dist:
                dist_id = len(self.distributions) - 1
                self.distributions[dist_id].extend(mini_batch)
                if len(self.distributions[dist_id]) >= self.min_dist_size:
                    self.update_threshold(dist_id)
                    self.is_creating_new_dist = False
                    self.current_dist = dist_id
            else:
                ratios = {dist_id: wassertein_distance(mini_batch, np.array(dist)) / self.thresholds[dist_id] for dist_id, dist
                          in self.distributions.items()}
                current_ratio = ratios[self.current_dist]
                if current_ratio < 1:  # the same dist
                    if len(self.distributions[self.current_dist]) < self.size_limit or self.size_limit == 0:
                        self.distributions[self.current_dist].extend(mini_batch.tolist())
                        if self.size_limit != 0:
                            self.update_threshold(self.current_dist)
                else:
                    best_dist, best_ratio = sorted(ratios.items(), key=lambda it: it[1])[0]
                    cp_index = mini_batch_id * self.sample_size
                    if best_ratio > 1:  # new dist
                        self.is_creating_new_dist = True
                        self.distributions[len(self.distributions)] = mini_batch.tolist()
                        cps.append(ChangePoint(index=cp_index, is_new_dist=True))
                    else:
                        self.current_dist = best_dist
                        if len(self.distributions[best_dist]) < self.size_limit or self.size_limit == 0:
                            self.distributions[best_dist].extend(mini_batch.tolist())
                            if self.size_limit != 0:
                                self.update_threshold(best_dist)
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
        self.is_creating_new_dist = False

    def name(self) -> str:
        return 'LIFEWATCH'

    def params(self) -> Dict:
        return {'threshold_ratio': self.threshold_ratio, 'sample_size': self.sample_size, 'size_limit': self.size_limit,
                'min_dist_size': self.min_dist_size}
