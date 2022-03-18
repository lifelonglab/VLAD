from typing import Optional, Dict, List

import numpy as np

from models.our.cpds.cpd import CPD, ChangePoint
from models.our.cpds.lifewatch.lifewatch import LIFEWATCH
from models.our.memories.hierarchical_organization import HierarchicalOrganization
from models.our.memories.memory import Memory
from models.our.memories.summarization.centroids import k_means_summarization
from models.our.memories.summarization.pyramid_calculator import compute_pyramid_size


class HierarchicalLifewatchMemory(CPD, Memory):
    def __init__(self, max_samples=3_000, threshold_ratio=2, max_size_ratio=2, subconcept_threshold_ratio=5, disable_cpd=False, disable_replay=False):
        self.lifewatch = LIFEWATCH(size_limit=0, threshold_ratio=threshold_ratio)
        self.hierarchy = HierarchicalOrganization(subconcept_threshold_ratio=subconcept_threshold_ratio)
        self.max_samples = max_samples
        self.max_size_ratio = max_size_ratio
        self.disable_cpd = disable_cpd
        self.disable_replay = disable_replay

    def detect_cp(self, data) -> List[ChangePoint]:
        if self.disable_cpd:
            self.lifewatch.detect_cp(data)
            return []
        return self.lifewatch.detect_cp(data)

    def name(self) -> str:
        no_cpd = '_no_cpd' if self.disable_cpd else ''
        no_replay = '_no_replay' if self.disable_replay else ''
        return f'HLW_fast_lim_{self.max_samples}_p{str(self.lifewatch.threshold_ratio)}_mf_{self.max_size_ratio}_str_{self.hierarchy.subconcept_threshold_ratio}{no_cpd}{no_replay}'

    def params(self) -> Dict:
        return {**self.lifewatch.params(), 'max_samples': self.max_samples, **self.hierarchy.params()}

    def new_data(self, data, is_new_dist: bool, distribution: Optional[int] = None):
        pass    # handled in lifewatch

    def get_replay(self) -> np.ndarray:
        if self.disable_replay:
            return np.empty(np.array(self.lifewatch.distributions[0]).shape)
        self.summarize()
        print(self.hierarchy.hierarchy)
        print(self.hierarchy.dists_by_layer())
        return np.concatenate(list(self.lifewatch.distributions.values()))

    def samples_number(self) -> int:
        return sum([len(d) for d in self.lifewatch.distributions.values()])

    def organize(self):
        if self.disable_replay: return
        print('Organizing hierarchy')
        self.hierarchy.organize(self.lifewatch.distributions, self.lifewatch.thresholds)

    def should_summarize(self) -> bool:
        return self.samples_number() > self.max_samples * self.max_size_ratio

    def summarize(self):
        if self.disable_replay:
            return
        print('Summarizing memory')
        sizes = compute_pyramid_size(self.hierarchy.dists_by_layer(), self.max_samples)

        new_distributions = {}
        for dist_id, data in self.lifewatch.distributions.items():
            summarized_data = k_means_summarization(data, max_data_length=sizes[dist_id])
            new_distributions[dist_id] = summarized_data.tolist()

        # new_distributions = {k: d for k, d in new_distributions.items() if len(d) >= 5}
        #
        self.lifewatch.set_distributions(new_distributions)

    def additional_measurements(self) -> Dict:
        return {'concepts': len(self.lifewatch.distributions), 'hierarchy': self.hierarchy.serializable_hierarchy()}
