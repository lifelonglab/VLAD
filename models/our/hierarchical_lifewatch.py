from typing import Optional, Dict, List

import numpy as np

from models.our.cpds.cpd import CPD, ChangePoint
from models.our.cpds.lifewatch.lifewatch import LIFEWATCH
from models.our.memories.hierarchical_organization import HierarchicalOrganization
from models.our.memories.memory import Memory
from models.our.memories.summarization.pyramid_calculator import compute_pyramid_size


class HierarchicalLifewatchMemory(CPD, Memory):
    def __init__(self, max_samples = 1000):
        self.lifewatch = LIFEWATCH(size_limit=0)
        self.hierarchy = HierarchicalOrganization()
        self.max_samples = max_samples

    def detect_cp(self, data) -> List[ChangePoint]:
        return self.lifewatch.detect_cp(data)

    def name(self) -> str:
        return 'Hierarchical_LIFEWATCH'

    def params(self) -> Dict:
        return self.lifewatch.params()

    def new_data(self, data, is_new_dist: bool, distribution: Optional[int] = None):
        pass    # handled in lifewatch

    def get_replay(self) -> np.ndarray:
        self._organize()
        print(self.hierarchy.hierarchy)
        print(self.hierarchy.dists_by_layer())
        return np.concatenate(list(self.lifewatch.distributions.values()))

    def samples_number(self) -> int:
        return sum([len(d) for d in self.lifewatch.distributions.values()])

    def _organize(self):
        self.hierarchy.organize(self.lifewatch.distributions)

    def _summarize(self):
        sizes = compute_pyramid_size(self.hierarchy.dists_by_layer(), self.max_samples)
