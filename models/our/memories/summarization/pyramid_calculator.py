from typing import List, Dict


def compute_pyramid_size(dists_by_layer: List[List[int]], max_samples: int) -> Dict[int, int]:
    count = sum([len(dists) / (lay_no + 1) for lay_no, dists in enumerate(dists_by_layer)])
    samples_per_main_concept = int(max_samples / count)
    return {dist_id: max(5, int(samples_per_main_concept/(lay_no + 1))) for lay_no, dists in enumerate(dists_by_layer) for dist_id in dists}

