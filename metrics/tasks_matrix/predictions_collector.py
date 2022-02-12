from collections import defaultdict
from typing import List, Dict
from typing_extensions import TypedDict

import numpy as np


class PredictionsDict(TypedDict):
    learned_task: str
    test_task: str
    y_pred: np.ndarray
    y_true: np.ndarray
    scores: np.ndarray


CollectedResults = List[List[PredictionsDict]]


class PredictionsCollector:
    def __init__(self):
        self._results: Dict[str, Dict[str, PredictionsDict]] = defaultdict(dict)
        self.order = []

    def add(self, learned_task: str, test_task: str, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray):
        if learned_task not in self.order:
            self.order.append(learned_task)

        self._results[learned_task][test_task] = {'learned_task': learned_task, 'test_task': test_task,
                                                  'y_pred': y_pred, 'y_true': y_true, 'scores': scores}

    def results(self) -> CollectedResults:
        return [[self._results[lt][tt] for tt in self.order] for lt in self.order]
