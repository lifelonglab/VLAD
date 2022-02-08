from collections import defaultdict
from typing import List, Dict
from typing_extensions import Literal, TypedDict

import numpy as np
from sklearn.metrics import accuracy_score

from metrics.metric_utils import prec_rec_f1
from metrics.tasks_matrix.predictions_collector import CollectedResults, PredictionsDict

BaseMetric = Literal['precision', 'recall', 'f1', 'accuracy']
base_metrics: List[BaseMetric] = ['precision', 'recall', 'f1', 'accuracy']


class BaseMetricsResults(TypedDict):
    precision: float
    recall: float
    f1: float
    accuracy: float


TasksMatrixResults = Dict[str, Dict[str, BaseMetricsResults]]


SingleMetricMatrix = np.ndarray # it is supposed to be: NxN matrix, where N - number of tasks


class BaseMetricsMatrixPerTask:
    def __init__(self):
        self._results = None
        self.order = []

    def process(self, results: CollectedResults):
        matrix = defaultdict(dict)
        for learned_task_evaluation in results:
            for evaluation in learned_task_evaluation:
                matrix[evaluation['learned_task']][evaluation['test_task']] = self._compute_metrics(evaluation)

        self._results = matrix
        self.order = [ev[0]['learned_task'] for ev in results]

    def _compute_metrics(self, results: PredictionsDict) -> BaseMetricsResults:
        y_true = results['y_true']
        y_pred = results['y_pred']
        prec, rec, f1 = prec_rec_f1(y_true=y_true, y_pred=y_pred)
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        return {
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'accuracy': accuracy
        }

    def results(self) -> TasksMatrixResults:
        return self._results

    def get_single_metric_matrix(self, metric: BaseMetric) -> SingleMetricMatrix:
        tasks_no = len(self.order)
        matrix = np.zeros((tasks_no, tasks_no))
        for i, learned_task in enumerate(self.order):
            for j, test_task in enumerate(self.order):
                matrix[i][j] = self._results[learned_task][test_task][metric]

        return matrix
