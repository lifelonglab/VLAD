from typing import Dict

from metrics.global_metrics import calculate_global_metrics
from metrics.postprocess.mean_metrics_per_task import MeanMetricsPerTask
from metrics.tasks_matrix.metrics_matrix_per_task import BaseMetricsMatrixPerTask
from metrics.tasks_matrix.predictions_collector import CollectedResults


def process_results(results: CollectedResults) -> Dict:
    base_metrics_per_task = BaseMetricsMatrixPerTask()
    base_metrics_per_task.process(results)

    avg_results = MeanMetricsPerTask().process(base_metrics_per_task.results())
    global_results = calculate_global_metrics(base_metrics_per_task)

    results_full = {
        'global': global_results,
        'task-avg': avg_results,
        'tasks-matrix': base_metrics_per_task.results()
    }
    return results_full
