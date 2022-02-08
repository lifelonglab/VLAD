from typing import Dict, get_args

from numpy import mean

from metrics.tasks_matrix.metrics_matrix_per_task import BaseMetricsResults, TasksMatrixResults, BaseMetric, \
    base_metrics


class MeanMetricsPerTask:
    def process(self, input_results: TasksMatrixResults):
        processed_results = {}

        for learned_task, task_results in input_results.items():
            processed_results[learned_task] = {metric: mean(values[metric]) for values in task_results.values() for
                                               metric in base_metrics}
        return processed_results
