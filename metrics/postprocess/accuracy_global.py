import numpy as np


class MetricGlobal:
    """Following: 'Don’t forget, there is more than forgetting: new metrics for Continual Learning.'"""
    def process(self, input_results: np.ndarray):
        tasks_no = input_results.shape[0]
        val_sum = 0
        for j in range(tasks_no):
            for i in range(j, tasks_no):
                val_sum += input_results[i][j]

        return val_sum / (tasks_no * (tasks_no + 1) / 2)

