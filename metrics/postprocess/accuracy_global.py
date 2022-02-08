import numpy as np


class AccuracyGlobal:
    """Following: 'Don’t forget, there is more than forgetting: new metrics for Continual Learning.'"""
    def process(self, input_results: np.ndarray):
        tasks_no = input_results.shape[0]
        acc_sum = 0
        for j in range(tasks_no):
            for i in range(j, tasks_no):
                acc_sum += input_results[i][j]

        return acc_sum / (tasks_no * (tasks_no + 1) / 2)