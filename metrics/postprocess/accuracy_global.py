import numpy as np


class MetricGlobal:
    """Following: 'Don’t forget, there is more than forgetting: new metrics for Continual Learning.'"""
    def process(self, input_results: np.ndarray):
        tasks_no = input_results.shape[0]

        values = []
        for j in range(tasks_no):
            for i in range(j, tasks_no):
                values.append(input_results[i][j])

        # mean = sum(values) / (tasks_no * (tasks_no + 1) / 2)
        # std = np.sqrt(sum([np.power(abs(val-mean), 2) for val in values]) / (tasks_no * (tasks_no + 1) / 2))
        return {
            'mean': np.mean(values),
            'std': np.std(values)
        }
