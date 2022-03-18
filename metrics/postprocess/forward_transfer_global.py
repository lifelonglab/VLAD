import numpy as np

from metrics.tasks_matrix.metrics_matrix_per_task import SingleMetricMatrix


class ForwardTransferGlobal:
    """Following: 'Don’t forget, there is more than forgetting: new metrics for Continual Learning.'"""

    def process(self, input_results: SingleMetricMatrix):
        tasks_no = input_results.shape[0]
        if tasks_no == 1: return 0

        values = []
        for j in range(tasks_no):
            for i in range(j):
                values.append(input_results[i][j])

        return {'mean': np.mean(values), 'std': np.std(values)}
