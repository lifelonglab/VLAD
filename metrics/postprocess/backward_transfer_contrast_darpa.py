import numpy as np

from metrics.tasks_matrix.metrics_matrix_per_task import SingleMetricMatrix


class BackwardTransferContrastDarpaGlobal:
    """Following: DARPA'"""

    def process(self, input_results: SingleMetricMatrix):
        tasks_no = input_results.shape[0]
        if tasks_no == 1: return 0

        values = []
        for i in range(1, tasks_no):
            for j in range(0, i):
                v1 = input_results[i][j]
                v0 = input_results[i-1][j]
                contrast_val = (v1 - v0) / (v0 + v1)
                values.append(contrast_val)
        return {'mean': np.mean(values), 'std': np.std(values)}
