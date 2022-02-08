import numpy as np

from metrics.tasks_matrix.metrics_matrix_per_task import SingleMetricMatrix


class BackwardTransferGlobal:
    """Following: 'Don’t forget, there is more than forgetting: new metrics for Continual Learning.'"""

    def process(self, input_results: SingleMetricMatrix):
        tasks_no = input_results.shape[0]

        sum_bwt = 0
        for i in range(1, tasks_no):
            for j in range(0, i):
                sum_bwt += input_results[i][j] - input_results[j][j]

        bwt = sum_bwt / (tasks_no * (tasks_no + 1) / 2)
        rem = 1 - abs(min(bwt, 0))
        bwt_plus = max(bwt, 0)
        return {'bwt': bwt, 'rem': rem, 'bwt+': bwt_plus}
