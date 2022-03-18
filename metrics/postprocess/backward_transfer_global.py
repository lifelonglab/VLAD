import numpy as np

from metrics.tasks_matrix.metrics_matrix_per_task import SingleMetricMatrix


class BackwardTransferGlobal:
    """Following: 'Don’t forget, there is more than forgetting: new metrics for Continual Learning.'"""

    def process(self, input_results: SingleMetricMatrix):
        tasks_no = input_results.shape[0]
        if tasks_no == 1: return {'bwt': 0, 'rem': 0, 'bwt+': 0}

        values = []
        for i in range(1, tasks_no):
            for j in range(0, i):
                values.append(input_results[i][j] - input_results[j][j])


        bwt = np.mean(values)
        rem = 1 - abs(min(bwt, 0))
        bwt_plus = max(bwt, 0)

        print(f'Standard calculated bwt', bwt)
        print(f'Numpy calculated bwt', np.mean(values))
        return {'bwt': {
            'mean': bwt,
            'std': np.std(values)
        }, 'rem': rem, 'bwt+': bwt_plus}
