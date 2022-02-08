from metrics.tasks_matrix.metrics_matrix_per_task import SingleMetricMatrix


class ForwardTransferGlobal:
    """Following: 'Don’t forget, there is more than forgetting: new metrics for Continual Learning.'"""

    def process(self, input_results: SingleMetricMatrix):
        tasks_no = input_results.shape[0]

        sum_fwt = 0
        for j in range(tasks_no):
            for i in range(j):
                sum_fwt += input_results[i][j]

        return sum_fwt / (tasks_no * (tasks_no - 1) / 2)
