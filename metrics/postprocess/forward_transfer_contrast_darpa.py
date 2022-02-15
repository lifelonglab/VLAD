from metrics.tasks_matrix.metrics_matrix_per_task import SingleMetricMatrix


class ForwardTransferContrastDarpaGlobal:
    """Following: DARPA'"""

    def process(self, input_results: SingleMetricMatrix):
        tasks_no = input_results.shape[0]

        sum_fwt = 0
        count = 0
        for i in range(1, tasks_no):       # after learning task i (cannot do this for the first task, as we dont have anything to compare before)
            for j in range(i + 1, tasks_no):    # for tasks with higher id than i
                v1 = input_results[i][j]
                v0 = input_results[i-1][j]
                contrast_val = (v1 - v0) / (v0 + v1)
                sum_fwt += contrast_val
                count += 1

        return sum_fwt / count

