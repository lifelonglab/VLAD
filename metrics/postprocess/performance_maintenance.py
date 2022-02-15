from metrics.tasks_matrix.metrics_matrix_per_task import SingleMetricMatrix


class PerformanceMaintenanceDarpa:
    def process(self, input_results: SingleMetricMatrix):
        tasks_no = input_results.shape[0]
        sum_pm = 0
        count = 0

        for i in range(tasks_no):
            for j in range(i, tasks_no):
                pm_after_learning = sum(input_results[i][j:])
                diffs = [sum(input_results[k][j:]) - pm_after_learning for k in range(i+1, tasks_no)]
                sum_pm += sum(diffs)
                count += len(diffs)

        return sum_pm / count
