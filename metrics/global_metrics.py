from metrics.postprocess.accuracy_global import MetricGlobal
from metrics.postprocess.backward_transfer_global import BackwardTransferGlobal
from metrics.postprocess.forward_transfer_global import ForwardTransferGlobal
from metrics.tasks_matrix.metrics_matrix_per_task import BaseMetricsMatrixPerTask, BaseMetric, base_metrics


def calculate_global_metrics(tasks_matrix_metrics: BaseMetricsMatrixPerTask):
    return {
        metric: {
            'value': (MetricGlobal().process(tasks_matrix_metrics.get_single_metric_matrix(metric))),
            'backward_transfer': BackwardTransferGlobal().process(
                tasks_matrix_metrics.get_single_metric_matrix(metric)),
            'forward_transfer': ForwardTransferGlobal().process(tasks_matrix_metrics.get_single_metric_matrix(metric))
        }
        for metric in base_metrics
    }
