import json
from pathlib import Path

from metrics.postprocess.mean_metrics_per_task import MeanMetricsPerTask


def recompute_for_file(results_file):
    print(results_file)
    data = None
    with open(results_file) as f:
        data = json.load(f)
    results_matrix = data['results']['tasks-matrix']
    recomputed_mean = MeanMetricsPerTask().process(results_matrix)
    data['results']['task-avg'] = recomputed_mean
    json_str = json.dumps(data, indent=4)
    Path(results_file).write_text(json_str, encoding='utf-8')



if __name__ == '__main__':
    p = Path('/home/nyder/research/lifelong learning/lifelong-anomaly-detection/out/results')
    for f in p.rglob('**/*.json'):
        recompute_for_file(f)