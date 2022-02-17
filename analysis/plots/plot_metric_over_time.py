import json


def plot_avg_metric_over_time(results_file, metric, out_dir):
    with open(results_file) as f:
        data = json.load(f)
