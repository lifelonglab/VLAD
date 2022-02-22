import json

import matplotlib.pyplot as plt
import seaborn as sns


def plot_avg_metric_over_time(results_file, metric, out_dir):
    print(results_file)
    plt.figure(figsize=(12, 8))
    with open(results_file) as f:
        data = json.load(f)
        tasks = data['metadata']['tasks']
        results = data['results']['task-avg']

        values = []
        for t in tasks:
            values.append(results[str(t)][metric])

        sns.lineplot(x=list(range(len(values))), y=values)
        plt.xlabel('Learning tasks')
        plt.ylim((0, 1))
        plt.ylabel(metric)
        plt.title(f'{metric} over time for {data["metadata"]["model"]} on {data["metadata"]["dataset"]};')
        plt.savefig(f'{out_dir}/avg-{metric}_over_time_{data["metadata"]["model"]}.png')
        plt.close()


def plot_metric_for_task_over_time(results_file, metric, out_dir, task_no):
    plt.figure(figsize=(12, 8))
    with open(results_file) as f:
        data = json.load(f)
        tasks = data['metadata']['tasks']
        results = data['results']['tasks-matrix']

        values = []
        task_to_draw = tasks[task_no]
        for t in tasks:
            values.append(results[str(t)][str(task_to_draw)][metric])

        sns.lineplot(x=list(range(len(values))), y=values)
        plt.xlabel('Learning tasks')
        plt.ylabel(metric)
        plt.ylim((0, 1))
        plt.title(f'{metric} over time for {data["metadata"]["model"]} on {data["metadata"]["dataset"]}; \n task no: {task_no} -- {task_to_draw}')
        plt.savefig(f'{out_dir}/{metric}_over_time_{data["metadata"]["model"]}_task_{task_no}.png')
        plt.close()

