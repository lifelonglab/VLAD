import json

import matplotlib.pyplot as plt
import seaborn as sns

def plot_avg_metric_over_time(results_file, metric, out_dir):
    with open(results_file) as f:
        data = json.load(f)



def plot_metric_for_task_over_time(results_file, metric, out_dir, task_no):
    plt.figure(figsize=(12, 8))
    with open(results_file) as f:
        data = json.load(f)
        tasks = data['metadata']['tasks']
        results = data['results']['tasks-matrix']

        values = []
        task_to_draw = tasks[task_no]
        for t in tasks:
            values.append(results[t][task_to_draw][metric])

        sns.lineplot(x=list(range(len(values))), y=values)
        plt.xlabel('Learning tasks')
        plt.ylabel(metric)
        plt.title(f'{metric} over time for {data["metadata"]["model"]} on {data["metadata"]["dataset"]}; task: {task_to_draw}')
        plt.savefig(f'{out_dir}/{metric}_over_time_{data["metadata"]["model"]}_{task_to_draw}.png')

