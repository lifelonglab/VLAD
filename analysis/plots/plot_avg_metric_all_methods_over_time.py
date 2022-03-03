import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_avg_metric_all_methods_over_time(results_path, metric, out_dir):
    print(results_path)
    plt.figure(figsize=(12, 8))
    values = {}
    for file in Path(results_path).glob('*.json'):
        print(file)
        with open(file) as f:
            data = json.load(f)
            model = data["metadata"]["model"]
            tasks = data['metadata']['tasks']
            results = data['results']['task-avg']

            method_values = []
            for t in tasks:
                method_values.append(results[str(t)][metric])

            values[model] = method_values

    df = pd.DataFrame()
    df['model'] = list(values.keys())
    df['values'] = list(values.values())
    df['time'] = [[str(i) for i in range(len(tasks))] for _ in range(len(values))]

    print(df)

    for model, model_values in values.items():
        sns.lineplot(x=list(range(len(tasks))), y=model_values, label=model)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('Learning tasks')
    plt.ylim((0, 1))
    plt.ylabel(metric)
    plt.title(f'{metric} over time on {data["metadata"]["dataset"]};')
    plt.savefig(f'{out_dir}/avg-{metric}_over_time_.png', bbox_inches='tight')
    plt.close()


def plot_avg_metric_all_methods_over_time_per_task(results_path, metric, out_dir):
    print(results_path)
    plt.figure(figsize=(12, 8))
    values_per_task = defaultdict(dict)
    for file in Path(results_path).glob('*.json'):
        print(file)
        with open(file) as f:
            data = json.load(f)
            model = data["metadata"]["model"]
            tasks = data['metadata']['tasks']
            results = data['results']['tasks-matrix']

            for task_to_draw in tasks:
                method_values = []
                for t in tasks:
                    method_values.append(results[str(t)][task_to_draw][metric])
                values_per_task[task_to_draw][model] = method_values

    for task_to_draw, values in values_per_task.items():
        df = pd.DataFrame()
        df['model'] = list(values.keys())
        df['values'] = list(values.values())
        df['time'] = [[str(i) for i in range(len(tasks))] for _ in range(len(values))]


        for model, model_values in values.items():
            sns.lineplot(x=list(range(len(tasks))), y=model_values, label=model)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.xlabel('Learning tasks')
        plt.ylim((0, 1))
        plt.ylabel(metric)
        plt.title(f'{metric} over time on {data["metadata"]["dataset"]}; \n Task: {task_to_draw}')
        plt.savefig(f'{out_dir}/avg-{metric}_task_{task_to_draw}_over_time_.png', bbox_inches='tight')
        plt.close()