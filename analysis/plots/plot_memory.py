import json
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(style="darkgrid")
sns.set(rc = {'figure.figsize':(12,6)})


datasets_translations = {
    'ngids_kfold_memory_adaptive': 'NGIDS',
    '3ids3_kfold_memory': '3IDS',
    'nsl_8_kfold_memory': 'NSL-KDD',
    'wind_5_kfold_memory': 'WIND',
    'unsw_10_kfold_memory': 'UNSW'
}


def _load_data_from_dataset(dataset):
    path = Path(f'out/results/{dataset}/IncrementalBatchLearner_0')

    if not path.exists():
        exit('no such directory')

    results = []

    for filepath in path.glob('*.json'):
        with open(filepath) as f:
            data = json.load(f)
            mem_size = data['metadata']['parameters']['memory']['max_samples']
            roc_auc = data['results']['global']['roc_auc']['value']['mean']
            bwt = data['results']['global']['roc_auc']['backward_transfer']['bwt']['mean']
            pretty_dataset = datasets_translations[dataset]
            results.append([mem_size, roc_auc, 'roc_auc', pretty_dataset])
            results.append([mem_size, bwt, 'bwt', pretty_dataset])

    return pd.DataFrame(results, columns=['memory', 'value', 'metric', 'dataset'])


def plot_single_memory(dataset):
    df = _load_data_from_dataset(dataset)

    g = sns.relplot(data=df, x='memory', y='value', hue='metric', kind='line')
    g.set_axis_labels('Memory size', 'Value')
    g.map(plt.axhline, y=0, color=".1", dashes=(2, 1), zorder=0)
    g.tight_layout()
    # g.map(plt.axhline, y=1, color=".1", dashes=(2, 1), zorder=0)
    # g.set(ylim=(-0.5, 1))
    plt.show()


def plot_all_memories(datasets):
    df = pd.concat([_load_data_from_dataset(dataset) for dataset in datasets], ignore_index=True)
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)

    for i, metric in enumerate(['roc_auc', 'bwt']):
        metric_df = df.loc[df['metric'] == metric]
        g = sns.lineplot(ax=axes[i], data=metric_df, x='memory', y='value', hue='dataset')
        axes[i].set_xlabel('Memory size')
        axes[i].set_ylabel(metric.upper())
        # g.map(plt.axhline, y=0, color=".1", dashes=(2, 1), zorder=0)
        # g.tight_layout()
        # g.map(plt.axhline, y=1, color=".1", dashes=(2, 1), zorder=0)
        # g.set(ylim=(-0.5, 1))

    for i in range(2):
        axes[i].get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.9))
    plt.show()
    plt.close()


datasets = ['ngids_kfold_memory_adaptive', 'nsl_8_kfold_memory', 'unsw_10_kfold_memory', '3ids3_kfold_memory', 'wind_5_kfold_memory']

plot_all_memories(datasets)