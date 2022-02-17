from matplotlib import pyplot as plt

from analysis.plots.results_loader import load_results
import seaborn as sns


def plot_metric_in_strategy(df, metric, out_dir):
    metrics_set = [metric, f'{metric}_bwt', f'{metric}_forward_transfer']
    metrics_ranges = [(0, 1), (-1, 1), (0, 1)]

    f, axes = plt.subplots(len(metrics_set), 1, figsize=(7, 10), sharex=True)

    for ax, m, metric_range in zip(axes, metrics_set, metrics_ranges):
        sns.barplot(x=df['pretty_name'], y=df[m], ax=ax)
        ax.set_ylim(*metric_range)
        ax.set(xlabel='')

    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.25, top=0.99)
    plt.savefig(f'{out_dir}/{metric}.png')
    plt.show()
