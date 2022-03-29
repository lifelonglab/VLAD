from matplotlib import pyplot as plt

from analysis.plots.results_loader import load_results
import seaborn as sns


def plot_metric_in_strategy(df, metric, out_dir, dataset, strategy):
    metrics_set = [metric, f'{metric}_bwt', f'{metric}_forward_transfer']
    # metrics_set = [metric]
    metrics_ranges = [(0, 1), None, (0, 1)]

    f, axes = plt.subplots(max(len(metrics_set), 2), 1, figsize=(7, 10), sharex=True)

    for ax, m, metric_range in zip(axes, metrics_set, metrics_ranges):
        std = df[f'{m}_std']
        print(std)
        sns.barplot(x=df['pretty_name'], y=df[m], yerr=std, ax=ax)
        if metric_range is not None:
            ax.set_ylim(*metric_range)
        ax.set(xlabel='')

    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.25, top=0.95)
    axes[0].set_title(f'{metric} for {dataset} in {strategy}')
    plt.savefig(f'{out_dir}/{metric}.png')
    # plt.show()
