from pathlib import Path

from analysis.plots.plot_metric_in_strategy import plot_metric_in_strategy
from analysis.plots.plot_metric_over_time import plot_metric_for_task_over_time
from analysis.plots.plot_times import plot_training_times
from analysis.plots.pretty_name import pretty_name
from analysis.plots.results_loader import load_results
import seaborn as sns

sns.set_theme(style="darkgrid")

def filter_by_strategy(df, strategy):
    return df.loc[df['strategy'] == strategy]


strategy = 'IncrementalTaskLearner'
dataset = 'creditcard_flat50'
metric = 'pr_auc'

out_dir = f'out/plots/{dataset}/{strategy}'
Path(out_dir).mkdir(parents=True, exist_ok=True)


results_df = filter_by_strategy(load_results(dataset, results_type='results'), strategy)
plot_metric_in_strategy(results_df, metric, out_dir, dataset, strategy)

times_df = filter_by_strategy(load_results(dataset, results_type='times'), strategy)

plot_training_times(times_df, out_dir, dataset, strategy)


plot_metric_for_task_over_time(f'out/results/{dataset}/IncrementalTaskLearner/IncrementalTaskLearner_COPOD.json', metric, out_dir, 0)
plot_metric_for_task_over_time(f'out/results/{dataset}/IncrementalTaskLearner/IncrementalTaskLearner_VAE.json', metric, out_dir, 0)