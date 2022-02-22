from pathlib import Path

from analysis.plots.plot_metric_in_strategy import plot_metric_in_strategy
from analysis.plots.plot_metric_over_time import plot_metric_for_task_over_time, plot_avg_metric_over_time
from analysis.plots.plot_times import plot_training_times
from analysis.plots.pretty_name import pretty_name
from analysis.plots.results_loader import load_results
import seaborn as sns

sns.set_theme(style="darkgrid")

def filter_by_strategy(df, strategy):
    return df.loc[df['strategy'] == strategy]

metrics = ['pr_auc', 'roc_auc', 'precision', 'recall']

strategy = 'IncrementalTaskLearner'
# strategy = 'IncrementalBatchLearner'
# datasets = ['creditcard_flat10', 'ngids_6', 'ngids_bosc_6', 'www_bosc_6_equalized_fixed']
datasets = ['energy_pv']
# datasets = ['full_adfa', 'full_adfa_bosc', 'full_adfa_bosc_unscaled']
# datasets = ['ngids_seq_5', 'ngids_clustered_5', 'ngids_clustered_5_closest_anomalies']

for dataset in datasets:
    for metric in metrics:
        out_dir = f'out/plots/{dataset}/{strategy}/{metric}'
        results_dir = f'out/results/{dataset}/{strategy}'
        Path(out_dir).mkdir(parents=True, exist_ok=True)


        results_df = filter_by_strategy(load_results(dataset, results_type='results'), strategy)
        plot_metric_in_strategy(results_df, metric, out_dir, dataset, strategy)

        times_df = filter_by_strategy(load_results(dataset, results_type='times'), strategy)

        plot_training_times(times_df, out_dir, dataset, strategy)

        detailed_out_dir = f'{out_dir}/detailed/'
        detailed_out_dir_tasks = f'{detailed_out_dir}/concepts/'
        Path(detailed_out_dir_tasks).mkdir(parents=True, exist_ok=True)

        for name in results_df['model']:
            plot_avg_metric_over_time(f'{results_dir}/{name}.json', metric, detailed_out_dir)
            for i in range(3):
                plot_metric_for_task_over_time(f'{results_dir}/{name}.json', metric, detailed_out_dir_tasks, i)