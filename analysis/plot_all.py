from pathlib import Path

from analysis.plots.plot_metric_in_strategy import plot_metric_in_strategy
from analysis.plots.pretty_name import pretty_name
from analysis.plots.results_loader import load_results

strategy = 'IncrementalBatchLearner'
dataset = 'adfa_30'
metric = 'pr_auc'

df = load_results(dataset)
filtered_df = df.loc[df['strategy'] == strategy]
filtered_df['pretty_name'] = filtered_df['model_name'].apply(lambda x: pretty_name(x))

out_dir = f'out/plots/{dataset}/{strategy}'
Path(out_dir).mkdir(parents=True, exist_ok=True)

plot_metric_in_strategy(filtered_df, metric, out_dir)