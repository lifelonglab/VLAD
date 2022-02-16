import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from metrics.tasks_matrix.metrics_matrix_per_task import base_metrics


def _extract_for(global_results, metric, results_store):
    metric_results = global_results[metric]
    results_store[metric].append(metric_results['value'])
    results_store[f'{metric}_bwt'].append(metric_results['backward_transfer']['bwt'])
    results_store[f'{metric}_rem'].append(metric_results['backward_transfer']['rem'])
    results_store[f'{metric}_bwt+'].append(metric_results['backward_transfer']['bwt+'])
    results_store[f'{metric}_forward_transfer'].append(metric_results['forward_transfer'])


def process_global_to_csv(path: Path):
    names = []
    strategy = []
    model_names = []
    results = defaultdict(list)
    for file in path.glob('*/*.json'):
        with open(file) as f:
            data = json.load(f)

            name = data['metadata']['name']
            names.append(name)
            strategy.append(data['metadata']['strategy'])
            model_names.append(data['metadata']['model'])

            global_results = data['results']['global']
            for metric in base_metrics:
                _extract_for(global_results, metric, results)

    df = pd.DataFrame()
    df['model'] = names
    df['strategy'] = strategy
    df['model_name'] = model_names
    for key, values in results.items():
        df[key] = values
    Path('out/results_analysis').mkdir(exist_ok=True)
    df.to_csv(f'out/results_analysis/analysis_{data["metadata"]["dataset"]}.csv', index=False)



if __name__ == '__main__':
    path = Path('out/results/credit_card')
    print(path.exists())
    process_global_to_csv(path)