import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from metrics.tasks_matrix.metrics_matrix_per_task import base_metrics


def _extract_for(global_results, metric, results_store):
    metric_results = global_results[metric]
    results_store[f'{metric}'].append(metric_results['value']['mean'])
    results_store[f'{metric}_std'].append(metric_results['value']['std'])
    results_store[f'{metric}_bwt'].append(metric_results['backward_transfer']['bwt']['mean'])
    results_store[f'{metric}_bwt_std'].append(metric_results['backward_transfer']['bwt']['std'])
    results_store[f'{metric}_rem'].append(metric_results['backward_transfer']['rem'])
    results_store[f'{metric}_bwt+'].append(metric_results['backward_transfer']['bwt+'])
    results_store[f'{metric}_forward_transfer'].append(metric_results['forward_transfer']['mean'])
    results_store[f'{metric}_forward_transfer_std'].append(metric_results['forward_transfer']['std'])
    # results_store[f'{metric}_forward_transfer_darpa'].append(metric_results['forward_transfer_darpa'])
    # results_store[f'{metric}_backward_transfer_darpa'].append(metric_results['backward_transfer_darpa'])
    # results_store[f'{metric}_performance_maintenance_darpa'].append(metric_results['performance_maintenance_darpa'])


def _extract_times(results, results_store):
    times_data = results['times']
    results_store['training_times'].append(times_data['all_trainings'])
    results_store['testing_times'].append(times_data['all_testings'])


def process_global_to_csv(path: Path):
    names = []
    strategy = []
    model_names = []
    results = defaultdict(list)
    times_results_store = defaultdict(list)

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

            _extract_times(data, times_results_store)

    for store, name in [(results, 'results'), (times_results_store, 'times')]:
        df = pd.DataFrame()
        df['model'] = names
        df['strategy'] = strategy
        df['model_name'] = model_names
        for key, values in store.items():
            df[key] = values
        Path(f'out/results_analysis/{data["metadata"]["dataset"]}').mkdir(exist_ok=True)
        df.to_csv(f'out/results_analysis/{data["metadata"]["dataset"]}/analysis_{name}.csv', index=False)


if __name__ == '__main__':
    # datasets = ['full_adfa', 'full_adfa_bosc', 'full_adfa_bosc_unscaled']
    datasets = ['full_ngids', 'ngids_seq_5', 'ngids_clustered_5', 'ngids_clustered_5_closest_anomalies']
    datasets = [
        # 'energy_pv_hours',
        # 'energy_pv_hours_short',
        # 'wind_rel_wind',
        # 'wind_clustered_10',
        # 'creditcard_flat10',
        # 'creditcard_5x5',
        # 'unsw_clustered_5_closest_anomaly'
        # 'unsw_10_small'
        # 'www_adfa_ngids_clustered',
        # 'www_clustered_5_closest_anomalies',
        # 'www_6x2_short'
        # 'ngids_5'
        # 'nsl_10_r'
        # '3ids3',
        # *[f'3ids3_order_{i}' for i in range(5)],
        # *[f'ngids_order_{i}' for i in range(5)],
        # *[f'nsl_8_order_{i}' for i in range(5)],
        # *[f'unsw_10_order_{i}' for i in range(5)],
        # *[f'wind_5_order_{i}' for i in range(5)],
        # *[f'ngids_kfold_{i}' for i in range(5)],
        # *[f'3ids3_kfold_{i}' for i in range(5)],
        # *[f'nsl_8_kfold_{i}' for i in range(5)],
        # *[f'unsw_10_kfold_{i}' for i in range(5)],
        # *[f'wind_5_kfold_{i}' for i in range(5)],
        # 'wind_5_kfold_memory',
        'ngids_kfold_memory_adaptive',
        'nsl_8_kfold_memory',
        # '3ids3_kfold_memory',
        # 'unsw_10_kfold_memory',

    ]
    for dataset in datasets:
        path = Path(f'out/results/{dataset}')
        process_global_to_csv(path)