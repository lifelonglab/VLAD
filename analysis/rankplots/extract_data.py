import json
import math
from collections import defaultdict

import pandas as pd

dir_path = 'out/results_analysis'

datasets = ['ngids', '3ids3', 'nsl_8', 'unsw_10', 'wind_5']
methods = ['IsolationForest', 'LocalOutlierFactor', 'OC-SVM', 'SUOD', 'COPOD', 'VAE_', 'Our']
method_translation = {'IsolationForest': 'IF', 'LocalOutlierFactor': 'LOF', 'OC-SVM': 'OC-SVM', 'SUOD': 'SUOD',
                      'COPOD': 'COPOD', 'VAE_': 'VAE'}

dataset_translation = {'ngids': 'NGIDS', '3ids3': '3IDS', 'nsl_8': 'NSL-KDD', 'unsw_10': 'UNSW', 'wind_5': 'WIND', '': ''}


def translate_method(name):
    for key, value in method_translation.items():
        if name.startswith(key):
            return value
    if name.startswith('Our'):
        # print(name)
        if 'no_cpd' in name:
            return 'OurNoCPD'
        elif 'no_replay' in name:
            return 'OurNoReplay'
        else:
            return 'Our'
    exit(f'WRONG NAME {name}')


mode = 'kfold'

results = {}
for dataset in datasets:
    for i in range(5):
        df = pd.read_csv(f'{dir_path}/{dataset}_{mode}_{i}/analysis_results.csv', header=[0])

        dataset_results = {}
        for method in methods:
            print(df['model_name'])
            rows = df.loc[df['model_name'].str.startswith(method)]
            for _, row in rows.iterrows():
                roc_auc = row['roc_auc']
                name = translate_method(row['model_name'])
                print(name)
                dataset_results[name] = roc_auc
        results[f'{dataset}_{i}'] = dataset_results


save_results = defaultdict(dict)
for d, single_results in results.items():
    for m, values in single_results.items():
        if 'NoCPD' not in m and 'NoReplay' not in m:
            save_results[d][m] = values

with open(f'{dir_path}/summary_results_all_{mode}.json', 'w') as f:
    json.dump(save_results, f)
