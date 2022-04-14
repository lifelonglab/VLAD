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


mode = 'kfolds'
# mode = 'orders'

results = {}
for dataset in datasets:
    df = pd.read_csv(f'{dir_path}/{dataset}_all_{mode}/results.csv', header=[0,1])

    dataset_results = {}
    for method in methods:
        rows = df.loc[df['model_name']['Unnamed: 1_level_1'].str.startswith(method)]
        for _, row in rows.iterrows():
            roc_auc = row['roc_auc']
            bwt = row['roc_auc_bwt']
            fwt = row['roc_auc_forward_transfer']
            name = translate_method(row['model_name']['Unnamed: 1_level_1'])
            print(name)
            dataset_results[name] = [roc_auc, bwt, fwt]
    results[dataset] = dataset_results


# save_results = defaultdict(dict)
# for d, single_results in results.items():
#     for m, values in single_results.items():
#         # if 'NoCPD' not in m and 'NoReplay' not in m:
#         roc_auc = values[0]['mean']
#         save_results[d][m] = roc_auc
#
# with open(f'{dir_path}/summary_results.json', 'w') as f:
#     json.dump(save_results, f)
#
# exit()

result_rows = []
for i in range(math.ceil(len(datasets)/2)):
    dataset_1 = datasets[i*2]
    dataset_2 = datasets[(i * 2) + 1] if (i * 2) + 1 < len(datasets) else ''
    result_rows.append(f' & \\multicolumn{{3}}{{c}}{{{dataset_translation[dataset_1]}}} & \\multicolumn{{3}}{{c}}{{{dataset_translation[dataset_2]}}}')
    mid_rules = '\cmidrule(lr){2-4} \cmidrule(lr){5-7}' if dataset_2 != '' else '\cmidrule(lr){2-4}'
    separator = mid_rules + ' \n  & ROC-AUC & BWT       & FWT'
    double_part_separator = 'ROC-AUC & BWT       & FWT \\\\' if dataset_2 != '' else '& & \\\\ \n'
    result_rows.append(f'{separator} & {double_part_separator} {mid_rules}')
    for name in results[dataset_1].keys():
        results_text_1 = ' & '.join([f"{metric['mean']:.3f}$\\pm {metric['std']:.3f}$" for metric in results[dataset_1][name]])
        if dataset_2 != '':
            results_text_2 = ' & '.join([f"{metric['mean']:.3f}$\\pm {metric['std']:.3f}$" for metric in results[dataset_2][name]])
        else:
            results_text_2 = ' & & '
        result_rows.append(f'\\texttt{{{name}}}  & {results_text_1} & {results_text_2}')
        if name == 'COPOD' or name == 'OurNoReplay':
            result_rows.append('\midrule')
    result_rows.append('\\\\')

with open(f'{dir_path}/out_analysis/{mode}.txt', 'w') as f:
    for row in result_rows:
        f.write(row + (' \\\\ \n' if row != '\midrule' else '\n'))

