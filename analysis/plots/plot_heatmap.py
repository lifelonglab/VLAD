import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

inputs = {
    'unsw_10':  {
        'clean': 'IncrementalBatchLearner_0_VAE_Params_64ep_32_8.json',
        'full': 'IncrementalBatchLearner_0_OurTestModel_VAE_Params_64ep_32_8_HLW_lim_1024_2000_p2_mf_5_str_5_steps_15000_.json'
    },
    '3ids3': {
        'clean': 'IncrementalBatchLearner_0_VAE_Params_64ep_8_4.json',
        'full': 'IncrementalBatchLearner_0_OurTestModel_VAE_Params_64ep_8_4_HLW_lim_1024_4000_p0.75_mf_5_str_1_steps_30000_.json'
    },
    'ngids': {
        'clean': 'IncrementalBatchLearner_0_VAE_Params_64ep_32_8.json',
        'full': 'IncrementalBatchLearner_0_OurTestModel_VAE_Params_64ep_32_8_HLW_lim_1024_500_p1.25_mf_5_str_1_steps_30000_.json'
    },
    'wind_5': {
        'clean': 'IncrementalBatchLearner_0_VAE_Params_64ep_16_4.json',
        'full': 'IncrementalBatchLearner_0_OurTestModel_VAE_Params_64ep_16_4_HLW_lim_1024_250_p1.25_mf_5_str_1.5_steps_10000_.json'
    },
    'nsl_8': {
        'clean': 'IncrementalBatchLearner_0_VAE_Params_64ep_32_8.json',
        'full': 'IncrementalBatchLearner_0_OurTestModel_VAE_Params_64ep_32_8_HLW_lim_1024_1000_p1.25_mf_5_str_1.25_steps_30000_.json'
    }
}


datasets = ['unsw_10', '3ids3', 'ngids', 'wind_5', 'nsl_8']
for dataset in datasets:
    for mode in ['clean', 'full']:
        results_filename = inputs[dataset][mode]
        sns.set_theme(style="darkgrid")
        # sns.set(rc = {'figure.figsize':(12,8)})

        filepath = f'out/results/{dataset}_kfold_0/IncrementalBatchLearner_0/{results_filename}'
        out_dir = 'out/plots/performance_heatmap'

        with open(filepath) as f:
            data = json.load(f)
            tasks = data['metadata']['tasks']
            tasks_mapping = {t: f'Task {i}' for i, t in enumerate(tasks)}
            results = data['results']['tasks-matrix']
            results_rows = []
            for learning_task, values in results.items():
                for eval_task, value in values.items():
                    roc_auc = value['roc_auc']
                    bwt = results[learning_task][eval_task]
                    results_rows.append([tasks_mapping[learning_task], tasks_mapping[eval_task], roc_auc])

            masks = np.zeros((len(tasks), len(tasks)))
            for i, lt in enumerate(tasks):
                for j, et in enumerate(tasks):
                    if j > i:
                        masks[i, j] = True

            df = pd.DataFrame(results_rows, columns=['learning_task', 'eval_task', 'roc_auc'])
            df = df.pivot(index='learning_task', columns='eval_task', values='roc_auc')
            fig, ax = plt.subplots()
            p = sns.heatmap(df, annot=True, vmin=0, vmax=1, center=0.5, mask=masks)

            p.set_xlabel('Evaluating on task')
            p.set_ylabel('After learning task')

            Path(out_dir).mkdir(exist_ok=True)
            # plt.show()
            plt.savefig(f'{out_dir}/{dataset}_{mode}.pdf', bbox_inches='tight')
            plt.close()