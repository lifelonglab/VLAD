import json
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(style="darkgrid")
sns.set(rc = {'figure.figsize':(12,8)})


def load_data(dataset_name, results_filename):
    initial_path = f'out/results/{dataset_name}_kfold_'

    gathered_data = []
    for i in range(5):
        with open(f'{initial_path}{i}/IncrementalBatchLearner_0/{results_filename}') as f:
            file_data = json.load(f)
            tasks = file_data['metadata']['tasks']
            tasks_mapping = {t: f'Task {i}' for i, t in enumerate(tasks)}
            results = file_data['results']['tasks-matrix']
            for k, (learning_task, task_results) in enumerate(results.items()):
                for l, (eval_task, values) in enumerate(task_results.items()):
                    if l <= k:
                        gathered_data.append([tasks_mapping[learning_task], tasks_mapping[eval_task], values['roc_auc']])

        # dfs.append(pd.read_csv(f'{initial_path}{i}/IncrementalBatchLearner_0/{results_filename}'))

    return pd.DataFrame(gathered_data, columns=['learning_task', 'eval_task', 'roc_auc'])


def plot_all_tasks_over_time(df, dataset_name, mode):
    fig, ax = plt.subplots()
    p = sns.lineplot(x='learning_task', y='roc_auc', hue='eval_task', data=df, sort=False, ci='sd', marker='o')
    p.set_xlabel('Learning task')
    p.set_ylabel('ROC-AUC')
    p.set(ylim=(0,1))
    ax.legend().set_title('')
    out_dir = 'out/plots/performance_over_time'
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    # plt.show()
    plt.savefig(f'{out_dir}/{dataset_name}_{mode}.pdf', bbox_inches='tight')
    plt.close()


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
    'nsl_10': {
        'clean': 'IncrementalBatchLearner_0_VAE_Params_64ep_32_8.json',
        'full': 'IncrementalBatchLearner_0_OurTestModel_VAE_Params_64ep_32_8_HLW_lim_1024_1000_p1.25_mf_5_str_1.25_steps_30000_.json'
    }
}

datasets = ['unsw_10', '3ids3', 'ngids', 'wind_5', 'nsl_10']
# datasets = ['3ids3']
for dataset_name in datasets:
    for mode in ['clean', 'full']:
        results_filename = inputs[dataset_name][mode]
        df = load_data(dataset_name, results_filename)
        plot_all_tasks_over_time(df, dataset_name, mode)