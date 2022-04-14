from pathlib import Path

import pandas as pd

# postfix = 'order'
postfix = 'kfold'

def analyze_multiple(dataset_name):
    full_init_path = f'out/results_analysis/{dataset_name}'
    dfs = []
    for i in range(5):
        file_path = f'{full_init_path}_{postfix}_{i}/analysis_results.csv'
        dfs.append(pd.read_csv(file_path))

    full_df = pd.concat(dfs)
    print(full_df)

    agg_dict = {col: ['mean', 'std'] for col in full_df.columns if col not in ['model', 'model_name', 'strategy']}
    avg_df = full_df.groupby(['model_name'], as_index=False).agg(agg_dict)

    res_dir = f'{full_init_path}_all_{postfix}s'
    Path(res_dir).mkdir(exist_ok=True)
    avg_df.to_csv(f'{res_dir}/results.csv')



datasets = ['ngids', '3ids3', 'nsl_8', 'unsw_10', 'wind_5']
for dataset_name in datasets:
    analyze_multiple(dataset_name)
