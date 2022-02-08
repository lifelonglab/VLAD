import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def process_global_to_csv(path: Path):
    models = []
    results = defaultdict(list)
    for file in path.glob('*.json'):
        with open(file) as f:
            data = json.load(f)

            model = data['metadata']['name']
            models.append(model)

            global_results = data['results']['global']
            results['accuracy'].append(global_results['accuracy'])
            results['bwt'].append(global_results['backward_transfer']['bwt'])
            results['rem'].append(global_results['backward_transfer']['rem'])
            results['bwt+'].append(global_results['backward_transfer']['bwt+'])
            results['forward_transfer'].append(global_results['forward_transfer'])

    df = pd.DataFrame()
    df['model'] = models
    for key, values in results.items():
        df[key] = values
    df.to_csv('results_analysis/analysis.csv', index=False)



if __name__ == '__main__':
    path = Path('results/ADFA_5')
    print(path.exists())
    process_global_to_csv(path)