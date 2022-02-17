from pathlib import Path

import pandas as pd


def load_results(dataset: str):
    path = Path(f'out/results_analysis/analysis_{dataset}.csv')
    return pd.read_csv(path)