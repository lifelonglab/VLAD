from pathlib import Path

import pandas as pd

from analysis.plots.pretty_name import pretty_name


def load_results(dataset: str, results_type: str):
    path = Path(f'out/results_analysis/{dataset}/analysis_{results_type}.csv')
    df = pd.read_csv(path)
    df['pretty_name'] = df['model_name'].apply(lambda x: pretty_name(x))
    return df
