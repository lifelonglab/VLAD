import pandas as pd

from analysis.plots.plot_utils import filter_by_strategy
from analysis.plots.results_loader import load_results


def plot_multiple(df):
    pass



def load_data():
    name = 'WIND-4'
    results = pd.read_csv(f'out/results_processed_manually/Results - {name}.csv')
    results['dataset'] = name

if __name__ == '__main__':
    load_data()
    data_1 = load_data()
    data_2 = load_data()