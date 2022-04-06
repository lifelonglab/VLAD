import pandas as pd


def analyze_executions(results_file):
    df = pd.read_csv(results_file)
    agg_dict = {col: ['mean', 'std'] for col in df.columns if col not in ['model', 'model_name', 'strategy']}
    df = df.groupby(['model_name'], as_index=False).agg(agg_dict)
    df.to_csv(f'{results_file}_summary.csv')







if __name__ == '__main__':
    analyze_executions('out/results_analysis/ngids_5/analysis_results.csv')