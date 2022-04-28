import pandas as pd

datasets = ['unsw_10', '3ids3', 'ngids', 'wind_5', 'nsl_8']
methods = ['IsolationForest', 'LocalOutlierFactor', 'OC-SVM', 'SUOD', 'COPOD', 'VAE_', 'Our']


def load_data():
    data = {}
    for dataset in datasets:
        path = f'out/results_analysis/{dataset}_all_kfolds/results.csv'
        df = pd.read_csv(path, header=[0,1])
        print(df.columns)
        for method in methods:
            row = df.loc[df['model_name']['Unnamed: 1_level_1'] == method].iloc[0]
            print(row['roc_auc']['mean'])


load_data()