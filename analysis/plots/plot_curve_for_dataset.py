import json
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import roc_curve


def plot_curve_for(path):
    values = []
    for file in path.glob('debug/*.npy'):
        print(file)
        data: Dict = np.load(file, allow_pickle=True).item()
        name = data['metadata']['name']
        results = data['collected_results'][0][0]
        y_true = results['y_true']
        y_score = results['scores']
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        print(fpr)
        print(tpr)

if __name__ == '__main__':
    path = Path('...')
    plot_curve_for(path)
