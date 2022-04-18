import json
from collections import defaultdict

from scipy.stats import wilcoxon

with open('out/results_analysis/summary_results_all_ordering.json') as f:
    data = json.load(f)
    results = defaultdict(list)

    for dataset, values in data.items():
        for method, value in values.items():
            results[method].append(value)

    print(results)
    our_results = results['Our']

    for method, values in results.items():
        print(len(values))
        print(values)
        if method != 'Our':
            diff = [o - v for v, o in zip(values, our_results)]
            print(diff)
            h, p = wilcoxon(diff)
            print(f'Comparing with {method}: h {h} p {p}')