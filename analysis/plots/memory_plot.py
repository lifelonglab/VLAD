import json


def plot_memory(results_file):
    with open(results_file) as f:
        data = json.load(f)
        measurements = data['other_measurements']

        memory_values = [val['memory_samples_number'] for val in measurements]

        print(memory_values)


if __name__ == '__main__':
    plot_memory('/home/nyder/research/lifelong learning/lifelong-anomaly-detection/out/results/wind_rel_wind/IncrementalTaskLearner/IncrementalTaskLearner_OurModel_AE_Hierarchical_LIFEWATCH_limit_1000_Hierarchical_LIFEWATCH_limit_1000.json')