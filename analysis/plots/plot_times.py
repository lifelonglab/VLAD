import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_times(df, out_dir, dataset, strategy):
    plt.figure(figsize=(12, 8))
    sns.barplot(x=df['pretty_name'], y=df['training_times'])
    plt.title(f'Training times for {dataset} in {strategy}')
    plt.ylabel('Training time (s)')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.4, top=0.95)

    plt.savefig(f'{out_dir}/training_times.png')

