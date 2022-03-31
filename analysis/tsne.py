import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from data_readers.credit_card_data_reader import CreditCardDataReader
from data_readers.mixed_ids_data_reader import MixedIdsDataReader


def draw_tsne(z, y, name):
    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", len(set(y))),
                    data=df).set(title=name)

    ax.set_ylim(-100, 100)
    ax.set_xlim(-100, 100)
    plt.show()


# data_reader = CreditCardDataReader('data/creditcard/creditcard_10_newcluster.npy', name='creditcard_5')
# data_reader = MixedIdsDataReader('data/mixed/3ids_2.npy', name='3ids2')
data_reader = MixedIdsDataReader('data/ngids/ngids_5.npy', name='ngids_5')

train_x = np.concatenate([t.data for t in data_reader.iterate_tasks()])
train_y = np.concatenate([[i] * len(t.data) for i, t in enumerate(data_reader.iterate_tasks())])

test_x = np.concatenate([t.data for t in data_reader.load_test_tasks()])
test_y = np.concatenate([[i if label == 0 else len(data_reader.load_test_tasks()) for label in t.labels] for i, t in enumerate(data_reader.load_test_tasks())])

tsne = TSNE(n_components=2, verbose=1, random_state=123)


full_x = np.concatenate([train_x, test_x])
print(train_y.shape)
print(test_y.shape)
full_y = np.concatenate([train_y, test_y])
print(set(full_y))
print(full_y.shape)

full_z = tsne.fit_transform(full_x)

draw_tsne(full_z, full_y, f'{data_reader.dataset_id()}_all')
draw_tsne(full_z[:len(train_y)], train_y, f'{data_reader.dataset_id()}_train')
draw_tsne(full_z[len(train_y):], test_y, f'{data_reader.dataset_id()}_test')

