import itertools
import os

from sklearn.metrics import roc_auc_score

from data_readers.credit_card_data_reader import CreditCardDataReader
from data_readers.energy_data_reader import EnergyDataReader
from data_readers.mixed_ids_data_reader import MixedIdsDataReader
from data_readers.nsl_data_reader import NslDataReader
from data_readers.unsw_data_reader import UnswDataReader
from data_readers.wind_rel_data_reader import WindEnergyDataReader
from models.classic.isolation_forest import IsolationForestAdapter
from models.classic.lof import LocalOutlierFactorAdapter
from models.classic.oc_svm import OneClassSVMAdapter
from models.modern.copod_adapter import COPODAdapter
from models.modern.suod_adapter import SUODAdapter
from models.our.models.vae_2 import VAEParams


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

data_reader = UnswDataReader('data/unsw/unsw_10_kfold_0.npy',
                             name='unsw_clustered_10_closest_anomaly')

data_reader = WindEnergyDataReader('data/wind/wind_5_kfold_0.npy', 'wind_clustered_5')
# data_reader = MixedIdsDataReader('data/3ids/3ids_3.npy', name='3ids')
# data_reader = EnergyDataReader('data/energy/energy_20.npy')
# data_reader = CreditCardDataReader('data/creditcard/creditcard_5.npy', name='creditcard_5')
#
# data_reader = MixedIdsDataReader('data/ngids/ngids_5.npy', name='ngids_5')
# data_reader = NslDataReader('data/nsl/nsl_10.npy', 'nsl_10')


input_features = data_reader.input_features()

train = data_reader.iterate_tasks()[0]
test = data_reader.load_test_tasks()[0]

print(train.name)
print(test.name)

models = {
    'vae': lambda inter, latent: VAEParams(input_features, inter, latent),
    'suod': lambda contamination: SUODAdapter(contamination=contamination),
    'if': lambda n_estimators, contamination: IsolationForestAdapter(n_estimators, contamination),
    'lof': lambda n_neighbors: LocalOutlierFactorAdapter(n_neighbors),
    'oc_svm': lambda nu, gamma: OneClassSVMAdapter(nu=nu, gamma=gamma),
    'copod': lambda contamination: COPODAdapter(contamination=contamination)
}

params = {
    'vae': [(64, 16), (48, 8), (32, 16), (32, 8), (16, 8), (16, 4), (8, 4)],
    'suod': [(0.001,), (0.0001,), (0.00001,)],
    'copod': [(0.001,), (0.0001,), (0.00001,)],
    'if': list(itertools.product([100, 200], [0.001, 0.0001])),
    'lof': [(2,), (5,), (10,)],
    'oc_svm': list(itertools.product([0.1, 0.01], [0.1, 0.01]))
}

best_model, best_value = {}, 0

models_list = [
    # 'suod', 'copod', 'if', 'lof',
    'oc_svm',
    # 'vae'
]

for model_name in models_list:
    best_value = 0
    model_params = params[model_name]
    for ps in model_params:
        print(ps)
        model_fn = models[model_name]
        model = models[model_name](*ps)
        model.learn(train.data)
        predictions, scores = model.predict(test.data)
        roc_auc = roc_auc_score(y_true=test.labels, y_score=scores)
        print('roc_auc', roc_auc)
        if roc_auc > best_value:
            best_value = roc_auc
            best_model[model_name] = ps

print('\n\n')
for key, val in best_model.items():
    print(f'{key}: {val}')
