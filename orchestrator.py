import itertools
import os

from best_competitor_models import best_unsw_competitors, best_wind_competitors, best_3ids_competitors, \
    best_energy_competitors, best_credit_card_competitors, best_nsl_competitors, best_ngids_competitors
from best_models import wind_rel_wind_models, unsw_5_models, energy_pv_models, three_ids_models, credit_card_models, \
    ngids_models, www_models, nsl_models
from data_readers.adfa_data_reader import AdfaDataReader
from data_readers.bosc_data_reader import BoscDataReader
from data_readers.credit_card_data_reader import CreditCardDataReader
from data_readers.energy_data_reader import EnergyDataReader
from data_readers.mixed_ids_data_reader import MixedIdsDataReader
from data_readers.nsl_data_reader import NslDataReader
from data_readers.smd_data_reader import SmdDataReader
from data_readers.unsw_data_reader import UnswDataReader
from data_readers.wind_rel_data_reader import WindEnergyDataReader
from experiment import experiment
from models.classic.always_value import AlwaysValueModel
from models.classic.isolation_forest import IsolationForestAdapter
from models.classic.lof import LocalOutlierFactorAdapter
from models.classic.oc_svm import OneClassSVMAdapter
from models.classic.random_model import RandomModel
from models.modern.copod_adapter import COPODAdapter
from models.modern.suod_adapter import SUODAdapter
from models.our.cpds.always_new_cpd import AlwaysNewCPD
from models.our.cpds.lifewatch.lifewatch import LIFEWATCH
from models.our.hierarchical_lifewatch import HierarchicalLifewatchMemory
from models.our.memories.flat_memory_with_summarization import FlatMemoryWithSummarization
from models.our.memories.simple_flat_memory import SimpleFlatMemory
from models.our.models.ae import AE
from models.our.models.vae import VAE
from models.our.models.vae_adfa import VAE_Adfa
from models.our.models.vae_pyod import VAEpyod
from models.our.our import OurModel, create_our_model_mixed
from models.our.our_adapter import OurModelAdapterBase
from strategies.ftl_wrapper import FirstTaskLearnerWrapper
from strategies.incremental_batch_wrapper import IncrementalBatchLearnerWrapper
from strategies.incremental_task_wrapper import IncrementalTaskLearnerWrapper
from strategies.know_it_all_wrapper import KnowItAllLearnerWrapper
from strategies.stl_wrapper import SingleTaskLearnerWrapper
from functools import partial


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_our_model(base_model_fn, cpd_fn, memory_fn):
    return lambda input_features: OurModel(base_model_fn(input_features), cpd=cpd_fn(), memory=memory_fn())


generated_mixed_models = []
for ratio in [
    1.5,
    2,
    2.5,
    3
    ]:
    for size in [
        # 3000,
        5000,
        10000]:
        for steps in [
            5000,
            15000]:
            for subconcept_ratio in [
                2,
                5]:
                generated_mixed_models.append(
                    lambda input_features, max_samples=size, threshold_ratio=ratio, subconcept_threshold_ratio=subconcept_ratio, steps_in=steps:
                    create_our_model_mixed(VAE(input_features),
                                           HierarchicalLifewatchMemory(
                                               max_samples=max_samples,
                                               threshold_ratio=threshold_ratio,
                                               subconcept_threshold_ratio=subconcept_threshold_ratio),
                                           steps=steps_in))


ngids_data_reader = lambda: MixedIdsDataReader('data/ngids/ngids_5.npy', name='ngids_5')
three_ids_not_closest_data_reader = lambda: MixedIdsDataReader('data/mixed/3ids_not_closest.npy', name='3ids_not_closest')
three_ids2_data_reader = lambda: MixedIdsDataReader('data/3ids/3ids_3_order_1.npy', name='3ids3')
three_ids_ordering = [lambda ordering=order: MixedIdsDataReader(f'data/3ids/3ids_3_order_{ordering}.npy', name=f'3ids3_order_{ordering}') for order in range(1, 5)]
adfa_ngids_www = lambda: MixedIdsDataReader('data/mixed/adfa_ngids_www.npy', name='adfa_ngids_www')

# adfa_data_reader = lambda: AdfaDataReader('data/adfa/full_adfa.npy', 'full_adfa')
smd_data_reader = lambda: SmdDataReader()
credit_card_data_reader = lambda: CreditCardDataReader('data/creditcard/creditcard_50.npy', name='creditcard_50')
data_reader1 = lambda: MixedIdsDataReader('data/ngids/full_ngids.npy', name='full_ngids')


www_data_reader = lambda: MixedIdsDataReader('data/www/www_6x2_short.npy', name='www_6x2_short')
adfa_data_reader = lambda: MixedIdsDataReader('data/adfa/adfa_clustered_5_closest_anomalies.npy',
                                              name='adfa_clustered_5_closest_anomalies')

unsw_data_reader = lambda: UnswDataReader('data/unsw/unsw_clustered_10_closest_anomalies.npy', name='unsw_clustered_10_closest_anomaly')
unsw_small_data_reader = lambda: UnswDataReader('data/unsw/unsw_10_small.npy', name='unsw_10_small')

energy_data_reader = lambda: EnergyDataReader('data/energy/energy_20.npy', 'energy_20')
wind_energy_data_reader = lambda: WindEnergyDataReader('data/energy/wind_nrel_seq_wind.npy')
wind_short_energy_data_reader = lambda: WindEnergyDataReader('data/energy/wind_clustered_10.npy', 'wind_clustered_10')

nsl_data_reader = lambda: NslDataReader('data/nsl/nsl_10.npy', 'nsl_10')
nslr_data_reader = lambda: NslDataReader('data/nsl/nsl_10_r.npy', 'nsl_10_r')

data_readers = [
    # energy_data_reader,
    # wind_energy_data_reader,
    # three_ids_not_closest_data_reader
    # ngids_data_reader,
    # www_data_reader,
    # unsw_data_reader,
    # unsw_small_data_reader
    # smd_data_reader,
    # mixed_ids_data_reader,
    # adfa_ngids_www
    # wind_short_energy_data_reader
    # credit_card_data_reader,
    # three_ids2_data_reader
    # nsl_data_reader
    # nslr_data_reader
    *three_ids_ordering
]
models_creators = [
    # *generated_mixed_models,
    # *unsw_5_models(),
    # *best_unsw_competitors()
    # *www_adfa_ngids()
    # *wind_rel_wind_models()
    # *best_wind_competitors()
    # *best_3ids_competitors(),
    *three_ids_models()
    # *best_energy_competitors()
    # *energy_pv_models()
    # *best_credit_card_competitors(),
    # *credit_card_models()
    # *ngids_models(),
    # *best_ngids_competitors()
    # *www_models()
    # *best_nsl_competitors()
    # *nsl_models()
]

print('models-creator', models_creators)
strategies = [
    # lambda model_fn, _: SingleTaskLearnerWrapper(model_fn),
    # lambda model_fn, _: FirstTaskLearnerWrapper(model_fn),
    # lambda model_fn, _: IncrementalTaskLearnerWrapper(model_fn),
    lambda model_fn, execution_no, _: IncrementalBatchLearnerWrapper(model_fn, execution_no=execution_no),
    # lambda model_fn, tasks_fn: KnowItAllLearnerWrapper(model_fn, learning_tasks=tasks_fn())
]


executions_number = 1

for data_reader_fn in data_readers:
    for strategy_fn in strategies:
        for model_fn in models_creators:
            for execution_i in range(executions_number):
                data_reader = data_reader_fn()
                print(model_fn)
                final_model = strategy_fn(lambda: model_fn(data_reader.input_features()), execution_i,
                                          lambda: list(data_reader.iterate_tasks()))
                print(f'Running {final_model.name()} on {data_reader.dataset_id()} execution {execution_i}')
                experiment(data_reader, final_model)
