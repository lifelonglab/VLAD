import itertools
import os

from data_readers.adfa_data_reader import AdfaDataReader
from data_readers.bosc_data_reader import BoscDataReader
from data_readers.credit_card_data_reader import CreditCardDataReader
from data_readers.energy_data_reader import EnergyDataReader
from data_readers.mixed_ids_data_reader import MixedIdsDataReader
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


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_our_model(base_model_fn, cpd_fn, memory_fn):
    return lambda input_features: OurModel(base_model_fn(input_features), cpd=cpd_fn(), memory=memory_fn())


our_models_base = [
    # lambda _: COPODAdapter(),
    lambda input_features: AE(input_features)
]
our_cpds = [
    lambda: AlwaysNewCPD(),
    # lambda: LIFEWATCH()
]
memories = [lambda: FlatMemoryWithSummarization()]
our_models = [create_our_model(base_model_fn, cpd_fn, memory_fn) for base_model_fn, cpd_fn, memory_fn in
              itertools.product(our_models_base, our_cpds, memories)]

generated_mixed_models = []
for ratio in [
    # 1.5,
    2,
    # 2.5,
    # 3
    ]:
    for size in [
        # 3000,
        # 5000,
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

our_ablation_limited_models = [
    lambda input_features: create_our_model_mixed(VAE(input_features), HierarchicalLifewatchMemory(
        max_samples=5000, threshold_ratio=2.5, subconcept_threshold_ratio=5, disable_cpd=True), steps=15000),
    lambda input_features: create_our_model_mixed(VAE(input_features), HierarchicalLifewatchMemory(
        max_samples=5000, threshold_ratio=2.5, subconcept_threshold_ratio=5, disable_replay=True), steps=15000)
]

our_mixed_models = [
    # lambda input_features: create_our_model_mixed(AE(input_features), HierarchicalLifewatchMemory()),
    lambda input_features: create_our_model_mixed(VAE(input_features),
                                                  HierarchicalLifewatchMemory(threshold_ratio=1.5)),
    # lambda input_features: create_our_model_mixed(VAE(input_features), HierarchicalLifewatchMemory(threshold_ratio=2)),
    # lambda input_features: create_our_model_mixed(VAE(input_features),
    #                                               HierarchicalLifewatchMemory(max_samples=10000, threshold_ratio=2)),
    # lambda input_features: create_our_model_mixed(VAE(input_features),
    # HierarchicalLifewatchMemory(threshold_ratio=2.5)),
    # lambda input_features: create_our_model_mixed(VAE(input_features),
    #                                               HierarchicalLifewatchMemory(max_samples=10000, threshold_ratio=2.5)),
    # lambda input_features: create_our_model_mixed(VAE(input_features), HierarchicalLifewatchMemory(threshold_ratio=3)),
    # lambda input_features: create_our_model_mixed(VAE(input_features),
    #                                               HierarchicalLifewatchMemory(threshold_ratio=3.5)),
    # lambda _: create_our_model_mixed(COPODAdapter(), HierarchicalLifewatchMemory()),
]

our_adfa_mixed_models = [
    lambda input_features: create_our_model_mixed(VAE_Adfa(input_features),
                                                  HierarchicalLifewatchMemory(max_samples=1000, threshold_ratio=2)),
    lambda input_features: create_our_model_mixed(VAE_Adfa(input_features),
                                                  HierarchicalLifewatchMemory(max_samples=500, threshold_ratio=2)),
    lambda input_features: create_our_model_mixed(VAE_Adfa(input_features),
                                                  HierarchicalLifewatchMemory(max_samples=1000, threshold_ratio=2.5)),
    lambda input_features: create_our_model_mixed(VAE_Adfa(input_features),
                                                  HierarchicalLifewatchMemory(max_samples=500, threshold_ratio=2.5)),
    lambda input_features: create_our_model_mixed(VAE_Adfa(input_features),
                                                  HierarchicalLifewatchMemory(max_samples=500, threshold_ratio=3)),
    lambda input_features: create_our_model_mixed(VAE_Adfa(input_features),
                                                  HierarchicalLifewatchMemory(max_samples=500, threshold_ratio=3.5)),
]

mixed_ids_data_reader = lambda: MixedIdsDataReader('data/mixed/www_adfa_ngids_clustered.npy',
                                                   name='www_adfa_ngids_clustered')

# adfa_data_reader = lambda: AdfaDataReader('data/adfa/full_adfa.npy', 'full_adfa')
smd_data_reader = lambda: SmdDataReader()
credit_card_data_reader = lambda: CreditCardDataReader('data/creditcard/creditcard_5x5.npy', name='creditcard_5x5')
data_reader1 = lambda: MixedIdsDataReader('data/ngids/full_ngids.npy', name='full_ngids')
data_reader4 = lambda: MixedIdsDataReader('data/ngids/ngids_seq_5.npy', name='ngids_seq_5')
data_reader2 = lambda: MixedIdsDataReader('data/ngids/ngids_clustered_5.npy', name='ngids_clustered_5')
data_reader3 = lambda: MixedIdsDataReader('data/ngids/ngids_clustered_5_closest_anomalies.npy',
                                          name='ngids_clustered_5_closest_anomalies')

www_data_reader = lambda: MixedIdsDataReader('data/www/www_clustered_5_closest_anomalies.npy',
                                             name='www_clustered_5_closest_anomalies')
adfa_data_reader = lambda: MixedIdsDataReader('data/adfa/adfa_clustered_5_closest_anomalies.npy',
                                              name='adfa_clustered_5_closest_anomalies')

unsw_data_reader = lambda: UnswDataReader('data/unsw/unsw_clustered_5_closest_anomalies.npy',
                                          name='unsw_clustered_5_closest_anomaly')

energy_data_reader = lambda: EnergyDataReader('data/energy/energy_pv_seq_hours.npy')
wind_energy_data_reader = lambda: WindEnergyDataReader('data/energy/wind_nrel_seq_wind.npy')

data_readers = [
    credit_card_data_reader,
    # energy_data_reader,
    # wind_energy_data_reader,
    # www_data_reader,
    # unsw_data_reader,
    # adfa_data_reader,
    # bosc_data_reader3,
    # bosc_data_reader1,
    # bosc_data_reader2, bosc_data_reader3, bosc_data_reader4
    # smd_data_reader,
    # mixed_ids_data_reader
]
models_creators = [
    # lambda _: IsolationForestAdapter(),
    # lambda _: LocalOutlierFactorAdapter(),
    # lambda _: OneClassSVMAdapter(),
    # # lambda _: RandomModel(),
    # lambda _: COPODAdapter(),
    # lambda _: SUODAdapter(),
    # lambda input_features: VAE(input_features),
    # lambda _: AlwaysValueModel(0),
    # lambda _: AlwaysValueModel(1),
    # lambda input_features: AE(input_features),
    # lambda input_features: VAEpyod(input_features),
    # *our_models,
    # *our_mixed_models,
    *generated_mixed_models,
    # *our_ablation_limited_models,
    # *our_adfa_mixed_models,
    # lambda: create_our_model_mixed(COPODAdapter(), HierarchicalLifewatchMemory())
    # lambda input_features: VAE_Adfa(input_features)
]

print('models-creator', models_creators)
strategies = [
    # lambda model_fn, _: SingleTaskLearnerWrapper(model_fn),
    # lambda model_fn, _: FirstTaskLearnerWrapper(model_fn),
    lambda model_fn, _: IncrementalTaskLearnerWrapper(model_fn),
    # lambda model_fn, _: IncrementalBatchLearnerWrapper(model_fn),
    # lambda model_fn, tasks_fn: KnowItAllLearnerWrapper(model_fn, learning_tasks=tasks_fn())
]

for data_reader_fn in data_readers:
    for strategy_fn in strategies:
        for model_fn in models_creators:
            data_reader = data_reader_fn()
            final_model = strategy_fn(lambda: model_fn(data_reader.input_features()),
                                      lambda: list(data_reader.iterate_tasks()))
            print(f'Running {final_model.name()} on {data_reader.dataset_id()}')
            experiment(data_reader, final_model)
