import itertools

from data_readers.adfa_data_reader import AdfaDataReader
from data_readers.bosc_data_reader import BoscDataReader
from data_readers.credit_card_data_reader import CreditCardDataReader
from data_readers.mixed_ids_data_reader import MixedIdsDataReader
from data_readers.smd_data_reader import SmdDataReader
from experiment import experiment
from models.classic.isolation_forest import IsolationForestAdapter
from models.classic.lof import LocalOutlierFactorAdapter
from models.classic.oc_svm import OneClassSVMAdapter
from models.modern.copod_adapter import COPODAdapter
from models.modern.suod_adapter import SUODAdapter
from models.our.cpds.always_new_cpd import AlwaysNewCPD
from models.our.cpds.lifewatch.lifewatch import LIFEWATCH
from models.our.hierarchical_lifewatch import HierarchicalLifewatchMemory
from models.our.memories.flat_memory_with_summarization import FlatMemoryWithSummarization
from models.our.memories.simple_flat_memory import SimpleFlatMemory
from models.our.models.ae import AE
from models.our.models.vae import VAE
from models.our.models.vae_pyod import VAEpyod
from models.our.our import OurModel, create_our_model_mixed
from models.our.our_adapter import OurModelAdapterBase
from strategies.ftl_wrapper import FirstTaskLearnerWrapper
from strategies.incremental_batch_wrapper import IncrementalBatchLearnerWrapper
from strategies.incremental_task_wrapper import IncrementalTaskLearnerWrapper
from strategies.know_it_all_wrapper import KnowItAllLearnerWrapper
from strategies.stl_wrapper import SingleTaskLearnerWrapper
from functools import partial


def create_our_model(base_model_fn, cpd_fn, memory_fn):
    return lambda input_features: OurModel(base_model_fn(input_features), cpd=cpd_fn(), memory=memory_fn())


our_models_base = [
    lambda _: COPODAdapter(),
    # lambda input_features: VAE(input_features)
]
our_cpds = [
    # lambda: AlwaysNewCPD(),
    lambda: LIFEWATCH()
]
memories = [lambda: FlatMemoryWithSummarization()]
our_models = [create_our_model(base_model_fn, cpd_fn, memory_fn) for base_model_fn, cpd_fn, memory_fn in
              itertools.product(our_models_base, our_cpds, memories)]

our_mixed_models = [
    lambda input_features: create_our_model_mixed(VAEpyod(input_features), HierarchicalLifewatchMemory()),
    # lambda _: create_our_model_mixed(COPODAdapter(), HierarchicalLifewatchMemory()),
]

# adfa_data_reader = lambda: AdfaDataReader('data/adfa/adfa_30.npy', 'adfa_30')
# smd_data_reader = lambda: SmdDataReader()
# credit_card_data_reader = lambda: CreditCardDataReader('data/creditcard/creditcard_flat_proportions_20.npy', name='creditcard_flat20_proportions')
# mixed_ids_data_reader = lambda: MixedIdsDataReader('data/mixed/adfa_www_ngids_1.npy', name='ADFA_WWW_NGIDS_1')
bosc_adfa_data_reader = lambda: BoscDataReader('data/mixed/adfa_ngids_www_bosc_equalized_ranged.npy', name='adfa_ngids_www_bosc_equalized_ranged')
data_readers = [bosc_adfa_data_reader]
models_creators = [
    # lambda _: IsolationForestAdapter(), lambda _: LocalOutlierFactorAdapter(), lambda _: OneClassSVMAdapter(),
    # lambda _: COPODAdapter(), lambda _: SUODAdapter(),
    # lambda input_features: AE(input_features),
    # lambda input_features: VAE(input_features),
    # lambda input_features: VAEpyod(input_features),
    # *our_models,
    *our_mixed_models,
    # lambda: create_our_model_mixed(COPODAdapter(), HierarchicalLifewatchMemory())
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
            print(strategy_fn)
            print(model_fn(0).name())
            data_reader = data_reader_fn()
            final_model = strategy_fn(lambda: model_fn(data_reader.input_features()),
                                      lambda: list(data_reader.iterate_tasks()))
            print(f'Running {final_model.name()} on {data_reader.dataset_id()}')
            experiment(data_reader, final_model)
