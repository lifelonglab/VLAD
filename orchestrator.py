from data_readers.adfa_data_reader import AdfaDataReader
from experiment import experiment
from models.classic.isolation_forest import IsolationForestAdapter
from models.classic.lof import LocalOutlierFactorAdapter
from models.classic.oc_svm import OneClassSVMAdapter
from models.modern.copod_adapter import COPODAdapter
from models.modern.suod_adapter import SUODAdapter
from models.our.our_adapter import OurModelAdapter
from strategies.ftl_wrapper import FirstTaskLearnerWrapper
from strategies.incremental_batch_wrapper import IncrementalBatchLearnerWrapper
from strategies.incremental_task_wrapper import IncrementalTaskLearnerWrapper
from strategies.know_it_all_wrapper import KnowItAllLearnerWrapper
from strategies.stl_wrapper import SingleTaskLearnerWrapper

adfa_data_reader = lambda: AdfaDataReader('data/adfa/Adduser_k_5_rate_10_iter_1.csv',
                                          'data/adfa/Adduser_k_5_rate_10')
# smd_data_reader = lambda: SmdDataReader()

data_readers = [adfa_data_reader]
models_creators = [
    lambda: IsolationForestAdapter(), lambda: LocalOutlierFactorAdapter(), lambda: OneClassSVMAdapter(),
    lambda: COPODAdapter(), lambda: SUODAdapter()
    # lambda: OurModelAdapter()
]
strategies = [
    lambda model_fn, _: SingleTaskLearnerWrapper(model_fn),
    lambda model_fn, _: FirstTaskLearnerWrapper(model_fn),
    lambda model_fn, _: IncrementalTaskLearnerWrapper(model_fn),
    lambda model_fn, _: IncrementalBatchLearnerWrapper(model_fn),
    lambda model_fn, tasks_fn: KnowItAllLearnerWrapper(model_fn, learning_tasks=tasks_fn())
]

for data_reader_fn in data_readers:
    for model_fn in models_creators:
        for strategy_fn in strategies:
            data_reader = data_reader_fn()
            final_model = strategy_fn(model_fn, lambda: list(data_reader.iterate_tasks()))
            print(f'Running {final_model.name()} on {data_reader.dataset_id()}')
            experiment(data_reader, final_model)
