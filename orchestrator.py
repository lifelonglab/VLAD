from data_readers.adfa_data_reader import AdfaDataReader
from data_readers.smd_data_reader import SmdDataReader
from experiment import experiment
from models.classic.lof import LocalOutlierFactorAdapter
from models.classic.oc_svm import OneClassSVMAdapter
from models.modern.copod_adapter import COPODAdapter
from models.classic.isolation_forest import IsolationForestAdapter
from models.strategies.ftl_wrapper import FirstTaskLearnerWrapper
from models.strategies.incremental_task_wrapper import IncrementalTaskLearnerWrapper
from models.strategies.know_it_all_wrapper import KnowItAllLearnerWrapper
from models.strategies.stl_wrapper import SingleTaskLearnerWrapper
from models.modern.suod_adapter import SUODAdapter

adfa_data_reader = lambda: AdfaDataReader('data_with_attacks/Adduser_k_5_rate_10_iter_1.csv',
                                          'data_with_attacks/adfa_ld_attacks/Adduser/k_5/rate_10/Adduser_k_5_rate_10')
smd_data_reader = lambda: SmdDataReader()

data_readers = [adfa_data_reader]
models_creators = [
    lambda: IsolationForestAdapter(), lambda: LocalOutlierFactorAdapter(), lambda: OneClassSVMAdapter(),
    # lambda: COPODAdapter(), lambda: SUODAdapter()
]
strategies = [
    lambda model_fn, _: SingleTaskLearnerWrapper(model_fn),
    lambda model_fn, _: FirstTaskLearnerWrapper(model_fn),
    lambda model_fn, _: IncrementalTaskLearnerWrapper(model_fn),
    lambda model_fn, tasks_fn: KnowItAllLearnerWrapper(model_fn, learning_tasks=tasks_fn())
]

for data_reader_fn in data_readers:
    for model_fn in models_creators:
        for strategy_fn in strategies:
            data_reader = data_reader_fn()
            final_model = strategy_fn(model_fn, lambda: list(data_reader.iterate_tasks()))
            print(f'Running {final_model.name()} on {data_reader.dataset_id()}')
            experiment(data_reader, final_model)
