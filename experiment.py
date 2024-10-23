from data_readers.adfa_data_reader import AdfaDataReader
from data_readers.data_reader import DataReader
from metrics.other_values_measurement import OtherValuesMeasurement
from metrics.tasks_matrix.predictions_collector import PredictionsCollector
from metrics.time.time_measurement import TimeMeasurement
from models.model_base import ModelBase
from models.our.cpds.lifewatch.lifewatch import LIFEWATCH
from models.our.memories.simple_flat_memory import SimpleFlatMemory
from models.our.models.vae import VAE
from models.our.our import OurModel
from models.our.our_adapter import OurModelAdapterBase
from strategies.stl_wrapper import SingleTaskLearnerWrapper
from results import process_results
from results_writer import save_results
from strategies.strategy import Strategy


def experiment(data_reader: DataReader, model: Strategy):
    # init
    results_collector = PredictionsCollector()
    time_measurement = TimeMeasurement()
    other_measurements = OtherValuesMeasurement()

    test_tasks = data_reader.load_test_tasks()

    # run
    time_measurement.start()
    for task in data_reader.iterate_tasks():
        # train
        time_measurement.start_training(task.name)
        model.learn(task.data)
        time_measurement.finish_training(task.name)
        other_measurements.add(model.additional_measurements(), task_name=task.name)

        # evaluate on all task
        time_measurement.start_testing_after(task.name)
        for test_task in test_tasks:
            predictions = model.predict(test_task.data)
            results_collector.add(task.name, test_task=test_task.name, y_true=test_task.labels, y_pred=predictions)
        time_measurement.finish_testing_after(task.name)

    time_measurement.finish()

    # postprocess metrics
    collected_results = results_collector.results()
    processed_results = process_results(collected_results)
    save_results(model, data_reader, processed_results=processed_results, collected_results=collected_results,
                 times=time_measurement.results(), other_measurements=other_measurements.results())


if __name__ == '__main__':
    reader = AdfaDataReader('data/adfa/Adduser_k_5_rate_10_iter_1.csv',
                            'data/adfa/Adduser_k_5_rate_10')
    # reader = SmdDataReader()
    # model = FirstTaskLearnerWrapper(lambda: IsolationForestAdapter())
    model = SingleTaskLearnerWrapper(lambda: OurModel(VAE(), cpd=LIFEWATCH(), memory=SimpleFlatMemory()))
    experiment(reader, model)
