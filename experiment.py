from data_readers.adfa_data_reader import AdfaDataReader
from data_readers.data_reader import DataReader
from data_readers.smd_data_reader import SmdDataReader
from metrics.tasks_matrix.predictions_collector import PredictionsCollector
from metrics.time.time_measurement import TimeMeasurement
from models.modern.copod_adapter import COPODAdapter
from models.model import Model
from models.our.our_adapter import OurModelAdapter
from models.strategies.stl_wrapper import SingleTaskLearnerWrapper
from results import process_results
from results_writer import save_results


def experiment(data_reader: DataReader, model: Model):
    # init
    results_collector = PredictionsCollector()
    time_measurement = TimeMeasurement()

    test_tasks = data_reader.load_test_tasks()

    # run
    time_measurement.start()
    for task in data_reader.iterate_tasks():
        # train
        time_measurement.start_training(task.name)
        model.learn(task.data)
        time_measurement.finish_training(task.name)

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
                 times=time_measurement.results())


if __name__ == '__main__':
    reader = AdfaDataReader('data/adfa/Adduser_k_5_rate_10_iter_1.csv',
                            'data/adfa/Adduser_k_5_rate_10')
    # reader = SmdDataReader()
    # model = FirstTaskLearnerWrapper(lambda: IsolationForestAdapter())
    model = SingleTaskLearnerWrapper(lambda: OurModelAdapter())
    experiment(reader, model)
