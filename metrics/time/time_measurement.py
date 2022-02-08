import time
from typing import Dict


class TimeMeasurement:
    def __init__(self):
        self.tmp_times = {}
        self.measurements = {'training': {}, 'testing_after': {}}

    def start(self):
        self.tmp_times['whole_process_start'] = time.time()

    def finish(self):
        self.measurements['whole_process'] = time.time() - self.tmp_times['whole_process_start']
        self.measurements['all_trainings'] = sum([v for v in self.measurements['training'].values()])
        self.measurements['all_testings'] = sum([v for v in self.measurements['testing_after'].values()])

    def start_training(self, task_name):
        self.tmp_times[f'training_{task_name}_start'] = time.time()

    def finish_training(self, task_name):
        self.measurements['training'][task_name] = time.time() - self.tmp_times[f'training_{task_name}_start']

    def start_testing_after(self, task_name):
        self.tmp_times[f'testing_after_{task_name}'] = time.time()

    def finish_testing_after(self, task_name):
        self.measurements['testing_after'][task_name] = time.time() - self.tmp_times[f'testing_after_{task_name}']

    def results(self) -> Dict:
        return self.measurements
