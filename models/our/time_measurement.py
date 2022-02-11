import time


class OurModelTimeMeasurement:
    def __init__(self):
        self.iteration_times = {}
        self.start_times = {}

    def reset(self):
        self.iteration_times = {}
        self.start_times = {}

    def start_cpd(self):
        self.start_times['cpd'] = time.time()

    def finish_cpd(self):
        self.iteration_times['cpd'] = time.time() - self.start_times['cpd']

    def start_training(self):
        self.start_times['training'] = time.time()

    def finish_training(self):
        self.iteration_times['training'] = time.time() - self.start_times['training']

    def start_memory_management(self):
        self.start_times['memory'] = time.time()

    def finish_memory_management(self):
        self.iteration_times['memory'] = time.time() - self.start_times['memory']

    def results(self):
        return self.iteration_times
