from typing import Dict, List

import numpy as np

from models.model_base import ModelBase
from models.our.cpds.always_new_cpd import AlwaysNewCPD
from models.our.cpds.cpd import CPD, ChangePoint
from models.our.memories.memory import Memory
from models.our.memories.simple_flat_memory import SimpleFlatMemory
from models.our.models.ae import AE

# when to retrain - we need to handle this
# use data and replay - should we use whole data or limit it?
#
from models.our.time_measurement import OurModelTimeMeasurement


class OurModel(ModelBase):
    def __init__(self, model, cpd, memory, enable_scaled_predictions=False, retrain_after_steps=1000):
        self.model: ModelBase = model
        self.cpd: CPD = cpd
        self.memory: Memory = memory

        self.retrain_after_steps = retrain_after_steps

        self.time_measurement = OurModelTimeMeasurement()
        self.ever_trained = False
        self.steps_from_last_retrain = 0

        self.iteration = 0

        self.max_reconstruction_error_per_dist = {}
        self.enable_scaled_predictions = enable_scaled_predictions

    def learn(self, data):
        should_retrain = not self.ever_trained

        self.time_measurement.reset()
        self.time_measurement.start_cpd()
        cps = self.cpd.detect_cp(data)
        if len(cps) > 0:
            should_retrain = True
        self.steps_from_last_retrain += len(data)
        self.time_measurement.finish_cpd()

        if self.steps_from_last_retrain >= self.retrain_after_steps:
            should_retrain = True

        self.time_measurement.start_memory_management()
        self._update_memory(cps, data)
        self.memory.organize()
        if self.memory.should_summarize() or should_retrain:
            self.memory.summarize()
            should_retrain = True
        self.time_measurement.finish_memory_management()

        if should_retrain:
            self.time_measurement.start_training()
            self._retrain_model(data)
            self.time_measurement.finish_training()
            self.ever_trained = True
            self.steps_from_last_retrain = 0

            # compute max reconstructione rrors for memory data
            if self.enable_scaled_predictions:
                for dist_id, dist in self.memory.distributions().items():
                    _, errors_for_dist = self.model.predict(np.array(dist))
                    self.max_reconstruction_error_per_dist[dist_id] = max(errors_for_dist)

    def predict(self, data, task_name=None):
        if not self.enable_scaled_predictions:
            return self.model.predict(data)
        else:
            print('scaling!')
            # model predictions and errors
            predictions, errors = self.model.predict(data)

            # assign to dist
            assigned_dists = self.memory.assign(data)

            # scale
            scaled_errors = [error/self.max_reconstruction_error_per_dist[dist_id] for dist_id, error in zip(assigned_dists, errors)]
            return predictions, np.array(scaled_errors)

    def name(self):
        scaled_text = '_ad_scaled' if self.enable_scaled_predictions else ''
        return f'OurTestModel_{self.model.name()}_{self.memory.name()}_steps_{self.retrain_after_steps}_{scaled_text}'

    def parameters(self) -> Dict:
        return {'model': self.model.parameters(), 'cpd': self.cpd.params(), 'memory': self.memory.params(),
                'retrain_after_steps': self.retrain_after_steps}

    def _update_memory(self, cps: List[ChangePoint], data):
        if len(cps) > 0:
            # before any cps
            if cps[0].index != 0:
                self.memory.new_data(data[0:cps[0].index], is_new_dist=False)
            for i, cp in enumerate(cps):
                start_index = cp.index
                end_index = len(data) if i == len(cps) - 1 else cps[i + 1].index
                task_data = data[start_index:end_index]
                self.memory.new_data(task_data, is_new_dist=cp.is_new_dist, distribution=cp.distribution)
        else:
            self.memory.new_data(data, is_new_dist=False)

    def _retrain_model(self, data):
        replay = self.memory.get_replay()
        print('length of replay', len(replay))
        retrain_data = np.concatenate((replay, data)) if len(replay) > 0 else data
        self.model.learn(retrain_data)  # or if there are too many iterations without change

    def additional_measurements(self) -> Dict:
        return {'memory_samples_number': self.memory.samples_number(), 'phases_times': self.time_measurement.results(),
                'memory': self.memory.additional_measurements()}


def create_our_model_mixed(model, cpd_memory, steps, enable_scaled_pred=False):
    return OurModel(model, cpd=cpd_memory, memory=cpd_memory, retrain_after_steps=steps, enable_scaled_predictions=enable_scaled_pred)
