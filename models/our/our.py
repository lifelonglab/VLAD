from typing import Dict, List

from models.model_base import ModelBase
from models.our.cpds.always_new_cpd import AlwaysNewCPD
from models.our.cpds.cpd import CPD, ChangePoint
from models.our.memories.simple_flat_memory import SimpleFlatMemory
from models.our.models.ae import AE


# when to retrain - we need to handle this
# use data and replay - should we use whole data or limit it?
#

class OurModelBase(ModelBase):
    def __init__(self):
        self.model = AE()
        self.cpd: CPD = AlwaysNewCPD()
        self.memory: SimpleFlatMemory = SimpleFlatMemory()

    def learn(self, data):
        cps = self.cpd.detect_cp(data)

        if len(cps) > 0:
            self._retrain_model(data)

        self._update_memory(cps, data)

        # train each N iterations # forced sleep

    def predict(self, data, task_name=None):
        return self.model.predict(data, )

    def name(self):
        return f'OurModel_{self.model.name()}_{self.cpd.name()}_{self.memory.name()}'

    def parameters(self) -> Dict:
        return {'model': self.model.params, 'cpd': self.cpd.params(), 'memory': self.memory.params()}

    def _update_memory(self, cps: List[ChangePoint], data):
        if len(cps) > 0:
            for i, cp in enumerate(cps):
                start_index = 0 if i == 0 else cps[i - 1].index
                end_index = cp.index
                task_data = data[start_index:end_index]
                self.memory.new_data(task_data, is_new_dist=cp.is_new_dist, distribution=cp.distribution)
        else:
            self.memory.new_data(data, is_new_dist=False)

    def _retrain_model(self, data):
        replay = self.memory.get_replay()
        retrain_data = replay + data
        self.model.learn(retrain_data)  # or if there are too many iterations without change
