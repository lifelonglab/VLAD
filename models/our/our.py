from typing import Dict

from models.model import Model
from models.our.cpds.always_new_cpd import AlwaysNewCPD
from models.our.cpds.cpd import CPD
from models.our.memories.simple_flat_memory import SimpleFlatMemory
from models.our.models.ae import AE


# when to retrain - we need to handle this
# use data and replay - should we use whole data or limit it?
#

class OurModel(Model):
    def __init__(self):
        self.model = AE()
        self.cpd: CPD = AlwaysNewCPD()
        self.memory: SimpleFlatMemory = SimpleFlatMemory()

    def learn(self, data):
        # check change points
        cps = self.cpd.detect_cp(data)

        # if there any changepoints
        if len(cps) > 0:
            # retrain with replay and new data
            replay = self.memory.get_replay()
            self.model.learn(replay + data)            # or if there are too many iterations without change
            # update memory
            for i, cp in enumerate(cps):
                start_index = 0 if i == 0 else cps[i-1].index
                end_index = cp.index
                task_data = data[start_index:end_index]
                self.memory.new_data(task_data, is_new=i != 0)
        else:
            self.memory.new_data(data, is_new=False)
            # train each N iterations # forced sleep

    def predict(self, data, task_name=None):
        return self.model.predict(data)

    def name(self):
        return 'OurModel-ae-test'

    def parameters(self) -> Dict:
        return self.model.params
