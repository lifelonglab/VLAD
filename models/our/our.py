from typing import Dict

from models.model import Model
from models.our.cpds.always_new_cpd import AlwaysNewCPD
from models.our.cpds.cpd import CPD
from models.our.memories.simple_flat_memory import SimpleFlatMemory
from models.our.models.ae import AE


class OurModel(Model):
    def __init__(self):
        self.model = AE()
        self.cpd: CPD = AlwaysNewCPD()
        self.memory: SimpleFlatMemory = SimpleFlatMemory()

    def learn(self, data):
        cps = self.cpd.detect_cp(data)
        if len(cps) > 0:
            for i, cp in enumerate(cps):
                start_index = 0 if i == 0 else cps[i-1].index
                end_index = cp.index
                task_data = data[start_index:end_index]
                self.memory.new_data(task_data, is_new=i != 0)
        else:
            self.memory.new_data(data, is_new=False)

        replay = self.memory.get_replay()
        self.model.learn(replay)

    def predict(self, data, task_name=None):
        return self.model.predict(data)

    def name(self):
        return 'OurModel-ae-test'

    def parameters(self) -> Dict:
        return self.model.params
