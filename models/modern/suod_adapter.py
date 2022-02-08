from typing import Dict

from pyod.models.suod import SUOD

from models.model import Model


class SUODAdapter(Model):
    def __init__(self):
        self.suod = SUOD()

    def name(self):
        return 'SUOD'

    def learn(self, data) -> None:
        self.suod.fit(data)

    def predict(self, data, task_name=None):
        return self.suod.predict(data)

    def parameters(self) -> Dict:
        return {'all': 'default'}
