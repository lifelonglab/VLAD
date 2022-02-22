from typing import Dict

from pyod.models.suod import SUOD

from models.model_base import ModelBase


class SUODAdapter(ModelBase):
    def __init__(self):
        self.suod = SUOD(contamination=0.00001)

    def name(self):
        return 'SUOD'

    def learn(self, data) -> None:
        self.suod.fit(data)

    def predict(self, data, task_name=None):
        return self.suod.predict(data), self.suod.predict_proba(data)[:, 1]

    def parameters(self) -> Dict:
        return {'all': 'default'}
