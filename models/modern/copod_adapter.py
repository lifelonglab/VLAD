from typing import Dict

from pyod.models.copod import COPOD

from models.model_base import ModelBase


class COPODAdapter(ModelBase):
    def __init__(self):
        self.copod = COPOD(contamination=0.01)

    def name(self):
        return 'COPOD'

    def learn(self, data) -> None:
        self.copod.fit(data)

    def predict(self, data, task_name=None):
        return self.copod.predict(data), self.copod.predict_proba(data)[:, 1]

    def parameters(self) -> Dict:
        return {'all': 'default'}
