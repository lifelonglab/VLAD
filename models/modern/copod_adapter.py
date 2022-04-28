from typing import Dict

from pyod.models.copod import COPOD

from models.model_base import ModelBase


class COPODAdapter(ModelBase):
    def __init__(self, contamination=0.01):
        self.params = {
            'contamination': contamination
        }
        self.copod = COPOD(contamination=self.params['contamination'])

    def name(self):
        return 'COPOD-v2'

    def learn(self, data) -> None:
        self.copod.fit(data)

    def predict(self, data, task_name=None):
        return self.copod.predict(data), self.copod.predict_proba(data)[:, 1]

    def parameters(self) -> Dict:
        return self.params
