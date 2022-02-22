from typing import Dict

from sklearn.neighbors import LocalOutlierFactor

from models.classic.utils import adjust_predictions
from models.model_base import ModelBase


class LocalOutlierFactorAdapter(ModelBase):
    def __init__(self):
        self.params = {
            'n_neighbors': 2,
            'novelty': True
        }
        self.lof = LocalOutlierFactor(n_neighbors=self.params['n_neighbors'], novelty=self.params['novelty'])

    def name(self):
        return 'LocalOutlierFactor'

    def learn(self, data) -> None:
        self.lof.fit(data)

    def predict(self, data, task_name=None):
        return adjust_predictions(self.lof.predict(data)), - self.lof.decision_function(data)

    def parameters(self) -> Dict:
        return self.params
