from typing import Dict

from sklearn.neighbors import LocalOutlierFactor

from models.model import Model


class LocalOutlierFactorAdapter(Model):
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
        return self.lof.predict(data)

    def parameters(self) -> Dict:
        return self.params
