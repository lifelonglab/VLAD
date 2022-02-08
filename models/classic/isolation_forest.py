import numpy as np
from sklearn.ensemble import IsolationForest

from models.model import Model


class IsolationForestAdapter(Model):
    def __init__(self):
        self.random_state = 0
        self.n_estimators = 25
        self.model = IsolationForest(random_state=self.random_state, n_estimators=self.n_estimators)

    def learn(self, data):
        self.model.fit(data)

    def predict(self, data, task_name=None):
        predictions = self.model.predict(data)
        predictions = np.where(predictions == 1, 0, predictions)
        predictions = np.where(predictions == -1, 1, predictions)
        return predictions

    def name(self):
        return 'IsolationForest'

    def parameters(self):
        return {
            'random_state': self.random_state,
            'n_estimators': self.n_estimators
        }
