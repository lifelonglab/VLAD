import numpy as np
from sklearn.ensemble import IsolationForest

from models.classic.utils import adjust_predictions
from models.model_base import ModelBase


class IsolationForestAdapter(ModelBase):
    def __init__(self):
        self.random_state = 0
        self.n_estimators = 25
        self.model = IsolationForest(random_state=self.random_state, n_estimators=self.n_estimators)

    def learn(self, data):
        self.model.fit(data)

    def predict(self, data, task_name=None):
        return adjust_predictions(self.model.predict(data))

    def name(self):
        return 'IsolationForest'

    def parameters(self):
        return {
            'random_state': self.random_state,
            'n_estimators': self.n_estimators
        }
