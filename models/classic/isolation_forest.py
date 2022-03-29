import numpy as np
from sklearn.ensemble import IsolationForest

from models.classic.utils import adjust_predictions
from models.model_base import ModelBase


class IsolationForestAdapter(ModelBase):
    def __init__(self, n_estimators=100, contamination=0.00001):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.model = IsolationForest(n_estimators=self.n_estimators, contamination=0.00001)

    def learn(self, data):
        self.model.fit(data)

    def predict(self, data, task_name=None):
        return adjust_predictions(self.model.predict(data)), -self.model.score_samples(data)

    def name(self):
        return 'IsolationForest'

    def parameters(self):
        return {
            'n_estimators': self.n_estimators,
            'contamination': self.contamination
        }
