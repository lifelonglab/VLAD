from typing import Dict

from sklearn.svm import OneClassSVM

from models.classic.utils import adjust_predictions
from models.model_base import ModelBase


class OneClassSVMAdapter(ModelBase):
    def __init__(self, nu=0.1, gamma=0.1):
        self.nu = nu
        self.kernel = 'rbf'
        self.gamma = gamma
        self.svm = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)

    def name(self):
        return 'OC-SVM'

    def learn(self, data) -> None:
        self.svm.fit(data)

    def predict(self, data, task_name=None):
        return adjust_predictions(self.svm.predict(data)), self.svm.decision_function(data)

    def parameters(self) -> Dict:
        return {
            'nu': self.nu,
            'kernel': self.kernel,
            'gamma': self.gamma
        }
