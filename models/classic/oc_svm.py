from typing import Dict

from sklearn.svm import OneClassSVM

from models.model_base import ModelBase


class OneClassSVMAdapter(ModelBase):
    def __init__(self):
        self.nu = 0.1
        self.kernel = 'rbf'
        self.gamma = 0.1
        self.svm = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)

    def name(self):
        return 'OC-SVM'

    def learn(self, data) -> None:
        self.svm.fit(data)

    def predict(self, data, task_name=None):
        return self.svm.predict(data)

    def parameters(self) -> Dict:
        return {
            'nu': self.nu,
            'kernel': self.kernel,
            'gamma': self.gamma
        }
