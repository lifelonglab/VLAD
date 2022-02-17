from typing import Dict

import numpy as np
from pyod.models.vae import VAE

from models.model_base import ModelBase


class VAEpyod(ModelBase):
    def __init__(self, input_features):
        self.params = {
            'encoder_neurons': [max(3, int(input_features / 2))],
            'decoder_neurons': [max(3, int(input_features / 2))],
            'latent_dim': max(2, int(input_features / 4)),
            'epochs': 5
        }
        self.model = VAE(encoder_neurons=self.params['encoder_neurons'], decoder_neurons=self.params['decoder_neurons'],
                         latent_dim=self.params['latent_dim'], epochs=self.params['epochs'])

    def name(self):
        return 'VAE_pyod'

    def learn(self, data) -> None:
        self.model.fit(data)

    def predict(self, data, task_name=None) -> (np.ndarray, np.ndarray):
        try:
            return self.model.predict(data), self.model.decision_function(data)
        except Exception as e:
            print(e)
            print('something is wrong')
            exit()

    def parameters(self) -> Dict:
        return self.params
