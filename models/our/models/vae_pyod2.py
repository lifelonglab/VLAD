from typing import Dict

import numpy as np
from pyod.models.vae import VAE

from models.model_base import ModelBase


class VAEpyodParams(ModelBase):
    def __init__(self, input_features, intermedient_dim, latent_dim):
        self.params = {
            'encoder_neurons': [intermedient_dim],
            'decoder_neurons': [intermedient_dim],
            'latent_dim': latent_dim,
            'epochs': 64
        }
        self.model = VAE(encoder_neurons=self.params['encoder_neurons'], decoder_neurons=self.params['decoder_neurons'],
                         latent_dim=self.params['latent_dim'], epochs=self.params['epochs'])

    def name(self):
        return 'VAE_pyod'

    def learn(self, data) -> None:
        self.model.fit(data)

    def predict(self, data, task_name=None) -> (np.ndarray, np.ndarray):
        try:
            print(data)
            return self.model.predict(data), self.model.decision_function(data)
        except Exception as e:
            print(e)
            print('something is wrong')
            exit()

    def parameters(self) -> Dict:
        return self.params
