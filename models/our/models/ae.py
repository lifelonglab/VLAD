from typing import Dict

import numpy as np
from keras import Input, Model
from keras.api.keras import optimizers
from keras.layers import Dense

from models.our.models.thresholds import max_threshold
from models.our.utils import mse
from models.model_base import ModelBase


class AE(ModelBase):
    def __init__(self, input_features):
        self.params = {
            'input_size': input_features,
            'intermediate_dim': 4,
            'encoding_dim': 2,
            'l_rate': 0.0001
        }
        self.model = self._create_model()

    def learn(self, data):
        self.model.fit(data, data,
                       shuffle=False,
                       epochs=128,
                       batch_size=256)
        predictions = self.model.predict(data)
        self.threshold = max_threshold(data, predictions)

    def predict(self, data, **kwargs):
        reconstruction = self.model.predict(data)
        errors = mse(data, reconstruction)
        return [1 if e > self.threshold else 0 for e in errors], errors

    def _create_model(self):
        input_size = self.params['input_size']
        intermediate_dim = self.params['intermediate_dim']
        encoding_dim = self.params['encoding_dim']

        input = Input(shape=(input_size,))

        intermediate_enc = Dense(intermediate_dim, activation='relu')(input)
        encoded = Dense(encoding_dim, activation='relu')(intermediate_enc)

        intermediate_dec = Dense(intermediate_dim, activation='relu')(encoded)
        decoded = Dense(input_size, activation='sigmoid')(intermediate_dec)

        autoencoder = Model(input, decoded)

        adam = optimizers.Adam(lr=self.params['l_rate'])
        autoencoder.compile(optimizer=adam, loss='mean_squared_error')
        return autoencoder

    def name(self):
        return 'AE'

    def parameters(self) -> Dict:
        return self.params

