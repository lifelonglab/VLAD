import numpy as np
from keras import Input, Model
from keras.api.keras import optimizers
from keras.layers import Dense

from models.our.utils import mse


class AE:
    def __init__(self):
        self.params = {
            'input_size': 6,
            'encoding_dim': 2,
            'l_rate': 0.01
        }
        self.model = self._create_model()

    def learn(self, data):
        self.model.fit(data, data,
                       shuffle=False,
                       epochs=32,
                       batch_size=256)
        predictions = self.model.predict(data)
        errors = mse(data, predictions)
        self.threshold = np.percentile(errors, 99)

    def predict(self, data):
        reconstruction = self.model.predict(data)
        errors = mse(data, reconstruction)
        return [1 if e > self.threshold else 0 for e in errors]

    def _create_model(self):
        input_size = self.params['input_size']
        encoding_dim = self.params['encoding_dim']

        input = Input(shape=(input_size,))

        encoded = Dense(encoding_dim, activation='relu')(input)
        decoded = Dense(input_size, activation='sigmoid')(encoded)
        autoencoder = Model(input, decoded)

        encoder = Model(input, encoded)
        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))

        adam = optimizers.Adam(lr=self.params['l_rate'])
        autoencoder.compile(optimizer=adam, loss='mean_squared_error')
        return autoencoder
