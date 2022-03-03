from typing import Dict

import numpy as np
from keras import Input, Model
from keras.api.keras import optimizers
import keras.backend as K
from keras.layers import Dense, Lambda
import tensorflow as tf

from models.model_base import ModelBase
from models.our.models.thresholds import max_threshold
from models.our.utils import mse


class VAE_Adfa(ModelBase):
    def __init__(self, input_features):
        self.params = {
            'input_size': input_features,
            'intermediate_dim': max(3, int(input_features/8)),
            'latent_dim': max(4, int(input_features/32)),
            'l_rate': 0.01
        }
        self.threshold = 0
        inputs, encoder = self._encoder()
        decoder = self._decoder()
        outputs = decoder(encoder(inputs))
        self.vae_model = Model(inputs, outputs, name='vae_mlp')
        opt = optimizers.Adam(learning_rate=self.params['l_rate'], clipvalue=0.5)

        self.vae_model.compile(optimizer=opt, loss=self.vae_loss)

    def vae_loss(self, x, x_decoded_mean):
        # compute the average MSE error, then scale it up, ie. simply sum on all axes
        # x = x.astype(np.float32)
        tf.cast(x, tf.float32)
        reconstruction_loss = K.sum(K.square(x - x_decoded_mean))
        # compute the KL loss
        kl_loss = - 0.5 * K.sum(1 + self.z_log_var - K.square(self.z_mean) - K.square(K.exp(self.z_log_var)), axis=-1)
        # return the average loss over all
        total_loss = K.mean(reconstruction_loss + kl_loss)
        return total_loss

    def _encoder(self):
        input_shape = self.params['input_size']
        intermediate_dim = self.params['intermediate_dim']
        latent_dim = self.params['latent_dim']

        inputs = Input(shape=input_shape, name='encoder_input')
        # x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(inputs)
        z_log_var = Dense(latent_dim, name='z_log_var')(inputs)
        # use the reparameterization trick and get the output from the sample() function
        z = Lambda(self._sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        encoder = Model(inputs, z, name='encoder')
        return inputs, encoder

    def _decoder(self):
        input_shape = self.params['input_size']
        intermediate_dim = self.params['intermediate_dim']
        latent_dim = self.params['latent_dim']

        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        # x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(input_shape, activation='sigmoid')(latent_inputs)
        # Instantiate the decoder model:
        decoder = Model(latent_inputs, outputs, name='decoder')
        return decoder

    def _sample(self, args):
        z_mean, z_log_var = args
        self.z_mean = z_mean
        self.z_log_var = z_log_var
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def learn(self, data):
        self.vae_model.fit(data, data,
                           shuffle=False,
                           epochs=128,
                           batch_size=64)
        predictions = self.vae_model.predict(data)
        self.threshold = max_threshold(data, predictions)

    def predict(self, data, **kwargs):
        reconstruction = self.vae_model.predict(data)
        errors = mse(data, reconstruction)
        return [1 if e > self.threshold else 0 for e in errors], errors

    def name(self):
        return 'VAE_adfa'

    def parameters(self) -> Dict:
        return self.params
