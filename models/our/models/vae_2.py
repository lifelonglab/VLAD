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


class VAEParams(ModelBase):
    def __init__(self, input_features, intermediate_dim, latent_dim):
        self.params = {
            'input_size': input_features,
            'intermediate_dim': intermediate_dim,
            'latent_dim': latent_dim,
            'l_rate': 0.001
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
        kl_loss = tf.clip_by_value(kl_loss, clip_value_min=-1_000_000, clip_value_max=1_000_000)
        kl_loss = tf.where(tf.math.is_nan(kl_loss), tf.zeros_like(kl_loss), kl_loss)
        # if tf.math.is_inf(kl_loss):
        # print('kl_loss', kl_loss)
        # print(kl_loss.shape)
        # print(kl_loss.numpy())
            # kl_loss = tf.constant(32 * [1_000_000_000_000.00])
        # return the average loss over all
        total_loss = K.mean(reconstruction_loss + kl_loss)

        # total_loss = tf.clip_by_value(total_loss, clip_value_min=-1_000_000_000, clip_value_max=1_000_000_000)
        # total_loss = tf.where(tf.math.is_nan(total_loss), tf.zeros_like(total_loss), total_loss)
        return total_loss

    def _encoder(self):
        input_shape = self.params['input_size']
        intermediate_dim = self.params['intermediate_dim']
        latent_dim = self.params['latent_dim']

        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_log_var = Dense(latent_dim, name='z_log_var')(x)
        # use the reparameterization trick and get the output from the sample() function
        z = Lambda(self._sample, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        encoder = Model(inputs, z, name='encoder')
        return inputs, encoder

    def _decoder(self):
        input_shape = self.params['input_size']
        intermediate_dim = self.params['intermediate_dim']
        latent_dim = self.params['latent_dim']

        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(input_shape, activation='sigmoid')(x)
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
        print('size of learning data', len(data))
        self.vae_model.fit(data, data,
                           shuffle=True,
                           epochs=64,
                           batch_size=32,
                           validation_split=0.1)
        predictions = self.vae_model.predict(data)
        self.threshold = max_threshold(data, predictions)

    def predict(self, data, **kwargs):
        reconstruction = self.vae_model.predict(data)
        errors = mse(data, reconstruction)
        return [1 if e > self.threshold else 0 for e in errors], errors

    def name(self):
        return f'VAE_Params_64ep_{self.params["intermediate_dim"]}_{self.params["latent_dim"]}'

    def parameters(self) -> Dict:
        return self.params
