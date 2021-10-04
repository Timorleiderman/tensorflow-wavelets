from abc import ABC

import tensorflow as tf
import Layers.DWT as DWT
from tensorflow import keras
from tensorflow.keras import layers
import Layers.Activation as Activation
from tensorflow.keras.models import Model


def basic_dwt_idw(input_shape, wave_name="db2", eagerly=False, soft_theshold=True):
    # load DWT IDWT model
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(DWT.DWT(name=wave_name))
    if soft_theshold:
        model.add(Activation.SureSoftThreshold())
    model.add(DWT.IDWT(name=wave_name))

    # for debug with break points
    model.run_eagerly = eagerly
    return model


class AutocodeBasic(Model):

    def __init__(self, latent_dim, width=28, height=28):
        super(AutocodeBasic, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(width*height, activation='sigmoid'),
            layers.Reshape((width, height, 1)),
        ])

    def get_config(self):
        return {"latent_dim", self.latent_dim}

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutocodeBasicDWT(Model):
    def get_config(self):
        pass

    def __init__(self, latent_dim, width=28, height=28, wave_name="db2"):
        super(AutocodeBasicDWT, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            DWT.DWT(name=wave_name),
            Activation.SureSoftThreshold(),
            DWT.IDWT(name=wave_name),
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(width*height, activation='sigmoid'),
            layers.Reshape((width, height, 1)),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded