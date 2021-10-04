import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_wavelets.Layers.DWT as DWT
import tensorflow_wavelets.Layers.DMWT as DMWT
import tensorflow_wavelets.Layers.DTCWT as DTCWT
import tensorflow_wavelets.Layers.Activation as Activation
from tensorflow.keras.models import Model


def basic_dwt_idwt(input_shape, wave_name="db2", eagerly=False, theshold=True, mode='soft'):
    # load DWT IDWT model
    model = keras.Sequential()
    model.add(keras.Input(shape=input_shape))
    model.add(DWT.DWT(name=wave_name))
    if theshold:
        model.add(Activation.Threshold(algo='sure', mode=mode))
    model.add(DWT.IDWT(name=wave_name))

    # for debug with break points
    model.run_eagerly = eagerly
    return model


def basic_dmwt(input_shape, nb_classes=10, wave_name="ghm", eagerly=False):

    x_input = layers.Input(shape=input_shape)
    x = DMWT.DMWT(wave_name)(x_input)
    x = layers.Flatten()(x)
    x = layers.Dense(nb_classes, activation="softmax")(x)
    model = Model(x_input, x, name="mymodel")
    # for debug with break points
    model.run_eagerly = eagerly
    return model


def basic_dtcwt(input_shape, nb_classes=10, level=2, eagerly=False):

    cplx_input = layers.Input(shape=input_shape)
    x = DTCWT.DTCWT(level)(cplx_input)
    x = layers.Flatten()(x)
    x = layers.Dense(nb_classes, activation="softmax")(x)
    model = Model(cplx_input, x, name="mymodel")
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
            Activation.Threshold(),
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