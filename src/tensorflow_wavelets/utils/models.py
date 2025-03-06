import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import AveragePooling2D

import tensorflow_wavelets.Layers.DWT as DWT
import tensorflow_wavelets.Layers.DMWT as DMWT
import tensorflow_wavelets.Layers.DTCWT as DTCWT
import tensorflow_wavelets.Layers.Threshold as Activation
from tensorflow.keras.models import Model


def basic_dwt_idwt(input_shape, wave_name="db2", eagerly=False, threshold=True, mode='soft', algo='sure', concat = True):
    # load DWT IDWT model
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(DWT.DWT(wavelet_name=wave_name, concat = concat))
    if threshold:
        model.add(Activation.Threshold(algo=algo, mode=mode))
    model.add(DWT.IDWT(wavelet_name=wave_name, concat = concat))

    # for debug with break points
    model.run_eagerly = eagerly
    return model

def basic_dwt_idwt_1d(input_shape, wave_name="db2", eagerly=False, threshold=True, mode='soft', algo='sure', concat = True):
    # load DWT IDWT model
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(DWT.DWT1D(wavelet_name=wave_name))
    if threshold:
        model.add(Activation.Threshold(algo=algo, mode=mode))
    model.add(DWT.IDWT1D(wavelet_name=wave_name))

    # for debug with break points
    model.run_eagerly = eagerly
    return model

def basic_dmwt(input_shape, nb_classes=10, wave_name="ghm", eagerly=False):

    x_input = keras.Input(shape=input_shape)
    x = DMWT.DMWT(wavelet_name=wave_name)(x_input)
    x = layers.Flatten()(x)
    x = layers.Dense(nb_classes, activation="softmax")(x)
    model = Model(x_input, x, name="mymodel")
    # for debug with break points
    model.run_eagerly = eagerly
    return model


def basic_dtcwt(input_shape, nb_classes=10, level=2, eagerly=False):

    cplx_input = keras.Input(shape=input_shape)
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
            DWT.DWT(wavelet_name=wave_name),
            Activation.Threshold(),
            DWT.IDWT(wavelet_name=wave_name),
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
    


class AveragePooling2DPyramid(tf.keras.Model):
    """ 
    """
    def __init__(self, batch_size, width, height,  **kwargs):
        super(AveragePooling2DPyramid, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.width = width
        self.height = height

    def build(self, input_shape):
        super(AveragePooling2DPyramid, self).build(input_shape)
        
    def call(self, inputs, training=None, mask=None):
        
        im1_4 = inputs
        im1_3 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_4)
        im1_2 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_3)
        im1_1 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_2)
        im1_0 = AveragePooling2D(pool_size=2, strides=2, padding='same')(im1_1)

        return im1_0, im1_1, im1_2, im1_3
    
    