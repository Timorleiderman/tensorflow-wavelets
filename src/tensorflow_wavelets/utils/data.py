import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def load_mnist(categorical=True, remove_n_samples=1000, expand_d=True):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Remove images to get smaller dataset
    if remove_n_samples != 0:
        x_train = x_train[:remove_n_samples, :, :]
        y_train = y_train[:remove_n_samples]
        x_test = x_test[:remove_n_samples//2, :, :]
        y_test = y_test[:remove_n_samples//2]

    if categorical:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    if expand_d:
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)