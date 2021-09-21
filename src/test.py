from tensorflow_wavelets.Layers.Activation import *
from tensorflow_wavelets.Layers.DMWT import *
from tensorflow.keras.models import Model
from tensorflow import keras


model = keras.Sequential()
model.add(keras.Input(shape=(28, 28, 1)))
model.add(DMWT(wave_name="ghm"))
model.add(RigreSure())
model.add(IDMWT(wave_name="ghm"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation="sigmoid"))
model.add(keras.layers.Dense(3136, activation="sigmoid"))
model.add(keras.layers.Reshape((56, 56, 1)))


model.summary()