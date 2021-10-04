import math
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist, cifar10

import DWT2
import DWT


model1 = keras.Sequential()
model1.add(keras.Input(shape=(32, 32, 3)))
model1.add(DWT2.DWT())
model1.summary()

model2 = keras.Sequential()
model2.add(keras.Input(shape=(32, 32, 3)))
model2.add(layers.MaxPooling2D())
model2.summary()

model3 = keras.Sequential()
model3.add(keras.Input(shape=(32, 32, 3)))
model3.add(DWT.DWT_Pooling())
model3.summary()