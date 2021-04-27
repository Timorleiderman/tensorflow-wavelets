import os
import pickle
import datetime
import numpy as np
import tensorflow as tf
import models.cifar10CNN
import models.WaveletCifar10CNN

from keras.datasets import cifar10
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical, Sequence


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # for tensor flow warning
weights_filepath = 'weights'
logs_filepath = 'logs'


if not os.path.exists(weights_filepath):
    os.makedirs(weights_filepath)

if not os.path.exists(logs_filepath):
    os.makedirs(logs_filepath)


nb_classes = 10
batch_size = 32
epochs = 30

lr = 1e-4  # learning rate
beta_1 = 0.9         # beta 1 - for adam optimizer
beta_2 = 0.96        # beta 2 - for adam optimizer
epsilon = 1e-7        # epsilon - for adam optimizer

trainFactor = 0.8
imageShape = (32, 32, 3)  # CIFAR-10 60,000 32X32 color

optimizer = Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)  # SGD()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# # Remove images to get smaller dataset
# x_train = x_train[:1000, :, :]
# y_train = y_train[:1000]
# x_test = x_test[:500, :, :]
# y_test = y_test[:500]


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Create and Train the Model
#model = models.cifar10CNN.cifar10CNN(imageShape, nb_classes)
#history_file_path = "trainHistoryCifar10CNN.txt"  # save loss and val loss

model = models.WaveletCifar10CNN.WaveletCNN(imageShape, nb_classes)
history_file_path = "trainHistoryWaveletCifar10CNN.txt" # save loss and val loss
# model.summary()

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

# log_dir = logs_filepath + r"\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
h5_tmp = "tmp.h5"
history = model.fit(x_train, y_train,
                    validation_split=1 - trainFactor,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=2,
                    callbacks=[
                        ModelCheckpoint("tmp.h5", monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                        # ModelCheckpoint(h5_tmp, monitor='loss', verbose=0, save_best_only=True, mode='auto'),
                        EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto')
                        # EarlyStopping(monitor='loss', min_delta=1e-3, patience=10, verbose=1, mode='auto'),
                        # tensorboard_callback
                        ],
                    )


model.load_weights("tmp.h5")
weights_path = os.path.join(weights_filepath + "WCNN.h5")
model.save(weights_path)

# Model Evaluation
result = model.evaluate(x_test, y_test)

with open(history_file_path, 'wb') as f:
    pickle.dump(history.history, f)
