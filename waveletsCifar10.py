import os
import glob
import pickle
import datetime
import tensorflow as tf

from keras.models import Model
from models.DWT import DWT_Pooling, IWT_UpSampling
from keras.datasets import cifar10
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical, Sequence
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Activation, Dropout


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # for tensor flow warning
history_file_path = "trainHistoryDict.txt"

if not os.path.exists('weights'):
    os.makedirs('weights')

if not os.path.exists('logs'):
    os.makedirs('weights')


def WaveletCNN(input_size=(224, 224, 3), nb_classes=120):
    inputs = Input(shape=input_size)

    output = Conv2D(32, (3, 3), padding="same")(inputs)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    output = Conv2D(32, (3, 3), padding="same")(output)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    output = DWT_Pooling()(output)  # dwt

    output = Conv2D(32, (3, 3), padding="same")(output)
    output = Dropout(0.25)(output)

    output = Conv2D(64, (3, 3), padding="same")(output)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    output = Conv2D(64, (3, 3), padding="same")(output)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    output = DWT_Pooling()(output)  # dwt

    output = Conv2D(64, (3, 3), padding="same")(output)

    output = Flatten()(output)
    output = Dropout(0.25)(output)

    output = Dense(512, activation="relu")(output)
    output = Dropout(0.5)(output)

    output = Dense(nb_classes, activation="softmax")(output)

    model = Model(inputs=inputs, outputs=output)

    return model


weights_filepath = 'weights/WCNN.h5'

nb_classes = 10
batch_size = 64
epochs = 400
lr = 0.01  # learning rate
trainFactor = 0.8
imageShape = (32, 32, 3)  # CIFAR10 60,000 32X32 color

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

cut_to = 500
x_train = x_train[:cut_to]
y_train = y_train[:cut_to]
x_test = x_test[:cut_to]
y_test = y_test[:cut_to]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Create and Train the Model
model = WaveletCNN(imageShape, nb_classes)
model.summary()

optimizer = Adam()  # SGD()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

log_dir = r"logs\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(x_train, y_train,
                    validation_split=1 - trainFactor,
                    epochs=epochs,
                    verbose=2,
                    callbacks=[
                        ModelCheckpoint("tmp.h5", monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                        EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto'),
                        tensorboard_callback
                        ],
                    )

model.load_weights("tmp.h5")
model.save(weights_filepath)

# Model Evaluation
result = model.evaluate(x_test, y_test)

print(result)


with open(history_file_path, 'wb') as f:
    pickle.dump(history.history, f)
