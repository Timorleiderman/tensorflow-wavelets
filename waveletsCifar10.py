# conda create --name dwtcnn python=3.6
# numpy==1.16.4
# keras == 2.2.2
# 'tensorflow=1.14.0=mkl*'
# scipy

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # for tensor flow warning
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical, Sequence
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Activation, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD, RMSprop
from keras.datasets import cifar100
from tensorflow.keras import backend
from models.DWT import DWT_Pooling

if not os.path.exists('weights'):
    os.makedirs('weights')

def WaveletCNN(input_size=(224, 224, 3), nb_classes=120):
    inputs = Input(shape=input_size)

    output = Conv2D(32, (3, 3), padding="same")(inputs)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)
    output = Conv2D(32, (3, 3), padding="same")(output)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    output = DWT_Pooling()(output)
    output = Conv2D(32, (3, 3), padding="same")(output)
    output = Dropout(0.25)(output)

    output = Conv2D(64, (3, 3), padding="same")(output)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)
    output = Conv2D(64, (3, 3), padding="same")(output)
    output = BatchNormalization()(output)
    output = Activation("relu")(output)

    output = DWT_Pooling()(output)
    output = Conv2D(64, (3, 3), padding="same")(output)
    output = Flatten()(output)
    output = Dropout(0.25)(output)

    output = Dense(512, activation="relu")(output)
    output = Dropout(0.5)(output)

    output = Dense(nb_classes, activation="softmax")(output)

    model = Model(inputs=inputs, outputs=output)

    return model


filepath = 'weights/WCNN.h5'
nb_classes = 100
batch_size = 64
epochs = 400
lr = 0.01
trainFactor = 0.8
imageShape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

cut_to = 500
x_train = x_train[:cut_to]
y_train = y_train[:cut_to]
x_test = x_test[:cut_to]
y_test = y_test[:cut_to]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

### Create and Train the Model
model = WaveletCNN(imageShape, nb_classes)
model.summary()
optimizer = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

history = model.fit(x_train, y_train,
                    validation_split=1 - trainFactor,
                    epochs=epochs,
                    verbose=2,
                    callbacks=[
                        ModelCheckpoint("tmp.h5", monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                        EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
                        ],
                    )

model.load_weights("tmp.h5")
model.save(filepath)

### Model Evaluation
result = model.evaluate(x_test, y_test)

print(result)

# plot train and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show();
plt.close()

