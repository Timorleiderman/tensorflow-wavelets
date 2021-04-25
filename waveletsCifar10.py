import os
import pickle
import datetime
import numpy as np
import tensorflow as tf
import models.cifar10CNN
import models.WaveletCifar10CNN

from keras.datasets import cifar10
from sklearn.model_selection import KFold
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
num_folds = 10
batch_size = 32
epochs = 50

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

# Define per-fold score containers <-- these are new
acc_per_fold = []
loss_per_fold = []

inputs = np.concatenate((x_train, x_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):

    # Create and Train the Model
    model = models.cifar10CNN.cifar10CNN(imageShape, nb_classes)
    history_file_path = str(fold_no) + "_trainHistoryCifar10CNN.txt"  # save loss and val loss

    # model = models.WaveletCifar10CNN.WaveletCNN(imageShape, nb_classes)
    # history_file_path = "trainHistoryWaveletCifar10CNN.txt" # save loss and val loss
    # model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

    log_dir = logs_filepath + r"\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    h5_tmp = "fold_" + str(fold_no) + "_tmp.h5"
    history = model.fit(inputs[train], targets[train],
                        validation_split=1 - trainFactor,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2,
                        callbacks=[
                            ModelCheckpoint("tmp.h5", monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                            # ModelCheckpoint(h5_tmp, monitor='loss', verbose=0, save_best_only=True, mode='auto'),
                            EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=10, verbose=1, mode='auto'),
                            # EarlyStopping(monitor='loss', min_delta=1e-3, patience=10, verbose=1, mode='auto'),
                            tensorboard_callback
                            ],
                        )

    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

    model.load_weights("tmp.h5")
    weights_path = os.path.join(weights_filepath, str(fold_no) + "_WCNNN.h5")
    model.save(weights_path)

    # Model Evaluation
    # result = model.evaluate(x_test, y_test)

    with open(history_file_path, 'wb') as f:
        pickle.dump(history.history, f)

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')


# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')