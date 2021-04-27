import os
import pickle
import matplotlib.pyplot as plt

num_folds = 10

history_file_path = r"../{}_trainHistoryCifar10CNN.txt"

fig, ax = plt.subplots(1, 10)
fig.suptitle('Horizontally stacked subplots')

for fold in range(num_folds):
    with open(history_file_path.format(fold+1), 'rb') as pickle_file:
        history = pickle.load(pickle_file)


    # plot train and validation loss
    ax[fold].plot(history['loss'])
    ax[fold].plot(history['val_loss'])
    ax[fold].set_title('model loss fold:'+str(fold+1))
    ax[fold].set_ylabel('loss')
    ax[fold].set_xlabel('epoch')
    ax[fold].legend(['train', 'validation'], loc='upper left')


fig.show()
plt.show()



