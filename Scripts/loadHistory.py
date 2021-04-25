import pickle
import matplotlib.pyplot as plt

history_file_path = r"..\trainHistoryCifar10CNN.txt"
history_file_path_wavelet = r"..\trainHistoryWaveletCifar10CNN.txt"

with open(history_file_path, 'rb') as pickle_file:
    history = pickle.load(pickle_file)

with open(history_file_path_wavelet, 'rb') as pickle_file_wavelet:
    history_wavelet = pickle.load(pickle_file_wavelet)

fig, (ax1, ax2) = plt.subplots(1, 2)

fig.suptitle('Horizontally stacked subplots')

# plot train and validation loss
ax1.plot(history['loss'])
ax1.plot(history['val_loss'])
ax1.set_title('Maxpooling2d model loss')
ax1.set_ylabel('loss')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')

# plot train and validation loss
ax2.plot(history_wavelet['loss'])
ax2.plot(history_wavelet['val_loss'])
ax2.set_title('DWT_Pooling model loss')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(['train', 'validation'], loc='upper left')

fig.show()
plt.show()



