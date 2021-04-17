import pickle
import matplotlib.pyplot as plt

history_file_path = r"..\trainHistoryDict.txt"

with open(history_file_path, 'rb') as pickle_file:
    history = pickle.load(pickle_file)

# plot train and validation loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show();
plt.close()
