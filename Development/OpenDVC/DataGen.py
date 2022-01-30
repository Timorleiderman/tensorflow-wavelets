import numpy as np
import tensorflow as tf
import tensorflow.keras
import load
import os
import sys
import fnmatch


class DataVimeo90kGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, np_folder, batch_size=32, dim=(240,240,32), n_channels=3, shuffle=True, I_QP=27): 
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.np_folder = np_folder
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.i_qp = I_QP
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(np.load(self.np_folder))/self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        print("get item:", index)

        # Generate data
        return self.__data_generation()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
           
        # path = self.paths[np.random.randint(len(self.paths))] + '/'
        pass

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = load.load_data_vimeo90k(self.np_folder, self.batch_size, self.dim[0], self.dim[1], self.dim[2], self.i_qp)

        return X

def generate_local_npy(pattern, path):
    result = list()
    print("")
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(root)
                sys.stdout.write('\r'+root + " ...")
    print("")
    return result

    return result
if __name__ == "__main__":
    print("hey")
    # a = generate_local_npy("f001.png", "/workspaces/tensorflow-wavelets/Development/OpenDVC")
    # np.save('local_basketball_cpy.npy', a)

    a = DataVimeo90kGenerator("local_basketball_cpy.npy")

    for data in a:
        print(data)