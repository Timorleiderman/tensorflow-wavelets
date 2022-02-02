import numpy as np
import tensorflow as tf
import tensorflow.keras
import load
import os
import sys
import fnmatch


class DataVimeo90kGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, np_folder, samples=32, dim=(240,240,32), n_channels=3, shuffle=True, I_QP=27): 
        'Initialization'
        self.dim = dim
        self.samples = samples
        self.np_folder = np_folder
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.i_qp = I_QP
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(np.load(self.np_folder))/self.samples)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # Generate data
        return self.__data_generation()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
           
        # path = self.paths[np.random.randint(len(self.paths))] + '/'
        pass

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        paths = np.load(self.np_folder) 
        path = paths[np.random.randint(len(paths))] + '/'

        # data = list()
        # data_out = list()
        I_QP = self.i_qp
        Width = self.dim[0]
        Height = self.dim[1]
        X0 = np.empty((self.samples, *self.dim))
        X1 = np.empty((self.samples, *self.dim))
        for sample in range(self.samples):
            f = np.random.randint(7)
            if f == 0:
                img_ref = load.read_png_crop_np(path + 'im1_bpg444_QP' + str(I_QP) + '.png', Width, Height)
                img_cur = load.read_png_crop_np(path + 'im' + str(f + 1) + '.png', Width, Height)
            else:
                img_ref = load.read_png_crop_np(path + 'im' + str(1) + '.png', Width, Height) 
                img_cur = load.read_png_crop_np(path + 'im' + str(f + 1) + '.png', Width, Height)
            X0[sample,] = img_ref / 255
            X1[sample,] = img_cur / 255

        # data.append()     
            # data_out.append(tf.expand_dims(img_cur/255, 0))
        # X = [tf.expand_dims(img_ref, 0), tf.expand_dims(img_cur, 0)]
        # return np.expand_dims(img_ref, 0), np.expand_dims(img_cur, 0), None
        return X0, X1, None
        # X = load.load_data_vimeo90k(self.np_folder, 1, self.dim[0], self.dim[1], self.dim[2], self.i_qp)
        
        # return X

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

    a = DataVimeo90kGenerator("/mnt/WindowsDev/Developer/tensorflow-wavelets/folder_cloud.npy", 4, (240,240,3), 3, True, 27)

    for data in a:
        print(data[0].shape)