import os
from typing import List

import imageio
import numpy as np
import tensorflow as tf


def load_local_data(data, frames, bach_size, height, width, channels, folder):
    for bch in range(bach_size):
        path = folder[np.random.randint(len(folder))] + '/'
        bb = np.random.randint(0,  415-width)
        for f in range(frames):
            img = imageio.imread(path + 'f' + str(f + 1).zfill(3) + '.png')
            data[f, bch, 0 : height, 0 : width, 0 : channels] = img[0 : height, bb : bb + width, 0 : channels]
    return data


def load_random_path(np_folder):
    paths = np.load(np_folder)   
    path = paths[np.random.randint(len(paths))] + '/'
    return path


def load_data(data, frames, batch_size, Height, Width, Channel, folder, I_QP):

    for b in range(batch_size):

        path = folder[np.random.randint(len(folder))] + '/'

        bb = np.random.randint(0, 447 - Width)

        for f in range(frames):

            if f == 0:
                img = imageio.imread(path + 'im1_bpg444_QP' + str(I_QP) + '.png')
                data[f, b, 0:Height, 0:Width, 0:Channel] = img[0:Height, bb: bb + Width, 0:Channel]
            else:
                img = imageio.imread(path + 'im' + str(f + 1) + '.png')
                data[f, b, 0:Height, 0:Width, 0:Channel] = img[0:Height, bb: bb + Width, 0:Channel]

    return data

def read_png_resize(filename, width, height):
    """Loads a PNG image file."""
    string = tf.io.read_file(filename)
    image = tf.image.decode_image(string, channels=3)
    img_res = tf.image.resize(image, [width,height])
    return tf.cast(img_res, dtype=tf.uint8)

def read_png_crop(filename, width, height):
    """Loads a PNG image file."""
    string = tf.io.read_file(filename)
    image = tf.image.decode_image(string, channels=3)
    img_crop = tf.image.crop_to_bounding_box(image, 0, 0, width, height)
    return tf.cast(img_crop, dtype=tf.uint8)


def load_data_vimeo90k(np_folder, samples, Height, Width, Channel, I_QP):

    paths = np.load(np_folder)   
    path = paths[np.random.randint(len(paths))] + '/'

    data = list()
    bb = np.random.randint(0, 447 - Width)
    for s in range(samples):
        f = np.random.randint(7)
        if f == 0:
            img_ref = read_png_crop(path + 'im1_bpg444_QP' + str(I_QP) + '.png', Width, Height)
            img_cur = read_png_crop(path + 'im' + str(f + 1) + '.png', Width, Height)
        else:
            img_ref = read_png_crop(path + 'im' + str(1) + '.png', Width, Height)
            img_cur = read_png_crop(path + 'im' + str(f + 1) + '.png', Width, Height)
        
        data.append([tf.expand_dims(img_ref, 0), tf.expand_dims(img_cur, 0)])     
        
    return data


class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, np_folder, batch_size=4, width=240, height=240, channels=3, I_QP=27):
    super().__init__()
    self.dir_paths = np.load(np_folder)
    self.batch_size = batch_size
    self.width = width
    self.height= height
    self.channels = channels
    self.on_epoch_end()
    self.i_qp = I_QP
  def __len__(self):
    return len(self.dir_paths)*7//self.batch_size

  def __getitem__(self, index):
    img_path = self.dir_paths[index] + "/"

    frame = np.random.randint(7)
    bb = np.random.randint(0, 447 - self.width)

    img_ref = imageio.imread(img_path + 'im1_bpg444_QP' + str(self.i_qp) + '.png')
    img_cur = imageio.imread(img_path + 'im' + str(frame + 1) + '.png')
    
    
    img_ref= img_ref[0:self.height, bb: bb + self.width, 0:self.channels]
    img_cur = img_cur[0:self.height, bb: bb + self.width, 0:self.channels]

    x = [img_ref, img_cur]
    y = img_cur
    return x, y

      




if __name__=="__main__":
    import numpy as np

    import load

    batch_size = 4
    Height = 240
    Width = 240
    Channel = 3
    lr_init = 1e-4
    frames=2
    samples=10
    I_QP=27

    # generator = DataGenerator("/mnt/WindowsDev/Developer/tensorflow-wavelets/folder_cloud.npy")
    # for data in generator:
    #     print(data)
    # folder = np.load("/mnt/WindowsDev/Developer/tensorflow-wavelets/folder_cloud.npy")
    # folder = ["/workspaces/tensorflow-wavelets/Development/OpenDVC/BasketballPass"]
    data = load_data_vimeo90k("/mnt/WindowsDev/Developer/tensorflow-wavelets/folder_cloud.npy",
                                samples, Height, Width, Channel, I_QP)

   
    for perm in data:
        print(perm[0].shape)
    #a = load.load_data()
    # data = np.zeros([frames, batch_size, Height, Width, Channel])
    # data - load_local_data(data, frames, batch_size, Height, Width, Channel, folder)
    # # data = load_data(data, frames, batch_size, Height, Width, Channel, folder, I_QP)
    # print("Data Load done! ...")