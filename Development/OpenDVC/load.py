import os
from typing import List

import imageio
import numpy as np
import tensorflow as tf




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

def read_png_crop_np(filename, width, height):
    """Loads a png file"""
    img = imageio.imread(filename)
    return img[0:height, 0: width, :]

def load_local_data(path, samples, height, width, channels):
    
    data = list()
    for s in range(2,samples):

        img_ref = read_png_crop(path + '/f' + str(1).zfill(3) + '.png', width, height)
        img_cur = read_png_crop(path + '/f' + str(s).zfill(3) + '.png', width, height)
    
        data.append([tf.expand_dims(img_ref, 0), tf.expand_dims(img_cur, 0)])     
    
    return data

def load_data_vimeo90k(np_folder, samples, Height, Width, Channel, I_QP):

    paths = np.load(np_folder)   
    path = paths[np.random.randint(len(paths))] + '/'

    data = list()
    # data_out = list()
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
        # data_out.append(tf.expand_dims(img_cur/255, 0))
    return data


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