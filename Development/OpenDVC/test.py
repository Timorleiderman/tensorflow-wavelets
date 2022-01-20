import OpenDVC
import numpy as np
from load import load_local_data, load_data
import tensorflow as tf
import matplotlib.pyplot as plt

tf.executing_eagerly()

batch_size = 2
EPOCHS = 10
Height = 64
Width = 64
Channel = 3
lr_init = 1e-4
frames=6
I_QP=27

folder = np.load("/mnt/WindowsDev/Developer/tensorflow-wavelets/folder_cloud.npy")

data = np.zeros([frames, batch_size, Height, Width, Channel])
# data = load_local_data(data, frames, batch_size, Height, Width, Channel, folder)
data = load_data(data, frames, batch_size, Height, Width, Channel, folder, I_QP)
dataset = tf.data.Dataset.from_tensor_slices(data).batch(2)

arg = OpenDVC.Arguments()
model = tf.saved_model.load(arg.model_save)

OpenDVC.compress(arg, model)
OpenDVC.decompress(arg, model)