import numpy as np

import load

batch_size = 4
Height = 256
Width = 256
Channel = 3
lr_init = 1e-4
frames=2
I_QP=27

folder = np.load("/mnt/WindowsDev/Developer/tensorflow-wavelets/folder_cloud.npy")
#a = load.load_data()
data = np.zeros([frames, batch_size, Height, Width, Channel])
data = load.load_data(data, frames, batch_size, Height, Width, Channel, folder, I_QP)
