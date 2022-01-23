import OpenDVC
import numpy as np
import load
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

path = load.load_random_path("/mnt/WindowsDev/Developer/tensorflow-wavelets/folder_cloud.npy")

arg = OpenDVC.Arguments()

model = tf.saved_model.load(arg.model_save)

i_frame = path + 'im1' + '.png'
p_frame = path + 'im2' + '.png'
print(i_frame)
out_bin = "/workspaces/tensorflow-wavelets/Development/OpenDVC/Test_com/test.bin"
out_decom = "/workspaces/tensorflow-wavelets/Development/OpenDVC/Test_com/testdcom.png"
p_on_test = "/workspaces/tensorflow-wavelets/Development/OpenDVC/Test_com/test_p_frame.png"
i_on_test = "/workspaces/tensorflow-wavelets/Development/OpenDVC/Test_com/test_i_frame.png"

OpenDVC.write_png(p_on_test, OpenDVC.read_png_crop(p_frame, 240, 240))
OpenDVC.write_png(i_on_test, OpenDVC.read_png_crop(i_frame, 240, 240))

OpenDVC.compress(model, i_frame, p_frame,out_bin, 240, 240)
OpenDVC.decompress(model, i_frame, out_bin, out_decom, 240, 240)