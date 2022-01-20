from os import path
import imageio
import numpy as np

def load_local_data(data, frames, bach_size, height, width, channels, folder):
    for bch in range(bach_size):
        path = folder[np.random.randint(len(folder))] + '/'
        bb = np.random.randint(0,  415-width)
        for f in range(frames):
            img = imageio.imread(path + 'f' + str(f + 1).zfill(3) + '.png')
            data[f, bch, 0 : height, 0 : width, 0 : channels] = img[0 : height, bb : bb + width, 0 : channels]
    return data

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


if __name__=="__main__":
    import numpy as np

    import load

    batch_size = 4
    Height = 240
    Width = 240
    Channel = 3
    lr_init = 1e-4
    frames=2
    I_QP=27

    # folder = np.load("/mnt/WindowsDev/Developer/tensorflow-wavelets/folder_cloud.npy")
    folder = ["/workspaces/tensorflow-wavelets/Development/OpenDVC/BasketballPass"]
    #a = load.load_data()
    data = np.zeros([frames, batch_size, Height, Width, Channel])
    data - load_local_data(data, frames, batch_size, Height, Width, Channel, folder)
    # data = load_data(data, frames, batch_size, Height, Width, Channel, folder, I_QP)
    print("Data Load done! ...")