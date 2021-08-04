
import numpy as np
import math


def cast_like_matlab_uint8_2d(data):

    h, w = data.shape
    for row in range(h):
        for col in range(w):
            frac, integ = math.modf(data[row,col])
            if frac >= 0.5:
                data[row, col] = np.ceil(data[row, col])
            elif frac < 0.5:
                data[row, col] = np.floor(data[row, col])

    data_clip = np.clip(data, 0, 255)
    return data_clip.astype('uint8')