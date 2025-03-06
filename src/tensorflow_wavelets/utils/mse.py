import numpy as np


def mse(imageA, imageB):

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err

def mse_1d(a, b):
    """Mean Squared Error calculation."""
    return np.mean((a - b) ** 2)