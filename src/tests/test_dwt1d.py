import pytest
import tensorflow as tf
import numpy as np
import sys


from tensorflow_wavelets.Layers.DWT import DWT1D, IDWT1D  # Replace 'your_module' with the actual module name


@pytest.mark.parametrize("wavelet_name", ["haar", "db2", "coif1"])
def test_wavelet_transform(wavelet_name):
    # Create a simple 1D signal
    signal = np.sin(np.linspace(0, 2*np.pi, 32))
    signal = signal.reshape(1, -1, 1).astype(np.float32)  # Shape: (batch_size, length, channels)

    # Create Wavelet Transform layers
    dwt = DWT1D(wavelet_name)
    idwt = IDWT1D(wavelet_name)

    # Pass through DWT and then inverse DWT
    transformed = dwt(signal)
    reconstructed = idwt(transformed).numpy()

    assert signal.shape == reconstructed.shape
    
    # Assert that the reconstructed signal is close to the original
    np.testing.assert_allclose(signal, reconstructed[0:, :, :], atol=1e-5)