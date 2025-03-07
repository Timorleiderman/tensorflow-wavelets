import pytest
import random
import logging
import numpy as np
import tensorflow as tf

from tensorflow_wavelets.Layers.DWT import DWT1D, IDWT1D

logger = logging.getLogger(__file__)

def create_signal(channel_count: int):
    # Create a simple 1D signal
    num_samples = random.randint(8, 128) * 2
    x = np.linspace(0, 2 * np.pi, num_samples)
    signal = np.sin(random.randint(1, 1e5) * x).astype(np.float32)  
    return np.repeat(signal[None, :, None], repeats=channel_count, axis=-1)


@pytest.mark.parametrize("wavelet_name", ["haar", "db2", "coif1"])
def test_wavelet_transform(wavelet_name):
    
    for idx in range(1, 8):
        signal = create_signal(idx)
        logger.info(f"{signal.shape}")
        # Create Wavelet Transform layers
        dwt = DWT1D(wavelet_name)
        idwt = IDWT1D(wavelet_name)

        # Pass through DWT and then inverse DWT
        transformed = dwt(signal)
        reconstructed = idwt(transformed).numpy()

        assert signal.shape == reconstructed.shape
        
        # Assert that the reconstructed signal is close to the original
        np.testing.assert_allclose(signal, reconstructed[0:, :, :], atol=1e-5)
        