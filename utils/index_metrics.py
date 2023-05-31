import numpy as np


def nbr(image):
    """Normalized Burn Ratio:
    (B8 - B12) / (B8 + B12)
    """
    num = image[:, :, 7] - image[:, :, 11]
    den = image[:, :, 7] + image[:, :, 11]
    return np.divide(num, den, out=np.zeros_like(num), where=den != 0)


def nbr2(image):
    """Normalized Burn Ratio 2:
    (B11 - B12) / (B11 + B12)
     """
    num = image[:, :, 10] - image[:, :, 11]
    den = image[:, :, 10] + image[:, :, 11]
    return np.divide(num, den, out=np.zeros_like(num), where=den != 0)


def bais2(image):
    """Burned Area Index for Sentinel-2:
    (1 - sqrt(B06 * B07 * B8A / B4)) * ((B12 - B8A) / sqrt(B12 + B8A) + 1)
    """
    num_a = image[:, :, 5] * image[:, :, 6] * image[:, :, 8]
    num_b = image[:, :, 3]
    den_a = image[:, :, 11] - image[:, :, 8]
    den_b = np.sqrt(image[:, :, 11] + image[:, :, 8])
    return (1 - np.sqrt(np.divide(num_a, num_b, out=np.zeros_like(num_a), where=num_b != 0))) * (
            1 + np.divide(den_a, den_b, out=np.zeros_like(num_a), where=den_b != 0))
