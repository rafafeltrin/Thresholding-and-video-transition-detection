from typing import Tuple

import numpy as np
from scipy import ndimage

def global_threshold_binarization(img: np.ndarray, threshold: int) -> np.ndarray:
    """Binariza a imagem a partir de um limiar.

    Pixels com intensidade maior que `threshold` recebem 255; os demais,
    0.

    :param img: Imagem de entrada.
    :param threshold: Limiar de corte para binarizacao.
    :returns: Imagem binaria em `uint8`.
    """
    result = np.where(img > threshold, 0, 255)
    return result.astype(np.uint8)

def otsu_threshold_binarization(img: np.ndarray) -> Tuple[np.ndarray, int]:
    histogram, _ = np.histogram(img.flatten(), bins=256, range=(0, 255))
    total_pixels = img.size

    p_i = histogram / total_pixels

    omega_1 = np.cumsum(p_i)
    omega_2 = 1 - omega_1

    i = np.arange(256)
    mu_s = np.cumsum(i * p_i)
    mu_t = np.sum(i * p_i)

    mu_1 = np.zeros_like(mu_s)
    mu_2 = np.zeros_like(mu_s)

    np.divide(mu_s, omega_1, out=mu_1, where=omega_1 > 0)
    np.divide(mu_t - mu_s, omega_2, out=mu_2, where=omega_2 > 0)

    sigma_b_squared = omega_1 * omega_2 * (mu_1 - mu_2) ** 2

    t = np.argmax(sigma_b_squared)

    result = np.where(img > t, 0, 255).astype(np.uint8)
    print(f"Otsu's threshold: {t}")
    return (result, int(t))

def bernsen_threshold_binarization(img: np.ndarray, window_size: int = 91) -> np.ndarray:
    z_max = ndimage.maximum_filter(img, size=window_size)
    z_min = ndimage.minimum_filter(img, size=window_size)

    z_max_float = z_max.astype(np.float32)
    z_min_float = z_min.astype(np.float32)

    t = (z_max_float + z_min_float) / 2

    result = np.where(img > t, 0, 255).astype(np.uint8)
    
    return result


def niblack_threshold_binarization(img: np.ndarray, window_size: int = 15, k: float = -0.2) -> np.ndarray:
    img_float = img.astype(np.float64)

    local_mean = ndimage.uniform_filter(img_float, size=window_size)

    local_mean_sq = ndimage.uniform_filter(img_float ** 2, size=window_size)

    local_variance = local_mean_sq - local_mean ** 2

    local_std = np.sqrt(np.maximum(local_variance, 0))

    t = local_mean + k * local_std

    result = np.where(img > t, 0, 255).astype(np.uint8)

    return result

def sauvola_threshold_binarization(img: np.ndarray, window_size: int = 15, k: float = 0.5, R: float = 128) -> np.ndarray:
    img_float = img.astype(np.float64)

    local_mean = ndimage.uniform_filter(img_float, size=window_size)

    local_mean_sq = ndimage.uniform_filter(img_float ** 2, size=window_size)

    local_variance = local_mean_sq - local_mean ** 2

    local_std = np.sqrt(np.maximum(local_variance, 0))

    t = local_mean * (1 + k * (local_std / R - 1))

    result = np.where(img > t, 0, 255).astype(np.uint8)

    return result