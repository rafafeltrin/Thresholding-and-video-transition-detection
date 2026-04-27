from typing import Tuple

import numpy as np

def global_threshold_binarization(img: np.ndarray, threshold: int) -> np.ndarray:
    """Binariza a imagem a partir de um limiar.

    Pixels com intensidade maior que `threshold` recebem 255; os demais,
    0.

    :param img: Imagem de entrada.
    :param threshold: Limiar de corte para binarizacao.
    :returns: Imagem binaria em `uint8`.
    """
    result = np.where(img > threshold, 255, 0)
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