import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

"""Funcoes utilitarias para E/S de imagens e visualizacao.

Este modulo concentra operacoes de carregamento, salvamento e exibicao
de imagens usadas no fluxo principal do projeto.
"""

def load_image(path: str, monochromatic: bool=True) -> np.ndarray:
    """Carrega uma imagem do disco usando OpenCV.

    :param path: Caminho do arquivo de imagem.
    :param monochromatic: Se verdadeiro, carrega em tons de cinza.
    :returns: Imagem carregada como arranjo NumPy.
    :raises FileNotFoundError: Se o arquivo nao puder ser lido.
    """
        
    if monochromatic:
        # Loads image as a single 2D matrix (M x N)
        flag = cv2.IMREAD_GRAYSCALE
    else:
        # Loads as a 3D matrix (M x N x 3) in BGR format
        flag = cv2.IMREAD_COLOR
    
    
    img = cv2.imread(path, flag)
    
    if img is None:
        raise FileNotFoundError(f"Could not find image at {path}")
    
    return img

def save_image(img: np.ndarray, folder: str, filename: str) -> None:
    """Salva uma imagem no disco.

    :param img: Imagem a ser salva.
    :param folder: Pasta de destino.
    :param filename: Nome do arquivo de saida.
    :returns: Nao retorna valor.
    :raises FileNotFoundError: Se a pasta de destino nao existir.
    """
    
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    
    full_path = os.path.join(folder, filename)
    
    # checks the file extension to decide the format
    image = cv2.imwrite(full_path, img)
    
    if not image:
        print(f"Failed to save image to {full_path}")
    else:
        print(f"Image saved: {full_path}")

def display_results(original: np.ndarray, processed: np.ndarray, title="Result"):
    """Exibe imagem original e processada lado a lado.

    :param original: Imagem original.
    :param processed: Imagem apos processamento.
    :param title: Titulo do painel da imagem processada.
    :returns: Nao retorna valor.
    """
    def to_matplotlib_color(img: np.ndarray) -> np.ndarray:
        """Converte imagem BGR para RGB quando necessario.

        :param img: Imagem de entrada.
        :returns: Imagem no formato esperado pelo Matplotlib.
        """
        if img.ndim == 3:
            # OpenCV usa BGR, enquanto Matplotlib espera RGB.
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(to_matplotlib_color(original), cmap='gray' if len(original.shape) == 2 else None)
    
    plt.subplot(1, 2, 2)
    plt.title(title)
    plt.imshow(to_matplotlib_color(processed), cmap='gray' if len(processed.shape) == 2 else None)
    
    plt.show()


def calculate_black_pixel_fraction(binarized_img: np.ndarray) -> float:
    total_pixels = binarized_img.size
    
    black_pixel_count = np.sum(binarized_img == 0)
    
    fraction = black_pixel_count / total_pixels
    
    return fraction

def display_histogram_and_stats(
    original_img: np.ndarray, 
    binarized_img: np.ndarray, 
    method_name: str, 
    threshold_value: float = None, 
    img_title: str = "Image"
):
    fraction = calculate_black_pixel_fraction(binarized_img)
    
    print(f"Método aplicado: {method_name}")
    if threshold_value is not None:
        print(f"Calculo do limiar: {threshold_value:.2f}")
    print(f"Fração de pixels do objeto (preto): {fraction:.4f} ({fraction * 100:.2f}%)\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(original_img.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.75)
    if threshold_value is not None:
        ax1.axvline(x=threshold_value, color='red', linestyle='--', linewidth=2, label=f'Limiar T={threshold_value:.1f}')
        ax1.legend()
    ax1.set_title(f'Histograma Original - {method_name}')
    ax1.set_xlabel('Intensidade (0-255)')
    ax1.set_ylabel('Número de pixels')
    ax1.grid(axis='y', alpha=0.3)
    
    ax2.hist(binarized_img.ravel(), bins=256, range=(0, 256), color='black', alpha=0.75, label=f'Objeto (Preto): {fraction * 100:.2f}%')
    ax2.set_title('Histograma Binarizado')
    ax2.set_xlabel('Intensidade (0-255)')
    ax2.set_ylabel('Número de pixels')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"output/histogram_{img_title}")
    plt.show()
    