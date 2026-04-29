"""Interface de linha de comando para aplicar transformacoes e filtros.

Este modulo recebe a imagem de entrada, seleciona a tarefa informada por
argumento e salva a imagem processada no caminho de saida.
"""

import argparse
import sys
import numpy as np
import cv2
from thresholding import (
    bernsen_threshold_binarization, 
    global_threshold_binarization, 
    niblack_threshold_binarization, 
    otsu_threshold_binarization, 
    sauvola_threshold_binarization,
    phansalskar_threshold_binarization,
    mean_threshold_binarization,
    median_threshold_binarization,
    contrast_threshold_binarization
)
from utils import display_histogram_and_stats, load_image, save_image, display_results, display_histograms_separate, save_histogram


def main():
    """Executa o fluxo principal da aplicacao via CLI.

    Configura e interpreta os argumentos, carrega a imagem de entrada,
    aplica a transformacao/filtro selecionado e salva o resultado.

    :returns: Nao retorna valor. O efeito e a escrita do arquivo de saida.
    """
    parser = argparse.ArgumentParser(description="MC920 - Trabalho 2")
    
    # Positional arguments for Input and Output paths
    parser.add_argument("input", help="Path to input PNG image")
    parser.add_argument("output", help="Path to save output PNG image")

    # Task selection argument
    parser.add_argument("--thresholding", type=str, required=True,
                        choices=[
                            "global", 
                            "otsu",
                            "bernsen",
                            "niblack",
                            "sauvola",
                            "phansalskar",
                            "mean",
                            "median",
                            "contrast",
                            "histogram"
                        ],
                        help="Select the thesholding method to apply")

    # Optional flag to display results
    parser.add_argument("--display", action="store_true", help="Display original and processed images side-by-side")
    
    # Optional flag for colored mode
    parser.add_argument("--colored", action="store_true", help="Load image in color mode instead of monochromatic")

    args = parser.parse_args()
    threshold = None

    if args.colored:
        img = load_image(args.input, monochromatic=False)
    else:
        img = load_image(args.input, monochromatic=True)
    
    if args.thresholding == "global":
        threshold = int(input("Enter the threshold value (0-255): "))
        result = global_threshold_binarization(img, threshold)
    elif args.thresholding == "otsu":
        result,threshold  = otsu_threshold_binarization(img)
    elif args.thresholding == "bernsen":
        window_size = int(input("Entre com o tamanho da janela (impar): "))
        result = bernsen_threshold_binarization(img, window_size)
    elif args.thresholding == "niblack":
        window_size = int(input("Entre com o tamanho da janela (impar): "))
        k = float(input("Entre com o valor de k (ex: -0.2): "))
        result = niblack_threshold_binarization(img, window_size, k)
    elif args.thresholding == "sauvola":
        window_size = int(input("Entre com o tamanho da janela (impar): "))
        k = float(input("Entre com o valor de k (ex: 0.5): "))
        r = float(input("Entre com o valor de R (ex: 128): "))
        result = sauvola_threshold_binarization(img, window_size, k, r)
    elif args.thresholding == "phansalskar":
        window_size = int(input("Entre com o tamanho da janela (impar): "))
        k = float(input("Entre com o valor de k (ex: 0.25): "))
        r = float(input("Entre com o valor de R (ex: 0.5): "))
        p = float(input("Entre com o valor de p (ex: 2.0): "))
        q = float(input("Entre com o valor de q (ex: 10.0): "))
        result = phansalskar_threshold_binarization(img, window_size, k, r, p, q)
    elif args.thresholding == "mean":
        window_size = int(input("Entre com o tamanho da janela (impar): "))
        c = float(input("Entre com o valor de C (ex: 10.0): "))
        result = mean_threshold_binarization(img, window_size, c)
    elif args.thresholding == "median":
        window_size = int(input("Entre com o tamanho da janela (impar): "))
        result = median_threshold_binarization(img, window_size)
    elif args.thresholding == "contrast":
        window_size = int(input("Entre com o tamanho da janela (impar): "))
        result = contrast_threshold_binarization(img, window_size)
    elif args.thresholding == "histogram":
        save_histogram(img, args.output)
        sys.exit(0)
    else:
        print("Invalid task.")
        sys.exit(1)

    save_image(result, "output", args.output)
    if args.display:
        if threshold is not None:
            display_histogram_and_stats(img, result, args.thresholding, threshold, args.output)
        else:
            display_histograms_separate(result, args.output)
        display_results(img, result, title=args.thresholding)
    
    print("Processing completed successfully.")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
if __name__ == "__main__":
    main()
