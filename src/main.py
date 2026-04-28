"""Interface de linha de comando para aplicar transformacoes e filtros.

Este modulo recebe a imagem de entrada, seleciona a tarefa informada por
argumento e salva a imagem processada no caminho de saida.
"""

import argparse
import sys
import numpy as np
import cv2
from thresholding import bernsen_threshold_binarization, global_threshold_binarization, niblack_threshold_binarization, otsu_threshold_binarization, sauvola_threshold_binarization
from utils import display_histogram_and_stats, load_image, save_image, display_results


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
                            "sauvola"
                        ],
                        help="Select the thesholding method to apply")

    # Optional flag to display results
    parser.add_argument("--display", action="store_true", help="Display original and processed images side-by-side")
    
    # Optional flag for colored mode
    parser.add_argument("--colored", action="store_true", help="Load image in color mode instead of monochromatic")

    args = parser.parse_args()

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
        threshold = 10
    elif args.thresholding == "niblack":
        window_size = int(input("Entre com o tamanho da janela (impar): "))
        k = float(input("Entre com o valor de k (ex: -0.2): "))
        result = niblack_threshold_binarization(img, window_size, k)
        threshold = 10
    elif args.thresholding == "sauvola":
        window_size = int(input("Entre com o tamanho da janela (impar): "))
        k = float(input("Entre com o valor de k (ex: 0.5): "))
        r = float(input("Entre com o valor de R (ex: 128): "))
        result = sauvola_threshold_binarization(img, window_size, k, r)
        threshold = 10
    else:
        print("Invalid task.")
        sys.exit(1)

    save_image(result, "output", args.output)
    if args.display:
        display_histogram_and_stats(img, result, threshold, args.thresholding, args.output)
        display_results(img, result, title=args.thresholding)
    
    print("Processing completed successfully.")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
if __name__ == "__main__":
    main()

