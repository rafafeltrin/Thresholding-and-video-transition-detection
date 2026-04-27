"""Interface de linha de comando para aplicar transformacoes e filtros.

Este modulo recebe a imagem de entrada, seleciona a tarefa informada por
argumento e salva a imagem processada no caminho de saida.
"""

import argparse
import sys
import numpy as np
import cv2
from utils import load_image, save_image, display_results


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
    parser.add_argument("--task", type=str, required=True,
                        choices=[
                            "x", 
                            "y"
                        ],
                        help="Select the transformation or filter to apply")

    # Optional flag to display results
    parser.add_argument("--display", action="store_true", help="Display original and processed images side-by-side")
    
    # Optional flag for colored mode
    parser.add_argument("--colored", action="store_true", help="Load image in color mode instead of monochromatic")

    args = parser.parse_args()

    if args.colored:
        img = load_image(args.input, monochromatic=False)
    else:
        img = load_image(args.input, monochromatic=True)
    
    if args.task == "x":
        result = img
    elif args.task == "y":
        result = img
    else:
        print("Invalid task.")
        sys.exit(1)

    save_image(result, "output", args.output)
    if args.display:
        display_results(img, result, title=args.task)
    
    print("Processing completed successfully.")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
if __name__ == "__main__":
    main()

