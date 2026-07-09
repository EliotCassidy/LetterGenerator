from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from letter_graph import _validate_binary_symmetric_matrix, save_letter_graph


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate random binary symmetric matrices, clean them, and save them as PNGs."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of PNGs to generate (default: 100).",
    )
    parser.add_argument(
        "--output-dir",
        default="random_matrix_pngs",
        help="Directory where PNG renders will be written.",
    )
    parser.add_argument(
        "--prob",
        type=float,
        default=0.3,
        help="Probability of an edge between any two nodes (default: 0.3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for repeatable generation.",
    )
    return parser


def generate_random_symmetric_matrix(rng: random.Random, prob: float) -> np.ndarray:
    """Generate a random binary symmetric 9x9 matrix with zero diagonal."""
    matrix = np.zeros((9, 9), dtype=int)
    for i in range(9):
        for j in range(i + 1, 9):
            val = 1 if rng.random() < prob else 0
            matrix[i, j] = val
            matrix[j, i] = val
    return matrix


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    print(f"Generating {args.count} random symmetric matrices with edge probability {args.prob}...")

    for index in range(1, args.count + 1):
        # 1. Generate a random binary symmetric matrix
        raw_matrix = generate_random_symmetric_matrix(rng, args.prob)
        
        # 2. Clean/validate it using the defined function
        cleaned_matrix = _validate_binary_symmetric_matrix(raw_matrix)
        
        # 3. Generate and save the PNG
        output_path = output_dir / f"random_{index:03d}.png"
        save_letter_graph(cleaned_matrix, str(output_path))
        
        if index % 10 == 0 or index == args.count:
            print(f"Saved {index}/{args.count} PNGs to {output_path}")

    print("Generation complete!")


if __name__ == "__main__":
    main()
