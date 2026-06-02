from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from letter_graph import save_letter_graph, _validate_binary_symmetric_matrix


UPPERCASE_ALPHABET = [chr(code_point) for code_point in range(ord("A"), ord("Z") + 1)]


def build_default_letter_matrices() -> dict[str, np.ndarray]:
    """Build a deterministic set of 26 valid 9x9 matrices for A-Z.

    The exact letter shapes are intentionally simple; they exist so the script
    can run out of the box and still produce 26 PNG files.
    """

    templates: dict[str, list[tuple[int, int]]] = {
        "A": [(0, 1), (1, 2), (0, 3), (2, 5), (3, 4), (4, 5), (3, 6), (5, 8)],
        "B": [(0, 3), (3, 6), (0, 1), (1, 4), (4, 2), (2, 5), (4, 7), (7, 8)],
        "C": [(0, 1), (1, 2), (0, 3), (3, 6), (6, 7), (7, 8)],
        "D": [(0, 1), (1, 2), (0, 3), (3, 6), (6, 7), (7, 8), (2, 5), (5, 8)],
        "E": [(0, 1), (1, 2), (0, 3), (3, 6), (3, 4), (4, 5), (6, 7), (7, 8)],
        "F": [(0, 1), (1, 2), (0, 3), (3, 6), (3, 4), (4, 5)],
        "G": [(0, 1), (1, 2), (0, 3), (3, 6), (6, 7), (7, 8), (4, 5), (5, 8)],
        "H": [(0, 3), (3, 6), (2, 5), (5, 8), (3, 4), (4, 5), (1, 4), (4, 7)],
        "I": [(0, 1), (1, 2), (1, 4), (4, 7), (6, 7), (7, 8)],
        "J": [(0, 1), (1, 2), (2, 5), (5, 8), (6, 7), (7, 8), (4, 7)],
        "K": [(0, 3), (3, 6), (1, 4), (4, 8), (2, 4), (4, 6)],
        "L": [(0, 3), (3, 6), (6, 7), (7, 8)],
        "M": [(0, 3), (3, 6), (0, 4), (4, 8), (2, 5), (5, 8), (4, 5)],
        "N": [(0, 3), (3, 6), (2, 5), (5, 8), (3, 4), (4, 5)],
        "O": [(0, 1), (1, 2), (0, 3), (3, 6), (6, 7), (7, 8), (2, 5), (5, 8)],
        "P": [(0, 1), (1, 2), (0, 3), (3, 6), (3, 4), (4, 5), (1, 4), (2, 5)],
        "Q": [(0, 1), (1, 2), (0, 3), (3, 6), (6, 7), (7, 8), (2, 5), (5, 8), (4, 8)],
        "R": [(0, 1), (1, 2), (0, 3), (3, 6), (3, 4), (4, 5), (1, 4), (2, 5), (4, 8)],
        "S": [(0, 1), (1, 2), (0, 3), (3, 4), (4, 5), (5, 8), (6, 7), (7, 8)],
        "T": [(0, 1), (1, 2), (1, 4), (4, 7)],
        "U": [(0, 3), (3, 6), (2, 5), (5, 8), (6, 7), (7, 8)],
        "V": [(0, 4), (4, 8), (2, 4), (4, 6), (6, 7), (7, 8)],
        "W": [(0, 3), (3, 6), (6, 4), (4, 8), (8, 5), (5, 2), (2, 7), (7, 6)],
        "X": [(0, 4), (4, 8), (2, 4), (4, 6)],
        "Y": [(0, 4), (2, 4), (4, 7)],
        "Z": [(0, 1), (1, 2), (2, 4), (4, 6), (6, 7), (7, 8)],
    }

    matrices: dict[str, np.ndarray] = {}
    for letter in UPPERCASE_ALPHABET:
        edges = templates[letter]
        matrix = np.zeros((9, 9), dtype=int)
        for a, b in edges:
            matrix[a, b] = 1
            matrix[b, a] = 1
        matrices[letter] = matrix
    return matrices


def load_letter_matrices(source_path: str) -> dict[str, np.ndarray]:
    """Load a mapping of uppercase letters to 9x9 matrices.

    Supported formats:
    - .json: {"A": [[...], ...], "B": [[...], ...], ...}
    - .npz: arrays stored under keys "A" ... "Z"
    """

    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")

    matrices: dict[str, np.ndarray] = {}

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("JSON input must be an object mapping letters to matrices.")
        for letter in UPPERCASE_ALPHABET:
            if letter not in payload:
                raise ValueError(f"Missing matrix for letter {letter}.")
            matrices[letter] = _validate_binary_symmetric_matrix(np.asarray(payload[letter]))
        return matrices

    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            for letter in UPPERCASE_ALPHABET:
                if letter not in data:
                    raise ValueError(f"Missing matrix for letter {letter}.")
                matrices[letter] = _validate_binary_symmetric_matrix(np.asarray(data[letter]))
        return matrices

    raise ValueError("Unsupported input format. Use .json or .npz.")


def generate_letter_pngs(letter_matrices: Mapping[str, np.ndarray], output_dir: str) -> list[Path]:
    """Generate one PNG per uppercase letter using the provided matrices."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    created_files: list[Path] = []
    for letter in UPPERCASE_ALPHABET:
        if letter not in letter_matrices:
            raise ValueError(f"Missing matrix for letter {letter}.")

        matrix = _validate_binary_symmetric_matrix(np.asarray(letter_matrices[letter]))
        file_path = output_path / f"{letter}.png"
        save_letter_graph(matrix, str(file_path))
        created_files.append(file_path)

    return created_files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 26 PNG files from 26 letter matrices.")
    parser.add_argument("--input", help="Path to a .json or .npz file containing 26 matrices.")
    parser.add_argument("--output-dir", default="generated_letters", help="Directory where PNG files will be written.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.input:
        letter_matrices = load_letter_matrices(args.input)
    else:
        letter_matrices = build_default_letter_matrices()
    created_files = generate_letter_pngs(letter_matrices, args.output_dir)
    print(f"Generated {len(created_files)} PNG files in {args.output_dir}")


if __name__ == "__main__":
    main()
