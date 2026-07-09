from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from letter_graph import (
    has_diagonal_connections,
    has_disconnected_parts,
    has_enclosed_space,
    has_eulerian_trail,
    has_horizontal_symmetry,
    has_vertical_symmetry,
    has_visual_crossings_extended,
    has_visual_crossings_restricted,
)


NODES = 9

# Each straight line of the 3x3 grid is represented by 4 states:
# 0 = empty
# 1 = first segment only
# 2 = second segment only
# 3 = full line (both segments, equivalent to the merged outer segment)
LINE_NODE_SEQUENCES = [
    (0, 1, 2),  # A-B-C
    (3, 4, 5),  # D-E-F
    (6, 7, 8),  # G-H-I
    (0, 3, 6),  # A-D-G
    (1, 4, 7),  # B-E-H
    (2, 5, 8),  # C-F-I
    (0, 4, 8),  # A-E-I
    (2, 4, 6),  # C-E-G
]
RULES: list[tuple[str, Callable[[np.ndarray], bool]]] = [
    ("Diagonal", has_diagonal_connections),
    ("Enclosed", has_enclosed_space),
    ("Disconnected", has_disconnected_parts),
    ("Vertical symmetry", has_vertical_symmetry),
    ("Horizontal symmetry", has_horizontal_symmetry),
    ("Eulerian trail", has_eulerian_trail),
    ("Visual crossings extended", has_visual_crossings_extended),
    ("Visual crossings restricted", has_visual_crossings_restricted),
]


def build_matrix_from_mask(mask: int) -> np.ndarray:
    """Build a symmetric 9x9 matrix from a 2-bit-per-line state mask."""

    matrix = np.zeros((NODES, NODES), dtype=np.uint8)

    for line_index, nodes in enumerate(LINE_NODE_SEQUENCES):
        state = (mask >> (2 * line_index)) & 0b11
        start, middle, end = nodes

        if state == 0:
            continue
        if state == 1:
            matrix[start, middle] = 1
            matrix[middle, start] = 1
        elif state == 2:
            matrix[middle, end] = 1
            matrix[end, middle] = 1
        elif state == 3:
            matrix[start, end] = 1
            matrix[end, start] = 1
    return matrix


def format_matrix_line(matrix: np.ndarray) -> str:
    """Format a 9x9 matrix as one line of row strings separated by spaces."""

    return " ".join("".join(str(int(value)) for value in row) for row in matrix)


def evaluate_rules(matrix: np.ndarray) -> str:
    """Return the compact rule signature, one bit per rule in RULES order."""

    return "".join("1" if rule(matrix) else "0" for _, rule in RULES)


def iter_masks(start: int = 0, stop: int | None = None) -> Iterable[int]:
    total = 1 << (2 * len(LINE_NODE_SEQUENCES))
    if stop is None or stop > total:
        stop = total
    for mask in range(start, stop):
        yield mask


def write_results(output_path: str, start: int = 0, stop: int | None = None, progress_every: int = 0) -> int:
    """Write all requested matrices and rule bits to a text file.

    Each line uses the format:
    <row0> <row1> ... <row8> #<rule_bits>
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    with path.open("w", encoding="utf-8") as handle:
        for index, mask in enumerate(iter_masks(start=start, stop=stop), start=1):
            matrix = build_matrix_from_mask(mask)
            line = f"{format_matrix_line(matrix)} #{evaluate_rules(matrix)}\n"
            handle.write(line)
            total_written += 1

            if progress_every and index % progress_every == 0:
                print(f"Wrote {index} matrices...")

    return total_written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate every reduced 9x9 matrix made of the 8 straight lines of the "
            "3x3 grid, with 4 states per line, and evaluate the graph rules for each one."
        )
    )
    parser.add_argument("--output", default="all_matrix_rules.txt", help="Destination text file.")
    parser.add_argument("--start", type=int, default=0, help="Start mask index (default: 0).")
    parser.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Exclusive stop mask index. Defaults to the full search space.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print a progress message every N matrices.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    total_written = write_results(
        output_path=args.output,
        start=args.start,
        stop=args.stop,
        progress_every=args.progress_every,
    )
    print(f"Wrote {total_written} lines to {args.output}")


if __name__ == "__main__":
    main()
