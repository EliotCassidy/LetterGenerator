from __future__ import annotations

import argparse
import random
from pathlib import Path

from exhaustive_matrix_rules import LINE_NODE_SEQUENCES, build_matrix_from_mask
from letter_graph import save_letter_graph


def generate_random_grids(count: int, output_dir: str, seed: int | None = None) -> list[Path]:
    """Generate `count` random grids (each of the 8 lines gets a random state:
    empty / first half / second half / full) and save each as a PNG showing
    the graph drawing plus the rules panel (save_letter_graph)."""

    rng = random.Random(seed)
    total_masks = 1 << (2 * len(LINE_NODE_SEQUENCES))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    created_files: list[Path] = []
    for index in range(1, count + 1):
        mask = rng.randrange(total_masks)
        matrix = build_matrix_from_mask(mask)
        file_path = output_path / f"grid_{index:03d}.png"
        save_letter_graph(matrix, str(file_path))
        created_files.append(file_path)

    return created_files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate random 3x3-grid graphs and save one PNG per grid (drawing + rules panel)."
    )
    parser.add_argument("--count", type=int, default=100, help="Number of random grids to generate.")
    parser.add_argument("--output-dir", default="test_value_grid", help="Directory where PNG files will be written.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    created_files = generate_random_grids(args.count, args.output_dir, seed=args.seed)
    print(f"Generated {len(created_files)} PNG files in {args.output_dir}")


if __name__ == "__main__":
    main()
