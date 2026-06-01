from __future__ import annotations

from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
POSITIONS = {
    "A": (0, 2),
    "B": (1, 2),
    "C": (2, 2),
    "D": (0, 1),
    "E": (1, 1),
    "F": (2, 1),
    "G": (0, 0),
    "H": (1, 0),
    "I": (2, 0),
}


def _validate_binary_symmetric_matrix(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix)
    if array.shape != (9, 9):
        raise ValueError(f"Expected a 9x9 matrix, got {array.shape!r}.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("Matrix must contain numeric values.")
    if not np.array_equal(array, array.T):
        raise ValueError("Matrix must be symmetric.")
    if not np.all(np.isin(array, [0, 1])):
        raise ValueError("Matrix must be binary (only 0 and 1 values).")
    return array.astype(int, copy=False)


def render_letter_graph(matrix: np.ndarray, dpi: int = 200) -> Image.Image:
    """Render a symmetric 9x9 binary adjacency matrix as a 3x3 letter graph."""

    adjacency = _validate_binary_symmetric_matrix(matrix)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    ax.set_aspect("equal")
    ax.axis("off")

    for x in range(3):
        ax.axvline(x, color="#E6E6E6", linewidth=1, zorder=0)
    for y in range(3):
        ax.axhline(y, color="#E6E6E6", linewidth=1, zorder=0)

    for i in range(9):
        for j in range(i + 1, 9):
            if adjacency[i, j] == 1:
                x1, y1 = POSITIONS[LABELS[i]]
                x2, y2 = POSITIONS[LABELS[j]]
                ax.plot([x1, x2], [y1, y2], color="#2E6F9E", linewidth=2.0, alpha=0.85, zorder=1)

    for label in LABELS:
        x, y = POSITIONS[label]
        circle = plt.Circle((x, y), 0.16, facecolor="#F7F7F7", edgecolor="#222222", linewidth=1.5, zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center", fontsize=14, fontweight="bold", color="#111111", zorder=3)

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    fig.tight_layout(pad=0.2)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer)
    image.load()
    return image


def save_letter_graph(matrix: np.ndarray, output_path: str, dpi: int = 200) -> None:
    """Render and save the graph as a PNG file."""

    image = render_letter_graph(matrix, dpi=dpi)
    image.save(output_path)


if __name__ == "__main__":
    sample = np.zeros((9, 9), dtype=int)
    sample[0, 1] = sample[1, 0] = 1
    sample[3, 4] = sample[4, 3] = 1
    sample[0, 4] = sample[4, 0] = 1
    sample[3, 8] = sample[8, 3] = 1
    save_letter_graph(sample, "letter_graph.png")