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


def has_diagonal_connections(matrix: np.ndarray) -> bool:
    """Return True if the graph contains at least one non-orthogonal edge."""

    adjacency = _validate_binary_symmetric_matrix(matrix)

    for i in range(9):
        x1, y1 = POSITIONS[LABELS[i]]
        for j in range(i + 1, 9):
            if adjacency[i, j] != 1:
                continue

            x2, y2 = POSITIONS[LABELS[j]]
            if x1 != x2 and y1 != y2:
                return True

    return False


def has_enclosed_space(matrix: np.ndarray) -> bool:
    """Return True if the graph contains at least one cycle/closed loop."""

    adjacency = _validate_binary_symmetric_matrix(matrix)
    visited = [False] * 9

    def depth_first_search(node: int, parent: int) -> bool:
        visited[node] = True

        for neighbor in range(9):
            if adjacency[node, neighbor] != 1:
                continue
            if not visited[neighbor]:
                if depth_first_search(neighbor, node):
                    return True
            elif neighbor != parent:
                return True

        return False

    for node in range(9):
        if not visited[node] and depth_first_search(node, -1):
            return True

    return False


def has_disconnected_parts(matrix: np.ndarray) -> bool:
    """Return True if the graph has more than one non-empty connected component."""

    adjacency = _validate_binary_symmetric_matrix(matrix)
    visited = [False] * 9

    def depth_first_search(node: int) -> None:
        visited[node] = True
        for neighbor in range(9):
            if adjacency[node, neighbor] == 1 and not visited[neighbor]:
                depth_first_search(neighbor)

    active_nodes = [node for node in range(9) if np.any(adjacency[node] == 1)]

    components = 0
    for node in active_nodes:
        if not visited[node]:
            components += 1
            if components > 1:
                return True
            depth_first_search(node)

    return False


def has_vertical_symmetry(matrix: np.ndarray) -> bool:
    """Return True if the graph is symmetric around the vertical B-E-H axis."""

    adjacency = _validate_binary_symmetric_matrix(matrix)

    mirrored_indices = {
        0: 2,
        1: 1,
        2: 0,
        3: 5,
        4: 4,
        5: 3,
        6: 8,
        7: 7,
        8: 6,
    }

    mirrored = adjacency.copy()
    for source_row, target_row in mirrored_indices.items():
        for source_col, target_col in mirrored_indices.items():
            mirrored[target_row, target_col] = adjacency[source_row, source_col]

    return np.array_equal(adjacency, mirrored)


def has_visual_crossings(matrix: np.ndarray) -> bool:
    """Return True if at least two drawn edges cross visually.

    This checks segment intersections in the 3x3 layout and ignores pairs of
    edges that only meet because they share a node or touch at an endpoint.
    It still counts cases where a segment passes through a node that is the
    endpoint of another segment, which is a visually meaningful crossing.
    """

    adjacency = _validate_binary_symmetric_matrix(matrix)
    edges = []

    for i in range(9):
        for j in range(i + 1, 9):
            if adjacency[i, j] == 1:
                edges.append((i, j))

    def orientation(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
            and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])
            and orientation(p, q, r) == 0
        )

    def segments_cross(p1, q1, p2, q2):
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
            return True

        # If a node lies on the interior of one segment and is an endpoint of
        # another, that is also a visible crossing in this drawing.
        for node in (p2, q2):
            if node != p1 and node != q1 and on_segment(p1, node, q1):
                return True

        for node in (p1, q1):
            if node != p2 and node != q2 and on_segment(p2, node, q2):
                return True

        return False

    for index, (i1, j1) in enumerate(edges):
        p1 = POSITIONS[LABELS[i1]]
        q1 = POSITIONS[LABELS[j1]]

        for i2, j2 in edges[index + 1 :]:
            if {i1, j1} & {i2, j2}:
                continue

            p2 = POSITIONS[LABELS[i2]]
            q2 = POSITIONS[LABELS[j2]]
            if segments_cross(p1, q1, p2, q2):
                return True

    return False


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
    sample[1, 8] = sample[8, 1] = 1
    save_letter_graph(sample, "letter_graph.png")
    has_diagonal = has_diagonal_connections(sample)
    has_cycle = has_enclosed_space(sample)
    has_crossings = has_visual_crossings(sample)
    has_disconnected = has_disconnected_parts(sample)
    has_vertical_symmetry = has_vertical_symmetry(sample)
    print(f"Has diagonal connections: {has_diagonal}")
    print(f"Has enclosed space: {has_cycle}")
    print(f"Has visual crossings: {has_crossings}")
    print(f"Has disconnected parts: {has_disconnected}")
    print(f"Has vertical symmetry: {has_vertical_symmetry}")