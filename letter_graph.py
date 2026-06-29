from __future__ import annotations

import math
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
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


def has_horizontal_symmetry(matrix: np.ndarray) -> bool:
    """Return True if the graph is symmetric around the horizontal D-E-F axis.

    This reflects rows: top<->bottom (A<->G, B<->H, C<->I), leaving D/E/F
    unchanged.
    """

    adjacency = _validate_binary_symmetric_matrix(matrix)

    mirrored_indices = {
        0: 6,
        1: 7,
        2: 8,
        3: 3,
        4: 4,
        5: 5,
        6: 0,
        7: 1,
        8: 2,
    }

    mirrored = adjacency.copy()
    for source_row, target_row in mirrored_indices.items():
        for source_col, target_col in mirrored_indices.items():
            mirrored[target_row, target_col] = adjacency[source_row, source_col]

    return np.array_equal(adjacency, mirrored)


def has_eulerian_trail(matrix: np.ndarray) -> bool:
    """Return True if the graph can be drawn in one stroke (Eulerian trail).

    Conditions for an undirected graph:
    - The graph is connected when ignoring isolated vertices, and
    - Either 0 or 2 vertices have odd degree.
    An empty graph (no edges) is considered drawable.
    """

    adjacency = _validate_binary_symmetric_matrix(matrix)
    degrees = adjacency.sum(axis=1)
    active = np.where(degrees > 0)[0]

    if active.size == 0:
        return True

    # check connectivity among active nodes
    visited = set()
    stack = [int(active[0])]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        neighbors = np.where(adjacency[node] == 1)[0]
        for nb in neighbors:
            if int(nb) not in visited:
                stack.append(int(nb))

    if set(active) - visited:
        return False

    odd = int(np.sum(degrees % 2 == 1))
    return odd == 0 or odd == 2


def has_visual_crossings(matrix: np.ndarray) -> bool:
    """Return True if at least two drawn edges cross visually.

    A visual crossing happens when two full straight lines intersect on the
    drawing. That includes:
    - a proper interior intersection between two edges,
    - a segment passing through a node used by another edge, and
    - two straight lines that cross at a node, even if each line is split into
      two edges by that node.
    """

    array = np.asarray(matrix).astype(int)
    A, B, C, D, E, F, G, H, I = range(9)

    # 1. TEST DU NOEUD CENTRAL (E) - Valable pour toute la grille
    lines_at_E = [
        array[H, B],  # HB
        array[D, F],  # DF
        array[A, I],  # AI
        array[C, G]   # CG
    ]
    if sum(lines_at_E) >= 2:
        return True

    # 2. CONFIGURATIONS SOURCE POUR LE CADRAN MAÎTRE : EHFI (Bas-Droite)
    simple_crossings = [
        ((I, E), (H, F)),  # N1
        ((I, B), (H, F)),  # N2
        ((I, D), (H, F)),  # N8
        ((G, F), (I, B)),  # N3
        ((C, H), (I, D))   # N7
    ]

    # Groupes concourants (N4, N5, N6) : "2 parmi les 3"
    concurent_groups = [
        [(I, E), (G, F), (H, C)],  # N5
        [(E, F), (H, C), (B, I)],  # N4
        [(E, H), (I, D), (G, F)]   # N6
    ]

    # 3. TRANSPOSITION ET APPLICATION AUX 4 CADRANS VIA LES SYMÉTRIES    
    cadrans_mappings = [
        # 1. Bas-Droite (EHFI) -> Identité
        {A:A, B:B, C:C, D:D, E:E, F:F, G:G, H:H, I:I},
        # 2. Bas-Gauche (DGHIE) -> Inversion Gauche/Droite
        {C:A, B:B, A:C, F:D, E:E, D:F, I:G, H:H, G:I},
        # 3. Haut-Droite (ABEDF) -> Inversion Haut/Bas
        {G:A, H:B, I:C, D:D, E:E, F:F, A:G, B:H, C:I},
        # 4. Haut-Gauche (ABED) -> Double Inversion
        {I:A, H:B, G:C, F:D, E:E, D:F, C:G, B:H, A:I}
    ]

    # Test pour chaque cadran géométrique
    for mapping in cadrans_mappings:
        # A. Test des croisements simples (N1, N2, N8, N3, N7)
        for (u1, v1), (u2, v2) in simple_crossings:
            edge1_active = array[mapping[u1], mapping[v1]] == 1
            edge2_active = array[mapping[u2], mapping[v2]] == 1
            if edge1_active and edge2_active:
                return True

        # B. Test des groupes concourants (N4, N5, N6)
        for edge_list in concurent_groups:
            active_count = 0
            for u, v in edge_list:
                if array[mapping[u], mapping[v]] == 1:
                    active_count += 1
                    if active_count >= 2:
                        return True

    return False


def _rule_results(matrix: np.ndarray) -> list[tuple[str, bool]]:
    return [
        ("Diagonal", has_diagonal_connections(matrix)),
        ("Enclosed", has_enclosed_space(matrix)),
        ("Disconnected", has_disconnected_parts(matrix)),
        ("Vertical symmetry", has_vertical_symmetry(matrix)),
        ("Horizontal symmetry", has_horizontal_symmetry(matrix)),
        ("Eulerian trail", has_eulerian_trail(matrix)),
        ("Visual crossings", has_visual_crossings(matrix)),
    ]


def _render_matrix_thumbnail(ax: plt.Axes, adjacency: np.ndarray) -> None:
    ax.imshow(adjacency, cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)

    for row in range(9):
        for col in range(9):
            ax.text(col, row, str(int(adjacency[row, col])), ha="center", va="center", fontsize=5, color="#111111")

    ax.set_title("Matrix 0/1", fontsize=10, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#888888")


def _render_rules_panel(ax: plt.Axes, adjacency: np.ndarray) -> None:
    ax.axis("off")
    ax.set_title("Rules", fontsize=12, fontweight="bold", loc="left", pad=8)

    y = 0.94
    for label, value in _rule_results(adjacency):
        status = "TRUE" if value else "FALSE"
        color = "#1B8A3D" if value else "#B42318"
        ax.text(0.0, y, f"{label}:", transform=ax.transAxes, fontsize=9, ha="left", va="top", color="#111111")
        ax.text(0.98, y, status, transform=ax.transAxes, fontsize=9, ha="right", va="top", color=color, fontweight="bold")
        y -= 0.105


def _render_graph_panel(ax: plt.Axes, adjacency: np.ndarray) -> None:
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


def render_letter_graph(matrix: np.ndarray, dpi: int = 200) -> Image.Image:
    """Render the graph plus a side panel with rule results and a matrix thumbnail."""

    adjacency = _validate_binary_symmetric_matrix(matrix)

    fig = plt.figure(figsize=(10, 5.8), dpi=dpi, constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig, width_ratios=[1.65, 1.0], height_ratios=[1.15, 0.85], wspace=0.18, hspace=0.28)

    ax_graph = fig.add_subplot(grid[:, 0])
    ax_rules = fig.add_subplot(grid[0, 1])
    ax_matrix = fig.add_subplot(grid[1, 1])

    _render_graph_panel(ax_graph, adjacency)
    _render_rules_panel(ax_rules, adjacency)
    _render_matrix_thumbnail(ax_matrix, adjacency)

    fig.suptitle("Letter Graph Summary", fontsize=14, fontweight="bold", y=0.98)

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
    has_vert_symmetry = has_vertical_symmetry(sample)
    has_hor_symmetry = has_horizontal_symmetry(sample)
    has_eulerian = has_eulerian_trail(sample)
    print(f"Has diagonal connections: {has_diagonal}")
    print(f"Has enclosed space: {has_cycle}")
    print(f"Has visual crossings: {has_crossings}")
    print(f"Has disconnected parts: {has_disconnected}")
    print(f"Has vertical symmetry: {has_vert_symmetry}")
    print(f"Has horizontal symmetry: {has_hor_symmetry}")
    print(f"Has Eulerian trail: {has_eulerian}")