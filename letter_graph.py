from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction
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

# Coordonnées exactes des 9 noeuds réels, indexées par 0-8. Nécessaire (en
# fractions, pas en float) pour identifier de manière unique les points de
# croisement lors de la construction de la matrice étendue : deux cas qui
# détectent le même point doivent produire la même clé (un seul noeud virtuel).
NODE_COORDS = [
    (Fraction(POSITIONS[label][0]), Fraction(POSITIONS[label][1]))
    for label in LABELS
]


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


A, B, C, D, E, F, G, H, I = range(9)

# Arêtes "pleines" (alignées avec la grille) et le noeud réel que chacune
# enjambe en son milieu (ex : A-C enjambe B). Sert à reconnaître qu'un
# demi-segment (ex A-B) est visuellement dessiné dès que la ligne pleine qui
# le contient (ex A-C) est active, même si A-B n'est pas encodé séparément
# dans la matrice.
TRAVERSED_NODES: dict[tuple[int, int], int] = {
    (A, C): B,
    (D, F): E,
    (G, I): H,
    (A, G): D,
    (B, H): E,
    (C, I): F,
    (A, I): E,
    (C, G): E,
}


def _canonical(u: int, v: int) -> tuple[int, int]:
    """Forme canonique (u < v) d'une arête."""
    return (u, v) if u < v else (v, u)


# Demi-segment -> ligne pleine qui le contient, dérivé de TRAVERSED_NODES.
HALF_TO_FULL: dict[tuple[int, int], tuple[int, int]] = {}
for _full, _mid in TRAVERSED_NODES.items():
    _u, _v = _full
    HALF_TO_FULL[_canonical(_u, _mid)] = _full
    HALF_TO_FULL[_canonical(_mid, _v)] = _full


def _realizing_edge(array: np.ndarray, u: int, v: int) -> tuple[int, int] | None:
    """Arête réellement dessinée qui matérialise le segment (u, v) : lui-même
    s'il est actif, sinon la ligne pleine qui le contient si elle l'est.
    None si (u, v) n'est visuellement dessiné par aucun trait de la matrice.
    """
    edge = _canonical(u, v)
    if array[edge] == 1:
        return edge
    full = HALF_TO_FULL.get(edge)
    if full is not None and array[full] == 1:
        return full
    return None


def _edge_drawn(array: np.ndarray, u: int, v: int) -> bool:
    """True si le segment (u, v) est visuellement dessiné : soit directement,
    soit parce que la ligne pleine qui le contient (ex A-C pour A-B) est
    active dans la matrice.
    """
    return _realizing_edge(array, u, v) is not None


# Variante "liste" de _realizing_edge, pour find_crossing_points /
# build_extended_matrix : si le demi-segment ET la ligne pleine qui le
# contient sont TOUS LES DEUX dessinés indépendamment (redondance dans la
# matrice), les deux doivent être coupés au point de croisement, pas
# seulement un des deux.
def _realizers(array: np.ndarray, edge: tuple[int, int]) -> list[tuple[int, int]]:
    """Arêtes réellement dessinées qui matérialisent le segment abstrait
    `edge` : l'arête elle-même, et/ou la ligne pleine qui la contient.
    Liste vide si le segment n'est réalisé par aucun trait de la matrice.
    """
    drawn = []
    if array[edge] == 1:
        drawn.append(edge)
    full = HALF_TO_FULL.get(edge)
    if full is not None and array[full] == 1:
        drawn.append(full)
    return drawn


# ---------------------------------------------------------------------------
# Configurations du cadran maître EHFI (bas-droite) transposées aux 4
# cadrans par symétrie. Réemployées par has_visual_crossings_extended,
# has_visual_crossings_restricted, has_disconnected_parts et
# build_extended_matrix : ce sont les mêmes points de croisement, qu'on les
# utilise pour détecter un croisement visuel, savoir si deux composantes du
# graphe se touchent en un point qui n'est pas un noeud réel, ou construire
# la matrice étendue avec un noeud virtuel à cet endroit.
#
# Le point de croisement de chaque cas n'est pas recopié à la main : il est
# calculé géométriquement (intersection de droites, en fractions exactes) à
# partir des coordonnées des noeuds réels que les arêtes relient.
# ---------------------------------------------------------------------------

Point = tuple[Fraction, Fraction]


def _line_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Point:
    """Point d'intersection des droites (p1p2) et (p3p4), en fractions exactes.

    Formule de l'intersection de deux droites définies chacune par deux
    points (déterminants) : https://en.wikipedia.org/wiki/Line-line_intersection
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        raise ValueError("Droites parallèles : pas de point d'intersection unique.")

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _crossing_point(edge1: tuple[int, int], edge2: tuple[int, int]) -> Point:
    """Point d'intersection des deux arêtes, à partir des coordonnées réelles
    de leurs 4 extrémités (NODE_COORDS)."""
    (u1, v1), (u2, v2) = edge1, edge2
    return _line_intersection(NODE_COORDS[u1], NODE_COORDS[v1], NODE_COORDS[u2], NODE_COORDS[v2])


def _assert_point_on_edge_line(point: Point, edge: tuple[int, int]) -> None:
    """Vérifie que `point` est bien aligné avec l'arête (garde-fou pour les
    groupes concourants : les 3 arêtes doivent passer par le même point)."""
    u, v = edge
    ux, uy = NODE_COORDS[u]
    vx, vy = NODE_COORDS[v]
    px, py = point
    cross_product = (vx - ux) * (py - uy) - (vy - uy) * (px - ux)
    if cross_product != 0:
        raise AssertionError(f"{point} n'est pas sur la droite de l'arête {edge}.")


# Croisements simples (N1, N2, N8, N3, N7) : une paire d'arêtes, dont on
# calcule le point de croisement à partir de leurs extrémités.
_SIMPLE_CROSSING_EDGES: list[tuple[tuple[int, int], tuple[int, int]]] = [
    ((I, E), (H, F)),  # N1
    ((I, B), (H, F)),  # N2
    ((I, D), (H, F)),  # N8
    ((G, F), (I, B)),  # N3
    ((C, H), (I, D)),  # N7
]
SIMPLE_CROSSINGS = [
    (edge1, edge2, _crossing_point(edge1, edge2))
    for edge1, edge2 in _SIMPLE_CROSSING_EDGES
]

# Groupes concourants (N4, N5, N6) : "2 parmi les 3" arêtes passent par le
# même point ; toute paire active y crée un croisement. Le point est calculé
# à partir des 2 premières arêtes, puis on vérifie que la 3ème y passe aussi.
_CONCURRENT_GROUP_EDGES: list[list[tuple[int, int]]] = [
    [(I, E), (G, F), (H, C)],  # N5
    [(E, F), (H, C), (B, I)],  # N4
    [(E, H), (I, D), (G, F)],  # N6
]
CONCURRENT_GROUPS = []
for _edges in _CONCURRENT_GROUP_EDGES:
    _point = _crossing_point(_edges[0], _edges[1])
    _assert_point_on_edge_line(_point, _edges[2])
    CONCURRENT_GROUPS.append((_edges, _point))

# Les 4 cadrans géométriques : mapping d'indices depuis le cadran maître,
# associé à la transformation géométrique correspondante (appliquée aux
# coordonnées des points de croisement). La grille s'étend de 0 à 2 : miroir
# gauche/droite <=> x -> 2 - x, miroir haut/bas <=> y -> 2 - y.
QUADRANTS = [
    # 1. Bas-Droite (EHFI) -> Identité
    ({A: A, B: B, C: C, D: D, E: E, F: F, G: G, H: H, I: I},
     lambda p: p),
    # 2. Bas-Gauche (DGHIE) -> Inversion Gauche/Droite
    ({C: A, B: B, A: C, F: D, E: E, D: F, I: G, H: H, G: I},
     lambda p: (2 - p[0], p[1])),
    # 3. Haut-Droite (ABEDF) -> Inversion Haut/Bas
    ({G: A, H: B, I: C, D: D, E: E, F: F, A: G, B: H, C: I},
     lambda p: (p[0], 2 - p[1])),
    # 4. Haut-Gauche (ABED) -> Double Inversion
    ({I: A, H: B, G: C, F: D, E: E, D: F, C: G, B: H, A: I},
     lambda p: (2 - p[0], 2 - p[1])),
]


def find_crossing_points(matrix: np.ndarray) -> dict[tuple, set]:
    """Identification des croisements HORS noeuds réels, par les tables.

    Retourne {point: {arêtes qui s'y croisent}}. Le dictionnaire déduplique
    automatiquement : un même point détecté par plusieurs cas ou plusieurs
    cadrans ne crée qu'une entrée, et toutes les arêtes impliquées s'y
    accumulent pour s'assurer ensuite de réaliser le bon découpage.
    """
    array = _validate_binary_symmetric_matrix(matrix)
    crossing_points: dict[tuple, set] = defaultdict(set)

    for mapping, transform in QUADRANTS:

        # A. Croisements simples (N1, N2, N8, N3, N7).
        # Un segment abstrait est actif s'il est réalisé par au moins un
        # trait dessiné (lui-même ou sa ligne pleine) ; on coupe le(s)
        # trait(s) réellement dessiné(s).
        for (u1, v1), (u2, v2), point in SIMPLE_CROSSINGS:
            drawn1 = _realizers(array, _canonical(mapping[u1], mapping[v1]))
            drawn2 = _realizers(array, _canonical(mapping[u2], mapping[v2]))
            if drawn1 and drawn2:
                crossing_points[transform(point)].update(drawn1 + drawn2)

        # B. Groupes concourants (N4, N5, N6) : croisement dès que >= 2
        # segments abstraits DISTINCTS du groupe sont réalisés (deux
        # traits superposés réalisant le même segment ne se croisent pas).
        for edge_list, point in CONCURRENT_GROUPS:
            realized = [
                _realizers(array, _canonical(mapping[u], mapping[v]))
                for (u, v) in edge_list
            ]
            realized = [drawn for drawn in realized if drawn]
            if len(realized) >= 2:
                for drawn in realized:
                    crossing_points[transform(point)].update(drawn)

    return dict(crossing_points)


def find_virtual_nodes(matrix: np.ndarray) -> list[tuple[tuple, list]]:
    """Version 'rapport' pour inspection : [(point, [arêtes]), ...] trié."""
    return sorted(
        (point, sorted(edges))
        for point, edges in find_crossing_points(matrix).items()
    )


def build_extended_matrix(matrix: np.ndarray):
    """Retourne (matrice étendue NxN, dictionnaire coordonnée -> indice).

    Matrice NxN vide, où chaque arête d'origine est remplacée par la chaîne
    de ses morceaux (coupes triées le long de l'arête) : les lignes pleines
    traversant un noeud réel (TRAVERSED_NODES) sont toujours coupées à ce
    noeud si elles sont actives, et les croisements hors noeuds
    (find_crossing_points) ajoutent un noeud virtuel à cet endroit. L'arête
    directe coupée n'est jamais écrite telle quelle.
    """
    array = _validate_binary_symmetric_matrix(matrix)
    crossing_points = find_crossing_points(array)

    # Enregistre les différents points de croisement pour chaque arête.
    cut_points: dict[tuple[int, int], set] = defaultdict(set)

    #   Table 1 : arêtes pleines coupées au noeud réel enjambé.
    for edge, node in TRAVERSED_NODES.items():
        if array[edge] == 1:
            cut_points[edge].add(NODE_COORDS[node])

    #   Table 2 (via find_crossing_points) : croisements hors noeuds.
    for point, edges in crossing_points.items():
        for edge in edges:
            cut_points[edge].add(point)

    # Numérotation : un indice par point de la nouvelle grille. Les 9 noeuds
    # de départ conservent leur indice (0-8), les points de croisement sont
    # ajoutés à la suite (9, 10, ...). Permet de ne pas réajouter un noeud
    # réel (ex E) avec un nouvel indice s'il est aussi victime d'un croisement.
    coord_to_index = {coord: index for index, coord in enumerate(NODE_COORDS)}
    for points in cut_points.values():
        for p in sorted(points):
            coord_to_index.setdefault(p, len(coord_to_index))

    # Crée la nouvelle matrice étendue à la bonne taille NxN.
    n_total = len(coord_to_index)
    extended = np.zeros((n_total, n_total), dtype=int)

    for u in range(9):  # parcourt la matrice d'origine et regarde les arêtes existantes.
        for v in range(u + 1, 9):
            if array[u, v] != 1:
                continue
            a = NODE_COORDS[u]

            def distance_along_edge(p):  # distance au point u de l'arête, pour ordonner les coupes s'il y en a plusieurs.
                return (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2

            interior = sorted(cut_points.get((u, v), ()), key=distance_along_edge)
            chain = [u] + [coord_to_index[p] for p in interior] + [v]  # chaîne (u, p1, p2, ..., v) le long de l'arête.
            for node_1, node_2 in zip(chain, chain[1:]):  # relie chaque paire de points consécutifs de la chaîne.
                extended[node_1, node_2] = extended[node_2, node_1] = 1

    return extended, coord_to_index


def _extended_has_cycle(adjacency: np.ndarray) -> bool:
    """True si le graphe non orienté (matrice NxN étendue) contient un cycle."""
    n_total = adjacency.shape[0]
    visited = [False] * n_total

    def depth_first_search(node: int, parent: int) -> bool:
        visited[node] = True
        for neighbor in range(n_total):
            if adjacency[node, neighbor] != 1:
                continue
            if not visited[neighbor]:
                if depth_first_search(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        return False

    return any(
        not visited[node] and depth_first_search(node, -1)
        for node in range(n_total)
    )


def has_enclosed_space(matrix: np.ndarray) -> bool:
    """True si le dessin contient une forme fermée, au sens visuel : une
    ligne qui traverse un noeud réel, ou deux lignes qui se croisent en un
    point qui n'est pas un noeud réel, peuvent fermer une boucle même sans
    arête directe entre les noeuds concernés (cf. build_extended_matrix)."""
    extended, _ = build_extended_matrix(matrix)
    return _extended_has_cycle(extended)


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


def has_disconnected_parts(matrix: np.ndarray) -> bool:
    """Return True if the graph has more than one non-empty connected component.

    Like has_enclosed_space_extended, this first builds the extended matrix:
    virtual crossing points and real nodes traversed by a full line become
    actual points, and each traversed/crossed edge is cut into its pieces.
    The original component-counting algorithm then runs on that extended
    graph, so two parts that only touch through a crossing point (not a
    shared endpoint in the raw 9x9 matrix) are correctly seen as connected.
    """

    adjacency = _validate_binary_symmetric_matrix(matrix)
    extended, _ = build_extended_matrix(adjacency)
    n_total = extended.shape[0]
    visited = [False] * n_total

    def depth_first_search(node: int) -> None:
        visited[node] = True
        for neighbor in range(n_total):
            if extended[node, neighbor] == 1 and not visited[neighbor]:
                depth_first_search(neighbor)

    active_nodes = [node for node in range(n_total) if np.any(extended[node] == 1)]

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


def has_visual_crossings_extended(matrix: np.ndarray) -> bool:
    """Return True if at least two drawn edges cross visually.

    A visual crossing extended happens when 4 segments visually intersect on the
    drawing. That includes:
    - a proper interior intersection between two edges,
    - a segment passing through a node used by another edge, and
    - two straight lines that cross at a node, even if each line is split into
      two edges by that node.
    """

    array = np.asarray(matrix).astype(int)

    # 1. TEST DU NOEUD CENTRAL (E)
    # Vérification des intersections directes et des configurations de lignes concourantes au centre
    # Évaluation de toutes les paires d'intersections possibles entre les 4 segments
    paires_crossings = [
        array[A, I] and array[C, G],  # AI et GC
        array[A, I] and array[H, B],  # AI et HB
        array[A, I] and array[D, F],  # AI et DF
        array[G, C] and array[H, B],  # GC et HB
        array[G, C] and array[D, F],  # GC et DF
        array[H, B] and array[D, F]   # HB et DF
    ]
    
    # Si au moins une de ces paires d'intersection est vraie, alors au moins 2 segments se croisent
    if any(paires_crossings):
        return True

    # Validation des sous-configurations spécifiques à 3 segments
    # (segments testés ici en demi-lignes : réalisés soit directement, soit
    # via la ligne pleine qui les contient, cf. _edge_drawn)
    complex_crossings_at_E = [
        _edge_drawn(array, I, E) and any([_edge_drawn(array, D, E), _edge_drawn(array, E, B), _edge_drawn(array, E, A)]) and array[C, G],
        _edge_drawn(array, G, E) and any([_edge_drawn(array, E, C), _edge_drawn(array, E, B), _edge_drawn(array, E, F)]) and array[A, I],
        _edge_drawn(array, A, E) and any([_edge_drawn(array, E, F), _edge_drawn(array, E, I), _edge_drawn(array, E, H)]) and array[C, G],
        _edge_drawn(array, C, E) and any([_edge_drawn(array, E, D), _edge_drawn(array, E, H), _edge_drawn(array, E, G)]) and array[A, I]
    ]
    if any(complex_crossings_at_E):
        return True

    # 2. CONFIGURATIONS SOURCE (cadran maître EHFI) + 3. TRANSPOSITION AUX 4
    # CADRANS VIA LES SYMÉTRIES : cf. SIMPLE_CROSSINGS, CONCURRENT_GROUPS et
    # QUADRANTS, réemployés aussi par has_visual_crossings_restricted,
    # has_disconnected_parts et build_extended_matrix (le point de croisement
    # n'est utile qu'à cette dernière, donc ignoré ici avec `_`).
    for mapping, _transform in QUADRANTS:
        # A. Test des croisements simples (N1, N2, N8, N3, N7)
        # (edge*_active tient compte des demi-segments réalisés par la
        # ligne pleine qui les contient, cf. _edge_drawn)
        for (u1, v1), (u2, v2), _point in SIMPLE_CROSSINGS:
            edge1_active = _edge_drawn(array, mapping[u1], mapping[v1])
            edge2_active = _edge_drawn(array, mapping[u2], mapping[v2])
            if edge1_active and edge2_active:
                return True

        # B. Test des groupes concourants (N4, N5, N6)
        for edge_list, _point in CONCURRENT_GROUPS:
            active_count = 0
            for u, v in edge_list:
                if _edge_drawn(array, mapping[u], mapping[v]):
                    active_count += 1
                    if active_count >= 2:
                        return True

    return False

def has_visual_crossings_restricted(matrix: np.ndarray) -> bool:
    """Return True if at least two drawn edges cross visually.

    A visual crossing "restricted" happens when two full straight lines intersect on the
    drawing. That includes:
    - a proper interior intersection between two edges,
    - a segment passing through a node used by another edge, and
    - two straight lines that cross at a node, even if each line is split into
      two edges by that node.
    """

    array = np.asarray(matrix).astype(int)

    # 1. TEST DU NOEUD CENTRAL (E)
    # Vérification des intersections directes et des configurations de lignes concourantes au centre
    # Évaluation de toutes les paires d'intersections possibles entre les 4 segments
    paires_crossings = [
        array[A, I] and array[C, G],  # AI et GC
        array[A, I] and array[H, B],  # AI et HB
        array[A, I] and array[D, F],  # AI et DF
        array[G, C] and array[H, B],  # GC et HB
        array[G, C] and array[D, F],  # GC et DF
        array[H, B] and array[D, F]   # HB et DF
    ]
    
    # Si au moins une de ces paires d'intersection est vraie, alors au moins 2 segments se croisent
    if any(paires_crossings):
        return True

    # 2. CONFIGURATIONS SOURCE (cadran maître EHFI) + 3. TRANSPOSITION AUX 4
    # CADRANS VIA LES SYMÉTRIES : cf. SIMPLE_CROSSINGS, CONCURRENT_GROUPS et
    # QUADRANTS, réemployés aussi par has_visual_crossings_extended,
    # has_disconnected_parts et build_extended_matrix (le point de croisement
    # n'est utile qu'à cette dernière, donc ignoré ici avec `_`).
    for mapping, _transform in QUADRANTS:
        # A. Test des croisements simples (N1, N2, N8, N3, N7)
        # (edge*_active tient compte des demi-segments réalisés par la
        # ligne pleine qui les contient, cf. _edge_drawn)
        for (u1, v1), (u2, v2), _point in SIMPLE_CROSSINGS:
            edge1_active = _edge_drawn(array, mapping[u1], mapping[v1])
            edge2_active = _edge_drawn(array, mapping[u2], mapping[v2])
            if edge1_active and edge2_active:
                return True

        # B. Test des groupes concourants (N4, N5, N6)
        for edge_list, _point in CONCURRENT_GROUPS:
            active_count = 0
            for u, v in edge_list:
                if _edge_drawn(array, mapping[u], mapping[v]):
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
        ("Visual crossings extended", has_visual_crossings_extended(matrix)),
        ("Visual crossings restricted", has_visual_crossings_restricted(matrix)),
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
    has_crossings_extended = has_visual_crossings_extended(sample)
    has_crossings_restricted = has_visual_crossings_restricted(sample)
    has_disconnected = has_disconnected_parts(sample)
    has_vert_symmetry = has_vertical_symmetry(sample)
    has_hor_symmetry = has_horizontal_symmetry(sample)
    has_eulerian = has_eulerian_trail(sample)
    print(f"Has diagonal connections: {has_diagonal}")
    print(f"Has enclosed space: {has_cycle}")
    print(f"Has visual crossings extended: {has_crossings_extended}")
    print(f"Has visual crossings restricted: {has_crossings_restricted}")
    print(f"Has disconnected parts: {has_disconnected}")
    print(f"Has vertical symmetry: {has_vert_symmetry}")
    print(f"Has horizontal symmetry: {has_hor_symmetry}")
    print(f"Has Eulerian trail: {has_eulerian}")