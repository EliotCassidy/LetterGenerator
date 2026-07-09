"""Détection de forme fermée avec noeuds virtuels — version TABLE (énumération).
 
Même pipeline que test_enclosed_extended.py, mais la PARTIE 1 (identification
des points de croisement) réutilise la logique d'énumération de
has_visual_crossings_extended : configurations du cadran maître EHFI
(N1-N8) transposées aux 4 cadrans par symétrie, au lieu d'un calcul
géométrique d'intersections.
 
Deux adaptations par rapport à la fonction booléenne d'origine :
 
  1. Chaque cas de la table porte désormais la COORDONNÉE de son croisement
     (dans le cadran maître) ; les 4 mappings d'indices sont associés à la
     transformation géométrique correspondante, appliquée à la coordonnée.
  2. Une seconde table énumérée liste les 8 arêtes "pleines" et le noeud
     réel qu'elles enjambent (A-C passe par B, A-I par E, ...). Elle
     remplace la section "croisements au noeud central E" de la fonction
     d'origine : couper les lignes pleines au noeud traversé connecte
     automatiquement tout ce qui s'y croise, sans noeud virtuel.
 
Les coordonnées sont des fractions exactes : indispensable pour que deux
cas détectant le même point produisent la même clé (un seul noeud virtuel).
 
Structure :
    PARTIE 1 — Identification des points de croisement (TABLES)
    PARTIE 2 — Transposition vers la matrice étendue NxN
    PARTIE 3 — DFS de détection de cycle
    PARTIE 4 — Fonction publique
    Zone de tests + VALIDATION CROISÉE contre la version géométrique.
"""
 
from __future__ import annotations
 
from collections import defaultdict
from fractions import Fraction
 
import numpy as np
 

# ===========================================================================
# NOTATIONS COMMUNES (identiques à letter_graph.py)
# ===========================================================================

LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
POSITIONS = {
    "A": (0, 2), "B": (1, 2), "C": (2, 2),
    "D": (0, 1), "E": (1, 1), "F": (2, 1),
    "G": (0, 0), "H": (1, 0), "I": (2, 0),
}

# Coordonnées exactes des 9 noeuds réels, indexées par 0-8.
# Une des nécessités est d'identifier de manière unique les noeuds virtuels pour former la bonne matrice pour le dfs.
NODE_COORDS = [
    (Fraction(POSITIONS[label][0]), Fraction(POSITIONS[label][1]))
    for label in LABELS
]

A, B, C, D, E, F, G, H, I = range(9)

# Vérifie que la matrice donnée en entrée correspond bien au format attendu pour une GRID
def _validate_binary_symmetric_matrix(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix)
    if array.shape != (9, 9):
        raise ValueError(f"Expected a 9x9 matrix, got {array.shape!r}.")
    if not np.array_equal(array, array.T):
        raise ValueError("Matrix must be symmetric.")
    if not np.all(np.isin(array, [0, 1])):
        raise ValueError("Matrix must be binary (only 0 and 1 values).")
    return array.astype(int, copy=False)

# Extrait la liste unique des arêtes (u,v) avec u<v.
def _canonical(u: int, v: int) -> tuple[int, int]:
    """Forme canonique (u < v) d'une arête."""
    return (u, v) if u < v else (v, u)

# 
# ===========================================================================
# PARTIE 1 — IDENTIFICATION DES POINTS DE CROISEMENT (TABLES ÉNUMÉRÉES)
# ===========================================================================
 
# ---------------------------------------------------------------------------
# TABLE 1 : arêtes "pleines" et le noeud réel qu'elles enjambent.
# Toujours coupées à ce noeud si elles sont actives (que le noeud soit
# utilisé par une autre arête ou non : la coupe est alors sans effet sur
# le résultat, mais elle rend le graphe fidèle au dessin).
# Cette table remplace la section "TEST DU NOEUD CENTRAL (E)" de
# has_visual_crossings_extended pour la construction de la matrice.
# ---------------------------------------------------------------------------

# Détermine toutes les arêtes qui enjambent un noeud reél.
TRAVERSED_NODES: dict[tuple[int, int], int] = {
    _canonical(A, C): B,   # horizontale haute pleine
    _canonical(D, F): E,   # horizontale médiane pleine
    _canonical(G, I): H,   # horizontale basse pleine
    _canonical(A, G): D,   # verticale gauche pleine
    _canonical(B, H): E,   # verticale médiane pleine
    _canonical(C, I): F,   # verticale droite pleine
    _canonical(A, I): E,   # diagonale principale pleine
    _canonical(C, G): E,   # diagonale secondaire pleine
}
 
# ---------------------------------------------------------------------------
# TABLE 1bis : containment. Chaque demi-segment d'une ligne pleine -> la
# ligne pleine qui le contient. Dérivée automatiquement de TRAVERSED_NODES.
# Nécessaire car un cas de la table 2 qui attend le demi-segment (E, F)
# est AUSSI réalisé si la matrice encode la ligne pleine (D, F) : le trait
# dessiné passe par le même point de croisement. Pour autant la fonction ne le repérerait pas pour l'instant.
# ---------------------------------------------------------------------------

HALF_TO_FULL: dict[tuple[int, int], tuple[int, int]] = {}
for _full, _mid in TRAVERSED_NODES.items():
    _u, _v = _full
    HALF_TO_FULL[_canonical(_u, _mid)] = _full
    HALF_TO_FULL[_canonical(_mid, _v)] = _full
 
 # Permet ainsi de considérer les segments réellement visualisés plutôt que les traits dans la représentation matricielle. Elle renvoie les traits réels qui caractérise la grille (ainsi si DF est présent, ce sera DE et EF)
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
# TABLE 2 : configurations du CADRAN MAÎTRE EHFI (bas-droite), reprises de
# has_visual_crossings_extended. Le point de croisement de chaque cas n'est
# plus recopié à la main : il est calculé géométriquement (intersection de
# droites, en fractions exactes) à partir des coordonnées des noeuds réels
# que les arêtes relient.
# ---------------------------------------------------------------------------

Point = tuple[Fraction, Fraction]

# Calcule les coordonnées des points d'intersection des droites (p1p2) et (p3p4), en fractions exactes (permet d'identifier les points de croisement après)
def _line_intersection(p1: Point, p2: Point, p3: Point, p4: Point) -> Point:
    """Point d'intersection des droites (p1p2) et (p3p4), en fractions exactes. 
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


# Croisements simples : une paire d'arêtes, dont on calcule le point de
# croisement à partir de leurs extrémités.
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

# Groupes concourants ("2 parmi 3" : les trois arêtes passent par le même
# point ; toute paire active y crée un croisement). Le point est calculé à
# partir des 2 premières arêtes, puis on vérifie que la 3ème y passe aussi.
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
 
# ---------------------------------------------------------------------------
# TABLE 3 : les 4 cadrans. Chaque mapping d'indices (repris tel quel de
# has_visual_crossings_extended) est associé à la transformation
# géométrique qui lui correspond, appliquée aux coordonnées des points.
# La grille s'étend de 0 à 2 : miroir gauche/droite <=> x -> 2 - x,
# miroir haut/bas <=> y -> 2 - y.
# ---------------------------------------------------------------------------
QUADRANTS = [
    # 1. Bas-Droite (EHFI) -> Identité
    ({A: A, B: B, C: C, D: D, E: E, F: F, G: G, H: H, I: I},
     lambda p: p),
    # 2. Bas-Gauche -> Inversion Gauche/Droite
    ({C: A, B: B, A: C, F: D, E: E, D: F, I: G, H: H, G: I},
     lambda p: (2 - p[0], p[1])),
    # 3. Haut-Droite -> Inversion Haut/Bas
    ({G: A, H: B, I: C, D: D, E: E, F: F, A: G, B: H, C: I},
     lambda p: (p[0], 2 - p[1])),
    # 4. Haut-Gauche -> Double Inversion
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
        # trait(s) réellement dessiné(s). On évalue ensuite chaque crossing points en vérifiant si les segments sont présents.
        for (u1, v1), (u2, v2), point in SIMPLE_CROSSINGS:
            drawn1 = _realizers(array, _canonical(mapping[u1], mapping[v1])) #drawn1 et drawns2 sont les traits réellement dessinés qui matérialisent le segment abstrait
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
 
# Permet d'obtenir la liste des points de croisement réels (y compris hors noeuds) (hors noeuds réels) et les arêtes qui s'y croisent.


def find_virtual_nodes(matrix: np.ndarray) -> list[tuple[tuple, list]]:
    """Version 'rapport' pour inspection : [(point, [arêtes]), ...] trié."""
    return sorted(
        (point, sorted(edges))
        for point, edges in find_crossing_points(matrix).items()
    )
# Renvoie la liste des points de croisement de notre matrice.
 
 
# ===========================================================================
# PARTIE 2 — TRANSPOSITION VERS LA MATRICE ÉTENDUE
# ===========================================================================
# Identique à la version géométrique : matrice NxN vide, chaque arête
# d'origine remplacée par la chaîne de ses morceaux (coupes triées le long
# de l'arête). L'arête directe coupée n'est jamais écrite.
# ---------------------------------------------------------------------------
 
def build_extended_matrix(matrix: np.ndarray):
    """Retourne (matrice étendue NxN, dictionnaire coordonnée -> indice)."""
    array = _validate_binary_symmetric_matrix(matrix)
    crossing_points = find_crossing_points(array)
 
    # Va enregistrer les différents points de croisement pour chaque arêtes.
    cut_points: dict[tuple[int, int], set] = defaultdict(set)
 
    #   Table 1 : arêtes pleines coupées au noeud réel enjambé.
    for edge, node in TRAVERSED_NODES.items():
        if array[edge] == 1:
            cut_points[edge].add(NODE_COORDS[node])
 
    #   Table 2 (via find_crossing_points) : croisements hors noeuds.
    for point, edges in crossing_points.items():
        for edge in edges:
            cut_points[edge].add(point)
 
    # Numérotation : Crée un indice pour chaque point de la nouvelle grid. Les 9 noeuds de départ conservent leur indice (0-8),
    # les points de croisement sont ajoutés à la suite (9, 10, ...). Permet notamment de ne pas réajouter le noeud E avec un nouvel indice dans le cas où ce dernier serait victime d'un croisement.

    coord_to_index = {coord: index for index, coord in enumerate(NODE_COORDS)}
    for points in cut_points.values():
        for p in sorted(points):
            coord_to_index.setdefault(p, len(coord_to_index))
 
 # Crée la nouvelle matrice étendue à la bonne taille NxN
    n_total = len(coord_to_index)
    extended = np.zeros((n_total, n_total), dtype=int)
 
 
    for u in range(9): #la boucle parcourt la matrice d'origine est regarde les arêtes existantes.
        for v in range(u + 1, 9):
            if array[u, v] != 1:
                continue
            a = NODE_COORDS[u]
 
            def distance_along_edge(p): #Calcule la distance d'une potentielle coupe p au point u de l'arête pour pouvoir ensuite ordonner les coupes dans le cas où il y en aurait plusieurs sur la même arête.
                return (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2
 
            interior = sorted(cut_points.get((u, v), ()),
                              key=distance_along_edge) # Ordonne les points de coupe le long de l'arête considérée.
            chain = [u] + [coord_to_index[p] for p in interior] + [v] # Crée la chaîne de points (u, p1, p2, ..., v) pour l'arête (u,v) et ses coupes.
            for node_1, node_2 in zip(chain, chain[1:]): # Ajoute les arêtes entre chaque paire de points consécutifs dans la chaîne à la matrice étendue.
                extended[node_1, node_2] = extended[node_2, node_1] = 1
 
    return extended, coord_to_index
 
 # On obtient la nouvelle matrice extended pour le dfs.
 
# ===========================================================================
# PARTIE 3 — DFS DE DÉTECTION DE CYCLE
# ===========================================================================
 
def _has_cycle(adjacency: np.ndarray) -> bool:
    """True si le graphe non orienté (matrice NxN) contient un cycle."""
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
 
 
# ===========================================================================
# PARTIE 4 — FONCTION FINALE
# ===========================================================================
 
def has_enclosed_space_extended(matrix: np.ndarray) -> bool:
    """True si le dessin contient une forme fermée, au sens visuel."""
    extended, _ = build_extended_matrix(matrix)
    return _has_cycle(extended)
 
 
