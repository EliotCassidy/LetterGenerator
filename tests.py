import numpy as np
from enclosed_space import has_enclosed_space_extended
from letter_graph import (
    has_disconnected_parts,
    has_visual_crossings_extended,
    has_visual_crossings_restricted,
)

# Initialisation des indices de la matrice (0 à 8)
A, B, C, D, E, F, G, H, I = range(9)

def create_empty_matrix() -> np.ndarray:
    """Initialise une matrice d'adjacence 9x9 remplie de zéros."""
    return np.zeros((9, 9), dtype=int)

def add_edge(matrix: np.ndarray, u: int, v: int) -> None:
    """Ajoute une arête non orientée entre deux nœuds d'une matrice symétrique."""
    matrix[u, v] = 1
    matrix[v, u] = 1

def run_visual_crossings_tests() -> None:
    """Exécute une suite de tests unitaires pour évaluer les fonctions de croisement et de cycle."""
    tests = {}

    # Cas 1 : Matrice vide
    m1 = create_empty_matrix()
    tests["Grille vide"] = {
        "matrix": m1,
        "expected_strict": False,
        "expected_extended": False,
        "expected_enclosed": False,
        "expected_disconnected": False
    }

    # Cas 2 : Intersection orthogonale au centre (Nœud E)
    m2 = create_empty_matrix()
    add_edge(m2, A, I)
    add_edge(m2, C, G)
    tests["Intersection en X (centre E)"] = {
        "matrix": m2,
        "expected_strict": True,
        "expected_extended": True,
        "expected_enclosed": False,
        "expected_disconnected": False
    }

    # Cas 3 : Intersection simple dans le cadran EHFI (Nœud N1)
    m3 = create_empty_matrix()
    add_edge(m3, I, E)
    add_edge(m3, H, F)
    tests["Intersection cadran Bas-Droite (N1)"] = {
        "matrix": m3,
        "expected_strict": True,
        "expected_extended": True,
        "expected_enclosed": False,
        "expected_disconnected": False
    }

    # Cas 4 : Intersection transposée dans le cadran ABED
    m4 = create_empty_matrix()
    add_edge(m4, A, E)
    add_edge(m4, B, D)
    tests["Intersection cadran Haut-Gauche (transposition)"] = {
        "matrix": m4,
        "expected_strict": True,
        "expected_extended": True,
        "expected_enclosed": False,
        "expected_disconnected": False
    }

    # Cas 5 : Alignements parallèles distants
    m5 = create_empty_matrix()
    add_edge(m5, A, G)
    add_edge(m5, B, H)
    tests["Lignes verticales parallèles"] = {
        "matrix": m5,
        "expected_strict": False,
        "expected_extended": False,
        "expected_enclosed": False,
        "expected_disconnected": True
    }

    # Cas 6 : Arêtes consécutives partageant une extrémité commune
    m6 = create_empty_matrix()
    add_edge(m6, A, B)
    add_edge(m6, B, C)
    tests["Lignes connectées sur un nœud (A-B-C)"] = {
        "matrix": m6,
        "expected_strict": False,
        "expected_extended": False,
        "expected_enclosed": False,
        "expected_disconnected": False
    }

    # Cas 7 : Condition minimale du groupe concourant N4
    m7 = create_empty_matrix()
    add_edge(m7, E, F)
    add_edge(m7, H, C)
    tests["Groupe concourant N4 (2 arêtes sur 3)"] = {
        "matrix": m7,
        "expected_strict": True,
        "expected_extended": True,
        "expected_enclosed": False,
        "expected_disconnected": False
    }

    # Cas 8 : Intersection traversante AF et GB
    m8 = create_empty_matrix()
    add_edge(m8, A, F)
    add_edge(m8, G, B)
    tests["Intersection traversante AF et GB"] = {
        "matrix": m8,
        "expected_strict": True,
        "expected_extended": True,
        "expected_enclosed": False,
        "expected_disconnected": False
    }

    # Cas 9 : Intersection traversante AF et DB
    m9 = create_empty_matrix()
    add_edge(m9, A, F)
    add_edge(m9, D, B)
    tests["Intersection traversante AF et DB"] = {
        "matrix": m9,
        "expected_strict": True,
        "expected_extended": True,
        "expected_enclosed": False,
        "expected_disconnected": False
    }

    # Cas 10 : AC, AE, EB -> Pas de croisement, mais forme fermée
    m10 = create_empty_matrix()
    add_edge(m10, A, C)
    add_edge(m10, A, E)
    add_edge(m10, E, B)
    tests["Triangle partiel (AC, AE, EB)"] = {
        "matrix": m10,
        "expected_strict": False,
        "expected_extended": False,
        "expected_enclosed": True,
        "expected_disconnected": False
    }

    # Cas 11 : AH, GE, DE -> Croisement et forme fermée
    m11 = create_empty_matrix()
    add_edge(m11, A, H)
    add_edge(m11, G, E)
    add_edge(m11, D, E)
    tests["Croisement + Forme fermée (AH, GE, DE)"] = {
        "matrix": m11,
        "expected_strict": True,
        "expected_extended": True,
        "expected_enclosed": True,
        "expected_disconnected": False
    }

    # Cas 12 : CH, DF -> Croisement mais pas de forme fermée
    m12 = create_empty_matrix()
    add_edge(m12, C, H)
    add_edge(m12, D, F)
    tests["Croisement sans forme fermée (CH, DF)"] = {
        "matrix": m12,
        "expected_strict": True,
        "expected_extended": True,
        "expected_enclosed": False,
        "expected_disconnected": False
    }

    # Cas 13 : AI, GE, EF -> Pas de forme fermée, croisement étendu mais pas de croisement strict
    m13 = create_empty_matrix()
    add_edge(m13, A, I)
    add_edge(m13, G, E)
    add_edge(m13, E, F)
    tests["Changement de direction (AI, GE, EF)"] = {
        "matrix": m13,
        "expected_strict": False,
        "expected_extended": True,
        "expected_enclosed": False,
        "expected_disconnected": False
    }

    # Cas 14 : AB, BE, EF, CH -> Pas de forme fermée, croisement étendu et
    # restreint, pas de disconnected parts
    m14 = create_empty_matrix()
    add_edge(m14, A, B)
    add_edge(m14, B, E)
    add_edge(m14, E, F)
    add_edge(m14, C, H)
    tests["Chemin + croisement (AB, BE, EF, CH)"] = {
        "matrix": m14,
        "expected_strict": True,
        "expected_extended": True,
        "expected_enclosed": False,
        "expected_disconnected": False
    }

    # Exécution et évaluation : chaque règle est vérifiée pour chaque cas.
    print("--- EVALUATION DES REGLES : CROSSINGS (STRICT/EXTENDED), ENCLOSED, DISCONNECTED ---")

    checks = [
        ("expected_strict", has_visual_crossings_restricted, "Strict"),
        ("expected_extended", has_visual_crossings_extended, "Extended"),
        ("expected_enclosed", has_enclosed_space_extended, "Enclosed"),
        ("expected_disconnected", has_disconnected_parts, "Disconnected"),
    ]

    success_count = 0
    total_count = 0

    for label, data in tests.items():
        for expected_key, func, name in checks:
            actual = func(data["matrix"])
            expected = data[expected_key]
            total_count += 1

            if actual == expected:
                print(f"OK   : [{label}] {name}")
                success_count += 1
            else:
                print(f"FAIL : [{label}] {name} -> Attendu : {expected} | Obtenu : {actual}")

    print("-" * 50)
    print(f"Bilan : {success_count}/{total_count} cas validés.")
    print("-" * 50)

if __name__ == "__main__":
    run_visual_crossings_tests()