"""
El grupo de similaridades T(x) = (sR)x + b — ejercicios de las diapositivas 11-13, 17 y 24.

Una similaridad es dilatación + rotación (u ortogonal) + traslación. Manda esferas
en esferas y rectas en rectas, así que es el primer candidato natural para llevar el
par (esfera, eje) de un húmero al de otro.

El resultado central de estas diapositivas es que NO alcanza: una similaridad
multiplica todas las distancias por el mismo factor s, luego preserva la razón
R/d (radio de la esfera sobre distancia del centro al eje). Si los dos pares tienen
razones distintas, ninguna similaridad los relaciona. Ahí es donde entra O(3,1).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Utilidades de recta
# ---------------------------------------------------------------------------

def project_point_on_line(p, line_point, line_dir):
    """Proyección ortogonal de p sobre la recta (line_point + t·line_dir)."""
    p = np.asarray(p, dtype=float)
    p0 = np.asarray(line_point, dtype=float)
    d = np.asarray(line_dir, dtype=float)
    d = d / np.linalg.norm(d)
    return p0 + float((p - p0) @ d) * d


def point_line_distance(p, line_point, line_dir):
    """Distancia de p a la recta."""
    return float(np.linalg.norm(np.asarray(p, dtype=float)
                                - project_point_on_line(p, line_point, line_dir)))


def line_line_distance(p1, d1, p2, d2, tol=1e-12):
    """
    Distancia entre dos rectas en 3D (paralelas o alabeadas).

    Sirve para medir cuánto le falta a T(L1) para coincidir con L2: ese hueco es la
    manifestación numérica del ejercicio de la diapositiva 17b.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    d1 = np.asarray(d1, dtype=float) / np.linalg.norm(d1)
    d2 = np.asarray(d2, dtype=float) / np.linalg.norm(d2)
    n = np.cross(d1, d2)
    nn = np.linalg.norm(n)
    if nn < tol:                       # paralelas
        return point_line_distance(p2, p1, d1)
    return float(abs((p2 - p1) @ (n / nn)))


# ---------------------------------------------------------------------------
# El par (esfera, eje)
# ---------------------------------------------------------------------------

def make_pair(center, radius, axis_point, axis_dir, **extra):
    """
    Constructor canónico del par (esfera, eje) que usa todo el paquete.

    Normaliza la dirección del eje, calcula la excentricidad d = dist(centro, eje) y
    la razón R/d, que es EL invariante de la diapositiva 13.
    """
    center = np.asarray(center, dtype=float)
    axis_point = np.asarray(axis_point, dtype=float)
    axis_dir = np.asarray(axis_dir, dtype=float)
    norm = np.linalg.norm(axis_dir)
    if norm < 1e-12:
        raise ValueError("🚨 dirección del eje degenerada (norma ≈ 0)")
    axis_dir = axis_dir / norm

    radius = float(radius)
    d = point_line_distance(center, axis_point, axis_dir)
    warnings = list(extra.pop("warnings", []))
    if d >= radius:
        warnings.append(
            f"⚠️  el eje NO corta la esfera (d={d:.2f} mm ≥ R={radius:.2f} mm): "
            "el boost de O(3,1) supone una recta secante"
        )

    pair = {
        "center": center,
        "radius": radius,
        "axis_point": axis_point,
        "axis_dir": axis_dir,
        "d": d,
        "ratio": (radius / d) if d > 1e-12 else float("inf"),
        "warnings": warnings,
    }
    pair.update(extra)
    return pair


def pair_report(pair):
    """Resumen legible de un par, para print y para los .txt de salida."""
    lines = [
        f"  📍 centro esfera : [{pair['center'][0]:9.4f}, {pair['center'][1]:9.4f}, "
        f"{pair['center'][2]:9.4f}] mm",
        f"  ⚪ radio         : {pair['radius']:.4f} mm",
        f"  📏 origen eje    : [{pair['axis_point'][0]:9.4f}, {pair['axis_point'][1]:9.4f}, "
        f"{pair['axis_point'][2]:9.4f}] mm",
        f"  ➡️  dirección eje : [{pair['axis_dir'][0]:8.4f}, {pair['axis_dir'][1]:8.4f}, "
        f"{pair['axis_dir'][2]:8.4f}]",
        f"  📐 excentricidad : d = {pair['d']:.4f} mm   |   R/d = {pair['ratio']:.4f}"
        f"   |   d/R = {pair['d'] / pair['radius']:.4f}",
    ]
    if pair.get("source"):
        lines.insert(0, f"  🗂️  fuente        : {pair['source']}")
    for w in pair.get("warnings", []):
        lines.append(f"  {w}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Construcción de la similaridad — diapositivas 11 y 12
# ---------------------------------------------------------------------------

def frame_from_pair(pair, tol=1e-9):
    """
    Marco ortonormal M = [w, v, n] (por COLUMNAS) asociado al par, tal como lo
    define la diapositiva 11:

        w = unitario del centro C hacia su proyección ortogonal sobre el eje L
        v = dirección unitaria del eje L
        n = w × v   (completa la base a derechas)

    Si el centro está SOBRE el eje (d ≈ 0) el vector w es indeterminado; en ese caso
    se elige uno arbitrario ortogonal a v y se avisa.
    """
    C = pair["center"]
    v = pair["axis_dir"]
    foot = project_point_on_line(C, pair["axis_point"], v)
    w_raw = foot - C
    degenerate = bool(np.linalg.norm(w_raw) < tol)

    if degenerate:
        aux = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        w = np.cross(v, aux)
        w = w / np.linalg.norm(w)
    else:
        w = w_raw / np.linalg.norm(w_raw)

    n = np.cross(w, v)
    n_norm = np.linalg.norm(n)
    if n_norm < tol:
        raise ValueError("🚨 marco degenerado: w y v son colineales")
    n = n / n_norm

    return {"M": np.column_stack([w, v, n]), "w": w, "v": v, "n": n,
            "degenerate": degenerate}


def similarity_from_pairs(pair1, pair2):
    """
    Ejercicio de las diapositivas 11-12: construye T(x) = (sR)x + b que lleva el par 1
    en el par 2, con la receta de la presentación:

        s = R₂ / R₁          R = M₂ M₁ᵀ          b = C₂ − s R C₁

    Devuelve un dict con s, R, b, el determinante, la desviación de ortogonalidad, y
    — lo importante — el hueco `axis_gap` que queda entre T(L₁) y L₂. Ese hueco es
    cero si y solo si R₁/d₁ = R₂/d₂ (diapositiva 13).
    """
    f1 = frame_from_pair(pair1)
    f2 = frame_from_pair(pair2)

    s = pair2["radius"] / pair1["radius"]
    R = f2["M"] @ f1["M"].T
    b = pair2["center"] - s * (R @ pair1["center"])

    orth = check_orthogonality(R)

    # Imagen del eje 1 bajo T, y cuánto le falta para ser el eje 2.
    img_point = apply_similarity(pair1["axis_point"], s, R, b)[0]
    img_dir = R @ pair1["axis_dir"]
    parallel_dev = float(1.0 - abs(img_dir @ pair2["axis_dir"]))
    axis_gap = line_line_distance(img_point, img_dir, pair2["axis_point"], pair2["axis_dir"])

    return {
        "s": float(s),
        "R": R,
        "b": b,
        "det": orth["det"],
        "orth_dev": orth["max_dev"],
        "is_rotation": orth["is_rotation"],
        "image_axis_point": img_point,
        "image_axis_dir": img_dir,
        "parallel_dev": parallel_dev,
        "axis_gap": axis_gap,
        "ratio1": pair1["ratio"],
        "ratio2": pair2["ratio"],
        "d_image": s * pair1["d"],
        "d2": pair2["d"],
    }


def apply_similarity(points, s, R, b):
    """(sR)x + b, vectorizado sobre (..., 3)."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    return s * (pts @ np.asarray(R, dtype=float).T) + np.asarray(b, dtype=float)


def invert_similarity(s, R, b):
    """Inversa de T: x ↦ (1/s) Rᵀ (x − b)."""
    R = np.asarray(R, dtype=float)
    return 1.0 / s, R.T, -(R.T @ np.asarray(b, dtype=float)) / s


def check_orthogonality(R, tol=1e-9):
    """Devuelve {max_dev, det, is_rotation, is_reflection} para una matriz 3x3."""
    R = np.asarray(R, dtype=float)
    dev = float(np.max(np.abs(R.T @ R - np.eye(3))))
    det = float(np.linalg.det(R))
    return {
        "max_dev": dev,
        "det": det,
        "is_orthogonal": bool(dev < tol),
        "is_rotation": bool(dev < tol and abs(det - 1.0) < 1e-6),
        "is_reflection": bool(dev < tol and abs(det + 1.0) < 1e-6),
    }


# ---------------------------------------------------------------------------
# El invariante R/d — diapositivas 13 y 17b
# ---------------------------------------------------------------------------

def pair_similarity_invariant(pair1, pair2, rtol=1e-9):
    """
    Ejercicio de la diapositiva 13: compara R₁/d₁ contra R₂/d₂.

    Una similaridad escala TODAS las longitudes por el mismo s, así que manda la
    esfera de radio R₁ en una de radio s·R₁ y el eje a distancia d₁ en uno a
    distancia s·d₁. La razón R/d es por tanto un INVARIANTE. Si las dos razones no
    coinciden, no existe ninguna similaridad que lleve un par exactamente en el otro
    — que es justo el ejercicio de la diapositiva 17b.
    """
    r1, r2 = pair1["ratio"], pair2["ratio"]
    rel_diff = abs(r1 - r2) / max(abs(r1), 1e-12)
    s = pair2["radius"] / pair1["radius"]
    return {
        "ratio1": r1,
        "ratio2": r2,
        "rel_diff": float(rel_diff),
        "equal": bool(rel_diff < rtol),
        "d_image": float(s * pair1["d"]),
        "d2": float(pair2["d"]),
        "gap_mm": float(abs(s * pair1["d"] - pair2["d"])),
        "similarity_suffices": bool(rel_diff < rtol),
    }


def rotation_between_vectors(a, b, tol=1e-12):
    """Rotación de Rodrigues que lleva el unitario a en el unitario b."""
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    b = np.asarray(b, dtype=float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < tol:
        if c > 0:
            return np.eye(3)
        # Antiparalelos: media vuelta alrededor de cualquier eje ortogonal a a.
        aux = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, aux)
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def similarity_axis_parallel(pair1, pair2):
    """
    Ejercicio de la diapositiva 17a: SIEMPRE existe una similaridad T con
    T(S₁) = S₂ y T(L₁) paralela a L₂.

    La construcción es la demostración: se toma s = R₂/R₁, se elige R como la
    rotación de Rodrigues que lleva la dirección del eje 1 en la del eje 2, y se
    fija b = C₂ − sR C₁ para clavar el centro. Entonces T(S₁) = S₂ exactamente y
    T(L₁) ∥ L₂ por construcción, quede donde quede.

    Lo que NO se puede pedir es que además coincidan: el hueco residual vale
    |s·d₁ − d₂| y solo se anula si R₁/d₁ = R₂/d₂.
    """
    s = pair2["radius"] / pair1["radius"]
    R = rotation_between_vectors(pair1["axis_dir"], pair2["axis_dir"])
    b = pair2["center"] - s * (R @ pair1["center"])

    img_point = apply_similarity(pair1["axis_point"], s, R, b)[0]
    img_dir = R @ pair1["axis_dir"]
    return {
        "s": float(s),
        "R": R,
        "b": b,
        "parallel_dev": float(1.0 - abs(img_dir @ pair2["axis_dir"])),
        "center_error": float(np.linalg.norm(
            apply_similarity(pair1["center"], s, R, b)[0] - pair2["center"])),
        "distance_gap": float(abs(s * pair1["d"] - pair2["d"])),
        "image_axis_point": img_point,
        "image_axis_dir": img_dir,
    }


def build_non_similar_example():
    """
    Ejercicio de la diapositiva 24 (vale doble): dos pares esfera/recta SECANTES que
    ninguna similaridad relaciona.

        Par 1: C = (0,0,0), R = 5, recta por (3,0,0) con dirección e_z  → d₁ = 3, R/d = 5/3
        Par 2: C = (10,0,0), R = 5, recta por (11,0,0) con dirección e_y → d₂ = 1, R/d = 5

    Ambas rectas cortan su esfera (d < R en los dos casos), así que el ejemplo es
    admisible. Como 5/3 ≠ 5, por la diapositiva 13 no hay similaridad posible. Pero
    sí existe una que lleva S₁ en S₂ y L₁ en una PARALELA a L₂ (diapositiva 17a).

    Devuelve (pair1, pair2, invariante, similaridad_paralela).
    """
    pair1 = make_pair([0, 0, 0], 5.0, [3, 0, 0], [0, 0, 1], source="ejemplo sintético 1")
    pair2 = make_pair([10, 0, 0], 5.0, [11, 0, 0], [0, 1, 0], source="ejemplo sintético 2")
    return pair1, pair2, pair_similarity_invariant(pair1, pair2), \
        similarity_axis_parallel(pair1, pair2)
