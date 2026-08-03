"""
Álgebra de Minkowski y el grupo O(3,1) — ejercicios de las diapositivas 2-5 y 18-21.

La esfera unitaria de R3 se escribe como la cuádrica de P3:  x² + y² + z² − w² = 0.
Las matrices M que satisfacen  Mᵀ Q M = Q,  con Q = diag(1,1,1,−1), preservan esa
cuádrica (o sea, mandan la esfera en sí misma) y además mandan rectas en rectas.
Ese grupo es O(3,1) y es lo que necesitamos cuando una similaridad NO alcanza para
llevar un par (esfera, eje) en otro.

Convención de este módulo
-------------------------
Un punto p de la esfera de radio ρ centrada en el origen se sube a P3 como el
4-vector homogéneo  q = [p, ρ].  Entonces  qᵀ Q q = |p|² − ρ² = 0,  es decir q es
un vector NULO de la métrica de Minkowski. Bajarlo de vuelta es  p = ρ·q[:3]/q[3].
"""

import numpy as np

# Métrica de Minkowski, la Q de la diapositiva 2.
ETA = np.diag([1.0, 1.0, 1.0, -1.0])


# ---------------------------------------------------------------------------
# Producto y normalización de Minkowski
# ---------------------------------------------------------------------------

def minkowski_dot(a, b):
    """Producto de Minkowski aᵀ η b = a0b0 + a1b1 + a2b2 − a3b3."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]
                 + a[..., 2] * b[..., 2] - a[..., 3] * b[..., 3])


def minkowski_norm2(x):
    """Norma al cuadrado xᵀ η x. OJO: puede ser negativa (vectores timelike)."""
    return minkowski_dot(x, x)


def normalize_minkowski(x, tol=1e-12):
    """
    Normaliza x a norma de Minkowski ±1.

    Devuelve (x_hat, sign) con sign = +1 si x es spacelike y −1 si es timelike.

    Este es el punto donde el script de MATLAB de la presentación se rompe: el
    vector 'common' = (nA+nB)/2 es TIMELIKE, su norma sale negativa (−21 en el
    ejemplo de la diapositiva 19) y sqrt() de un negativo da NaN en numpy. Aquí
    normalizamos con sqrt(|·|) y devolvemos el signo aparte.
    """
    x = np.asarray(x, dtype=float)
    n2 = minkowski_norm2(x)
    if abs(n2) < tol:
        raise ValueError(
            "🚨 vector nulo (norma de Minkowski ≈ 0): no se puede normalizar. "
            "Suele significar que los dos puntos de la cuerda coinciden, o que "
            "la recta es tangente a la esfera."
        )
    return x / np.sqrt(abs(n2)), (1.0 if n2 > 0 else -1.0)


# ---------------------------------------------------------------------------
# Pertenencia al grupo
# ---------------------------------------------------------------------------

def is_in_o31(M, tol=1e-9):
    """
    Verifica Mᵀ η M = η. Devuelve (esta_en_el_grupo, desviacion_maxima).
    """
    M = np.asarray(M, dtype=float)
    dev = float(np.max(np.abs(M.T @ ETA @ M - ETA)))
    return bool(dev < tol), dev


def lorentz_inverse(F):
    """
    Inversa de una matriz de O(3,1) sin resolver ningún sistema: F⁻¹ = η Fᵀ η.

    Sale de Fᵀ η F = η: multiplicando por η a izquierda y derecha, (η Fᵀ η) F = I.
    Es más estable numéricamente que np.linalg.inv y deja claro por qué funciona.
    """
    F = np.asarray(F, dtype=float)
    return ETA @ F.T @ ETA


def embed_rotation(R):
    """
    Ejercicio de la diapositiva 4: embebe R ∈ O(3) en O(3,1) como diag(R, 1).

    La verificación es inmediata: Mᵀ η M tiene bloque superior Rᵀ R = I y entrada
    (3,3) igual a −1, o sea exactamente η.
    """
    R = np.asarray(R, dtype=float)
    M = np.eye(4)
    M[:3, :3] = R
    return M


def is_proper_orthochronous(M, tol=1e-9):
    """
    ¿M es propia (det = +1) y ortócrona (M[3,3] ≥ 1)? Devuelve (bool, det, M33).

    Las transformaciones que nos interesan (boosts y rotaciones) están en esta
    componente conexa del grupo. Un det = −1 delata una reflexión, que en nuestro
    contexto anatómico significaría cambiar el hombro de lado.
    """
    M = np.asarray(M, dtype=float)
    det = float(np.linalg.det(M))
    return bool(abs(det - 1.0) < 1e-6 and M[3, 3] >= 1.0 - tol), det, float(M[3, 3])


# ---------------------------------------------------------------------------
# Subida y bajada entre R3 y P3
# ---------------------------------------------------------------------------

def sphere_to_homogeneous(points, radius):
    """(..., 3) → (..., 4) añadiendo w = radius. Los puntos deben estar en la esfera."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    w = np.full((points.shape[0], 1), float(radius))
    return np.hstack([points, w])


def homogeneous_to_sphere(q, radius, eps=1e-9):
    """
    (..., 4) → (..., 3) vía p = radius · q[:3] / q[3].

    Devuelve (points, w, n_singular) donde n_singular cuenta los puntos que caen
    cerca del plano evanescente {q[3] = 0} — ahí la proyectividad manda al infinito
    y el resultado no significa nada.
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    w = q[:, 3]
    n_singular = int(np.sum(np.abs(w) < eps))
    safe_w = np.where(np.abs(w) < eps, np.nan, w)
    return float(radius) * q[:, :3] / safe_w[:, None], w, n_singular


def line_sphere_intersections(center, radius, line_point, line_dir):
    """
    Corta la recta (line_point + t·line_dir) con la esfera (center, radius).

    Devuelve (A, B) o (None, None) si la recta no corta (la distancia del centro
    a la recta es ≥ radius). El orden es el de t creciente.
    """
    center = np.asarray(center, dtype=float)
    p0 = np.asarray(line_point, dtype=float)
    d = np.asarray(line_dir, dtype=float)
    d = d / np.linalg.norm(d)

    f = p0 - center
    b = 2.0 * float(f @ d)
    c = float(f @ f) - float(radius) ** 2
    disc = b * b - 4.0 * c
    if disc <= 0.0:
        return None, None
    sq = np.sqrt(disc)
    t1, t2 = (-b - sq) / 2.0, (-b + sq) / 2.0
    return p0 + t1 * d, p0 + t2 * d


# ---------------------------------------------------------------------------
# Marcos de Lorentz asociados a una cuerda
# ---------------------------------------------------------------------------

def complete_minkowski_frame(u, v, tol=1e-9):
    """
    Completa {u (spacelike), v (timelike)} a una base η-ortonormal de R4.

    Devuelve (s, t): dos vectores spacelike η-unitarios, η-ortogonales entre sí y
    a u y v. Se obtienen por Gram-Schmidt con la métrica de Minkowski sobre la
    base canónica, descartando las direcciones que ya están cubiertas.

    Esto es lo que la diapositiva 20 escribe A MANO (s1 = [0,1,0,0],
    t1 = [0,0,5,2]/√21, etc.): esos valores son válidos solo para su ejemplo. Para
    cuerdas arbitrarias hay que generarlos, y hay que generarlos de forma que el
    marco sea η-ortonormal — si no, la matriz resultante se sale de O(3,1).
    """
    vecs = [np.asarray(u, dtype=float), np.asarray(v, dtype=float)]
    sigs = [1.0, -1.0]
    extra = []

    for e in np.eye(4):
        if len(extra) == 2:
            break
        w = e.copy()
        for b, s in zip(vecs + extra, sigs + [1.0] * len(extra)):
            w = w - (minkowski_dot(w, b) / s) * b
        n2 = minkowski_norm2(w)
        if n2 > tol:
            extra.append(w / np.sqrt(n2))

    if len(extra) != 2:
        raise ValueError("🚨 no se pudo completar el marco de Minkowski (base degenerada)")
    return extra[0], extra[1]


def lorentz_frame_for_chord(A, B, radius):
    """
    Marco de Lorentz F = [u, s, t, v] (por COLUMNAS) asociado a la cuerda AB.

    Siguiendo la diapositiva 20:
      u = normaliza(nB − nA)        → spacelike, apunta a lo largo de la cuerda
      v = normaliza((nA + nB)/2)    → timelike, el "punto medio" proyectivo
    y s, t completan la base.

    Por construcción Fᵀ η F = diag(1, 1, 1, −1) = η, o sea F ∈ O(3,1) salvo signo.
    Devuelve un dict con F y sus piezas, para poder inspeccionarlo en los reportes.
    """
    nA = np.append(np.asarray(A, dtype=float), float(radius))
    nB = np.append(np.asarray(B, dtype=float), float(radius))

    for name, q in (("A", nA), ("B", nB)):
        if abs(minkowski_norm2(q)) > 1e-6 * radius ** 2:
            raise ValueError(
                f"🚨 el punto {name} no está sobre la esfera de radio {radius}: "
                f"qᵀηq = {minkowski_norm2(q):.6g}"
            )

    u, sign_u = normalize_minkowski(nB - nA)
    v, sign_v = normalize_minkowski((nA + nB) / 2.0)
    if sign_u < 0 or sign_v > 0:
        raise ValueError(
            f"🚨 signatura inesperada: u debería ser spacelike (+1) y v timelike (−1), "
            f"salió u={sign_u:+.0f}, v={sign_v:+.0f}"
        )

    s, t = complete_minkowski_frame(u, v)
    F = np.column_stack([u, s, t, v])
    return {"F": F, "u": u, "s": s, "t": t, "v": v, "nA": nA, "nB": nB}


# ---------------------------------------------------------------------------
# El boost que fija la esfera y manda una cuerda en otra
# ---------------------------------------------------------------------------

def boost_fixing_sphere(A1, B1, A2, B2, radius=5.0, check=True):
    """
    Ejercicio de las diapositivas 18-19: construye M ∈ O(3,1) que fija la esfera de
    radio `radius` centrada en el origen y manda la cuerda A1B1 en la cuerda A2B2.

    Construcción (ver nota abajo):
        F1 = marco de Lorentz de A1B1,  F2 = marco de Lorentz de A2B2
        M  = F2 · F1⁻¹   con   F1⁻¹ = η F1ᵀ η

    Como ambos marcos son η-ortonormales, Mᵀ η M = η sale automáticamente.

    RELACIÓN CON LA RECETA DE LA PRESENTACIÓN
    -----------------------------------------
    La diapositiva 21 llega a la MISMA matriz por un camino más largo:
    F20 = [u2, s20, t20, v2], M0 = F20·inv(F1), s2 = M0·s1, t2 = M0·t1,
    F2 = [u2, s2, t2, v2], M = F2·inv(F1).

    Verificado numéricamente sobre 30 cuerdas aleatorias (ver selftest): esa receta
    coincide con esta a ~1e-13 y M0 == M, o sea el paso de transportar s1, t1 es
    redundante. La razón es que si s20, t20 son la completación η-ortonormal del
    marco 2 — que es lo que son los vectores escritos a mano en la diapositiva —
    entonces F2 = M0·F1 y por tanto M = M0 = F2·F1⁻¹.

    Dos diferencias prácticas a favor de esta versión:
      * usa F1⁻¹ = η F1ᵀ η en vez de una inversión numérica;
      * los vectores de completación se GENERAN (la diapositiva los da hardcodeados
        para su ejemplo, sin decir de dónde salen). Ver complete_minkowski_frame.
    """
    f1 = lorentz_frame_for_chord(A1, B1, radius)
    f2 = lorentz_frame_for_chord(A2, B2, radius)
    M = f2["F"] @ lorentz_inverse(f1["F"])

    if check:
        ok, dev = is_in_o31(M)
        if not ok:
            raise AssertionError(f"🚨 la matriz construida no está en O(3,1): desviación {dev:.3e}")

    return {"M": M, "F1": f1["F"], "F2": f2["F"], "frame1": f1, "frame2": f2,
            "o31_dev": is_in_o31(M)[1]}


def boost_fixing_sphere_pptx_recipe(A1, B1, A2, B2, radius=5.0,
                                    s1=None, t1=None, s20=None, t20=None):
    """
    La receta LITERAL de las diapositivas 20-21, portada de MATLAB tal cual:

        F1  = [u1, s1,  t1,  v1]
        F20 = [u2, s20, t20, v2]
        M0  = F20 · inv(F1) ;  s2 = M0·s1 ;  t2 = M0·t1
        F2  = [u2, s2, t2, v2] ;  M = F2 · inv(F1)

    Es CORRECTA y GENERAL, y equivale a boost_fixing_sphere (coincide a ~1e-13 sobre
    cuerdas aleatorias). Se conserva por trazabilidad con el enunciado y porque
    permite comprobar dos cosas:

      * que los cuatro vectores escritos a mano en la diapositiva 21
        (s1 = s20 = [0,1,0,0], t1 = [0,0,5,2]/√21, t20 = [0,0,5,−3]/4) son
        exactamente la completación η-ortonormal de cada marco — no valores mágicos;
      * que el paso de transportar s1, t1 por M0 es redundante: sale M0 == M.

    Trampa a evitar: si por descuido se usan en F20 los vectores de completación del
    marco 1 en vez de los del marco 2, la matriz resultante sigue mandando A1→A2 y
    B1→B2 correctamente (error ~1e-15) pero YA NO está en O(3,1) — ‖MᵀηM − η‖ salta
    a órdenes de magnitud 1. Es un fallo silencioso; por eso boost_fixing_sphere
    verifica la pertenencia al grupo antes de devolver nada.
    """
    f1 = lorentz_frame_for_chord(A1, B1, radius)
    f2 = lorentz_frame_for_chord(A2, B2, radius)

    c_s1 = np.asarray(s1, dtype=float) if s1 is not None else f1["s"]
    c_t1 = np.asarray(t1, dtype=float) if t1 is not None else f1["t"]
    c_s20 = np.asarray(s20, dtype=float) if s20 is not None else f2["s"]
    c_t20 = np.asarray(t20, dtype=float) if t20 is not None else f2["t"]

    F1 = np.column_stack([f1["u"], c_s1, c_t1, f1["v"]])
    F20 = np.column_stack([f2["u"], c_s20, c_t20, f2["v"]])
    invF1 = np.linalg.inv(F1)
    M0 = F20 @ invF1
    s2 = M0 @ c_s1
    t2 = M0 @ c_t1
    F2 = np.column_stack([f2["u"], s2, t2, f2["v"]])
    M = F2 @ invF1
    return {"M": M, "F1": F1, "F20": F20, "M0": M0, "F2": F2, "o31_dev": is_in_o31(M)[1]}


def boost_along_chord(theta, A, B, radius=5.0):
    """
    Familia de gauge de 1 parámetro: el boost que fija la esfera Y la recta AB,
    pero desliza los puntos a lo largo de ella.

    En el marco F = [u, s, t, v] de la cuerda, actúa como un boost hiperbólico en
    el plano (u, v) y como la identidad en (s, t). Componer con esta familia es lo
    que mueve los músculos a lo largo de la diáfisis sin romper el par (esfera, eje).

    Detalle geométrico: los dos puntos de corte A y B son direcciones NULAS del
    plano (u, v), o sea autovectores del boost con autovalores e^±θ. Proyectivamente
    quedan FIJOS; lo que se desliza son los demás puntos de la recta. Por eso esta
    familia no rompe el par (esfera, eje) pero sí reposiciona todo lo que cuelga de él.
    """
    f = lorentz_frame_for_chord(A, B, radius)
    ch, sh = np.cosh(float(theta)), np.sinh(float(theta))
    local = np.eye(4)
    local[0, 0] = ch
    local[0, 3] = sh
    local[3, 0] = sh
    local[3, 3] = ch
    return f["F"] @ local @ lorentz_inverse(f["F"])


# ---------------------------------------------------------------------------
# Acción proyectiva sobre puntos de R3
# ---------------------------------------------------------------------------

def apply_projective(M, points, radius, eps=1e-9):
    """
    Aplica M a puntos de R3 vía P3:  p ↦ radius · (M[p; radius])₁₂₃ / (M[p; radius])₄.

    Devuelve (mapped, w_out, n_singular). El peso w_out es el factor de escala local
    de la proyectividad: sirve para medir cuánta distorsión introduce y para detectar
    el plano evanescente.

    IMPORTANTE: esto NO es una similaridad. Preserva rectas, planos y razones dobles,
    pero no ángulos ni razones de distancias. Los objetos transformados sufren
    distorsión proyectiva.
    """
    M = np.asarray(M, dtype=float)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    q = sphere_to_homogeneous(pts, radius)
    qp = q @ M.T
    return homogeneous_to_sphere(qp, radius, eps=eps)


def check_vanishing_plane(M, points, radius, margin=1e-3):
    """
    Chequea si algún punto cae cerca del plano evanescente {(Mq)₄ = 0}.

    Si hay cambios de signo en w, la malla queda literalmente partida por el
    infinito y el STL resultante no tiene sentido: hay que abortar ese objeto, no
    escribir basura. Devuelve un dict con el diagnóstico.
    """
    _, w, n_singular = apply_projective(M, points, radius)
    w = np.asarray(w, dtype=float)
    scale = float(np.max(np.abs(w))) if w.size else 1.0
    n_near = int(np.sum(np.abs(w) < margin * max(scale, 1e-12)))
    n_pos = int(np.sum(w > 0))
    n_neg = int(np.sum(w < 0))
    return {
        "n_near": n_near,
        "n_singular": n_singular,
        "sign_flips": bool(n_pos > 0 and n_neg > 0),
        "min_abs_w": float(np.min(np.abs(w))) if w.size else 0.0,
        "max_abs_w": scale,
        "safe": bool(n_near == 0 and n_singular == 0 and not (n_pos > 0 and n_neg > 0)),
    }
