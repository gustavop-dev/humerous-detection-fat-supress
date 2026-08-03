"""
Extracción del par (esfera, eje) del húmero — ejercicio de la diapositiva 25b.

Dos fuentes:
  * una malla .stl completa (el húmero standard),
  * los CSV de círculos por slice que ya produce humerus_boundary_analysis.py
    (el húmero del paciente, visto por MRI).

La esfera aproxima la superficie articular de la cabeza; el eje aproxima el eje del
brazo. La diapositiva 14 recuerda que el centro de la esfera es sustancialmente
EXCÉNTRICO respecto al eje (d ≈ 9 mm, d/R ≈ 0.4), y esa excentricidad es lo que
decide si una similaridad basta o hace falta O(3,1).

Nota importante sobre el eje en MRI: los volúmenes axiales de este proyecto son
cortos (~68 mm) y solo cubren el húmero proximal, así que los cortes que aquí se
llaman "diafisarios" son en realidad cuello quirúrgico / metáfisis. El eje sale
sesgado y d subestimado. Por eso todo estimador de eje devuelve además su
incertidumbre, y hay una curva de convergencia para ver si llegó a estabilizarse.
"""

import csv
import os

import numpy as np

from Geometry.similarity import make_pair, point_line_distance

# Criterios de calidad, fijados a priori (ver el plan). Un dataset que no los pasa
# se marca, no se arregla a mano.
MIN_ARTICULAR_CIRCLES = 4
MIN_SHAFT_CIRCLES = 5
SPHERE_RADIUS_RANGE_MM = (20.0, 28.0)
MAX_AXIS_RMS_MM = 2.5
MAX_AXIS_SIGMA_DEG = 5.0


# ---------------------------------------------------------------------------
# Primitivas
# ---------------------------------------------------------------------------

def principal_axis(points):
    """PCA. Devuelve (centroide, dirección principal unitaria, autovalores desc.)."""
    P = np.asarray(points, dtype=float)
    c = P.mean(axis=0)
    u, s, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[0] / np.linalg.norm(vt[0]), s ** 2


def line_fit_rms(points, line_point, line_dir):
    """RMS de la distancia de los puntos a la recta."""
    d = np.array([point_line_distance(p, line_point, line_dir) for p in np.asarray(points)])
    return float(np.sqrt(np.mean(d ** 2))) if d.size else float("nan")


def section_centroids(points, axis_point, axis_dir, n_slabs=12, t_range=None,
                      min_pts_per_slab=4, trim=2.0):
    """
    Corta la nube en losas perpendiculares al eje y devuelve el centroide y el radio
    transversal de cada una.

    `t_range` es una fracción (lo, hi) del recorrido a lo largo del eje: sirve para
    quedarse con la parte cilíndrica (diáfisis) y dejar fuera la cabeza.

    Dentro de cada losa se parte de la MEDIANA (no la media) y se descartan los
    puntos a más de `trim` medianas: `Humero.stl` tiene geometría espuria en algunas
    bandas (radios de 32-56 mm donde la diáfisis mide 11) y una media cruda se va
    detrás de ella.

    Devuelve (centroids (K,3), radii (K,), t (K,), counts (K,)).
    """
    P = np.asarray(points, dtype=float)
    v = np.asarray(axis_dir, dtype=float)
    v = v / np.linalg.norm(v)
    t = (P - np.asarray(axis_point, dtype=float)) @ v

    lo, hi = t.min(), t.max()
    if t_range is not None:
        span = hi - lo
        lo, hi = lo + t_range[0] * span, lo + t_range[1] * span

    def transverse_radius(pts, c):
        d = pts - c
        return np.linalg.norm(d - np.outer(d @ v, v), axis=1)

    edges = np.linspace(lo, hi, n_slabs + 1)
    cents, radii, ts, counts = [], [], [], []
    for k in range(n_slabs):
        pts = P[(t >= edges[k]) & (t < edges[k + 1])]
        if len(pts) < min_pts_per_slab:
            continue
        c = np.median(pts, axis=0)
        r = transverse_radius(pts, c)
        med = np.median(r)
        if med > 0:
            pts = pts[r < trim * med]
            if len(pts) < min_pts_per_slab:
                continue
            c = pts.mean(axis=0)
        cents.append(c)
        radii.append(float(np.median(transverse_radius(pts, c))))
        ts.append(0.5 * (edges[k] + edges[k + 1]))
        counts.append(len(pts))

    return (np.asarray(cents), np.asarray(radii), np.asarray(ts), np.asarray(counts))


def detect_head_side(points, axis_point, axis_dir, edge_fraction=0.12):
    """
    ¿Hacia qué lado del eje está la cabeza humeral? Devuelve +1 o −1.

    Criterio: la cabeza es el extremo con MAYOR radio transversal medio (es un
    casquete esférico de ~24 mm frente a una diáfisis de ~11 mm).
    """
    P = np.asarray(points, dtype=float)
    v = np.asarray(axis_dir, dtype=float)
    v = v / np.linalg.norm(v)
    t = (P - np.asarray(axis_point, dtype=float)) @ v
    lo, hi = t.min(), t.max()
    span = hi - lo

    def mean_radius(mask):
        pts = P[mask]
        if len(pts) < 4:
            return 0.0
        c = pts.mean(axis=0)
        dd = pts - c
        perp = dd - np.outer(dd @ v, v)
        return float(np.mean(np.linalg.norm(perp, axis=1)))

    r_hi = mean_radius(t > hi - edge_fraction * span)
    r_lo = mean_radius(t < lo + edge_fraction * span)
    return 1 if r_hi >= r_lo else -1


def estimate_shaft_axis(points, n_slabs=12, shaft_fraction=0.62, min_pts_per_slab=4,
                        radius_band=(0.6, 1.8)):
    """
    Eje de la diáfisis: losas perpendiculares → centroide robusto por losa → recta
    por esos centroides.

    Un PCA global sobre todo el hueso NO sirve: la cabeza es un bulto lateral y
    excéntrico que tuerce la dirección principal. Y tampoco sirve iterar la
    selección de losas sobre el eje refinado: al reseleccionar, el eje se realimenta
    y diverge (medido: 8.5° → 8.4° → 11.1° en `Humero.stl`). Así que la región
    diafisaria se fija UNA vez con el eje preliminar y solo se refina el ajuste.

    La estabilidad viene de `radius_band`: se descartan las losas cuyo radio
    transversal se aparta de la mediana (en `Humero.stl` hay bandas con radios de
    32-56 mm frente a los ~11 mm reales, geometría espuria de la malla).

    `shaft_fraction` es la fracción del hueso, medida desde el extremo OPUESTO a la
    cabeza, que se considera diáfisis.

    Devuelve dict con axis_point, axis_dir (orientada de la diáfisis hacia la
    cabeza), rms, n_slabs_used, n_slabs_rejected y los radios por losa.
    """
    P = np.asarray(points, dtype=float)
    c0, v, _ = principal_axis(P)
    v = detect_head_side(P, c0, v) * v          # v apunta hacia la cabeza

    cents, radii, ts, counts = section_centroids(
        P, c0, v, n_slabs=n_slabs, t_range=(0.0, shaft_fraction),
        min_pts_per_slab=min_pts_per_slab)

    n_rejected = 0
    if len(radii):
        med = float(np.median(radii))
        keep = (radii > radius_band[0] * med) & (radii < radius_band[1] * med)
        n_rejected = int((~keep).sum())
        cents, radii, ts, counts = cents[keep], radii[keep], ts[keep], counts[keep]

    if len(cents) < 3:
        raise ValueError(
            f"🚨 solo {len(cents)} losas útiles para el eje: la malla es demasiado "
            "basta o la región diafisaria está mal delimitada")

    axis_point, axis_dir, _ = principal_axis(cents)
    if axis_dir @ v < 0:
        axis_dir = -axis_dir

    return {
        "axis_point": axis_point,
        "axis_dir": axis_dir,
        "rms": line_fit_rms(cents, axis_point, axis_dir),
        "n_slabs_used": int(len(cents)),
        "n_slabs_rejected": n_rejected,
        "slab_radii": radii,
        "centroids": cents,
        "preliminary_dir": v,
        "tilt_vs_preliminary_deg": float(np.degrees(np.arccos(np.clip(axis_dir @ v, -1, 1)))),
    }


def fit_sphere_algebraic(points):
    """
    Ajuste algebraico de esfera por mínimos cuadrados (el mismo método que
    humerus_boundary_analysis.fit_sphere_to_points). Devuelve (centro, radio).
    """
    P = np.asarray(points, dtype=float)
    if len(P) < 4:
        return None, None
    A = np.hstack([2.0 * P, np.ones((len(P), 1))])
    b = np.sum(P ** 2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    center = sol[:3]
    r2 = sol[3] + float(center @ center)
    return (center, float(np.sqrt(r2))) if r2 > 0 else (None, None)


def ransac_sphere(points, n_iter=2000, inlier_dist=1.5, min_inliers=10,
                  radius_range=(5.0, 60.0), seed=0):
    """
    RANSAC de esfera. Devuelve (centro, radio, máscara de inliers, rms) o Nones.

    Mismos parámetros de espíritu que el pipeline existente, pero con semilla fija
    para que los reportes sean reproducibles.
    """
    P = np.asarray(points, dtype=float)
    if len(P) < 4:
        return None, None, None, None
    rng = np.random.default_rng(seed)
    best = (0, None, None)

    for _ in range(n_iter):
        c, r = fit_sphere_algebraic(P[rng.choice(len(P), 4, replace=False)])
        if r is None or not (radius_range[0] <= r <= radius_range[1]):
            continue
        inl = np.abs(np.linalg.norm(P - c, axis=1) - r) < inlier_dist
        n = int(inl.sum())
        if n > best[0]:
            best = (n, c, r)

    if best[0] < min_inliers:
        return None, None, None, None

    # Refit sobre los inliers.
    c, r = best[1], best[2]
    inl = np.abs(np.linalg.norm(P - c, axis=1) - r) < inlier_dist
    c, r = fit_sphere_algebraic(P[inl])
    if r is None:
        return None, None, None, None
    inl = np.abs(np.linalg.norm(P - c, axis=1) - r) < inlier_dist
    rms = float(np.sqrt(np.mean((np.linalg.norm(P[inl] - c, axis=1) - r) ** 2)))
    return c, float(r), inl, rms


def fit_head_sphere(points, axis_point, axis_dir, head_fraction=0.32, **kwargs):
    """
    Ajusta la esfera de la cabeza usando solo el casquete: la fracción superior del
    hueso a lo largo del eje (con axis_dir apuntando hacia la cabeza).
    """
    P = np.asarray(points, dtype=float)
    v = np.asarray(axis_dir, dtype=float)
    v = v / np.linalg.norm(v)
    t = (P - np.asarray(axis_point, dtype=float)) @ v
    cut = t.max() - head_fraction * (t.max() - t.min())
    head = P[t > cut]
    center, radius, inl, rms = ransac_sphere(head, **kwargs)
    return center, radius, inl, rms, head


# ---------------------------------------------------------------------------
# Par (esfera, eje) desde una malla — el húmero standard
# ---------------------------------------------------------------------------

def sphere_axis_from_mesh(path, head_fraction=0.32, n_slabs=12, shaft_fraction=0.62,
                          n_samples=40000, seed=0):
    """
    Ejercicio de la diapositiva 8: "si tienes el stl del húmero, la esfera
    aproximante y el eje se pueden aproximar directamente de la triangulación".

    Se trabaja sobre una nube MUESTREADA DE LA SUPERFICIE, no sobre los vértices:
    en `Humero.stl` el 70 % de los 1301 vértices están en la cabeza y la diáfisis se
    queda con ~10 por banda. Muestreando por área, el RMS del eje baja de 2.09 mm a
    0.75 mm y se aprovechan las 12 losas en vez de 10.

    Devuelve un `pair` (ver Geometry.similarity.make_pair) con diagnósticos.
    """
    from Geometry.stl_io import read_stl, sample_surface

    tri, _ = read_stl(path)
    V = sample_surface(tri, n_points=n_samples, seed=seed)

    axis = estimate_shaft_axis(V, n_slabs=n_slabs, shaft_fraction=shaft_fraction)
    center, radius, inl, rms_sphere, head_pts = fit_head_sphere(
        V, axis["axis_point"], axis["axis_dir"], head_fraction=head_fraction, seed=seed)

    if radius is None:
        raise ValueError(f"🚨 no se pudo ajustar la esfera de la cabeza en {path}")

    warnings = []
    if not (SPHERE_RADIUS_RANGE_MM[0] <= radius <= SPHERE_RADIUS_RANGE_MM[1]):
        warnings.append(f"⚠️  radio {radius:.2f} mm fuera del rango anatómico "
                        f"{SPHERE_RADIUS_RANGE_MM}")
    if not np.isnan(axis["rms"]) and axis["rms"] > MAX_AXIS_RMS_MM:
        warnings.append(f"⚠️  RMS del eje {axis['rms']:.2f} mm > {MAX_AXIS_RMS_MM} mm")

    return make_pair(
        center, radius, axis["axis_point"], axis["axis_dir"],
        source=f"mesh:{os.path.basename(path)}",
        rms_sphere=rms_sphere,
        rms_axis=axis["rms"],
        n_sphere_inliers=int(inl.sum()) if inl is not None else 0,
        n_head_points=int(len(head_pts)),
        n_slabs_used=axis["n_slabs_used"],
        n_slabs_rejected=axis["n_slabs_rejected"],
        tilt_vs_preliminary_deg=axis["tilt_vs_preliminary_deg"],
        n_vertices=int(len(V)),
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Par (esfera, eje) desde los CSV — el húmero del paciente
# ---------------------------------------------------------------------------

def calibrate_axis_bias_from_mesh(path, z_extents_mm=(55.0, 60.0, 68.0, 80.0, 100.0, 130.0),
                                  head_fraction=0.32, n_samples=40000, seed=0):
    """
    Mide el sesgo que introduce ver solo el húmero proximal — el pendiente que el
    profesor anotó en la diapositiva 16.

    Los volúmenes MRI de este proyecto cubren ~68 mm y solo llegan a cuello
    quirúrgico / metáfisis, así que el "eje diafisario" que se estima de ellos no es
    el eje del brazo. Como el húmero standard SÍ está completo, el sesgo es medible:

      1. eje verdadero con la malla entera;
      2. truncar la malla a `z_extent` mm desde el extremo de la cabeza;
      3. re-estimar con el MISMO estimador;
      4. comparar ángulo, d y la razón R/d.

    Si `d_scale` es estable a lo largo del barrido, la corrección es defendible; si
    no lo es, el resultado es que NO se puede corregir, y eso también se reporta.
    """
    from Geometry.stl_io import read_stl, sample_surface

    tri, _ = read_stl(path)
    V = sample_surface(tri, n_points=n_samples, seed=seed)

    full = estimate_shaft_axis(V)
    center, radius, _, _, _ = fit_head_sphere(V, full["axis_point"], full["axis_dir"],
                                              head_fraction=head_fraction, seed=seed)
    d_true = point_line_distance(center, full["axis_point"], full["axis_dir"])

    v = full["axis_dir"]
    t = (V - full["axis_point"]) @ v
    t_top = t.max()

    rows = []
    for extent in z_extents_mm:
        sub = V[t > t_top - extent]
        if len(sub) < 500:
            rows.append({"z_extent_mm": float(extent), "ok": False,
                         "reason": f"solo {len(sub)} vértices"})
            continue
        try:
            # Con menos hueso, la "diáfisis" disponible es una fracción mayor del
            # recorte: hay que pedirle al estimador que mire más arriba.
            trunc = estimate_shaft_axis(sub, n_slabs=10, shaft_fraction=0.75)
            c2, r2, _, _, _ = fit_head_sphere(sub, trunc["axis_point"], trunc["axis_dir"],
                                              head_fraction=head_fraction, seed=seed)
            d_tr = point_line_distance(c2, trunc["axis_point"], trunc["axis_dir"])
            angle = float(np.degrees(np.arccos(np.clip(abs(trunc["axis_dir"] @ v), -1, 1))))
            rows.append({
                "z_extent_mm": float(extent), "ok": True,
                "angle_deg": angle,
                "d_trunc": float(d_tr), "d_scale": float(d_tr / d_true),
                "radius_trunc": float(r2),
                "ratio_trunc": float(r2 / d_tr) if d_tr > 1e-9 else float("inf"),
                "n_slabs": trunc["n_slabs_used"],
            })
        except Exception as exc:                                   # noqa: BLE001
            rows.append({"z_extent_mm": float(extent), "ok": False, "reason": str(exc)})

    scales = [r["d_scale"] for r in rows if r.get("ok")]

    # El rango que de verdad importa: la cobertura de los MRI de este proyecto es de
    # ~68 mm. Por debajo de ~58 mm el estimador se rompe del todo (a 55 mm el eje sale
    # con 55° de error), así que la ventana útil de calibración es 58-100 mm.
    mri_like = [r for r in rows if r.get("ok") and 58.0 <= r["z_extent_mm"] <= 100.0]
    d_scale_mri = float(np.mean([r["d_scale"] for r in mri_like])) if mri_like else float("nan")
    ratio_mri = float(np.mean([r["ratio_trunc"] for r in mri_like])) if mri_like else float("nan")

    return {
        "path": path,
        "d_true": float(d_true),
        "radius_true": float(radius),
        "ratio_true": float(radius / d_true),
        "rows": rows,
        "d_scale_mean": float(np.mean(scales)) if scales else float("nan"),
        "d_scale_std": float(np.std(scales)) if scales else float("nan"),
        "d_scale_mri_like": d_scale_mri,
        "d_scale_mri_like_std": (float(np.std([r["d_scale"] for r in mri_like]))
                                 if mri_like else float("nan")),
        "ratio_mri_like": ratio_mri,
        "n_mri_like": len(mri_like),
        "stable": bool(len(mri_like) >= 2
                       and np.std([r["d_scale"] for r in mri_like]) < 0.20 * abs(d_scale_mri)),
    }


def bias_corrected_pair(pair, d_scale):
    """
    Devuelve una copia del par con la excentricidad corregida por el sesgo de
    truncamiento medido en `calibrate_axis_bias_from_mesh`.

    Se desplaza el eje alejándolo del centro hasta que d = d_leído / d_scale,
    conservando la dirección. NO corrige el error angular (~10°), que requeriría
    saber en qué plano se inclinó — así que esto es una corrección de primer orden
    sobre la magnitud, no sobre la orientación.
    """
    from Geometry.similarity import frame_from_pair, make_pair

    if not np.isfinite(d_scale) or d_scale <= 0:
        raise ValueError(f"🚨 d_scale inválido: {d_scale}")

    f = frame_from_pair(pair)
    d_new = pair["d"] / float(d_scale)
    new_point = pair["center"] + d_new * f["w"]

    extra = {k: v for k, v in pair.items()
             if k not in ("center", "radius", "axis_point", "axis_dir", "d", "ratio",
                          "warnings", "source")}
    extra["bias_corrected"] = True
    extra["d_scale_applied"] = float(d_scale)
    extra["d_raw_mm"] = float(pair["d"])
    return make_pair(pair["center"], pair["radius"], new_point, pair["axis_dir"],
                     source=f"{pair.get('source', '?')} (corregido d_scale={d_scale:.3f})",
                     warnings=list(pair.get("warnings", [])), **extra)


def load_slice_circles(csv_path):
    """Lee slice_circles.csv y devuelve una lista de dicts tipados."""
    rows = []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "slice_idx": int(r["slice_idx"]),
                "circle_idx": int(r["circle_idx"]),
                "is_articular": bool(int(r["is_articular"])),
                "area_px": float(r["area_px"]),
                "center": np.array([float(r["center_x_mm"]), float(r["center_y_mm"]),
                                    float(r["center_z_mm"])]),
                "normal": np.array([float(r["normal_x"]), float(r["normal_y"]),
                                    float(r["normal_z"])]),
                "radius_mm": float(r["radius_mm"]),
                "num_inliers": int(r["num_inliers"]),
                "inlier_ratio": float(r["inlier_ratio"]),
                "fit_residual_mm": float(r["fit_residual_mm"]),
            })
    return rows


def _common_normal(circles):
    """
    Normal común de las slices, con los signos homogeneizados.

    El CSV guarda la normal con signo arbitrario por slice (viene del SVD), así que
    hay que alinearlas antes de promediar o el resultado se cancela.
    """
    N = np.array([c["normal"] for c in circles], dtype=float)
    ref = N[0]
    N = N * np.sign(N @ ref)[:, None]
    n = N.mean(axis=0)
    return n / np.linalg.norm(n)


def fit_sphere_from_articular_circles(circles):
    """
    Esfera a partir de los círculos articulares — la idea de la diapositiva 8:
    "a partir de estos [arcos] puedes generar la esfera aproximante".

    Como las slices axiales son paralelas y comparten normal n, cada círculo de
    radio r_i a altura t_i = n·C_i cumple  r_i² = R² − (t_i − t₀)²  donde t₀ es la
    altura del centro de la esfera. Eso es lineal en (t₀, R² − t₀²):

        r_i² + t_i² = 2 t₀ t_i + (R² − t₀²)

    Las coordenadas del centro dentro del plano se toman como la media de los
    centros de los círculos (todos deberían proyectarse al mismo punto).

    Devuelve (centro, radio, rms, n_usados).
    """
    if len(circles) < 3:
        return None, None, None, 0

    n = _common_normal(circles)
    C = np.array([c["center"] for c in circles], dtype=float)
    r = np.array([c["radius_mm"] for c in circles], dtype=float)
    t = C @ n

    A = np.column_stack([2.0 * t, np.ones(len(t))])
    b = r ** 2 + t ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    t0, K = float(sol[0]), float(sol[1])
    R2 = K + t0 ** 2
    if R2 <= 0:
        return None, None, None, 0
    R = float(np.sqrt(R2))

    mean_c = C.mean(axis=0)
    center = mean_c + (t0 - float(mean_c @ n)) * n

    pred = R2 - (t - t0) ** 2
    rms = float(np.sqrt(np.mean((r - np.sqrt(np.clip(pred, 0, None))) ** 2)))
    return center, R, rms, len(circles)


def fit_axis_from_shaft_circles(circles, n_boot=300, seed=0):
    """
    Eje a partir de los centros de los círculos NO articulares (diáfisis/metáfisis).

    Devuelve (axis_point, axis_dir, rms, sigma_deg, n). `sigma_deg` es la desviación
    angular por bootstrap: es la barra de error que hay que arrastrar hasta R/d.
    """
    if len(circles) < 3:
        return None, None, None, None, len(circles)

    C = np.array([c["center"] for c in circles], dtype=float)
    p0, v, _ = principal_axis(C)
    rms = line_fit_rms(C, p0, v)

    rng = np.random.default_rng(seed)
    angles = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(C), len(C))
        if len(np.unique(idx)) < 3:
            continue
        _, vb, _ = principal_axis(C[idx])
        if vb @ v < 0:
            vb = -vb
        angles.append(np.degrees(np.arccos(np.clip(vb @ v, -1, 1))))

    sigma = float(np.std(angles)) if angles else float("nan")
    return p0, v, rms, sigma, len(C)


def axis_convergence_curve(circles):
    """
    ¿Ha convergido el eje con la extensión disponible?

    Recalcula el eje usando los k centros más distales (k = 3..N) y devuelve, para
    cada k, el ángulo respecto al eje calculado con todos. Si la curva no se aplana,
    el brazo de palanca es insuficiente — que es exactamente lo que pasa con
    volúmenes axiales cortos.

    Devuelve (ks, angles_deg, d_values) con d_values = None (lo rellena el llamador
    si tiene la esfera).
    """
    if len(circles) < 4:
        return np.array([]), np.array([]), None

    C = np.array([c["center"] for c in circles], dtype=float)
    _, v_full, _ = principal_axis(C)

    # Ordenar por posición a lo largo del eje, empezando por el extremo distal.
    order = np.argsort(C @ v_full)
    C = C[order]

    ks, angles = [], []
    for k in range(3, len(C) + 1):
        _, vk, _ = principal_axis(C[:k])
        if vk @ v_full < 0:
            vk = -vk
        ks.append(k)
        angles.append(float(np.degrees(np.arccos(np.clip(vk @ v_full, -1, 1)))))
    return np.asarray(ks), np.asarray(angles), None


def sphere_axis_from_slice_circles(csv_path, dataset_name=None):
    """
    Ejercicio de la diapositiva 25: el par (esfera, eje) del paciente, a partir de
    los arcos circulares detectados en los cortes axiales.

    Los círculos articulares (los de la cabeza) dan la esfera; los no articulares
    (los del brazo, bajo el cuello) dan el eje.

    Devuelve un `pair` con `quality_flags`: los criterios de exclusión están fijados
    a priori y se reportan, no se aplican a ojo.
    """
    circles = load_slice_circles(csv_path)
    art = [c for c in circles if c["is_articular"]]
    shaft = [c for c in circles if not c["is_articular"]]

    center, radius, rms_sphere, n_art = fit_sphere_from_articular_circles(art)
    if radius is None:
        raise ValueError(f"🚨 no se pudo ajustar la esfera en {csv_path} "
                         f"({len(art)} círculos articulares)")

    axis_point, axis_dir, rms_axis, sigma_deg, n_shaft = fit_axis_from_shaft_circles(shaft)
    if axis_dir is None:
        raise ValueError(f"🚨 no hay suficientes círculos diafisarios en {csv_path} "
                         f"({len(shaft)})")

    # Orientar el eje hacia la cabeza, para que el marco sea comparable con el del STL.
    if (center - axis_point) @ axis_dir < 0:
        axis_dir = -axis_dir

    ks, angles, _ = axis_convergence_curve(shaft)

    flags = []
    if n_art < MIN_ARTICULAR_CIRCLES:
        flags.append(f"pocos_circulos_articulares ({n_art} < {MIN_ARTICULAR_CIRCLES})")
    if n_shaft < MIN_SHAFT_CIRCLES:
        flags.append(f"pocos_circulos_diafisarios ({n_shaft} < {MIN_SHAFT_CIRCLES})")
    if not (SPHERE_RADIUS_RANGE_MM[0] <= radius <= SPHERE_RADIUS_RANGE_MM[1]):
        flags.append(f"radio_fuera_de_rango ({radius:.2f} mm)")
    if rms_axis is not None and rms_axis > MAX_AXIS_RMS_MM:
        flags.append(f"eje_mal_ajustado (rms {rms_axis:.2f} mm)")
    if sigma_deg is not None and not np.isnan(sigma_deg) and sigma_deg > MAX_AXIS_SIGMA_DEG:
        flags.append(f"eje_incierto (sigma {sigma_deg:.2f}°)")
    if len(angles) and angles[-2:].max() > 2.0:
        flags.append(f"eje_no_convergido (ultimo salto {angles[-2:].max():.2f}°)")

    return make_pair(
        center, radius, axis_point, axis_dir,
        source=f"csv:{dataset_name or os.path.basename(os.path.dirname(csv_path))}",
        rms_sphere=rms_sphere,
        rms_axis=rms_axis,
        axis_sigma_deg=sigma_deg,
        n_articular=n_art,
        n_shaft=n_shaft,
        convergence_k=ks,
        convergence_angles_deg=angles,
        quality_flags=flags,
        usable=bool(not flags),
    )
