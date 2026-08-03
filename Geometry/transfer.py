"""
Transporte standard → paciente — actividad de la diapositiva 29.

Componemos una transformación Φ que lleva el par (esfera, eje) del húmero standard
al par del paciente, y con ella arrastramos las mallas de los músculos del manguito
rotador.

La receta es una conjugación:

    Φ = K_pac⁻¹ ∘ M ∘ K_std

donde K lleva un par a la forma canónica (esfera de radio ρ en el origen, eje
paralelo a e₁ a altura h = ρ·d/R sobre el plano z = 0) y M es el boost de O(3,1) que
fija esa esfera y manda la cuerda canónica del standard en la del paciente.

Si las razones R/d coinciden, h_std = h_pac, sale M = I y Φ es una simple
similaridad. Si no coinciden, hace falta el boost — y entonces Φ NO preserva ángulos
ni razones de distancias, solo rectas, planos y razones dobles.
"""

import csv
import os

import numpy as np

from Geometry.lorentz import (
    apply_projective, boost_along_chord, boost_fixing_sphere, check_vanishing_plane,
    is_in_o31,
)
from Geometry.similarity import frame_from_pair, point_line_distance

TARGET_RADIUS = 5.0          # ρ de la forma canónica, el mismo de la diapositiva 18

# Sectores nominales de la presentación (presentation/sections/11-geometric-trick.tex).
# Son una HIPÓTESIS no validada; aquí solo sirven de referencia para comparar.
NOMINAL_SECTORS_DEG = {
    "Subescapular": (0, 45),
    "Supraespinoso": (45, 135),
    "Infraespinoso": (135, 225),
    "RedondoMenor": (225, 270),
}


# ---------------------------------------------------------------------------
# Forma canónica
# ---------------------------------------------------------------------------

def canonical_similarity(pair, target_radius=TARGET_RADIUS):
    """
    Similaridad K que lleva el par a la forma canónica.

    Con Q = [v, n, w] por columnas (v = dirección del eje, w = unitario del centro
    hacia el pie de la perpendicular, n = w × v) y K(x) = (ρ/R)·Qᵀ(x − C):

      * el centro de la esfera va al origen y su radio pasa de R a ρ;
      * el eje va a la recta {(t, 0, h)} con h = ρ·d/R, dirección e₁.

    Devuelve dict con s, R, b, h, y los dos puntos A, B de corte de la recta
    canónica con la esfera canónica.
    """
    f = frame_from_pair(pair)
    Q = np.column_stack([f["v"], f["n"], f["w"]])       # [e1←v, e2←n, e3←w]
    s = target_radius / pair["radius"]
    Rm = Q.T
    b = -s * (Rm @ pair["center"])

    h = target_radius * pair["d"] / pair["radius"]
    half = float(np.sqrt(max(target_radius ** 2 - h ** 2, 0.0)))
    A = np.array([-half, 0.0, h])
    B = np.array([half, 0.0, h])
    return {"s": float(s), "R": Rm, "b": b, "h": float(h), "A": A, "B": B,
            "half_chord": half, "Q": Q}


def homogenize_similarity(s, R, b, target_radius=TARGET_RADIUS):
    """
    Sube la similaridad x ↦ sRx + b a una 4×4 que actúa en el chart w = ρ.

    Sobre [p; ρ] da [sRp + b; ρ], que es lo que queremos, y sobre pesos w ≠ ρ da la
    extensión proyectiva correcta. Así toda la composición es UNA sola matriz.
    """
    H = np.eye(4)
    H[:3, :3] = s * np.asarray(R, dtype=float)
    H[:3, 3] = np.asarray(b, dtype=float) / float(target_radius)
    return H


def invert_homogeneous(H):
    """Inversa de la 4×4 (es una similaridad homogeneizada, siempre invertible)."""
    return np.linalg.inv(np.asarray(H, dtype=float))


def build_full_transform(pair_std, pair_pat, target_radius=TARGET_RADIUS,
                         gauge_theta=0.0, ratio_rtol=1e-6):
    """
    Construye Φ = K_pac⁻¹ ∘ M(gauge) ∘ K_std como una única matriz 4×4 proyectiva.

    Devuelve dict con:
      Phi              la 4×4 (aplicar con apply_projective en el chart ρ)
      needs_boost      si las razones R/d difieren de verdad
      M                el boost de O(3,1) (identidad si no hace falta)
      K_std, K_pat     las similaridades canónicas
      checks           residuos de verificación (esfera y eje)
    """
    K_std = canonical_similarity(pair_std, target_radius)
    K_pat = canonical_similarity(pair_pat, target_radius)

    rel = abs(pair_std["ratio"] - pair_pat["ratio"]) / max(abs(pair_std["ratio"]), 1e-12)
    needs_boost = bool(rel > ratio_rtol)

    if needs_boost:
        boost = boost_fixing_sphere(K_std["A"], K_std["B"], K_pat["A"], K_pat["B"],
                                    target_radius)
        M = boost["M"]
    else:
        M = np.eye(4)

    if abs(gauge_theta) > 0:
        M = boost_along_chord(gauge_theta, K_pat["A"], K_pat["B"], target_radius) @ M

    H_std = homogenize_similarity(K_std["s"], K_std["R"], K_std["b"], target_radius)
    H_pat = homogenize_similarity(K_pat["s"], K_pat["R"], K_pat["b"], target_radius)
    Phi = invert_homogeneous(H_pat) @ M @ H_std

    return {
        "Phi": Phi,
        "M": M,
        "K_std": K_std,
        "K_pat": K_pat,
        "needs_boost": needs_boost,
        "ratio_std": float(pair_std["ratio"]),
        "ratio_pat": float(pair_pat["ratio"]),
        "ratio_rel_diff": float(rel),
        "gauge_theta": float(gauge_theta),
        "target_radius": float(target_radius),
        "o31_dev": float(is_in_o31(M)[1]),
        "checks": verify_transform(Phi, pair_std, pair_pat, target_radius),
    }


def apply_transform(transform, points):
    """Aplica Φ a puntos de R³. Devuelve (mapped, w, n_singular)."""
    return apply_projective(transform["Phi"], points, transform["target_radius"])


def verify_transform(Phi, pair_std, pair_pat, target_radius=TARGET_RADIUS, n=400, seed=0):
    """
    Comprueba que Φ hace lo que promete: manda la esfera del standard en la del
    paciente y el eje del standard en el eje del paciente.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    on_sphere = pair_std["center"] + pair_std["radius"] * v / np.linalg.norm(v, axis=1)[:, None]
    mapped, _, n_sing = apply_projective(Phi, on_sphere, target_radius)
    sphere_err = float(np.max(np.abs(
        np.linalg.norm(mapped - pair_pat["center"], axis=1) - pair_pat["radius"])))

    ts = np.linspace(-60.0, 60.0, 200)
    on_axis = pair_std["axis_point"] + ts[:, None] * pair_std["axis_dir"]
    mapped_axis, _, _ = apply_projective(Phi, on_axis, target_radius)
    axis_err = float(np.max([point_line_distance(p, pair_pat["axis_point"],
                                                 pair_pat["axis_dir"]) for p in mapped_axis]))

    return {"sphere_max_err_mm": sphere_err, "axis_max_err_mm": axis_err,
            "n_singular": int(n_sing),
            "ok": bool(sphere_err < 1e-6 and axis_err < 1e-6 and n_sing == 0)}


# ---------------------------------------------------------------------------
# Mallas
# ---------------------------------------------------------------------------

def transform_mesh(triangles, transform):
    """
    Aplica Φ a una malla, vértice a vértice.

    Una proyectividad manda planos en planos, así que los triángulos siguen siendo
    triángulos planos y NO hace falta remallar. Sí hay que recalcular las normales.

    Devuelve (triangles_out, diag) con el diagnóstico del plano evanescente.
    """
    tri = np.asarray(triangles, dtype=float)
    flat = tri.reshape(-1, 3)
    diag = check_vanishing_plane(transform["Phi"], flat, transform["target_radius"])
    mapped, w, n_sing = apply_projective(transform["Phi"], flat, transform["target_radius"])
    diag["n_singular"] = int(n_sing)
    diag["w_ratio"] = float(diag["max_abs_w"] / max(diag["min_abs_w"], 1e-12))
    return mapped.reshape(-1, 3, 3), diag


def map_muscle_stls(std_dir, transform, out_dir,
                    names=("Infraespinoso", "RedondoMenor", "Subescapular"),
                    humerus_name="Humero"):
    """
    Transporta las mallas del standard al paciente y las escribe como STL binario.

    Si el plano evanescente parte una malla, esa malla se ABORTA con warning en vez
    de escribir geometría sin sentido.
    """
    from Geometry.stl_io import read_stl, write_stl_binary

    os.makedirs(out_dir, exist_ok=True)
    results = []
    for name in (humerus_name,) + tuple(names):
        src = os.path.join(std_dir, f"{name}.stl")
        if not os.path.exists(src):
            results.append({"name": name, "ok": False, "reason": "no existe el .stl"})
            print(f"  ⚠️  {name}: no se encontró {src}")
            continue

        tri, _ = read_stl(src)
        out_tri, diag = transform_mesh(tri, transform)
        if not diag["safe"]:
            results.append({"name": name, "ok": False, "reason": "plano evanescente",
                            "diag": diag})
            print(f"  🚨 {name}: la malla cruza el plano evanescente, se aborta")
            continue

        dst = os.path.join(out_dir, f"{name}_mapped.stl")
        write_stl_binary(dst, out_tri)
        results.append({"name": name, "ok": True, "path": dst,
                        "n_triangles": int(len(out_tri)),
                        "w_ratio": diag["w_ratio"],
                        "triangles": out_tri})
        print(f"  ✅ {name}: {len(out_tri)} triángulos → {os.path.basename(dst)} "
              f"(distorsión proyectiva w_max/w_min = {diag['w_ratio']:.4f})")
    return results


# ---------------------------------------------------------------------------
# Eficacia de modelación
# ---------------------------------------------------------------------------

def load_boundary_points(csv_path):
    """Lee boundary_world_coords.csv → (points (N,3), slice_idx (N,))."""
    pts, idx = [], []
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            pts.append([float(r["x_mm"]), float(r["y_mm"]), float(r["z_mm"])])
            idx.append(int(r["slice_idx"]))
    return np.asarray(pts, dtype=float), np.asarray(idx, dtype=int)


def _surface_cloud(triangles, n_points=200000, seed=0):
    """Nube densa sobre la superficie de una malla, con la normal de cada muestra."""
    from Geometry.stl_io import compute_normals, sample_surface, triangle_areas

    tri = np.asarray(triangles, dtype=float)
    areas = triangle_areas(tri)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(tri), size=int(n_points), p=areas / areas.sum())
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1.0
    u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
    a, b, c = tri[idx, 0], tri[idx, 1], tri[idx, 2]
    pts = a + u[:, None] * (b - a) + v[:, None] * (c - a)
    return pts, compute_normals(tri)[idx]


def bone_fidelity(mapped_humerus_tri, boundary_pts, articular_slices=None,
                  slice_idx=None, n_samples=200000, seed=0):
    """
    M1 — la ÚNICA métrica con ground truth real.

    El húmero standard sí tiene contraparte en el paciente, así que medimos la
    distancia de la nube de borde del MRI a la superficie del húmero transportado.

    Se restringe a la banda z que el MRI cubre: si no, el máximo lo domina la parte
    del hueso que la resonancia nunca vio y el número no significa nada.

    Como los músculos viajan solidarios al hueso por la misma Φ, este error es una
    COTA INFERIOR del error muscular.
    """
    from scipy.spatial import cKDTree

    cloud, _ = _surface_cloud(mapped_humerus_tri, n_points=n_samples, seed=seed)
    z_lo, z_hi = boundary_pts[:, 2].min(), boundary_pts[:, 2].max()
    band = cloud[(cloud[:, 2] >= z_lo - 2.0) & (cloud[:, 2] <= z_hi + 2.0)]
    if len(band) < 100:
        return {"ok": False, "reason": "el húmero transportado no cubre la banda del MRI"}

    tree = cKDTree(band)
    dist, _ = tree.query(boundary_pts)

    out = {
        "ok": True,
        "n_boundary_points": int(len(boundary_pts)),
        "n_surface_samples": int(len(band)),
        "rms_mm": float(np.sqrt(np.mean(dist ** 2))),
        "median_mm": float(np.median(dist)),
        "p95_mm": float(np.percentile(dist, 95)),
        "max_mm": float(dist.max()),
        "z_band_mm": (float(z_lo), float(z_hi)),
    }
    if articular_slices is not None and slice_idx is not None:
        art = np.isin(slice_idx, list(articular_slices))
        if art.any():
            out["rms_articular_mm"] = float(np.sqrt(np.mean(dist[art] ** 2)))
        if (~art).any():
            out["rms_shaft_mm"] = float(np.sqrt(np.mean(dist[~art] ** 2)))
    return out


def muscle_plausibility(mapped_humerus_tri, muscle_tri, pair_pat,
                        footprint_mm=2.0, n_samples=150000, seed=0):
    """
    M2 — plausibilidad anatómica del músculo transportado. Dos números:

      (a) penetración: fracción de vértices que quedan DENTRO del hueso. El test de
          dentro/fuera usa la normal de la muestra más cercana (pseudo-normal); es
          aproximado cerca de aristas, pero suficiente para detectar un mapeo roto.
      (b) huella de inserción: posición angular θ del centroide de los vértices que
          quedan a menos de `footprint_mm` del hueso, medida en el marco anatómico
          del paciente (θ = atan2(p·n, p·w) con w hacia el pie de la perpendicular
          del eje y n = w × v).

    (b) es FALSABLE: si un músculo aterriza en el sector equivocado, el mapeo está
    mal — típicamente por reflexión o por lateralidad.
    """
    from scipy.spatial import cKDTree

    from Geometry.stl_io import unique_vertices

    cloud, normals = _surface_cloud(mapped_humerus_tri, n_points=n_samples, seed=seed)
    V, _ = unique_vertices(muscle_tri)
    tree = cKDTree(cloud)
    dist, near = tree.query(V)

    inside = np.einsum("ij,ij->i", V - cloud[near], normals[near]) < 0.0
    foot = V[dist < footprint_mm]

    res = {
        "n_vertices": int(len(V)),
        "penetration_fraction": float(inside.mean()),
        "penetration_mean_mm": float(dist[inside].mean()) if inside.any() else 0.0,
        "n_footprint": int(len(foot)),
        "min_dist_mm": float(dist.min()),
    }

    if len(foot):
        f = frame_from_pair(pair_pat)
        rel = foot - pair_pat["center"]
        theta = np.degrees(np.arctan2(rel @ f["n"], rel @ f["w"])) % 360.0
        # Media circular, para que el envoltorio en 0/360 no invente un sector falso.
        ang = np.radians(theta)
        res["footprint_theta_deg"] = float(
            np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360.0)
        res["footprint_theta_p10_p90"] = (float(np.percentile(theta, 10)),
                                          float(np.percentile(theta, 90)))
        res["footprint_centroid_mm"] = foot.mean(axis=0)
    return res


def modeling_efficacy(mapped, pair_pat, boundary_csv, articular_slices=None):
    """
    Punto 7 de la diapositiva 29: "estimar la eficacia de modelación".

    ⚠️  NO HAY SEGMENTACIÓN MUSCULAR DEL PACIENTE. Ninguna de estas métricas mide
    exactitud anatómica de los músculos: miden CONSISTENCIA GEOMÉTRICA del transporte.
    Dice/IoU muscular, espesor tendinoso y detección de desgarro son N/A sin sustituto.
    """
    boundary_pts, slice_idx = load_boundary_points(boundary_csv)
    hum = next((m for m in mapped if m.get("ok") and m["name"] == "Humero"), None)
    if hum is None:
        return {"ok": False, "reason": "no se pudo transportar el húmero"}

    out = {
        "ok": True,
        "M1_bone_fidelity": bone_fidelity(hum["triangles"], boundary_pts,
                                          articular_slices, slice_idx),
        "M2_muscles": {},
        "notes": [
            "sin ground truth muscular del paciente: esto mide consistencia geométrica",
            "falta el supraespinoso (sector 45-135°, el clínicamente dominante)",
        ],
    }
    for m in mapped:
        if m.get("ok") and m["name"] != "Humero":
            out["M2_muscles"][m["name"]] = muscle_plausibility(
                hum["triangles"], m["triangles"], pair_pat)
    return out
