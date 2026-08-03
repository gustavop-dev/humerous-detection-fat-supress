"""
Verificación del núcleo geométrico.

El test de oro es test_boost_ground_truth: reproduce la matriz publicada en la
diapositiva 19 dígito a dígito. Todo lo demás se apoya en él.

Uso:  python exercises_geometry.py --selftest
"""

import math

import numpy as np

from Geometry.lorentz import (
    ETA, apply_projective, boost_along_chord, boost_fixing_sphere,
    boost_fixing_sphere_pptx_recipe, complete_minkowski_frame, embed_rotation,
    is_in_o31, is_proper_orthochronous, line_sphere_intersections,
    lorentz_frame_for_chord, lorentz_inverse, minkowski_dot, normalize_minkowski,
)
from Geometry.similarity import (
    apply_similarity, build_non_similar_example, check_orthogonality, frame_from_pair,
    line_line_distance, make_pair, pair_similarity_invariant, point_line_distance,
    similarity_axis_parallel, similarity_from_pairs,
)

# Datos del ejemplo de las diapositivas 18-19.
RHO = 5.0
A1 = np.array([-4.58257569495584, 0.0, 2.0])
B1 = np.array([4.58257569495584, 0.0, 2.0])
A2 = np.array([-4.0, 0.0, -3.0])
B2 = np.array([4.0, 0.0, -3.0])

# Datos del ejemplo de la diapositiva 12 (dos húmeros reales).
PAIR_SLIDE12_1 = dict(center=[-163.1600, -76.0344, 1313.9157], radius=23.0954,
                      axis_point=[-198.0894, -69.5927, 1162.7306],
                      axis_dir=[0.1394, -0.0747, 0.9874])
PAIR_SLIDE12_2 = dict(center=[80.4906, -13.2714, 121.7528], radius=22.2809,
                      axis_point=[70.5923, -1.7971, 64.2265],
                      axis_dir=[0.1353, -0.0479, -0.9896])


def _random_rotation(rng):
    """Rotación uniforme vía QR, con det = +1."""
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def _random_chord(rng, radius):
    """Dos puntos aleatorios distintos sobre la esfera de radio `radius`."""
    def pt():
        v = rng.normal(size=3)
        return radius * v / np.linalg.norm(v)
    a, b = pt(), pt()
    while np.linalg.norm(a - b) < 1e-3 * radius:
        b = pt()
    return a, b


# ---------------------------------------------------------------------------
# Ejercicio diapositiva 4 — O(3) ⊂ O(3,1)
# ---------------------------------------------------------------------------

def test_embed_rotation():
    """Si R ∈ O(3) entonces diag(R,1) ∈ O(3,1). Chequeo numérico de la demostración."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        M = embed_rotation(_random_rotation(rng))
        ok, dev = is_in_o31(M)
        assert ok, f"diag(R,1) no cayó en O(3,1): desviación {dev:.3e}"
        worst = max(worst, dev)
    assert worst < 1e-12, f"desviación máxima {worst:.3e} demasiado grande"
    return f"200 rotaciones aleatorias, ‖MᵀηM − η‖ ≤ {worst:.2e}"


# ---------------------------------------------------------------------------
# Ejercicio diapositivas 18-19 — el boost, TEST DE ORO
# ---------------------------------------------------------------------------

def test_boost_ground_truth():
    """Reproduce la matriz publicada en la diapositiva 19, entrada por entrada."""
    res = boost_fixing_sphere(A1, B1, A2, B2, RHO)
    M = res["M"]

    k = math.sqrt(21.0)
    expected = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 31 * k / 84, -25 * k / 84],
        [0.0, 0.0, -25 * k / 84, 31 * k / 84],
    ])
    assert np.allclose(M, expected, atol=1e-9), \
        f"M no coincide con la publicada:\n{M}\nesperada:\n{expected}"

    ok, dev = is_in_o31(M)
    assert ok and dev < 1e-12, f"‖MᵀηM − η‖ = {dev:.3e}"

    proper, det, m33 = is_proper_orthochronous(M)
    assert proper, f"no es propia ortócrona: det={det:.6f}, M[3,3]={m33:.6f}"

    # cosh² − sinh² = 1 en el bloque hiperbólico
    ch, sh = M[2, 2], M[2, 3]
    assert abs(ch * ch - sh * sh - 1.0) < 1e-12

    # A1 → A2 y B1 → B2, proyectivamente, con el factor 4/√21
    mapped, w, n_sing = apply_projective(M, [A1, B1], RHO)
    assert n_sing == 0
    assert np.allclose(mapped, np.vstack([A2, B2]), atol=1e-9), f"mapeo incorrecto:\n{mapped}"
    factor = RHO / w[0]
    assert abs(factor - 4.0 / k) < 1e-12, f"factor proyectivo {factor:.12f} ≠ {4.0 / k:.12f}"

    return (f"M reproducida a {np.max(np.abs(M - expected)):.1e}, "
            f"‖MᵀηM − η‖ = {dev:.1e}, factor = {factor:.12f}")


def test_boost_preserves_sphere():
    """500 puntos de la esfera siguen sobre la esfera tras el boost."""
    rng = np.random.default_rng(1)
    M = boost_fixing_sphere(A1, B1, A2, B2, RHO)["M"]
    v = rng.normal(size=(500, 3))
    pts = RHO * v / np.linalg.norm(v, axis=1)[:, None]
    mapped, _, n_sing = apply_projective(M, pts, RHO)
    assert n_sing == 0
    err = float(np.max(np.abs(np.linalg.norm(mapped, axis=1) - RHO)))
    assert err < 1e-9, f"la esfera no se preserva: error máximo {err:.3e}"
    return f"500 puntos, |‖p′‖ − ρ| ≤ {err:.2e}"


def test_boost_line_to_line():
    """La imagen de la cuerda sigue siendo una recta, y es la cuerda destino."""
    M = boost_fixing_sphere(A1, B1, A2, B2, RHO)["M"]
    ts = np.linspace(-1.0, 2.0, 200)
    pts = A1 + ts[:, None] * (B1 - A1)
    mapped, _, n_sing = apply_projective(M, pts, RHO)
    assert n_sing == 0

    # Colinealidad: el rango de la nube centrada debe ser 1.
    centered = mapped - mapped.mean(axis=0)
    sv = np.linalg.svd(centered, compute_uv=False)
    assert sv[1] < 1e-9 * max(sv[0], 1.0), f"la imagen no es colineal: σ₂ = {sv[1]:.3e}"

    # Y esa recta es la que pasa por A2 y B2.
    gap = line_line_distance(mapped[0], mapped[-1] - mapped[0], A2, B2 - A2)
    assert gap < 1e-9, f"la recta imagen no es la destino: hueco {gap:.3e}"
    return f"200 puntos colineales (σ₂ = {sv[1]:.1e}), hueco a la recta destino {gap:.1e}"


def test_boost_generalizes():
    """
    Sobre cuerdas aleatorias, la construcción se mantiene en O(3,1) y mapea bien —
    y la receta de las diapositivas 20-21 da EXACTAMENTE la misma matriz.

    Además comprueba que el paso intermedio de la receta (transportar s1, t1 por M0)
    es redundante: sale M0 == M.
    """
    rng = np.random.default_rng(3)
    worst_dev = worst_err = worst_diff = worst_m0 = 0.0
    for _ in range(30):
        a1, b1 = _random_chord(rng, RHO)
        a2, b2 = _random_chord(rng, RHO)

        res = boost_fixing_sphere(a1, b1, a2, b2, RHO)
        ok, dev = is_in_o31(res["M"])
        assert ok, f"se salió de O(3,1): {dev:.3e}"
        worst_dev = max(worst_dev, dev)

        mapped, _, _ = apply_projective(res["M"], [a1, b1], RHO)
        worst_err = max(worst_err, float(np.max(np.abs(mapped - np.vstack([a2, b2])))))

        pptx = boost_fixing_sphere_pptx_recipe(a1, b1, a2, b2, RHO)
        assert pptx["o31_dev"] < 1e-9, f"la receta del pptx se salió del grupo: {pptx['o31_dev']:.3e}"
        worst_diff = max(worst_diff, float(np.max(np.abs(pptx["M"] - res["M"]))))
        worst_m0 = max(worst_m0, float(np.max(np.abs(pptx["M0"] - pptx["M"]))))

    assert worst_err < 1e-8, f"mapeo incorrecto: {worst_err:.3e}"
    assert worst_diff < 1e-9, f"la receta del pptx difiere: {worst_diff:.3e}"
    assert worst_m0 < 1e-9, f"M0 debería ser igual a M: {worst_m0:.3e}"
    return (f"30 cuerdas aleatorias · ‖MᵀηM − η‖ ≤ {worst_dev:.1e} · mapeo ≤ {worst_err:.1e} · "
            f"receta del pptx idéntica ({worst_diff:.1e}) y M0 == M ({worst_m0:.1e})")


def test_pptx_recipe_handmade_vectors_are_the_completion():
    """
    Los cuatro vectores escritos a mano en la diapositiva 21 no son valores mágicos:
    son exactamente la completación η-ortonormal de cada marco. Con ellos, la receta
    da la matriz publicada.
    """
    s1 = np.array([0.0, 1.0, 0.0, 0.0])
    t1 = np.array([0.0, 0.0, 5.0, 2.0]) / math.sqrt(21.0)
    s20 = np.array([0.0, 1.0, 0.0, 0.0])
    t20 = np.array([0.0, 0.0, 5.0, -3.0]) / 4.0

    # Son η-unitarios spacelike y η-ortogonales a su propio marco.
    for (s, t, A, B) in ((s1, t1, A1, B1), (s20, t20, A2, B2)):
        f = lorentz_frame_for_chord(A, B, RHO)
        for vec in (s, t):
            assert abs(minkowski_dot(vec, vec) - 1.0) < 1e-12, "no es spacelike unitario"
            for other in (f["u"], f["v"]):
                assert abs(minkowski_dot(vec, other)) < 1e-12, "no es η-ortogonal al marco"
        assert abs(minkowski_dot(s, t)) < 1e-12, "s y t no son η-ortogonales entre sí"

    res = boost_fixing_sphere_pptx_recipe(A1, B1, A2, B2, RHO,
                                          s1=s1, t1=t1, s20=s20, t20=t20)
    ref = boost_fixing_sphere(A1, B1, A2, B2, RHO)["M"]
    diff = float(np.max(np.abs(res["M"] - ref)))
    assert diff < 1e-9, f"debería coincidir con la matriz publicada; difiere en {diff:.3e}"
    assert res["o31_dev"] < 1e-12, f"‖MᵀηM − η‖ = {res['o31_dev']:.3e}"
    return f"los 4 vectores del pptx son la completación η-ortonormal; M coincide a {diff:.1e}"


def test_wrong_completion_is_a_silent_failure():
    """
    La trampa: usar en F20 la completación del marco 1 en vez de la del marco 2.

    El resultado sigue mandando A1→A2 y B1→B2 con error ~1e-15, pero se sale de
    O(3,1). Es un fallo silencioso, y por eso boost_fixing_sphere verifica la
    pertenencia al grupo antes de devolver nada.
    """
    f1 = lorentz_frame_for_chord(A1, B1, RHO)
    res = boost_fixing_sphere_pptx_recipe(A1, B1, A2, B2, RHO,
                                          s20=f1["s"], t20=f1["t"])
    mapped, _, _ = apply_projective(res["M"], [A1, B1], RHO)
    map_err = float(np.max(np.abs(mapped - np.vstack([A2, B2]))))
    assert map_err < 1e-8, f"aun así debería mapear bien: {map_err:.3e}"
    assert res["o31_dev"] > 1e-3, (
        f"debería salirse del grupo, pero ‖MᵀηM − η‖ = {res['o31_dev']:.3e}")

    # Y la función buena rechaza ese tipo de matriz.
    ok, _ = is_in_o31(res["M"])
    assert not ok
    return (f"mapea bien ({map_err:.1e}) pero ‖MᵀηM − η‖ = {res['o31_dev']:.2f}: "
            "fallo silencioso capturado por el assert de O(3,1)")


def test_frame_completion():
    """El marco η-ortonormal cumple Fᵀ η F = η y su inversa de Lorentz es correcta."""
    f = lorentz_frame_for_chord(A1, B1, RHO)
    F = f["F"]
    dev = float(np.max(np.abs(F.T @ ETA @ F - ETA)))
    assert dev < 1e-12, f"el marco no es η-ortonormal: {dev:.3e}"
    inv_dev = float(np.max(np.abs(lorentz_inverse(F) @ F - np.eye(4))))
    assert inv_dev < 1e-12, f"η Fᵀ η no es la inversa: {inv_dev:.3e}"

    # El 2-plano generado por (s, t) es el mismo que el de la diapositiva 20.
    ref = np.column_stack([np.array([0.0, 1.0, 0.0, 0.0]),
                           np.array([0.0, 0.0, 5.0, 2.0]) / math.sqrt(21.0)])
    got = np.column_stack([f["s"], f["t"]])
    P_ref = ref @ np.linalg.pinv(ref)
    P_got = got @ np.linalg.pinv(got)
    plane_dev = float(np.max(np.abs(P_ref - P_got)))
    assert plane_dev < 1e-9, f"el 2-plano de completación no coincide: {plane_dev:.3e}"
    return f"Fᵀ η F = η a {dev:.1e}; el 2-plano (s,t) coincide con el del pptx a {plane_dev:.1e}"


def test_minkowski_normalization():
    """El vector 'common' es timelike: normalizarlo sin cuidado daría NaN."""
    nA = np.append(A1, RHO)
    nB = np.append(B1, RHO)
    common = (nA + nB) / 2.0
    n2 = minkowski_dot(common, common)
    assert n2 < 0, f"se esperaba norma negativa (timelike), salió {n2:.6f}"
    assert abs(n2 + 21.0) < 1e-9, f"en este ejemplo debería valer −21, salió {n2:.6f}"
    v, sign = normalize_minkowski(common)
    assert sign < 0 and abs(minkowski_dot(v, v) + 1.0) < 1e-12
    assert np.all(np.isfinite(v)), "salieron NaN al normalizar"
    try:
        normalize_minkowski(np.append(A1, RHO))       # vector nulo
        raise AssertionError("normalizar un vector nulo debería fallar explícitamente")
    except ValueError:
        pass
    return f"⟨common, common⟩ = {n2:.1f} < 0, normalizado sin NaN y el nulo falla limpio"


def test_gauge_family():
    """
    boost_along_chord fija la esfera y la recta, deja FIJOS los dos puntos de corte
    (son sus direcciones nulas) y desliza los puntos interiores a lo largo de la recta.
    """
    interior = A1 + np.array([0.2, 0.5, 0.8])[:, None] * (B1 - A1)
    max_shift = 0.0
    for theta in np.linspace(-0.8, 0.8, 9):
        G = boost_along_chord(theta, A1, B1, RHO)
        ok, dev = is_in_o31(G)
        assert ok, f"el gauge se salió de O(3,1): {dev:.3e}"

        # Los puntos de corte quedan fijos.
        ends, _, n_sing = apply_projective(G, [A1, B1], RHO)
        assert n_sing == 0
        end_err = float(np.max(np.abs(ends - np.vstack([A1, B1]))))
        assert end_err < 1e-9, f"los puntos de corte deberían quedar fijos: {end_err:.3e}"

        # Los interiores siguen sobre la recta...
        moved, _, _ = apply_projective(G, interior, RHO)
        gap = max(point_line_distance(p, A1, B1 - A1) for p in moved)
        assert gap < 1e-9, f"el gauge sacó los puntos de la recta: {gap:.3e}"
        # ...pero se han deslizado sobre ella.
        max_shift = max(max_shift, float(np.max(np.abs(moved - interior))))

    assert max_shift > 1e-3, "el gauge no movió nada: la familia sería trivial"
    return (f"9 valores de θ: cortes fijos, recta preservada, "
            f"deslizamiento interior hasta {max_shift:.3f} mm")


def test_line_sphere_intersections():
    """Los cortes recta-esfera caen sobre la esfera y sobre la recta."""
    A, B = line_sphere_intersections([0, 0, 0], RHO, [3, 0, 0], [0, 0, 1])
    assert A is not None
    for p in (A, B):
        assert abs(np.linalg.norm(p) - RHO) < 1e-12
        assert point_line_distance(p, [3, 0, 0], [0, 0, 1]) < 1e-12
    # Recta que no corta.
    assert line_sphere_intersections([0, 0, 0], RHO, [9, 0, 0], [0, 0, 1])[0] is None
    return "cortes sobre la esfera y sobre la recta; el caso no secante devuelve None"


# ---------------------------------------------------------------------------
# Ejercicios diapositivas 11-13, 17 y 24 — similaridades
# ---------------------------------------------------------------------------

def test_similarity_slide12():
    """La similaridad de la diapositiva 12, con sus datos exactos."""
    p1 = make_pair(**PAIR_SLIDE12_1)
    p2 = make_pair(**PAIR_SLIDE12_2)
    res = similarity_from_pairs(p1, p2)

    assert res["orth_dev"] < 1e-12, f"R no es ortogonal: {res['orth_dev']:.3e}"
    assert abs(abs(res["det"]) - 1.0) < 1e-9, f"|det R| = {abs(res['det']):.9f}"
    assert abs(res["s"] - 22.2809 / 23.0954) < 1e-12

    # T lleva el centro 1 exactamente al centro 2...
    err = float(np.linalg.norm(apply_similarity(p1["center"], res["s"], res["R"], res["b"])[0]
                               - p2["center"]))
    assert err < 1e-9, f"T(C₁) ≠ C₂: error {err:.3e} mm"

    # ...y la esfera 1 exactamente en la esfera 2.
    rng = np.random.default_rng(5)
    v = rng.normal(size=(300, 3))
    on_s1 = p1["center"] + p1["radius"] * v / np.linalg.norm(v, axis=1)[:, None]
    img = apply_similarity(on_s1, res["s"], res["R"], res["b"])
    rad_err = float(np.max(np.abs(np.linalg.norm(img - p2["center"], axis=1) - p2["radius"])))
    assert rad_err < 1e-9, f"la esfera imagen no es S₂: error {rad_err:.3e} mm"

    # El eje imagen es paralelo al eje 2 pero NO coincide.
    assert res["parallel_dev"] < 1e-9, f"el eje imagen no es paralelo: {res['parallel_dev']:.3e}"
    return (f"s = {res['s']:.6f}, det R = {res['det']:+.6f}, ‖RᵀR − I‖ = {res['orth_dev']:.1e}, "
            f"hueco T(L₁)↔L₂ = {res['axis_gap']:.4f} mm")


def test_ratio_invariant():
    """R/d es invariante bajo similaridades; si difiere, no hay similaridad posible."""
    p1 = make_pair(**PAIR_SLIDE12_1)
    p2 = make_pair(**PAIR_SLIDE12_2)
    inv = pair_similarity_invariant(p1, p2)

    # Caso construido a propósito para que SÍ se cumpla: par 2 = imagen exacta del par 1.
    s, R = 1.7, _random_rotation(np.random.default_rng(9))
    b = np.array([12.0, -3.0, 5.0])
    p3 = make_pair(s * (R @ p1["center"]) + b, s * p1["radius"],
                   s * (R @ p1["axis_point"]) + b, R @ p1["axis_dir"])
    inv_ok = pair_similarity_invariant(p1, p3)
    assert inv_ok["equal"], f"la razón debería preservarse: {inv_ok['rel_diff']:.3e}"
    assert inv_ok["gap_mm"] < 1e-9, f"el hueco debería ser 0, salió {inv_ok['gap_mm']:.3e}"

    res = similarity_from_pairs(p1, p3)
    assert res["axis_gap"] < 1e-9, f"T(L₁) debería coincidir con L₃: {res['axis_gap']:.3e}"

    return (f"pptx diapositiva 12: R/d = {inv['ratio1']:.4f} vs {inv['ratio2']:.4f} "
            f"(dif. rel. {inv['rel_diff']:.2%}, hueco {inv['gap_mm']:.4f} mm) · "
            f"par transformado a propósito: razón preservada y hueco nulo")


def test_axis_parallel_always_exists():
    """Diapositiva 17a: la similaridad con eje paralelo existe SIEMPRE."""
    rng = np.random.default_rng(11)
    worst_par, worst_center = 0.0, 0.0
    for _ in range(50):
        p1 = make_pair(rng.normal(size=3) * 20, abs(rng.normal(25, 4)) + 5,
                       rng.normal(size=3) * 20, rng.normal(size=3))
        p2 = make_pair(rng.normal(size=3) * 20, abs(rng.normal(25, 4)) + 5,
                       rng.normal(size=3) * 20, rng.normal(size=3))
        res = similarity_axis_parallel(p1, p2)
        assert check_orthogonality(res["R"])["is_rotation"], "R no es una rotación"
        worst_par = max(worst_par, res["parallel_dev"])
        worst_center = max(worst_center, res["center_error"])
    assert worst_par < 1e-9 and worst_center < 1e-8
    return f"50 pares aleatorios: paralelismo ≤ {worst_par:.1e}, error de centro ≤ {worst_center:.1e}"


def test_non_similar_example():
    """Diapositiva 24 (vale doble): el contraejemplo y su similaridad paralela."""
    p1, p2, inv, par = build_non_similar_example()
    assert p1["d"] < p1["radius"] and p2["d"] < p2["radius"], "las rectas deben ser secantes"
    assert abs(p1["ratio"] - 5.0 / 3.0) < 1e-12 and abs(p2["ratio"] - 5.0) < 1e-12
    assert not inv["equal"], "el contraejemplo debería violar la igualdad de razones"
    assert inv["gap_mm"] > 1e-6, f"el hueco debería ser no nulo: {inv['gap_mm']:.3e}"

    # Pero sí existe la versión con eje paralelo.
    assert par["parallel_dev"] < 1e-9 and par["center_error"] < 1e-9
    assert abs(par["distance_gap"] - inv["gap_mm"]) < 1e-9

    # Y ninguna similaridad cierra el hueco: la de la receta tampoco.
    res = similarity_from_pairs(p1, p2)
    assert res["axis_gap"] > 1e-6, "alguna similaridad cerró el hueco: revisar el ejemplo"
    return (f"R₁/d₁ = {inv['ratio1']:.4f} ≠ R₂/d₂ = {inv['ratio2']:.4f}; "
            f"hueco irreducible {inv['gap_mm']:.4f} mm; la versión paralela sí existe")


def test_frame_from_pair_degenerate():
    """Si el centro está sobre el eje, el marco se marca degenerado y no revienta."""
    p = make_pair([0, 0, 0], 10.0, [0, 0, 0], [0, 0, 1])
    f = frame_from_pair(p)
    assert f["degenerate"], "debería marcarse degenerado"
    assert check_orthogonality(f["M"])["is_orthogonal"], "el marco debe seguir siendo ortogonal"
    return "centro sobre el eje: marco degenerado marcado, ortogonalidad preservada"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("Ej-4    O(3) ⊂ O(3,1)", test_embed_rotation),
    ("Ej-18/19 GROUND TRUTH del boost", test_boost_ground_truth),
    ("Ej-18   el boost preserva la esfera", test_boost_preserves_sphere),
    ("Ej-18   la cuerda va a la cuerda", test_boost_line_to_line),
    ("Ej-19   generaliza y coincide con la receta", test_boost_generalizes),
    ("Ej-19   los vectores del pptx son la completación",
     test_pptx_recipe_handmade_vectors_are_the_completion),
    ("Ej-19   completación equivocada = fallo silencioso",
     test_wrong_completion_is_a_silent_failure),
    ("Ej-19   marco η-ortonormal e inversa", test_frame_completion),
    ("Ej-19   normalización timelike sin NaN", test_minkowski_normalization),
    ("Ej-19   familia de gauge", test_gauge_family),
    ("aux     cortes recta-esfera", test_line_sphere_intersections),
    ("Ej-11/12 similaridad de la diapositiva 12", test_similarity_slide12),
    ("Ej-13   invariante R/d", test_ratio_invariant),
    ("Ej-17a  similaridad con eje paralelo", test_axis_parallel_always_exists),
    ("Ej-24   contraejemplo (vale doble)", test_non_similar_example),
    ("aux     marco degenerado", test_frame_from_pair_degenerate),
]


def run_all_tests(verbose=True):
    """Corre todos los tests. Devuelve (n_pass, n_fail)."""
    n_pass = n_fail = 0
    print("=" * 78)
    print("🔬 VERIFICACIÓN DEL NÚCLEO GEOMÉTRICO")
    print("=" * 78)
    for name, fn in TESTS:
        try:
            detail = fn()
            n_pass += 1
            print(f"  ✅ {name}")
            if verbose and detail:
                print(f"       {detail}")
        except Exception as exc:                                  # noqa: BLE001
            n_fail += 1
            print(f"  ❌ {name}")
            print(f"       {type(exc).__name__}: {exc}")
    print("-" * 78)
    icon = "🎉" if n_fail == 0 else "🚨"
    print(f"  {icon} {n_pass} pasaron, {n_fail} fallaron")
    print("=" * 78)
    return n_pass, n_fail
