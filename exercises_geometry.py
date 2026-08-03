#!/usr/bin/env python3
"""
Runner de los ejercicios de geometría proyectiva del pptx de Marco Paluszny.

No necesita SAM, torch ni GPU: trabaja sobre los CSV que ya produjo
humerus_boundary_analysis.py y sobre los .stl de excercises/.

Uso:
    python exercises_geometry.py --selftest            # verifica el núcleo (Ej-4..24)
    python exercises_geometry.py --std-pair            # par (esfera, eje) del standard
    python exercises_geometry.py --calibrate           # sesgo del eje por truncamiento
    python exercises_geometry.py --patient-pairs       # par de cada dataset
    python exercises_geometry.py --transfer            # actividad completa (dia. 29)
    python exercises_geometry.py --all                 # todo lo anterior
"""

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
STD_DIR = os.path.join(ROOT, "excercises")
OUT_ROOT = os.path.join(ROOT, "Datasets", "Out")
STD_MESH = os.path.join(STD_DIR, "Humero.stl")


def _jsonable(obj):
    """Convierte arrays de numpy y tipos numpy a algo serializable."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k not in ("triangles",)}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _dump(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_jsonable(data), fh, indent=2, ensure_ascii=False)
    return path


def cmd_selftest():
    from Geometry.selftest import run_all_tests
    _, n_fail = run_all_tests()
    return 1 if n_fail else 0


def cmd_std_pair():
    from Geometry.similarity import pair_report
    from Geometry.sphere_axis import sphere_axis_from_mesh
    print("=" * 78)
    print("🦴 PAR (ESFERA, EJE) DEL HÚMERO STANDARD — diapositiva 8")
    print("=" * 78)
    pair = sphere_axis_from_mesh(STD_MESH)
    print(pair_report(pair))
    print(f"  📊 rms esfera {pair['rms_sphere']:.3f} mm | rms eje {pair['rms_axis']:.3f} mm "
          f"| losas {pair['n_slabs_used']} usadas, {pair['n_slabs_rejected']} descartadas")
    path = _dump(os.path.join(STD_DIR, "out", "standard_frame.json"), pair)
    print(f"  💾 {os.path.relpath(path, ROOT)}")
    return pair


def cmd_calibrate():
    from Geometry.sphere_axis import calibrate_axis_bias_from_mesh
    print("=" * 78)
    print("📐 CALIBRACIÓN DEL SESGO DEL EJE POR TRUNCAMIENTO — pendiente de la dia. 16")
    print("=" * 78)
    cal = calibrate_axis_bias_from_mesh(STD_MESH)
    print(f"  malla completa: R = {cal['radius_true']:.2f} mm, d = {cal['d_true']:.2f} mm, "
          f"R/d = {cal['ratio_true']:.2f}")
    print("   extensión   ángulo     d      d_scale   R/d leído")
    for r in cal["rows"]:
        if r.get("ok"):
            print(f"    {r['z_extent_mm']:6.0f} mm  {r['angle_deg']:6.2f}°  "
                  f"{r['d_trunc']:6.2f}  {r['d_scale']:7.3f}   {r['ratio_trunc']:8.2f}")
        else:
            print(f"    {r['z_extent_mm']:6.0f} mm  -- {r['reason']}")
    print(f"  ⚠️  el propio standard, con cobertura de MRI, APARENTA R/d ≈ "
          f"{cal['ratio_mri_like']:.2f} cuando vale {cal['ratio_true']:.2f}")
    path = _dump(os.path.join(STD_DIR, "out", "axis_bias_calibration.json"), cal)
    print(f"  💾 {os.path.relpath(path, ROOT)}")
    return cal


def cmd_patient_pairs(cal=None):
    from Geometry.similarity import pair_report
    from Geometry.sphere_axis import sphere_axis_from_slice_circles
    print("=" * 78)
    print("🧑‍⚕️  PAR (ESFERA, EJE) DE CADA PACIENTE — diapositiva 25")
    print("=" * 78)
    pairs = {}
    for csv_path in sorted(glob.glob(os.path.join(OUT_ROOT, "*", "slice_circles.csv"))):
        ds = os.path.basename(os.path.dirname(csv_path))
        print("-" * 78)
        print(f"  {ds[:72]}")
        try:
            pair = sphere_axis_from_slice_circles(csv_path, ds)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  🚨 {exc}")
            continue
        print(pair_report(pair))
        print(f"  📊 n_art={pair['n_articular']} n_diáf={pair['n_shaft']} "
              f"σ_eje={pair['axis_sigma_deg']:.2f}°")
        if cal:
            dc = pair["d"] / cal["d_scale_mri_like"]
            print(f"  🔧 corregido por sesgo: d = {dc:.2f} mm, R/d = {pair['radius'] / dc:.2f}")
        print(f"  {'✅ usable' if pair['usable'] else '⛔ excluido'}: "
              f"{pair['quality_flags'] or 'sin banderas'}")
        pairs[ds] = pair
        _dump(os.path.join(OUT_ROOT, ds, "geometry", "patient_frame.json"), pair)
    return pairs


def cmd_transfer(std_pair, pairs):
    from Geometry.sphere_axis import load_slice_circles
    from Geometry.transfer import build_full_transform, map_muscle_stls, modeling_efficacy
    print("=" * 78)
    print("🔀 TRANSPORTE STANDARD → PACIENTE — actividad de la diapositiva 29")
    print("=" * 78)
    reports = {}
    for ds, pair in pairs.items():
        if not pair["usable"]:
            print(f"  ⏭️  {ds[:60]}: excluido por calidad, no se transporta")
            continue
        print("-" * 78)
        print(f"  {ds[:72]}")

        tr = build_full_transform(std_pair, pair)
        chk = tr["checks"]
        print(f"  🧮 R/d standard {tr['ratio_std']:.3f} vs paciente {tr['ratio_pat']:.3f} "
              f"(dif. {tr['ratio_rel_diff']:.1%}) → boost {'SÍ' if tr['needs_boost'] else 'NO'}")
        print(f"  ✔️  esfera → esfera: {chk['sphere_max_err_mm']:.2e} mm | "
              f"eje → eje: {chk['axis_max_err_mm']:.2e} mm | O(3,1): {tr['o31_dev']:.1e}")
        if not chk["ok"]:
            print("  🚨 la verificación de Φ falló, se salta este dataset")
            continue

        out_dir = os.path.join(OUT_ROOT, ds, "geometry", "muscles_mapped")
        mapped = map_muscle_stls(STD_DIR, tr, out_dir)

        boundary = os.path.join(OUT_ROOT, ds, "boundary_world_coords.csv")
        art = {c["slice_idx"] for c in load_slice_circles(
            os.path.join(OUT_ROOT, ds, "slice_circles.csv")) if c["is_articular"]}
        eff = modeling_efficacy(mapped, pair, boundary, art)

        m1 = eff.get("M1_bone_fidelity", {})
        if m1.get("ok"):
            print(f"  📏 M1 fidelidad ósea: RMS {m1['rms_mm']:.2f} mm, "
                  f"mediana {m1['median_mm']:.2f} mm, P95 {m1['p95_mm']:.2f} mm")
            if "rms_articular_mm" in m1:
                print(f"      articulares {m1['rms_articular_mm']:.2f} mm | "
                      f"diafisarias {m1.get('rms_shaft_mm', float('nan')):.2f} mm")
        for name, m2 in eff.get("M2_muscles", {}).items():
            th = m2.get("footprint_theta_deg")
            th_s = f"θ={th:.1f}°" if th is not None else "sin huella"
            print(f"  💪 M2 {name:15s} penetración {m2['penetration_fraction']:.1%}, "
                  f"huella {m2['n_footprint']:5d} vért., {th_s}")

        report = {"dataset": ds, "transform": tr, "efficacy": eff,
                  "muscles": [{k: v for k, v in m.items() if k != "triangles"}
                              for m in mapped]}
        reports[ds] = report
        _dump(os.path.join(OUT_ROOT, ds, "geometry", "transfer_report.json"), report)
    return reports


def cmd_bias_comparison(std_pair, pairs, cal):
    """
    Compara el transporte con el `d` crudo contra el `d` corregido por el sesgo de
    truncamiento. Es la validación cruzada del hallazgo: la corrección se deriva del
    STL standard y se valida contra la nube de borde del PACIENTE, que no intervino
    en derivarla.
    """
    from Geometry.sphere_axis import bias_corrected_pair, load_slice_circles
    from Geometry.stl_io import read_stl
    from Geometry.transfer import (bone_fidelity, build_full_transform,
                                   load_boundary_points, transform_mesh)

    print("=" * 78)
    print("⚖️  CRUDO vs CORREGIDO POR SESGO — validación cruzada")
    print("=" * 78)
    print(f"  d_scale = {cal['d_scale_mri_like']:.3f} ± {cal['d_scale_mri_like_std']:.3f} "
          f"(ventana 58-100 mm, n={cal['n_mri_like']})")
    print(f"  {'dataset':30s} {'variante':10s} {'R/d':>6s} {'w_max/w_min':>11s} "
          f"{'RMS':>6s} {'med':>6s} {'P95':>6s}")

    tri, _ = read_stl(STD_MESH)
    rows = []
    for ds, pair in pairs.items():
        if not pair["usable"]:
            continue
        out_dir = os.path.join(OUT_ROOT, ds)
        bpts, sidx = load_boundary_points(os.path.join(out_dir, "boundary_world_coords.csv"))
        art = {c["slice_idx"] for c in load_slice_circles(
            os.path.join(out_dir, "slice_circles.csv")) if c["is_articular"]}

        for label, p in (("crudo", pair),
                         ("corregido", bias_corrected_pair(pair, cal["d_scale_mri_like"]))):
            tr = build_full_transform(std_pair, p)
            mapped, diag = transform_mesh(tri, tr)
            m1 = bone_fidelity(mapped, bpts, art, sidx)
            print(f"  {ds[:30]:30s} {label:10s} {p['ratio']:6.2f} {diag['w_ratio']:11.2f} "
                  f"{m1['rms_mm']:6.2f} {m1['median_mm']:6.2f} {m1['p95_mm']:6.2f}")
            rows.append({"dataset": ds, "variant": label, "ratio": p["ratio"],
                         "w_ratio": diag["w_ratio"], **{k: m1[k] for k in
                                                        ("rms_mm", "median_mm", "p95_mm")}})

    path = _dump(os.path.join(STD_DIR, "out", "bias_comparison.json"),
                 {"d_scale": cal["d_scale_mri_like"], "rows": rows})
    print(f"  💾 {os.path.relpath(path, ROOT)}")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Ejercicios de geometría proyectiva sobre el húmero")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--std-pair", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--patient-pairs", action="store_true")
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return 0

    if args.selftest or args.all:
        if cmd_selftest():
            return 1

    std_pair = cal = None
    if args.std_pair or args.transfer or args.all:
        std_pair = cmd_std_pair()
    if args.calibrate or args.all:
        cal = cmd_calibrate()
    if args.patient_pairs or args.transfer or args.all:
        pairs = cmd_patient_pairs(cal)
        if (args.transfer or args.all) and std_pair:
            cmd_transfer(std_pair, pairs)
            if cal:
                cmd_bias_comparison(std_pair, pairs, cal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
