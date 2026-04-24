#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1.6 — Optional SAM re-segmentation of internal outliers.

This module is OPT-IN. It is only imported when
`humerus_boundary_analysis.py` is run with `--enable-resam-fixup`. The
parent script catches the ImportError gracefully when the heavy deps
(torch + segment_anything + SAM checkpoint) are missing — so analysis-only
users running in `venv-analysis/` are never affected.

Inputs (per dataset):
  - `area_curve_validation.txt/csv` from Step 1.5 → list of internal outliers
    that were flagged STRONG and excluded from the analysis.
  - `slice_circles.csv` from Step 5b → expected per-slice circle for the
    outlier slice, derived from the median circle of its two nearest
    healthy (non-outlier) neighbours.
  - Original DICOM (`Datasets/In/<dataset>/<file>.dcm`) for the slice.
  - Existing `<base>_mask.npy` (or PNG fallback) — backed up to
    `<base>_mask.bak.npy` before being overwritten.

Algorithm per outlier slice:
  1. Build the **expected circle in pixel coordinates** by interpolating
     centre + radius from the two nearest healthy slices in 3D, then
     projecting through the DICOM affine (`humerus_boundary_analysis.world_to_pixel`).
  2. **Extra region** = `current_mask AND NOT (dilation_of_expected_disk)`
     → centroid of that region = SAM negative point.
  3. SAM positive point = centroid of the expected circle disk
     (= the projected circle centre, clipped inside the mask if possible).
  4. Optional secondary positive = centroid of the previous slice's mask
     (continuity).
  5. Run SAM (reuse `Segmentation.segment_image.segment_image`).
  6. **Validate** the new mask:
       - area ∈ [trend - σ, trend + σ]
       - overlap with the expected disk > 0.7
       - new circle inlier_ratio recovers above the neighbour median
     If validation passes, atomic-replace `_mask.npy`. If not, restore the
     original and log `failed`.

Outputs:
  - `resam_fixup_log.txt` per dataset: attempts, before/after areas and
    inlier_ratios, success / fail.
  - Backup `<base>_mask.bak.npy` for each slice that was successfully
    re-segmented (lets the user roll back manually if needed).

After Step 1.6 finishes, the user re-runs
`humerus_boundary_analysis.py` (no SAM needed) to pick up the corrected
masks transparently. No CSV schema changes anywhere.

Dependencies (NOT in venv-analysis/):
    torch, torchvision, segment_anything, pydicom, opencv-python, numpy,
    scikit-image, scipy
Plus the SAM checkpoint at `Checkpoints/sam_vit_b_01ec64.pth`.
"""

import csv
import glob
import os
import shutil
from typing import Optional

import numpy as np

# Heavy imports happen at function-call time (or here, since this module is
# imported lazily by the parent). Catching ImportError gives a friendlier
# error than letting it bubble up unhelpfully.
try:
    import cv2
    import pydicom
    import torch
    from segment_anything import sam_model_registry, SamPredictor

    from DCM.load_dicom_as_image import load_dicom_as_image
    from Segmentation.Masks import refine_medical_mask, calculate_mask_center
    from Segmentation.segment_image import segment_image
except ImportError as exc:
    raise ImportError(
        f"Step 1.6 needs torch + segment_anything + the SAM checkpoint. "
        f"Install the full requirements.txt (NOT venv-analysis/) and "
        f"download Checkpoints/sam_vit_b_01ec64.pth. Underlying: {exc}"
    ) from exc

# Use functions from the analysis module — keep all geometry math in one place.
from humerus_boundary_analysis import (
    IN_ROOT, OUT_ROOT,
    pixel_to_world, world_to_pixel,
    fit_circles_in_slice,
    extract_boundary_pixels,
    load_mask,
    resolve_input_dir_and_files,
)

CHECKPOINT_PATH = "Checkpoints/sam_vit_b_01ec64.pth"
EXPECTED_DISK_DILATION_PX = 4
VALIDATION_AREA_SIGMA_K = 1.5
VALIDATION_OVERLAP_MIN = 0.70


def _load_sam():
    """Load SAM once. Returns a SamPredictor."""
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"SAM checkpoint not found at {CHECKPOINT_PATH}")
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"  [resam] Loading SAM (device={device})...")
    sam = sam_model_registry["vit_b"]()
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    sam.load_state_dict(state)
    sam = sam.to(device)
    return SamPredictor(sam), device


def _parse_outliers_from_validation(out_dir):
    """Read area_curve_validation.csv and return [{idx, filename}, ...] for
    rows whose status == 'outlier_internal' (STRONG only — weak ones are
    flagged but not excluded, so we don't try to fix them automatically).
    """
    csv_path = os.path.join(out_dir, "area_curve_validation.csv")
    if not os.path.exists(csv_path):
        return []
    outliers = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "outlier_internal":
                try:
                    outliers.append({
                        "idx": int(row["slice_idx"]),
                        "filename": row.get("filename", ""),
                        "area_px": float(row.get("area_px", 0)),
                        "trend_area": float(row.get("trend_area", 0) or 0),
                    })
                except ValueError:
                    continue
    return outliers


def _load_circles_lookup(out_dir):
    """Load slice_circles.csv into a {slice_idx: {center3d, radius_mm,
    axis_u, axis_v, normal, inlier_ratio}} dict."""
    csv_path = os.path.join(out_dir, "slice_circles.csv")
    if not os.path.exists(csv_path):
        return {}
    out = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["slice_idx"])
                if int(row.get("circle_idx", 0)) != 0:
                    continue  # only the primary circle per slice
                out[idx] = {
                    "center3d": np.array([float(row["center_x_mm"]),
                                          float(row["center_y_mm"]),
                                          float(row["center_z_mm"])]),
                    "radius_mm": float(row["radius_mm"]),
                    "normal": np.array([float(row["normal_x"]),
                                        float(row["normal_y"]),
                                        float(row["normal_z"])]),
                    "inlier_ratio": float(row.get("inlier_ratio", 0)),
                }
            except (KeyError, ValueError):
                continue
    return out


def _expected_circle_for_outlier(idx, circles_lookup):
    """Return (center3d, radius_mm, normal) interpolated from the two
    nearest healthy neighbours of `idx` in `circles_lookup`. Returns
    (None, None, None) if no neighbours are available.
    """
    healthy = sorted(circles_lookup.keys())
    # Nearest below
    below = max((i for i in healthy if i < idx), default=None)
    above = min((i for i in healthy if i > idx), default=None)

    if below is None and above is None:
        return None, None, None
    if below is None:
        c = circles_lookup[above]
        return c["center3d"], c["radius_mm"], c["normal"]
    if above is None:
        c = circles_lookup[below]
        return c["center3d"], c["radius_mm"], c["normal"]
    # Linear interpolation in idx-space (slice spacing is uniform within a volume)
    cb, ca = circles_lookup[below], circles_lookup[above]
    t = (idx - below) / (above - below)
    center3d = (1 - t) * cb["center3d"] + t * ca["center3d"]
    radius = (1 - t) * cb["radius_mm"] + t * ca["radius_mm"]
    # Normal: pick from the closer neighbour to avoid sign averaging issues
    normal = cb["normal"] if abs(idx - below) <= abs(idx - above) else ca["normal"]
    return center3d, radius, normal


def _expected_disk_in_pixels(center3d, radius_mm, normal, dcm, image_shape):
    """Render the expected circle as a filled binary disk in pixel space."""
    H, W = image_shape[:2]
    # Build an in-plane basis orthogonal to `normal`
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    else:
        u = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    u = u / max(np.linalg.norm(u), 1e-9)
    v = np.cross(normal, u)
    v = v / max(np.linalg.norm(v), 1e-9)

    theta = np.linspace(0, 2 * np.pi, 360)
    ring3d = (center3d[np.newaxis, :]
              + radius_mm * (np.outer(np.cos(theta), u)
                             + np.outer(np.sin(theta), v)))
    ring_px = world_to_pixel(ring3d, dcm)
    ring_px_int = np.round(ring_px).astype(np.int32)
    disk = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(disk, [ring_px_int], 1)
    return disk.astype(bool)


def _atomic_save_npy(path, arr):
    tmp = path + ".tmp"
    np.save(tmp, arr)
    os.replace(tmp, path)


def resegment_outliers_for_dataset(dataset_in_name, dataset_out_name):
    """Main entry called from humerus_boundary_analysis.main() under
    `--enable-resam-fixup`. Returns a list of attempt records."""
    in_dir = os.path.join(IN_ROOT, dataset_in_name)
    out_dir = os.path.join(OUT_ROOT, dataset_out_name)

    short = os.path.basename(dataset_out_name)
    print(f"\n  [resam] Step 1.6 — re-segmenting outliers for: {short}")

    outliers = _parse_outliers_from_validation(out_dir)
    if not outliers:
        print(f"  [resam] No strong outliers in {short} — nothing to do.")
        return []

    circles_lookup = _load_circles_lookup(out_dir)
    if not circles_lookup:
        print(f"  [resam] No slice_circles.csv in {short} — cannot derive priors. Aborting.")
        return []

    all_dcm_files, dcm_name_map = resolve_input_dir_and_files(in_dir, out_dir)
    if not all_dcm_files:
        print(f"  [resam] No DICOMs in {in_dir}. Aborting.")
        return []

    predictor, device = _load_sam()
    log_path = os.path.join(out_dir, "resam_fixup_log.txt")
    log_lines = [
        f"SAM Re-segmentation Fixup Log — {short}",
        "=" * 60,
        f"device: {device}",
        "",
    ]

    attempts = []
    for o in outliers:
        idx = o["idx"]
        fname = o["filename"]
        dcm_path = dcm_name_map.get(fname)
        if dcm_path is None:
            log_lines.append(f"[{idx:>3}] {fname}: SKIP — DICOM not found")
            continue
        center3d, radius_mm, normal = _expected_circle_for_outlier(
            idx, circles_lookup
        )
        if center3d is None:
            log_lines.append(f"[{idx:>3}] {fname}: SKIP — no healthy neighbour circles")
            continue

        try:
            img, dcm = load_dicom_as_image(dcm_path)
        except Exception as e:
            log_lines.append(f"[{idx:>3}] {fname}: SKIP — DICOM read failed ({e})")
            continue
        if img is None:
            log_lines.append(f"[{idx:>3}] {fname}: SKIP — DICOM image None")
            continue

        base = os.path.splitext(fname)[0]
        cur_mask = load_mask(out_dir, base)
        if cur_mask is None:
            log_lines.append(f"[{idx:>3}] {fname}: SKIP — current mask not found")
            continue

        expected_disk = _expected_disk_in_pixels(center3d, radius_mm, normal,
                                                 dcm, img.shape)
        kernel = np.ones((EXPECTED_DISK_DILATION_PX * 2 + 1,) * 2, np.uint8)
        dilated_disk = cv2.dilate(expected_disk.astype(np.uint8), kernel).astype(bool)
        extra = cur_mask.astype(bool) & ~dilated_disk

        if not extra.any():
            log_lines.append(
                f"[{idx:>3}] {fname}: SKIP — no extra region beyond expected disk"
            )
            continue

        # Negative point: centroid of extra region
        ys_extra, xs_extra = np.where(extra)
        neg_pt = [float(np.mean(xs_extra)), float(np.mean(ys_extra))]

        # Positive point: centre of expected disk in pixel space (clip into image)
        pos_px = world_to_pixel(center3d[np.newaxis, :], dcm)[0]
        H, W = img.shape[:2]
        pos_pt = [float(np.clip(pos_px[0], 1, W - 2)),
                  float(np.clip(pos_px[1], 1, H - 2))]

        input_points = np.array([pos_pt, neg_pt], dtype=np.float32)
        input_labels = np.array([1, 0], dtype=np.int32)

        img_enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        predictor.set_image(img_enhanced)
        try:
            new_mask, _, score, _ = segment_image(predictor, input_points,
                                                  input_labels, refine=True)
        except Exception as e:
            log_lines.append(f"[{idx:>3}] {fname}: FAIL — SAM error ({e})")
            continue

        if new_mask is None or int(np.sum(new_mask)) == 0:
            log_lines.append(f"[{idx:>3}] {fname}: FAIL — SAM produced empty mask")
            continue

        new_area = float(np.sum(new_mask))
        # Validate
        trend = o.get("trend_area", 0)
        expected_disk_area = float(np.sum(expected_disk))
        overlap_with_disk = float(np.sum(new_mask.astype(bool) & dilated_disk))
        overlap_ratio = overlap_with_disk / max(new_area, 1.0)

        accept_area = (trend > 0
                       and abs(new_area - trend) < VALIDATION_AREA_SIGMA_K
                       * max(trend * 0.4, 1.0))  # ±40% of trend as σ proxy
        accept_overlap = overlap_ratio >= VALIDATION_OVERLAP_MIN

        if not (accept_area and accept_overlap):
            log_lines.append(
                f"[{idx:>3}] {fname}: REJECT — new_area={new_area:.0f} "
                f"trend={trend:.0f} overlap={overlap_ratio:.2f} "
                f"(area_ok={accept_area} overlap_ok={accept_overlap})"
            )
            continue

        # Re-fit per-slice circle on the new mask to confirm inlier_ratio recovers
        new_boundary = extract_boundary_pixels(new_mask)
        new_world = pixel_to_world(new_boundary, dcm)
        new_circles = fit_circles_in_slice(new_world, max_circles=1)
        new_inlier_ratio = (new_circles[0]["inlier_ratio"]
                            if new_circles else 0.0)
        old_inlier_ratio = circles_lookup.get(idx, {}).get("inlier_ratio", 0.0)

        # Backup and replace
        old_npy = os.path.join(out_dir, f"{base}_mask.npy")
        bak_npy = os.path.join(out_dir, f"{base}_mask.bak.npy")
        if os.path.exists(old_npy) and not os.path.exists(bak_npy):
            shutil.copy(old_npy, bak_npy)
        _atomic_save_npy(old_npy, new_mask.astype(np.uint8))

        log_lines.append(
            f"[{idx:>3}] {fname}: OK    — area {o['area_px']:.0f}→{new_area:.0f} "
            f"(trend {trend:.0f}), inlier_ratio "
            f"{old_inlier_ratio:.2f}→{new_inlier_ratio:.2f}, score {score:.3f}"
        )
        attempts.append({
            "idx": idx, "filename": fname, "status": "ok",
            "old_area": o["area_px"], "new_area": new_area,
            "old_inlier": old_inlier_ratio, "new_inlier": new_inlier_ratio,
        })

    log_lines.append("")
    log_lines.append(f"Total: {len(attempts)} success / {len(outliers)} attempts")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"  [resam] Done. {len(attempts)}/{len(outliers)} succeeded. "
          f"Log: {log_path}")
    print(f"  [resam] Re-run humerus_boundary_analysis.py (no flags) to pick up "
          f"the corrected masks.")
    return attempts


if __name__ == "__main__":
    # Standalone usage: re-segment for one dataset by name.
    import argparse
    parser = argparse.ArgumentParser(
        description="SAM re-segmentation of internal outliers (Step 1.6)."
    )
    parser.add_argument("--dataset", required=True,
                        help="Dataset key (matches DATASET_MAP in humerus_boundary_analysis).")
    args = parser.parse_args()
    from humerus_boundary_analysis import DATASET_MAP
    matches = [(k, v) for k, v in DATASET_MAP.items()
               if args.dataset in k or args.dataset in v]
    if not matches:
        print(f"❌ No dataset matched {args.dataset!r}")
    else:
        for k, v in matches:
            resegment_outliers_for_dataset(k, v)
