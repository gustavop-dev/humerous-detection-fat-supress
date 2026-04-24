#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Humerus Boundary Analysis Pipeline
====================================
Steps:
  1) Classify segmented slices as CORRECT / AMBIGUOUS / FAILED based on Dice.
  2) List ambiguous slices that need manual review.
  3) Extract boundary (contour) pixels from correctly segmented slices.
  4) Convert boundary pixels to real-world 3D coordinates using DICOM headers
     (ImagePositionPatient, ImageOrientationPatient, PixelSpacing).
  5) Detect circular arcs in the 3D boundary point cloud (the humeral head
     articular surface is approximately a spherical cap).
  6) Generate a per-dataset summary diagram.
"""

import os
import glob
import csv
import numpy as np
import pydicom
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for projection='3d'
from itertools import combinations

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
IN_ROOT = os.path.join(ROOT_DIR, "Datasets", "In")
OUT_ROOT = os.path.join(ROOT_DIR, "Datasets", "Out")

# Dice thresholds for classification
DICE_CORRECT_THRESHOLD = 0.80    # Dice >= 0.80  -> CORRECT
DICE_AMBIGUOUS_THRESHOLD = 0.55  # 0.55 <= Dice < 0.80  -> AMBIGUOUS
# Dice < 0.55  -> FAILED

# Minimum points to consider a candidate circular arc on the global sphere
MIN_ARC_POINTS = 10              # bumped from 6 — single-point arcs were spurious
MIN_ARC_INLIER_RATIO = 0.30      # an arc needs at least 30 % of its slice points on the sphere

# RANSAC parameters for the global sphere fit (humeral head cap, "feature 1")
RANSAC_ITERATIONS = 2000
RANSAC_INLIER_DISTANCE = 2.0  # mm

# Per-slice circle fitting ("feature 2": circles lying on the articular surface)
MIN_CIRCLE_INLIERS = 10              # min inliers per per-slice circle
MIN_CIRCLE_INLIER_RATIO = 0.35       # inliers / boundary_points_in_slice
CIRCLE_INLIER_DISTANCE_MM = 1.5      # mm
CIRCLE_RANSAC_ITERATIONS = 600
MAX_CIRCLES_PER_SLICE = 3            # iterative RANSAC: try to find up to N circles per slice
ARTICULAR_AREA_RATIO = 0.70          # slice flagged as articular if area ≥ 70 % of max area

# Step 1.5 — direction-aware area-curve validation
ENABLE_AREA_VALIDATION = True            # master switch (CLI: --no-area-validation)
TREND_WINDOW_HALF = 1                    # rolling median window = 2*half + 1 = 3
BELL_PEAK_MIN_RATIO = 0.50               # peak must be >= this fraction of global max
# Hybrid bell-extent: trim = static-floor-from-edge ∪ reversal-from-peak.
# Static floor catches the "long tail of small areas" pattern at the head
# (e.g. GE slices 1–7 are all in the noise floor). Reversal catches the
# "area went back UP" pattern (e.g. P5SE1 slice 19 reaches above the prior
# slice). On the shaft side, the static floor is much looser because the
# diaphysis tapers gradually below 30 % of peak legitimately.
BELL_BOUNDARY_HEAD_RATIO = 0.30          # head: trim if smoothed < 30% of peak (run-from-edge)
BELL_BOUNDARY_SHAFT_RATIO = 0.10         # shaft: trim if smoothed < 10% of peak (run-from-edge)
BELL_MIN_RUN_LENGTH = 2                  # min run length under threshold for static trim
BELL_REVERSAL_RATIO_HEAD = 1.10          # head: walk from peak, trim where area grows ≥ this factor
BELL_REVERSAL_RATIO_SHAFT = 1.20         # shaft: more tolerant of small bumps (tuberosities)
BELL_REVERSAL_MIN_ABS = 50.0             # ignore reversals with abs jump < this many px (numerical wobble)
INTERNAL_OUTLIER_RESID_SIGMA = 2.5       # robust z-score threshold for area-side outlier
INTERNAL_OUTLIER_INLIER_DROP = 0.15      # circle inlier_ratio drop vs neighbours
INTERNAL_OUTLIER_DICE_DROP = 0.10        # Dice drop vs neighbours
INTERNAL_OUTLIER_MIN_SIGMA_PX = 50.0     # MAD floor in pixel units — avoids σ blow-up on smooth curves
DIRECTION_UNKNOWN_USES_SHAFT = True      # safe default if DICOM direction is missing

# CLI overrides (populated by main()'s argparse). Module-level so process_dataset
# can read them without changing its signature.
_CLI_OVERRIDES = {
    "head_trim_ratio": None,
    "shaft_trim_ratio": None,
    "no_area_validation": False,
    "enable_resam_fixup": False,
    "dataset": None,
}

# Dataset mapping  (input subfolder -> output subfolder)
DATASET_MAP = {
    "axial_A_to_T_fatSupressed_dicoms-20260214T164130Z-1-001/axial_A_to_T_fatSupressed_dicoms":
        "axial_A_to_T_fatSupressed_dicoms-20260214T164130Z-1-001",
    "AXIAL fs PD FSE 256x256 GE MEDICAL SYSTEMS Optima MR360 CLINICA SANTA ANA hombroDerecho":
        "AXIAL fs PD FSE 256x256 GE MEDICAL SYSTEMS Optima MR360 CLINICA SANTA ANA hombroDerecho",
    "AXIAL-P5SE1 Clinica Especialistas Reina Virgen Maria QUIBDO HombroDer pd_tse_fs_tra_320 SIEMENS":
        "AXIAL-P5SE1 Clinica Especialistas Reina Virgen Maria QUIBDO HombroDer pd_tse_fs_tra_320 SIEMENS",
    "pd_tse_fs_tra_320_fov150_4":
        "pd_tse_fs_tra_320_fov150_4",
    "t1_tse_fs_tra_320_fov150_5":
        "t1_tse_fs_tra_320_fov150_5",
    "t1_tse_fs_tra_320_fov150_11":
        "t1_tse_fs_tra_320_fov150_11",
}


# ===================================================================
# Step 1 & 2 – Parse propagation_summary.txt -> classify slices
# ===================================================================

def parse_summary(summary_path):
    """Return list of dicts with per-slice info parsed from propagation_summary.txt."""
    slices = []
    failed_names = set()
    in_results = False
    in_failed = False

    with open(summary_path, "r") as f:
        for line in f:
            line = line.rstrip("\n")

            # Detect failed-slices section
            if line.startswith("Failed slices:"):
                in_failed = True
                continue
            if in_failed:
                if line.strip().startswith("- ["):
                    # e.g. "  - [10] J.dcm: ..."
                    parts = line.strip().split("]")
                    idx_str = parts[0].replace("- [", "").strip()
                    fname = parts[1].split(":")[0].strip()
                    failed_names.add(fname)
                    continue
                elif line.strip() == "" or line.startswith("="):
                    in_failed = False

            # Detect per-image results section
            if line.startswith("Results per image:"):
                in_results = True
                continue
            if line.startswith("---") and not in_results:
                continue

            if in_results and line.strip():
                # Format: " 13. I000013.dcm     | Area: ... | Dice: ... | Score: ... | ..."
                if "|" not in line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                # First part: "  1. A.dcm"
                idx_filename = parts[0]
                dot_pos = idx_filename.find(".")
                if dot_pos == -1:
                    continue
                idx = int(idx_filename[:dot_pos].strip())
                filename = idx_filename[dot_pos + 1:].strip()

                info = {"idx": idx, "filename": filename}
                for p in parts[1:]:
                    if p.startswith("Area:"):
                        info["area"] = float(p.replace("Area:", "").replace("px", "").strip())
                    elif p.startswith("Dice:"):
                        val = p.replace("Dice:", "").strip()
                        info["dice"] = float(val) if val != "REF" else 1.0
                    elif p.startswith("Score:"):
                        info["score"] = float(p.replace("Score:", "").strip())

                # Markers
                info["warning"] = "⚠️" in line
                info["severe"] = "🚨" in line
                info["neighbor"] = "⭕" in line

                slices.append(info)

    return slices, failed_names


def classify_slices(slices, failed_names):
    """Classify each slice as 'correct', 'ambiguous', or 'failed'."""
    for s in slices:
        dice = s.get("dice", 0.0)
        if dice >= DICE_CORRECT_THRESHOLD:
            s["class"] = "correct"
        elif dice >= DICE_AMBIGUOUS_THRESHOLD:
            s["class"] = "ambiguous"
        else:
            s["class"] = "failed"

    # Also add entries for slices that are not in results at all (hard failures)
    result_names = {s["filename"] for s in slices}
    for fn in failed_names:
        if fn not in result_names:
            slices.append({"filename": fn, "dice": 0.0, "class": "failed",
                           "area": 0, "score": 0.0, "idx": -1})

    return slices


# ===================================================================
# Step 3 – Extract boundary pixels from correct slices
# ===================================================================

def load_mask(output_dir, filename_base):
    """Load the binary mask saved as .npy or .png."""
    # Prefer .npy (exact binary mask)
    npy_path = os.path.join(output_dir, f"{filename_base}_mask.npy")
    if os.path.exists(npy_path):
        return np.load(npy_path).astype(np.uint8)

    mask_path = os.path.join(output_dir, f"{filename_base}_mask.png")
    if os.path.exists(mask_path):
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return (m > 127).astype(np.uint8)

    return None


def extract_boundary_pixels(mask):
    """Return (N, 2) array of boundary pixel coordinates (col, row) = (x, y)."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = []
    for c in contours:
        for pt in c:
            points.append(pt[0])  # [x, y]
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(points, dtype=np.float64)


def load_contour_points_by_slice(output_dir, files, middle_idx, z_spacing=12):
    """
    Fallback: load contour_points_3d.csv (pixel-space x,y,z) and group by
    slice index.  Returns dict {slice_idx: (N,2) array of (col, row)}.
    """
    csv_path = os.path.join(output_dir, "contour_points_3d.csv")
    npy_path = os.path.join(output_dir, "contour_points_3d.npy")

    if os.path.exists(npy_path):
        pts = np.load(npy_path)
    elif os.path.exists(csv_path):
        pts = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    else:
        return {}

    if pts.ndim != 2 or pts.shape[1] != 3:
        return {}

    result = {}
    z_values = np.unique(pts[:, 2])
    for z in z_values:
        slice_idx = int(round(z / z_spacing)) + middle_idx
        mask_pts = pts[pts[:, 2] == z]
        result[slice_idx] = mask_pts[:, :2]  # (x, y) in pixel coords

    return result


# ===================================================================
# Step 4 – Convert pixel coords -> real-world 3D (mm) via DICOM header
# ===================================================================

def _slice_normal(dicom_ds):
    """Return the slice-plane normal (unit vector) from ImageOrientationPatient.

    Cross product of row_cosine and col_cosine. Generalises the "Z direction"
    framing to coronal/oblique acquisitions.
    """
    iop = np.array([float(v) for v in dicom_ds.ImageOrientationPatient])
    row_cosine = iop[:3]
    col_cosine = iop[3:6]
    n = np.cross(row_cosine, col_cosine)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None
    return n / norm


def detect_acquisition_direction(dcm_lookup):
    """Detect whether alphabetical file order traverses the volume superior→inferior
    or inferior→superior, using DICOM ImagePositionPatient projected on the slice
    normal (handles axial / coronal / oblique uniformly).

    Args:
        dcm_lookup: dict {slice_idx: pydicom.Dataset}. Indices are 1-based and
                    match the ordering in slices_info / propagation_summary.txt.

    Returns:
        {
            "direction":   "alphabetical_to_superior" | "alphabetical_to_inferior" | "unknown",
            "head_side":   "low_idx" | "high_idx" | "unknown",
            "z_per_idx":   {idx: float (signed traversal position, mm)},
            "source":      "ImagePositionPatient" | "SliceLocation" | "InstanceNumber_fallback",
            "delta_total": float | None,   # signed end-to-end traversal
        }

    Conventions: in DICOM patient coordinates +Z is SUPERIOR (cranial). The
    HEAD of the humerus sits at the SUPERIOR end of the bone. So:
      - if alphabetical order goes toward SUPERIOR  →  head_side = "high_idx"
      - if alphabetical order goes toward INFERIOR  →  head_side = "low_idx"
    For unknown direction we leave head_side = "unknown" and downstream uses
    shaft tolerance on both sides (loose / safe).
    """
    if not dcm_lookup:
        return {"direction": "unknown", "head_side": "unknown",
                "z_per_idx": {}, "source": "ImagePositionPatient",
                "delta_total": None}

    sorted_idx = sorted(dcm_lookup.keys())
    z_per_idx = {}
    source = "ImagePositionPatient"

    # Reference normal from the first usable IOP (slices in the same volume
    # share the same orientation, in practice).
    ref_normal = None
    ref_ipp = None
    for idx in sorted_idx:
        ds = dcm_lookup[idx]
        if hasattr(ds, "ImageOrientationPatient") and hasattr(ds, "ImagePositionPatient"):
            n = _slice_normal(ds)
            if n is not None:
                ref_normal = n
                ref_ipp = np.array([float(v) for v in ds.ImagePositionPatient])
                break

    if ref_normal is not None:
        for idx in sorted_idx:
            ds = dcm_lookup[idx]
            if not hasattr(ds, "ImagePositionPatient"):
                continue
            ipp = np.array([float(v) for v in ds.ImagePositionPatient])
            # Signed projection along the slice normal (positive = same direction
            # as the normal). This is monotonic in slice position.
            z_per_idx[idx] = float(np.dot(ipp - ref_ipp, ref_normal))
    else:
        # Fallback: use SliceLocation (already a scalar along the slice axis)
        source = "SliceLocation"
        for idx in sorted_idx:
            ds = dcm_lookup[idx]
            if hasattr(ds, "SliceLocation"):
                z_per_idx[idx] = float(ds.SliceLocation)
        if not z_per_idx:
            # Last resort: InstanceNumber (won't tell us the axis sign vs patient)
            source = "InstanceNumber_fallback"
            for idx in sorted_idx:
                ds = dcm_lookup[idx]
                if hasattr(ds, "InstanceNumber"):
                    z_per_idx[idx] = float(ds.InstanceNumber)

    if len(z_per_idx) < 2:
        return {"direction": "unknown", "head_side": "unknown",
                "z_per_idx": z_per_idx, "source": source, "delta_total": None}

    usable_idx = sorted(z_per_idx.keys())
    z_first = z_per_idx[usable_idx[0]]
    z_last = z_per_idx[usable_idx[-1]]
    delta = z_last - z_first

    if abs(delta) < 1e-3:
        return {"direction": "unknown", "head_side": "unknown",
                "z_per_idx": z_per_idx, "source": source, "delta_total": delta}

    # Project the patient SUPERIOR direction (+Z in patient coords) onto the
    # slice normal to know which sign of `delta` means "going superior". For
    # axial acquisitions ref_normal[2] is ±1 and this collapses to sign(delta).
    if ref_normal is not None:
        sup_sign = 1.0 if ref_normal[2] >= 0 else -1.0
        signed_delta = delta * sup_sign
    else:
        # SliceLocation / InstanceNumber: we cannot tell which way is superior,
        # so fall back to "unknown".
        return {"direction": "unknown", "head_side": "unknown",
                "z_per_idx": z_per_idx, "source": source, "delta_total": delta}

    if signed_delta > 0:
        direction = "alphabetical_to_superior"
        head_side = "high_idx"
    else:
        direction = "alphabetical_to_inferior"
        head_side = "low_idx"

    return {"direction": direction, "head_side": head_side,
            "z_per_idx": z_per_idx, "source": source,
            "delta_total": signed_delta}


def pixel_to_world(pixel_coords, dicom_ds):
    """
    Convert pixel coordinates (col, row) to patient coordinate system (mm).

    Uses the standard DICOM affine transform:
        P_xyz = ImagePositionPatient + col * PixelSpacing[1] * row_cosines
                                     + row * PixelSpacing[0] * col_cosines

    Parameters
    ----------
    pixel_coords : (N, 2) array  – columns (x), rows (y)
    dicom_ds     : pydicom Dataset

    Returns
    -------
    world_coords : (N, 3) array  – (X, Y, Z) in mm
    """
    ipp = np.array([float(v) for v in dicom_ds.ImagePositionPatient])
    iop = np.array([float(v) for v in dicom_ds.ImageOrientationPatient])
    ps = np.array([float(v) for v in dicom_ds.PixelSpacing])

    row_cosine = iop[:3]  # direction cosines of the row direction
    col_cosine = iop[3:6]  # direction cosines of the column direction

    # PixelSpacing = [row_spacing, col_spacing]
    dr = ps[0]  # spacing along rows (between columns)
    dc = ps[1]  # spacing along columns (between rows)

    cols = pixel_coords[:, 0]
    rows = pixel_coords[:, 1]

    world = (ipp[np.newaxis, :]
             + np.outer(cols, dc * row_cosine)
             + np.outer(rows, dr * col_cosine))

    return world


def get_dicom_for_slice(in_dir, filename):
    """Read the DICOM dataset corresponding to `filename`."""
    dcm_path = os.path.join(in_dir, filename)
    if not os.path.exists(dcm_path):
        # Try without extension
        base = os.path.splitext(filename)[0]
        dcm_path = os.path.join(in_dir, base + ".dcm")
    if not os.path.exists(dcm_path):
        return None
    return pydicom.dcmread(dcm_path)


# ===================================================================
# Step 5 – Detect circular arcs in 3D via RANSAC on a sphere model
# ===================================================================

def fit_sphere_to_points(pts):
    """
    Least-squares fit of a sphere to 3D points.

    Minimise  ||p - c||^2 = R^2  =>  linear system in (cx, cy, cz, R^2 - c.c).

    Returns (center, radius) or (None, None) if degenerate.
    """
    n = pts.shape[0]
    if n < 4:
        return None, None

    A = np.zeros((n, 4))
    A[:, 0] = 2 * pts[:, 0]
    A[:, 1] = 2 * pts[:, 1]
    A[:, 2] = 2 * pts[:, 2]
    A[:, 3] = 1.0

    b = np.sum(pts ** 2, axis=1)

    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, None

    cx, cy, cz = result[0], result[1], result[2]
    R_sq = result[3] + cx ** 2 + cy ** 2 + cz ** 2
    if R_sq <= 0:
        return None, None
    return np.array([cx, cy, cz]), np.sqrt(R_sq)


def ransac_sphere(points, n_iter=RANSAC_ITERATIONS, inlier_dist=RANSAC_INLIER_DISTANCE,
                  min_inliers=MIN_ARC_POINTS):
    """
    RANSAC sphere fitting.  Returns (center, radius, inlier_mask) for the
    best sphere, or (None, None, None).
    """
    n = points.shape[0]
    if n < 4:
        return None, None, None

    best_inliers = None
    best_count = 0
    best_center = None
    best_radius = None

    for _ in range(n_iter):
        idx = np.random.choice(n, 4, replace=False)
        sample = points[idx]
        center, radius = fit_sphere_to_points(sample)
        if center is None or radius <= 0 or radius > 200:  # sanity: radius < 200 mm
            continue

        dists = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        inlier_mask = dists < inlier_dist
        count = np.sum(inlier_mask)

        if count > best_count:
            best_count = count
            best_inliers = inlier_mask
            best_center = center
            best_radius = radius

    if best_count < min_inliers:
        return None, None, None

    # Refit on all inliers
    center, radius = fit_sphere_to_points(points[best_inliers])
    if center is None:
        return best_center, best_radius, best_inliers

    dists = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    inlier_mask = dists < inlier_dist

    return center, radius, inlier_mask


def find_circular_arcs_per_slice(world_points, slice_labels, inlier_mask):
    """
    Among the inlier points (those on the global head sphere), group by
    slice and require both an absolute count (>= MIN_ARC_POINTS) and a
    minimum fraction of the slice's boundary points (MIN_ARC_INLIER_RATIO).
    Slices that contribute a single stray point on the sphere are dropped —
    those used to show up as spurious 1-point arcs in the 3D viz.
    """
    arcs = []
    unique_slices = np.unique(slice_labels[inlier_mask])
    for sl in unique_slices:
        sl_total = int(np.sum(slice_labels == sl))
        if sl_total == 0:
            continue
        mask_sl = (slice_labels == sl) & inlier_mask
        pts = world_points[mask_sl]
        n = len(pts)
        if n < MIN_ARC_POINTS:
            continue
        if (n / sl_total) < MIN_ARC_INLIER_RATIO:
            continue
        arcs.append({
            "slice": int(sl),
            "num_points": n,
            "points": pts,
        })
    return arcs


# ===================================================================
# Step 5b – Per-slice circle fitting ("feature 2": articular surface)
# ===================================================================
#
# The humeral head is approximated globally by ONE sphere (Step 5a, above).
# But the *articular* surface — the cartilage-on-glenoid contact area — sits
# only on a handful of slices, those with the largest cross-sectional area
# (the "equator" of the head).  On each such slice the lateral / superior
# portion of the bone outline is a circular arc that LIES ON the articular
# surface, it is NOT an approximation of the head sphere.  These per-slice
# circles are the second geometric feature used to represent the humerus.

def fit_circle_2d_kasa(xy):
    """Algebraic least-squares circle fit (Kåsa).

    Returns (cx, cy, r) or (None, None, None) if degenerate.
    Solves the linear system  2*x*cx + 2*y*cy + (R^2 - cx^2 - cy^2) = x^2+y^2.
    """
    n = xy.shape[0]
    if n < 3:
        return None, None, None
    x = xy[:, 0]
    y = xy[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones(n)])
    b = x ** 2 + y ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, None
    cx, cy, c = sol
    r_sq = c + cx ** 2 + cy ** 2
    if r_sq <= 0:
        return None, None, None
    return float(cx), float(cy), float(np.sqrt(r_sq))


def ransac_circle_2d(xy, n_iter=CIRCLE_RANSAC_ITERATIONS,
                     inlier_dist=CIRCLE_INLIER_DISTANCE_MM,
                     min_inliers=MIN_CIRCLE_INLIERS,
                     max_radius_mm=200.0,
                     rng=None):
    """RANSAC circle fit on 2D points. Returns (cx, cy, r, inlier_mask)."""
    n = xy.shape[0]
    if n < 3:
        return None, None, None, None
    if rng is None:
        rng = np.random.default_rng(42)

    best_count = 0
    best_inliers = None
    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        cx, cy, r = fit_circle_2d_kasa(xy[idx])
        if cx is None or r > max_radius_mm:
            continue
        d = np.abs(np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2) - r)
        inl = d < inlier_dist
        c = int(inl.sum())
        if c > best_count:
            best_count = c
            best_inliers = inl

    if best_count < min_inliers:
        return None, None, None, None

    # Refit on all inliers
    cx, cy, r = fit_circle_2d_kasa(xy[best_inliers])
    if cx is None:
        return None, None, None, None
    d = np.abs(np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2) - r)
    inliers = d < inlier_dist
    if int(inliers.sum()) < min_inliers:
        return None, None, None, None
    return cx, cy, r, inliers


def project_to_slice_plane(pts3d):
    """Use SVD to project 3D points onto their best-fit plane.

    Returns:
        xy        : (N, 2) coords in the plane
        centroid  : (3,)   plane origin
        u, v      : (3,)   in-plane basis vectors
        normal    : (3,)   plane normal
    """
    centroid = pts3d.mean(axis=0)
    centered = pts3d - centroid
    # SVD: rows of vt are right singular vectors. The smallest is the plane normal.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    u, v, normal = vt[0], vt[1], vt[2]
    xy = np.column_stack([centered @ u, centered @ v])
    return xy, centroid, u, v, normal


def fit_circles_in_slice(pts3d, max_circles=MAX_CIRCLES_PER_SLICE):
    """Iterative RANSAC: extract up to `max_circles` distinct circles from
    the boundary points of a single slice. After each fit, inliers are
    removed and the next circle is sought in what remains.

    Returns list of dicts with: center3d, radius_mm, axis_u, axis_v,
        normal, inliers_world, num_inliers, num_total_in_slice,
        fit_residual_mm, inlier_ratio.
    """
    if len(pts3d) < MIN_CIRCLE_INLIERS:
        return []

    xy, centroid, u, v, normal = project_to_slice_plane(pts3d)
    total_in_slice = len(pts3d)
    available = np.ones(len(pts3d), dtype=bool)
    rng = np.random.default_rng(42)
    circles = []

    for _ in range(max_circles):
        if int(available.sum()) < MIN_CIRCLE_INLIERS:
            break
        cx, cy, r, inl = ransac_circle_2d(xy[available], rng=rng)
        if cx is None:
            break
        n_inl = int(inl.sum())
        n_avail = int(available.sum())
        ratio_local = n_inl / n_avail
        ratio_slice = n_inl / total_in_slice
        # Accept if it explains a meaningful slice fraction. We compare against
        # the FULL slice point count (not just remaining) to keep the ratio
        # meaningful across iterations.
        if ratio_slice < MIN_CIRCLE_INLIER_RATIO and len(circles) > 0:
            break

        # Map inliers back to the original slice indexing
        avail_idx = np.where(available)[0]
        global_inlier_idx = avail_idx[inl]
        inliers3d = pts3d[global_inlier_idx]
        center3d = centroid + cx * u + cy * v

        d = np.abs(np.sqrt((xy[global_inlier_idx, 0] - cx) ** 2
                           + (xy[global_inlier_idx, 1] - cy) ** 2) - r)
        residual = float(d.mean())

        circles.append({
            "center3d": center3d,
            "radius_mm": float(r),
            "axis_u": u,
            "axis_v": v,
            "normal": normal,
            "inliers_world": inliers3d,
            "num_inliers": n_inl,
            "num_total_in_slice": total_in_slice,
            "fit_residual_mm": residual,
            "inlier_ratio": ratio_slice,
            "inlier_ratio_local": ratio_local,
            # In-plane center (for later world→pixel projection)
            "center_in_plane": (cx, cy),
        })

        # Remove these inliers from the pool and try again
        available[global_inlier_idx] = False

    return circles


def fit_circles_per_slice(world_pts, slice_labels, areas_by_slice):
    """Run the per-slice circle fitter over every slice that has enough
    boundary points, and tag each circle as articular (lies on the joint
    surface) based on the slice's mask area.
    """
    if len(world_pts) == 0:
        return []

    if areas_by_slice:
        max_area = max(areas_by_slice.values())
    else:
        max_area = 0.0
    area_threshold = ARTICULAR_AREA_RATIO * max_area

    all_circles = []
    for sl in np.unique(slice_labels):
        sl = int(sl)
        mask = slice_labels == sl
        pts = world_pts[mask]
        if len(pts) < MIN_CIRCLE_INLIERS:
            continue
        slice_circles = fit_circles_in_slice(pts)
        area = float(areas_by_slice.get(sl, 0.0))
        is_articular = area_threshold > 0 and area >= area_threshold
        for ci, c in enumerate(slice_circles):
            c["slice_idx"] = sl
            c["circle_idx"] = ci
            c["area_px"] = area
            c["is_articular"] = bool(is_articular)
            all_circles.append(c)

    return all_circles


# ===================================================================
# World ↔ pixel inverse (needed to draw 3D circles back on a slice PNG)
# ===================================================================

def world_to_pixel(world_pts, dicom_ds):
    """Inverse of `pixel_to_world`. Assumes the input points lie on the
    slice plane defined by the DICOM header (true for points produced by
    `pixel_to_world` from this same slice)."""
    ipp = np.array([float(v) for v in dicom_ds.ImagePositionPatient])
    iop = np.array([float(v) for v in dicom_ds.ImageOrientationPatient])
    ps = np.array([float(v) for v in dicom_ds.PixelSpacing])

    row_cosine = iop[:3]
    col_cosine = iop[3:6]
    dr = ps[0]
    dc = ps[1]

    delta = world_pts - ipp[np.newaxis, :]
    cols = (delta @ row_cosine) / dc
    rows = (delta @ col_cosine) / dr
    return np.column_stack([cols, rows])


# ===================================================================
# Step 1.5 — Direction-aware area-curve validation
# ===================================================================
#
# Three failure patterns the raw curve doesn't catch on its own:
#   (A) Noise BEFORE the humerus actually starts — false-positive
#       segmentations on slices that don't contain humerus.
#   (B) Internal outliers where SAM grabbed extra muscle/glenoid (area
#       jumps above the local trend after a clear deceleration).
#   (C) Noise AFTER the humerus has truly ended (tail).
#
# Anatomy provides asymmetry: the humeral HEAD closes ABRUPTLY (~spherical
# cap), the SHAFT tapers GRADUALLY (~cylindrical diaphysis). We use
# tighter / looser thresholds per side based on the acquisition direction
# inferred from DICOM headers.

def _rolling_median(values_by_idx, half_window=TREND_WINDOW_HALF):
    """Per-slice rolling median over a window of width 2*half_window+1.

    Operates on the SORTED slice indices but pulls neighbours by INTEGER
    INDEX in the sorted list — gaps in slice_idx are tolerated.
    """
    sorted_idx = sorted(values_by_idx.keys())
    n = len(sorted_idx)
    out = {}
    for i, idx in enumerate(sorted_idx):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        window = [values_by_idx[sorted_idx[j]] for j in range(lo, hi)]
        out[idx] = float(np.median(window))
    return out


def _residual_sigma(values_by_idx, trend_by_idx, n_passes=2, clip_sigma=5.0):
    """Robust per-slice residual: (raw - trend) / (1.4826 * MAD).

    Two-pass: clip residuals at ±clip_sigma between passes so the MAD is
    not inflated by a long contaminated stretch. A pixel-level sigma floor
    (INTERNAL_OUTLIER_MIN_SIGMA_PX) prevents nonsensical blow-ups on very
    smooth curves where MAD ≈ 0.
    """
    sorted_idx = sorted(values_by_idx.keys())
    raw = np.array([values_by_idx[i] for i in sorted_idx], dtype=float)
    trend = np.array([trend_by_idx[i] for i in sorted_idx], dtype=float)
    resid = raw - trend
    sigma = 1.0
    for _ in range(n_passes):
        mad = float(np.median(np.abs(resid)))
        sigma = max(1.4826 * mad, INTERNAL_OUTLIER_MIN_SIGMA_PX)
        resid_clipped = np.clip(resid, -clip_sigma * sigma, clip_sigma * sigma)
        mad = float(np.median(np.abs(resid_clipped)))
        sigma = max(1.4826 * mad, INTERNAL_OUTLIER_MIN_SIGMA_PX)
    return {idx: float((values_by_idx[idx] - trend_by_idx[idx]) / sigma)
            for idx in sorted_idx}


def _walk_trim_run(side_indices_from_edge, smoothed, threshold, min_run):
    """Walk from the volume edge inward; collect contiguous leading run of
    slices whose smoothed area is below `threshold`. If run length reaches
    `min_run`, return the trimmed indices, else return [] (avoid trimming a
    single dip)."""
    run = []
    for idx in side_indices_from_edge:
        if smoothed.get(idx, 0.0) < threshold:
            run.append(idx)
        else:
            break
    if len(run) >= min_run:
        return run
    return []


def _walk_trim_from_peak(peak_idx, side_indices_outward, smoothed,
                         reversal_ratio):
    """Walk from the peak slice outward through `side_indices_outward`
    (already sorted in walking order: e.g. for the LEFT side these are
    [peak-1, peak-2, ..., 1]; for the RIGHT side [peak+1, peak+2, ...]).

    Keep slices as long as the smoothed area continues to descend (with
    tolerance — small upward wiggles below `reversal_ratio` and below
    `BELL_REVERSAL_MIN_ABS` absolute pixels are accepted as noise). At the
    first GENUINE reversal (smoothed[next] / smoothed[curr] >= reversal_ratio
    AND the absolute jump exceeds the floor), trim everything from `next`
    outward to the volume edge.

    This matches anatomy:
      - HEAD side: ~spherical cap closes abruptly. Many slices descend
        smoothly to small areas — keep them all. Trim only when curve
        clearly turns back up (= we left the bone).
      - SHAFT side: ~cylindrical, area decreases gradually with optional
        small bumps from tuberosities. Same trim criterion: real reversal,
        not bumpiness.
    """
    if not side_indices_outward:
        return []
    curr_smooth = smoothed.get(peak_idx, 0.0)
    trimmed = []
    triggered = False
    for idx in side_indices_outward:
        next_smooth = smoothed.get(idx, 0.0)
        if triggered:
            trimmed.append(idx)
            continue
        ratio = (next_smooth / curr_smooth) if curr_smooth > 0 else 0.0
        abs_jump = next_smooth - curr_smooth
        if ratio >= reversal_ratio and abs_jump > BELL_REVERSAL_MIN_ABS:
            triggered = True
            trimmed.append(idx)
            continue
        # Keep this slice; it becomes the new reference for the descent
        curr_smooth = next_smooth
    return trimmed


def validate_area_curve(slices_info, direction_info,
                        slice_circles_lookup=None,
                        head_trim_ratio=None,
                        shaft_trim_ratio=None):
    """Direction-aware bell-extent + internal-outlier detection.

    Args:
        slices_info: list of dicts with at least {idx, area, dice} for every
            classified slice (CORRECT / AMBIGUOUS / FAILED). Only entries
            with idx > 0 and finite area are considered.
        direction_info: output of detect_acquisition_direction().
        slice_circles_lookup: optional dict {slice_idx: circle_dict} from a
            prior Step 5b run. When None, internal-outlier detection still
            runs but cannot use the inlier_ratio cross-check (Pattern B
            confirmation falls back to Dice-only).
        head_trim_ratio / shaft_trim_ratio: optional per-call overrides
            (CLI knobs); fall back to module constants when None.

    Returns: dict with the AreaValidation schema documented in the plan.
    """
    # CLI overrides → static floor ratios (kept as the user-facing knobs)
    head_floor = (head_trim_ratio if head_trim_ratio is not None
                  else BELL_BOUNDARY_HEAD_RATIO)
    shaft_floor = (shaft_trim_ratio if shaft_trim_ratio is not None
                   else BELL_BOUNDARY_SHAFT_RATIO)
    head_reversal = BELL_REVERSAL_RATIO_HEAD
    shaft_reversal = BELL_REVERSAL_RATIO_SHAFT
    head_side = direction_info.get("head_side", "unknown")

    # Build per-idx maps from positive, area-bearing entries
    area_by_idx = {}
    dice_by_idx = {}
    for s in slices_info:
        idx = s.get("idx", -1)
        if idx <= 0:
            continue
        area = float(s.get("area", 0.0) or 0.0)
        if area <= 0:
            continue
        area_by_idx[idx] = area
        if "dice" in s and s["dice"] is not None:
            dice_by_idx[idx] = float(s["dice"])

    decision_log = []

    if len(area_by_idx) < 6:
        decision_log.append(f"volume too short ({len(area_by_idx)} slices); skipping validation")
        return {
            "bell_peak_idx": None, "bell_peak_area": None,
            "bell_start_idx": None, "bell_end_idx": None,
            "kept_indices": set(area_by_idx.keys()),
            "trimmed_pre_bell": [], "trimmed_post_bell": [],
            "internal_outliers": [],
            "head_side": head_side, "trend_per_idx": {},
            "residual_sigma_per_idx": {}, "decision_log": decision_log,
            "skipped": True, "reason_skipped": "too_few_slices",
        }

    sorted_idx = sorted(area_by_idx.keys())
    smoothed = _rolling_median(area_by_idx, half_window=TREND_WINDOW_HALF)

    # Locate peak from the SMOOTHED curve so a single-slice spike doesn't
    # define the bell. Resolve ties by picking the centre-most index.
    peak_smoothed_value = max(smoothed.values())
    peak_candidates = [i for i, v in smoothed.items() if v == peak_smoothed_value]
    peak_idx = peak_candidates[len(peak_candidates) // 2]
    peak_area_raw = area_by_idx[peak_idx]
    global_max_area = max(area_by_idx.values())

    decision_log.append(
        f"peak (smoothed) at slice {peak_idx}, raw area = {peak_area_raw:.0f}"
    )

    if peak_area_raw < BELL_PEAK_MIN_RATIO * global_max_area:
        decision_log.append(
            "no clear bell — peak smoothed area < BELL_PEAK_MIN_RATIO * global max; "
            "validation passes through"
        )
        return {
            "bell_peak_idx": peak_idx, "bell_peak_area": peak_area_raw,
            "bell_start_idx": sorted_idx[0], "bell_end_idx": sorted_idx[-1],
            "kept_indices": set(sorted_idx),
            "trimmed_pre_bell": [], "trimmed_post_bell": [],
            "internal_outliers": [],
            "head_side": head_side, "trend_per_idx": smoothed,
            "residual_sigma_per_idx": {}, "decision_log": decision_log,
            "skipped": True, "reason_skipped": "no_clear_bell",
        }

    # Side classification: which side of the peak is head, which is shaft?
    # head_side="low_idx" → indices < peak are HEAD, > peak are SHAFT
    # head_side="high_idx" → indices > peak are HEAD, < peak are SHAFT
    # head_side="unknown" → both sides treated as shaft (loose / safe)
    if head_side == "low_idx":
        left_is_head = True
    elif head_side == "high_idx":
        left_is_head = False
    else:
        left_is_head = None  # both sides shaft

    left_indices = [i for i in sorted_idx if i < peak_idx]
    right_indices = [i for i in sorted_idx if i > peak_idx]

    def _side_floor(is_head_side):
        if is_head_side is None:
            return shaft_floor
        return head_floor if is_head_side else shaft_floor

    def _side_reversal(is_head_side):
        if is_head_side is None:
            return shaft_reversal
        return head_reversal if is_head_side else shaft_reversal

    left_floor = _side_floor(left_is_head)
    right_floor = _side_floor(left_is_head if left_is_head is None
                              else (not left_is_head))
    left_rev = _side_reversal(left_is_head)
    right_rev = _side_reversal(left_is_head if left_is_head is None
                               else (not left_is_head))
    left_threshold = left_floor * peak_smoothed_value
    right_threshold = right_floor * peak_smoothed_value

    # LEFT side: edge → peak is sorted ascending (slice 1, 2, ..., peak-1)
    # RIGHT side: edge → peak is sorted descending (slice last, ..., peak+1)
    left_from_edge = list(left_indices)               # ascending = from low edge inward
    right_from_edge = list(reversed(right_indices))    # descending = from high edge inward
    left_outward = list(reversed(left_indices))        # peak → low edge
    right_outward = list(right_indices)                # peak → high edge

    # Static floor (catches "long tail of small areas" — head noise pattern)
    left_static = _walk_trim_run(left_from_edge, smoothed, left_threshold,
                                 BELL_MIN_RUN_LENGTH)
    right_static = _walk_trim_run(right_from_edge, smoothed, right_threshold,
                                  BELL_MIN_RUN_LENGTH)
    # Reversal-from-peak (catches "area went back up" — late-noise pattern)
    left_rev_trim = _walk_trim_from_peak(peak_idx, left_outward, smoothed, left_rev)
    right_rev_trim = _walk_trim_from_peak(peak_idx, right_outward, smoothed, right_rev)

    left_trim = sorted(set(left_static) | set(left_rev_trim))
    right_trim = sorted(set(right_static) | set(right_rev_trim))

    decision_log.append(
        f"left side ({'head' if left_is_head else ('shaft' if left_is_head is False else 'unknown→shaft')}): "
        f"floor={left_threshold:.0f} (={left_floor:.2f}*peak), reversal_ratio={left_rev}; "
        f"static={left_static}, reversal={left_rev_trim} → trimmed={left_trim}"
    )
    decision_log.append(
        f"right side ({'shaft' if left_is_head else ('head' if left_is_head is False else 'unknown→shaft')}): "
        f"floor={right_threshold:.0f} (={right_floor:.2f}*peak), reversal_ratio={right_rev}; "
        f"static={right_static}, reversal={right_rev_trim} → trimmed={right_trim}"
    )

    trimmed = set(left_trim) | set(right_trim)
    kept_after_trim = [i for i in sorted_idx if i not in trimmed]
    bell_start = kept_after_trim[0] if kept_after_trim else sorted_idx[0]
    bell_end = kept_after_trim[-1] if kept_after_trim else sorted_idx[-1]

    # Map trimmed slices to "pre-bell" (low-idx side) vs "post-bell" (high-idx)
    trimmed_pre = sorted(set(left_trim))
    trimmed_post = sorted(set(right_trim))

    # ---- Internal outlier detection (Pattern B) ----
    kept_area = {i: area_by_idx[i] for i in kept_after_trim}
    if len(kept_area) >= 5:
        kept_smoothed = _rolling_median(kept_area, half_window=TREND_WINDOW_HALF)
        residual_sigma = _residual_sigma(kept_area, kept_smoothed)
    else:
        kept_smoothed = {i: kept_area[i] for i in kept_area}
        residual_sigma = {i: 0.0 for i in kept_area}
        decision_log.append("kept slices < 5 — outlier detection skipped")

    internal_outliers = []
    for idx in sorted(kept_area.keys()):
        rs = residual_sigma.get(idx, 0.0)
        if rs <= INTERNAL_OUTLIER_RESID_SIGMA:
            continue  # under threshold OR negative-side (under-segmentation)
        if kept_area[idx] <= kept_smoothed[idx]:
            continue  # area didn't actually exceed trend (numerical edge)
        # Cross-check 1: per-slice circle inlier_ratio drop vs neighbours
        confirmed_by_circle = False
        if slice_circles_lookup is not None and idx in slice_circles_lookup:
            ir_self = float(slice_circles_lookup[idx].get("inlier_ratio", 1.0))
            neighbours = []
            for j in (idx - 1, idx + 1, idx - 2, idx + 2):
                if j in slice_circles_lookup and j not in {idx}:
                    neighbours.append(float(slice_circles_lookup[j].get("inlier_ratio", 1.0)))
            if neighbours:
                ir_neighbour = float(np.median(neighbours))
                if ir_neighbour - ir_self > INTERNAL_OUTLIER_INLIER_DROP:
                    confirmed_by_circle = True
        # Cross-check 2: Dice drop vs neighbours
        confirmed_by_dice = False
        if idx in dice_by_idx:
            d_neighbours = [dice_by_idx[j] for j in (idx - 1, idx + 1)
                            if j in dice_by_idx]
            if d_neighbours:
                d_med = float(np.median(d_neighbours))
                if d_med - dice_by_idx[idx] > INTERNAL_OUTLIER_DICE_DROP:
                    confirmed_by_dice = True
        if confirmed_by_circle or confirmed_by_dice:
            internal_outliers.append({
                "idx": idx,
                "residual_sigma": rs,
                "area": kept_area[idx],
                "trend": kept_smoothed[idx],
                "confirmed_by_circle": confirmed_by_circle,
                "confirmed_by_dice": confirmed_by_dice,
                "confidence": "strong",
            })
        else:
            # Weak: flag in report, do NOT exclude (no second signal)
            internal_outliers.append({
                "idx": idx,
                "residual_sigma": rs,
                "area": kept_area[idx],
                "trend": kept_smoothed[idx],
                "confirmed_by_circle": False,
                "confirmed_by_dice": False,
                "confidence": "weak",
            })

    # Final kept_indices excludes STRONG outliers; weak outliers stay in
    strong_outlier_idx = {o["idx"] for o in internal_outliers
                          if o["confidence"] == "strong"}
    kept_indices = set(kept_after_trim) - strong_outlier_idx

    decision_log.append(
        f"internal outliers: {len(internal_outliers)} flagged, "
        f"{len(strong_outlier_idx)} confirmed and excluded"
    )

    return {
        "bell_peak_idx": peak_idx,
        "bell_peak_area": peak_area_raw,
        "bell_start_idx": bell_start,
        "bell_end_idx": bell_end,
        "kept_indices": kept_indices,
        "trimmed_pre_bell": trimmed_pre,
        "trimmed_post_bell": trimmed_post,
        "internal_outliers": internal_outliers,
        "head_side": head_side,
        "trend_per_idx": smoothed,
        "residual_sigma_per_idx": residual_sigma,
        "decision_log": decision_log,
        "skipped": False,
        "reason_skipped": None,
        "left_is_head": left_is_head,
        "left_threshold": left_threshold,
        "right_threshold": right_threshold,
        "left_reversal": left_rev,
        "right_reversal": right_rev,
    }


def write_validation_report(out_dir, short_name, validation, direction_info,
                            slices_info):
    """Write a human-readable area_curve_validation.txt and a machine-
    readable area_curve_validation.csv. Sibling files; do not modify any
    existing CSV schema."""
    txt_path = os.path.join(out_dir, "area_curve_validation.txt")
    csv_path = os.path.join(out_dir, "area_curve_validation.csv")

    # Build a quick {idx -> filename} lookup
    name_by_idx = {}
    for s in slices_info:
        idx = s.get("idx", -1)
        if idx > 0 and "filename" in s:
            name_by_idx[idx] = s["filename"]

    head_side = validation.get("head_side", "unknown")
    if head_side == "low_idx":
        head_label, shaft_label = "low_idx", "high_idx"
    elif head_side == "high_idx":
        head_label, shaft_label = "high_idx", "low_idx"
    else:
        head_label = shaft_label = "unknown"

    with open(txt_path, "w") as f:
        f.write(f"Area-Curve Validation Report — {short_name}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Acquisition direction: {direction_info.get('direction', 'unknown')}  "
                f"(source: {direction_info.get('source', 'n/a')})\n")
        f.write(f"Head side: {head_label}     Shaft side: {shaft_label}\n")
        if validation.get("skipped"):
            f.write(f"\nValidation SKIPPED — reason: {validation.get('reason_skipped')}\n")
            for line in validation.get("decision_log", []):
                f.write(f"  - {line}\n")
            f.write("=" * 60 + "\n")
        else:
            peak_idx = validation["bell_peak_idx"]
            f.write(f"Bell peak: slice {peak_idx}  area={validation['bell_peak_area']:.0f} px\n\n")
            f.write("Bell extent (kept):\n")
            f.write(f"  start = {validation['bell_start_idx']}  "
                    f"end = {validation['bell_end_idx']}  "
                    f"(count = {len(validation['kept_indices'])})\n\n")

            pre = validation["trimmed_pre_bell"]
            f.write("Trimmed PRE-bell (low_idx side):\n")
            if pre:
                for i in pre:
                    f.write(f"  - [{i:>3}] {name_by_idx.get(i, '?')}  "
                            f"smoothed={validation['trend_per_idx'].get(i, 0):.0f}\n")
            else:
                f.write("  - none\n")
            f.write("\nTrimmed POST-bell (high_idx side):\n")
            post = validation["trimmed_post_bell"]
            if post:
                for i in post:
                    f.write(f"  - [{i:>3}] {name_by_idx.get(i, '?')}  "
                            f"smoothed={validation['trend_per_idx'].get(i, 0):.0f}\n")
            else:
                f.write("  - none\n")

            f.write("\nInternal outliers (Pattern B):\n")
            outs = validation["internal_outliers"]
            if not outs:
                f.write("  - none\n")
            for o in outs:
                excluded = "EXCLUDED" if o["confidence"] == "strong" else "FLAGGED (weak)"
                marks = []
                if o["confirmed_by_circle"]:
                    marks.append("circle")
                if o["confirmed_by_dice"]:
                    marks.append("dice")
                marks_str = ",".join(marks) if marks else "no cross-check"
                f.write(f"  - [{o['idx']:>3}] {name_by_idx.get(o['idx'], '?')}  "
                        f"area={o['area']:.0f} trend={o['trend']:.0f} "
                        f"σ=+{o['residual_sigma']:.2f}  "
                        f"confirm=[{marks_str}]  {excluded}\n")

            f.write("\nDecision log:\n")
            for line in validation.get("decision_log", []):
                f.write(f"  - {line}\n")
            f.write("=" * 60 + "\n")

    # CSV mirror
    out_status = {}
    out_reason = {}
    for i in validation.get("trimmed_pre_bell", []):
        out_status[i] = "trimmed_pre"
        out_reason[i] = "below head/shaft threshold from edge"
    for i in validation.get("trimmed_post_bell", []):
        out_status[i] = "trimmed_post"
        out_reason[i] = "below head/shaft threshold from edge"
    for o in validation.get("internal_outliers", []):
        if o["confidence"] == "strong":
            out_status[o["idx"]] = "outlier_internal"
        else:
            out_status[o["idx"]] = "outlier_internal_weak"
        marks = []
        if o["confirmed_by_circle"]:
            marks.append("circle_inlier_drop")
        if o["confirmed_by_dice"]:
            marks.append("dice_drop")
        out_reason[o["idx"]] = ("residual_sigma>thr; " +
                                ("/".join(marks) if marks else "no_cross_check"))
    if validation.get("bell_peak_idx") is not None:
        out_status.setdefault(validation["bell_peak_idx"], "kept")

    # Determine "side" for each slice
    peak = validation.get("bell_peak_idx")
    head_side_val = validation.get("head_side", "unknown")
    def _side_for_idx(i):
        if peak is None or i == peak:
            return "peak"
        if head_side_val == "low_idx":
            return "head" if i < peak else "shaft"
        if head_side_val == "high_idx":
            return "shaft" if i < peak else "head"
        return "unknown"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["slice_idx", "filename", "area_px", "dice", "status",
                         "reason", "residual_sigma", "trend_area", "side"])
        # Sort by idx, include every slice in slices_info with idx>0
        rows_idx = sorted({s["idx"] for s in slices_info if s.get("idx", -1) > 0})
        for i in rows_idx:
            s = next((x for x in slices_info if x.get("idx") == i), None)
            area = float(s.get("area", 0.0) or 0.0) if s else 0.0
            dice_val = float(s.get("dice", 0.0)) if s and "dice" in s else ""
            status = out_status.get(i, "kept")
            reason = out_reason.get(i, "")
            rs = validation.get("residual_sigma_per_idx", {}).get(i, "")
            trend = validation.get("trend_per_idx", {}).get(i, "")
            writer.writerow([
                i, s.get("filename", "") if s else "",
                f"{area:.0f}", f"{dice_val:.3f}" if dice_val != "" else "",
                status, reason,
                f"{rs:.3f}" if isinstance(rs, float) else rs,
                f"{trend:.1f}" if isinstance(trend, float) else trend,
                _side_for_idx(i),
            ])

    return txt_path, csv_path


# ===================================================================
# Render: 5-panel slice PNG with the fitted circle drawn on the image
# ===================================================================

def _load_slice_image(in_dir, fname):
    """Load the original slice as RGB for visualization."""
    dcm_path = os.path.join(in_dir, fname)
    if not os.path.exists(dcm_path):
        base = os.path.splitext(fname)[0]
        dcm_path = os.path.join(in_dir, base + ".dcm")
    if not os.path.exists(dcm_path):
        return None
    try:
        from DCM.load_dicom_as_image import load_dicom_as_image
        img, _ = load_dicom_as_image(dcm_path)
        return img
    except Exception:
        return None


def render_slice_with_circles(in_dir, out_dir, fname, mask, dcm,
                              slice_circles, info="", boundary_px=None):
    """Render a PNG (5-panel if `mask` is available, 3-panel fallback otherwise)
    showing the fitted articular circles drawn back on the original DICOM slice.

    Layouts:
      mask present : Original | Overlay | Mask | Contour | Circles
      mask absent  : Original | Boundary points | Circles
    """
    img = _load_slice_image(in_dir, fname)
    if img is None or dcm is None:
        return None

    base = os.path.splitext(fname)[0]
    has_mask = mask is not None

    if has_mask:
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        axes_iter = list(axes)
        circle_ax = axes[4]

        axes[0].imshow(img)
        axes[0].set_title(f"{base}\nOriginal")
        axes[0].axis("off")

        axes[1].imshow(img)
        axes[1].imshow(mask, alpha=0.45, cmap="Blues")
        axes[1].set_title(f"Overlay\n{info}")
        axes[1].axis("off")

        axes[2].imshow(mask, cmap="gray")
        axes[2].set_title(f"Mask\nArea: {int(np.sum(mask))} px")
        axes[2].axis("off")

        contours, _ = cv2.findContours((mask * 255).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img_contours = img.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
        axes[3].imshow(img_contours)
        axes[3].set_title(f"Contour\n{len(contours)} contour(s)")
        axes[3].axis("off")

        circle_ax.imshow(img)
        circle_ax.set_title(f"Articular circles\n{len(slice_circles)} circle(s)")
        circle_ax.axis("off")
    else:
        # Fallback layout when binary mask is unavailable
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        circle_ax = axes[2]

        axes[0].imshow(img)
        axes[0].set_title(f"{base}\nOriginal")
        axes[0].axis("off")

        axes[1].imshow(img)
        if boundary_px is not None and len(boundary_px) > 0:
            axes[1].plot(boundary_px[:, 0], boundary_px[:, 1],
                         ".", color="cyan", markersize=2, alpha=0.7)
        axes[1].set_title(f"Boundary points\n(no .npy mask available)")
        axes[1].axis("off")

        circle_ax.imshow(img)
        circle_ax.set_title(f"Articular circles\n{len(slice_circles)} circle(s)\n{info}")
        circle_ax.axis("off")
    axes = None  # silence linter; we use circle_ax for panel 5

    if not slice_circles:
        circle_ax.text(0.5, 0.5, "no circle fitted", transform=circle_ax.transAxes,
                       ha="center", va="center", color="red", fontsize=10)
    else:
        # Convert each circle (3D) back to pixel coords by sampling its
        # parametric form in 3D and projecting through the DICOM affine.
        theta = np.linspace(0, 2 * np.pi, 240)
        colors = ["red", "orange", "magenta"]
        for ci, c in enumerate(slice_circles):
            color = colors[ci % len(colors)]
            ring3d = (c["center3d"][np.newaxis, :]
                      + c["radius_mm"]
                        * (np.outer(np.cos(theta), c["axis_u"])
                           + np.outer(np.sin(theta), c["axis_v"])))
            ring_px = world_to_pixel(ring3d, dcm)
            circle_ax.plot(ring_px[:, 0], ring_px[:, 1], "-", color=color,
                           lw=2, alpha=0.85,
                           label=f"#{ci}: R={c['radius_mm']:.1f} mm, "
                                 f"{c['num_inliers']}/{c['num_total_in_slice']} pts")
            # Center and inlier dots
            cen_px = world_to_pixel(c["center3d"][np.newaxis, :], dcm)[0]
            circle_ax.plot(cen_px[0], cen_px[1], "+", color=color,
                           markersize=14, markeredgewidth=2)
            inl_px = world_to_pixel(c["inliers_world"], dcm)
            circle_ax.plot(inl_px[:, 0], inl_px[:, 1], ".", color=color,
                           markersize=4, alpha=0.9)
        circle_ax.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{base}_circle.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ===================================================================
# Step 6 – Summary diagram per dataset
# ===================================================================

def generate_summary_diagram(dataset_name, slices_info, all_world_pts,
                             sphere_center, sphere_radius, inlier_mask,
                             arcs, slice_circles, output_dir,
                             validation=None, direction_info=None):
    """Create a single summary PNG for the dataset.

    `slice_circles` is the list of per-slice circle fits (the second
    geometric feature — circles lying on the articular surface).
    `validation` (optional) is the AreaValidation dict from Step 1.5; when
    provided, panel 3 overlays bell-extent + status colours and panel 7
    appends a validation block."""
    fig = plt.figure(figsize=(22, 20))
    fig.suptitle(f"Humerus Boundary Analysis – {os.path.basename(dataset_name)}",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3,
                          height_ratios=[1, 1.4, 1])

    # ---- Panel 1: Slice classification bar chart ----
    ax1 = fig.add_subplot(gs[0, 0])
    correct = [s for s in slices_info if s["class"] == "correct"]
    ambiguous = [s for s in slices_info if s["class"] == "ambiguous"]
    failed = [s for s in slices_info if s["class"] == "failed"]
    counts = [len(correct), len(ambiguous), len(failed)]
    colors_bar = ["#2ecc71", "#f1c40f", "#e74c3c"]
    bars = ax1.bar(["Correct", "Ambiguous", "Failed"], counts, color=colors_bar, edgecolor="black")
    for bar, c in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(c), ha="center", fontweight="bold")
    ax1.set_ylabel("Number of slices")
    ax1.set_title("1. Slice Classification (Dice-based)")

    # ---- Panel 2: Dice score per slice ----
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_sl = sorted([s for s in slices_info if s.get("idx", -1) > 0],
                       key=lambda s: s["idx"])
    if sorted_sl:
        xs = [s["idx"] for s in sorted_sl]
        ys = [s.get("dice", 0) for s in sorted_sl]
        cls = [s["class"] for s in sorted_sl]
        color_map = {"correct": "#2ecc71", "ambiguous": "#f1c40f", "failed": "#e74c3c"}
        cs = [color_map[c] for c in cls]
        ax2.bar(xs, ys, color=cs, edgecolor="black", linewidth=0.5)
        ax2.axhline(DICE_CORRECT_THRESHOLD, color="#2ecc71", ls="--", lw=1,
                     label=f"Correct ≥ {DICE_CORRECT_THRESHOLD}")
        ax2.axhline(DICE_AMBIGUOUS_THRESHOLD, color="#f1c40f", ls="--", lw=1,
                     label=f"Ambiguous ≥ {DICE_AMBIGUOUS_THRESHOLD}")
        ax2.set_xlabel("Slice index")
        ax2.set_ylabel("Dice score")
        ax2.set_title("2. Dice Score per Slice")
        ax2.legend(fontsize=7, loc="lower left")

    # ---- Panel 3: Area curve (with validation overlay if available) ----
    ax3 = fig.add_subplot(gs[0, 2])
    if sorted_sl:
        areas = [s.get("area", 0) for s in sorted_sl]
        # Determine per-slice status from validation
        status_by_idx = {}
        if validation is not None and not validation.get("skipped", True):
            for i in validation.get("trimmed_pre_bell", []):
                status_by_idx[i] = "trim_pre"
            for i in validation.get("trimmed_post_bell", []):
                status_by_idx[i] = "trim_post"
            for o in validation.get("internal_outliers", []):
                status_by_idx[o["idx"]] = ("outlier_strong"
                                           if o["confidence"] == "strong"
                                           else "outlier_weak")
        # Connect-the-dots line first, dim
        ax3.plot(xs, areas, "-", color="#3498db", lw=1, alpha=0.4)
        # Overlay rolling-median trend if available
        if validation is not None and validation.get("trend_per_idx"):
            tx = sorted(validation["trend_per_idx"].keys())
            ty = [validation["trend_per_idx"][i] for i in tx]
            ax3.plot(tx, ty, "--", color="#7f8c8d", lw=1.2,
                     alpha=0.8, label="trend (rolling median)")
        # Status-coloured markers
        color_for = {
            "trim_pre": "#bdc3c7",
            "trim_post": "#7f8c8d",
            "outlier_strong": "#e74c3c",
            "outlier_weak": "#f39c12",
        }
        for x, y in zip(xs, areas):
            st = status_by_idx.get(x, "kept")
            c = color_for.get(st, "#27ae60")  # default green = kept
            sz = 60 if st.startswith("outlier") else 35
            edge = "black" if st.startswith("outlier") else "none"
            ax3.scatter([x], [y], color=c, s=sz, zorder=3,
                        edgecolor=edge, linewidth=0.6)
        # Vertical bell-extent lines
        if validation is not None and not validation.get("skipped", True):
            bs, be = validation.get("bell_start_idx"), validation.get("bell_end_idx")
            if bs is not None:
                ax3.axvline(bs - 0.5, color="#16a085", ls=":", lw=1, alpha=0.7)
            if be is not None:
                ax3.axvline(be + 0.5, color="#16a085", ls=":", lw=1, alpha=0.7)
            # Direction caret
            head_side = validation.get("head_side", "unknown")
            peak = validation.get("bell_peak_idx")
            if peak is not None and head_side in ("low_idx", "high_idx"):
                arrow = "← H" if head_side == "low_idx" else "H →"
                ax3.text(0.02 if head_side == "low_idx" else 0.98, 0.95, arrow,
                         transform=ax3.transAxes,
                         ha="left" if head_side == "low_idx" else "right",
                         va="top", fontsize=10, fontweight="bold",
                         color="#c0392b",
                         bbox=dict(boxstyle="round,pad=0.2",
                                   facecolor="white", edgecolor="#c0392b",
                                   alpha=0.7))
        ax3.set_xlabel("Slice index")
        ax3.set_ylabel("Mask area (px)")
        title_extra = ""
        if validation is not None and not validation.get("skipped", True):
            title_extra = (f"  bell=[{validation['bell_start_idx']}.."
                           f"{validation['bell_end_idx']}]")
        ax3.set_title(f"3. Mask Area per Slice{title_extra}")
        ax3.grid(True, alpha=0.3)
        if validation is not None and validation.get("trend_per_idx"):
            ax3.legend(fontsize=7, loc="lower right")

    # ---- Panel 4: 3D boundary points (world coords) ----
    ax4 = fig.add_subplot(gs[1, 0], projection="3d")
    if all_world_pts is not None and len(all_world_pts) > 0:
        ax4.scatter(all_world_pts[:, 0], all_world_pts[:, 1], all_world_pts[:, 2],
                    c=all_world_pts[:, 2], cmap="viridis", s=0.5, alpha=0.5)
        ax4.set_xlabel("X (mm)")
        ax4.set_ylabel("Y (mm)")
        ax4.set_zlabel("Z (mm)")
    ax4.set_title("4. Boundary Points (DICOM coords)")

    # ---- Panel 5: Sphere fit + inliers ----
    ax5 = fig.add_subplot(gs[1, 1], projection="3d")
    if all_world_pts is not None and inlier_mask is not None and sphere_center is not None:
        outliers = all_world_pts[~inlier_mask]
        inliers = all_world_pts[inlier_mask]
        if len(outliers) > 0:
            ax5.scatter(outliers[:, 0], outliers[:, 1], outliers[:, 2],
                        c="gray", s=0.3, alpha=0.2, label="Non-sphere")
        if len(inliers) > 0:
            ax5.scatter(inliers[:, 0], inliers[:, 1], inliers[:, 2],
                        c="red", s=1.5, alpha=0.8, label="Sphere inliers")

        # Draw wireframe sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        sx = sphere_center[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
        sy = sphere_center[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
        sz = sphere_center[2] + sphere_radius * np.outer(np.ones_like(u), np.cos(v))
        ax5.plot_wireframe(sx, sy, sz, color="blue", alpha=0.08, linewidth=0.3)

        ax5.set_xlabel("X (mm)")
        ax5.set_ylabel("Y (mm)")
        ax5.set_zlabel("Z (mm)")
        ax5.legend(fontsize=7)
    ax5.set_title(
        f"5. Sphere Fit  R={sphere_radius:.1f} mm" if sphere_radius else "5. Sphere Fit (N/A)"
    )

    # ---- Panel 6: Per-slice circles in 3D (Feature 2) ----
    ax6 = fig.add_subplot(gs[1, 2], projection="3d")
    if slice_circles:
        theta = np.linspace(0, 2 * np.pi, 200)
        for c in slice_circles:
            color = "red" if c["is_articular"] else "lightgray"
            lw = 1.8 if c["is_articular"] else 0.6
            alpha = 0.9 if c["is_articular"] else 0.45
            ring = (c["center3d"][np.newaxis, :]
                    + c["radius_mm"]
                      * (np.outer(np.cos(theta), c["axis_u"])
                         + np.outer(np.sin(theta), c["axis_v"])))
            ax6.plot(ring[:, 0], ring[:, 1], ring[:, 2],
                     color=color, lw=lw, alpha=alpha)
            ax6.scatter([c["center3d"][0]], [c["center3d"][1]], [c["center3d"][2]],
                        color=color, s=15, alpha=alpha)
        ax6.set_xlabel("X (mm)")
        ax6.set_ylabel("Y (mm)")
        ax6.set_zlabel("Z (mm)")
        # Custom legend
        from matplotlib.lines import Line2D
        ax6.legend(handles=[
            Line2D([0], [0], color="red", lw=2, label="Articular slice"),
            Line2D([0], [0], color="lightgray", lw=1, label="Non-articular slice"),
        ], fontsize=7, loc="upper right")
    else:
        ax6.text(0.5, 0.5, 0.5, "no circles fitted",
                 transform=ax6.transAxes, ha="center")
    n_articular = sum(1 for c in slice_circles if c["is_articular"])
    ax6.set_title(f"6. Per-slice Circles (Feature 2)\n"
                  f"{len(slice_circles)} circles, {n_articular} on articular surface")

    # ---- Panel 7: Analysis summary text (full row) ----
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")
    text_lines = [
        "FEATURE 1 — Global sphere fit (humeral head approximation)",
        "-" * 78,
    ]
    if sphere_center is not None and sphere_radius is not None:
        text_lines += [
            f"  center = ({sphere_center[0]:.1f}, {sphere_center[1]:.1f}, "
            f"{sphere_center[2]:.1f}) mm    radius = {sphere_radius:.1f} mm",
            f"  inliers on sphere     : {int(np.sum(inlier_mask))} / "
            f"{len(all_world_pts) if all_world_pts is not None else 0} "
            f"boundary points",
            f"  arcs (>= {MIN_ARC_POINTS} pts, >= {int(MIN_ARC_INLIER_RATIO*100)}% of slice) : "
            f"{len(arcs)}",
        ]
    else:
        text_lines.append("  (sphere fit unavailable)")

    text_lines += [
        "",
        "FEATURE 2 — Per-slice circles (lying on articular surface)",
        "-" * 78,
        f"  {len(slice_circles)} circles fitted across "
        f"{len({c['slice_idx'] for c in slice_circles})} slices  "
        f"(articular threshold: area >= {int(ARTICULAR_AREA_RATIO*100)}% of max area)",
        f"  {n_articular} circles flagged as articular",
        "",
    ]
    if slice_circles:
        text_lines.append(
            f"  {'sl':>3} {'k':>2} {'art':>3} {'R(mm)':>7} "
            f"{'cx':>7} {'cy':>7} {'cz':>7} {'inl/tot':>9} {'res(mm)':>8}"
        )
        for c in slice_circles[:30]:
            text_lines.append(
                f"  {c['slice_idx']:>3} {c['circle_idx']:>2} "
                f"{'  *' if c['is_articular'] else '':>3} "
                f"{c['radius_mm']:>7.2f} "
                f"{c['center3d'][0]:>7.1f} {c['center3d'][1]:>7.1f} "
                f"{c['center3d'][2]:>7.1f} "
                f"{c['num_inliers']:>4}/{c['num_total_in_slice']:<3} "
                f"{c['fit_residual_mm']:>8.3f}"
            )
        if len(slice_circles) > 30:
            text_lines.append(f"  ... ({len(slice_circles) - 30} more)")

    # ----- AREA-CURVE VALIDATION block -----
    if validation is not None:
        text_lines += [
            "",
            "AREA-CURVE VALIDATION (Step 1.5)",
            "-" * 78,
        ]
        if validation.get("skipped"):
            text_lines.append(
                f"  skipped — reason: {validation.get('reason_skipped')}"
            )
        else:
            dir_str = "n/a"
            head_str = "n/a"
            if direction_info is not None:
                dir_str = direction_info.get("direction", "unknown")
                head_str = direction_info.get("head_side", "unknown")
            text_lines += [
                f"  direction = {dir_str}    head_side = {head_str}",
                f"  bell extent: [{validation['bell_start_idx']}.."
                f"{validation['bell_end_idx']}]   "
                f"peak = slice {validation['bell_peak_idx']} "
                f"({validation['bell_peak_area']:.0f} px)",
                f"  trimmed pre-bell : {len(validation['trimmed_pre_bell'])}    "
                f"trimmed post-bell : {len(validation['trimmed_post_bell'])}",
            ]
            outs = validation.get("internal_outliers", [])
            n_strong = sum(1 for o in outs if o["confidence"] == "strong")
            n_weak = len(outs) - n_strong
            text_lines.append(
                f"  internal outliers: {n_strong} strong (excluded), "
                f"{n_weak} weak (flagged only)"
            )
            for o in outs[:6]:
                marks = []
                if o["confirmed_by_circle"]:
                    marks.append("circle")
                if o["confirmed_by_dice"]:
                    marks.append("dice")
                marks_str = ",".join(marks) if marks else "no_cross_check"
                tag = "STRONG" if o["confidence"] == "strong" else "weak  "
                text_lines.append(
                    f"    [{o['idx']:>3}] {tag}  area={o['area']:.0f} "
                    f"trend={o['trend']:.0f} σ=+{o['residual_sigma']:.2f} "
                    f"[{marks_str}]"
                )

    text_lines += [
        "",
        f"Slice classification: {len(correct)} correct, "
        f"{len(ambiguous)} ambiguous, {len(failed)} failed",
    ]

    ax7.text(0.01, 0.98, "\n".join(text_lines), transform=ax7.transAxes,
             fontsize=8.5, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax7.set_title("7. Analysis Summary", loc="left")

    out_path = os.path.join(output_dir, "boundary_analysis_summary.png")
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  💾 Summary diagram saved: {out_path}")


# ===================================================================
# Main pipeline
# ===================================================================

def resolve_input_dir_and_files(in_dir, out_dir):
    """
    Resolve the actual input directory containing DICOM files and return
    a sorted list of file paths plus a mapping from basename -> path.
    Handles nested directory structures.
    """
    exts = ["*.dcm", "*.DCM"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(in_dir, ext)))
        files.extend(glob.glob(os.path.join(in_dir, "*", ext)))
    files = sorted(set(files), key=lambda x: os.path.basename(x))
    name_to_path = {os.path.basename(f): f for f in files}
    return files, name_to_path


def process_dataset(dataset_in_name, dataset_out_name):
    """Run the full analysis pipeline for one dataset."""
    in_dir = os.path.join(IN_ROOT, dataset_in_name)
    out_dir = os.path.join(OUT_ROOT, dataset_out_name)

    short_name = os.path.basename(dataset_out_name)
    print(f"\n{'='*70}")
    print(f"Processing: {short_name}")
    print(f"{'='*70}")

    summary_path = os.path.join(out_dir, "propagation_summary.txt")
    if not os.path.exists(summary_path):
        # Try one level deeper (nested output dirs)
        nested = glob.glob(os.path.join(out_dir, "*", "propagation_summary.txt"))
        if nested:
            out_dir = os.path.dirname(nested[0])
            summary_path = nested[0]
        else:
            print(f"  ⚠️  No propagation_summary.txt found – skipping")
            return

    # Resolve input DICOM files
    all_dcm_files, dcm_name_map = resolve_input_dir_and_files(in_dir, out_dir)
    if not all_dcm_files:
        print(f"  ⚠️  No DICOM files found in {in_dir} – skipping")
        return

    # Determine middle index for contour fallback
    middle_idx = len(all_dcm_files) // 2

    # --- Step 1 & 2: Parse + Classify ---
    slices_info, failed_names = parse_summary(summary_path)
    slices_info = classify_slices(slices_info, failed_names)

    correct = [s for s in slices_info if s["class"] == "correct"]
    ambiguous = [s for s in slices_info if s["class"] == "ambiguous"]
    failed = [s for s in slices_info if s["class"] == "failed"]

    print(f"  Step 1: Classification – {len(correct)} correct, {len(ambiguous)} ambiguous, {len(failed)} failed")
    print(f"  Step 2: Ambiguous slices needing review:")
    for s in sorted(ambiguous, key=lambda x: x.get("idx", 0)):
        print(f"          [{s.get('idx', '?')}] {s['filename']}  Dice={s.get('dice', 0):.3f}")

    # --- Step 1.5 (PASS 1): direction + bell-extent trimming ---
    direction_info = {"direction": "unknown", "head_side": "unknown",
                      "z_per_idx": {}, "source": "n/a", "delta_total": None}
    validation = None
    area_validation_enabled = ENABLE_AREA_VALIDATION and not _CLI_OVERRIDES.get("no_area_validation")
    if area_validation_enabled:
        # Load DICOMs for the CORRECT slices once, to detect direction.
        dcm_lookup_for_dir = {}
        for s in correct:
            fname = s.get("filename", "")
            idx = s.get("idx", -1)
            if idx <= 0:
                continue
            dpath = dcm_name_map.get(fname)
            if dpath is None:
                continue
            try:
                dcm = pydicom.dcmread(dpath, stop_before_pixels=True)
                dcm_lookup_for_dir[idx] = dcm
            except Exception:
                continue
        direction_info = detect_acquisition_direction(dcm_lookup_for_dir)
        print(f"  Step 1.5: direction={direction_info['direction']}  "
              f"head_side={direction_info['head_side']}  "
              f"(source={direction_info['source']})")
        # Pass 1 — area+dice only; circles not yet computed
        validation = validate_area_curve(slices_info, direction_info,
                                         slice_circles_lookup=None,
                                         head_trim_ratio=_CLI_OVERRIDES.get("head_trim_ratio"),
                                         shaft_trim_ratio=_CLI_OVERRIDES.get("shaft_trim_ratio"))
        if validation.get("skipped"):
            print(f"    ⚠️  validation skipped: {validation.get('reason_skipped')}")
        else:
            print(f"    bell extent: [{validation['bell_start_idx']}..{validation['bell_end_idx']}]  "
                  f"trimmed pre={len(validation['trimmed_pre_bell'])}  "
                  f"post={len(validation['trimmed_post_bell'])}")
        # Filter correct → only kept slices contribute to boundary loop
        kept = validation.get("kept_indices") or {s["idx"] for s in correct if s.get("idx", -1) > 0}
        correct = [s for s in correct if s.get("idx", -1) in kept]

    # --- Step 3 & 4: Extract boundary pixels & convert to world coords ---
    print(f"  Step 3-4: Extracting boundary pixels and converting to 3D world coords...")

    # Build a mapping from slice index -> filename for correct slices
    correct_by_idx = {s["idx"]: s for s in correct if s.get("idx", -1) > 0}

    # Try loading masks first; if none exist, fall back to contour_points_3d
    contour_fallback = None  # dict {slice_idx: (N,2) pixel coords}

    all_world_points = []
    all_slice_labels = []
    per_slice_data = []
    masks_found = 0

    for s in sorted(correct, key=lambda x: x.get("idx", 0)):
        fname = s["filename"]
        base = os.path.splitext(fname)[0]
        slice_idx = s.get("idx", 0)

        # Load mask
        mask = load_mask(out_dir, base)
        boundary_px = None
        if mask is not None:
            masks_found += 1
            boundary_px = extract_boundary_pixels(mask)
            if len(boundary_px) == 0:
                boundary_px = None

        # Fallback: use pre-saved contour points
        if boundary_px is None:
            if contour_fallback is None:
                contour_fallback = load_contour_points_by_slice(
                    out_dir, all_dcm_files, middle_idx, z_spacing=12)
            if slice_idx in contour_fallback:
                boundary_px = contour_fallback[slice_idx]

        if boundary_px is None or len(boundary_px) == 0:
            continue

        # Load DICOM header
        dcm_path = dcm_name_map.get(fname)
        if dcm_path is None:
            continue
        try:
            dcm = pydicom.dcmread(dcm_path)
        except Exception:
            continue

        # Check required DICOM attributes
        if not all(hasattr(dcm, attr) for attr in
                   ["ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing"]):
            print(f"    ⚠️  {fname}: missing DICOM spatial attributes – skipping")
            continue

        # Convert to world coords
        world_pts = pixel_to_world(boundary_px, dcm)
        all_world_points.append(world_pts)
        all_slice_labels.append(np.full(len(world_pts), slice_idx))

        per_slice_data.append({
            "filename": fname,
            "idx": slice_idx,
            "num_boundary_pts": len(boundary_px),
            "num_world_pts": len(world_pts),
            "mask": mask,           # may be None if loaded via fallback
            "dcm": dcm,
            "area_px": float(s.get("area", 0.0)),
            "boundary_px": boundary_px,  # for the no-mask render fallback
        })

    if all_world_points:
        all_world_points = np.vstack(all_world_points)
        all_slice_labels = np.concatenate(all_slice_labels)
    else:
        all_world_points = np.empty((0, 3))
        all_slice_labels = np.empty(0)

    print(f"    Masks loaded: {masks_found}, contour fallback slices: {len(contour_fallback) if contour_fallback else 0}")
    print(f"    Total boundary points in world coords: {len(all_world_points)}")

    # Save world coordinates
    if len(all_world_points) > 0:
        csv_path = os.path.join(out_dir, "boundary_world_coords.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x_mm", "y_mm", "z_mm", "slice_idx"])
            for pt, sl in zip(all_world_points, all_slice_labels):
                writer.writerow([f"{pt[0]:.4f}", f"{pt[1]:.4f}", f"{pt[2]:.4f}", int(sl)])
        print(f"    💾 World coords saved: {csv_path}")

        npy_path = os.path.join(out_dir, "boundary_world_coords.npy")
        np.save(npy_path, all_world_points)

    # Save classification report
    report_path = os.path.join(out_dir, "slice_classification.txt")
    with open(report_path, "w") as f:
        f.write(f"Slice Classification Report – {short_name}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Thresholds: CORRECT >= {DICE_CORRECT_THRESHOLD}, "
                f"AMBIGUOUS >= {DICE_AMBIGUOUS_THRESHOLD}, FAILED < {DICE_AMBIGUOUS_THRESHOLD}\n\n")
        for label, group, color in [("CORRECT", correct, "green"),
                                     ("AMBIGUOUS", ambiguous, "yellow"),
                                     ("FAILED", failed, "red")]:
            f.write(f"[{label}] ({len(group)} slices)\n")
            for s in sorted(group, key=lambda x: x.get("idx", 0)):
                f.write(f"  [{s.get('idx', '?'):>3}] {s['filename']:<20}  "
                        f"Dice={s.get('dice', 0):.3f}  Area={s.get('area', 0):.0f}\n")
            f.write("\n")
    print(f"    💾 Classification report: {report_path}")

    # --- Step 5: Detect circular arcs (RANSAC sphere fit) ---
    print(f"  Step 5: Detecting circular arcs (sphere RANSAC)...")
    sphere_center = None
    sphere_radius = None
    inlier_mask = None
    arcs = []

    if len(all_world_points) >= MIN_ARC_POINTS:
        sphere_center, sphere_radius, inlier_mask = ransac_sphere(all_world_points)
        if sphere_center is not None:
            print(f"    Sphere: center=({sphere_center[0]:.1f}, {sphere_center[1]:.1f}, "
                  f"{sphere_center[2]:.1f}), R={sphere_radius:.1f} mm, "
                  f"inliers={np.sum(inlier_mask)}/{len(all_world_points)}")
            arcs = find_circular_arcs_per_slice(all_world_points, all_slice_labels, inlier_mask)
            print(f"    Circular arcs (>= {MIN_ARC_POINTS} pts per slice): {len(arcs)}")
            for a in arcs:
                print(f"      Slice {int(a['slice'])}: {a['num_points']} points on sphere")
        else:
            print(f"    ⚠️  RANSAC sphere fit did not converge.")
    else:
        print(f"    ⚠️  Not enough boundary points for sphere fitting ({len(all_world_points)})")

    # Save arcs data
    if arcs:
        arcs_path = os.path.join(out_dir, "circular_arcs.csv")
        with open(arcs_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["slice_idx", "num_points", "x_mm", "y_mm", "z_mm"])
            for a in arcs:
                for pt in a["points"]:
                    writer.writerow([int(a["slice"]), a["num_points"],
                                     f"{pt[0]:.4f}", f"{pt[1]:.4f}", f"{pt[2]:.4f}"])
        print(f"    💾 Circular arcs saved: {arcs_path}")

    # --- Step 5b: Per-slice circles (FEATURE 2 — articular surface) ---
    print(f"  Step 5b: Fitting per-slice circles (Feature 2)...")
    areas_by_slice = {d["idx"]: d["area_px"] for d in per_slice_data}
    slice_circles = fit_circles_per_slice(
        all_world_points, all_slice_labels, areas_by_slice
    )
    n_articular = sum(1 for c in slice_circles if c["is_articular"])
    print(f"    Per-slice circles: {len(slice_circles)} total "
          f"({n_articular} on articular surface)")
    for c in slice_circles[:20]:
        flag = "🔴 articular" if c["is_articular"] else "·"
        print(f"      Slice {c['slice_idx']:>3}#{c['circle_idx']}: "
              f"R={c['radius_mm']:6.2f} mm, "
              f"inliers={c['num_inliers']:>3}/{c['num_total_in_slice']:<3}, "
              f"residual={c['fit_residual_mm']:.2f} mm  {flag}")

    if slice_circles:
        # circles.csv: one row per fitted circle (centers + radii — what the
        # user explicitly asked for)
        circles_path = os.path.join(out_dir, "slice_circles.csv")
        with open(circles_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "slice_idx", "circle_idx", "is_articular", "area_px",
                "center_x_mm", "center_y_mm", "center_z_mm",
                "normal_x", "normal_y", "normal_z",
                "radius_mm", "num_inliers", "num_total_in_slice",
                "inlier_ratio", "fit_residual_mm",
            ])
            for c in slice_circles:
                writer.writerow([
                    c["slice_idx"], c["circle_idx"],
                    int(c["is_articular"]), f"{c['area_px']:.0f}",
                    f"{c['center3d'][0]:.4f}", f"{c['center3d'][1]:.4f}",
                    f"{c['center3d'][2]:.4f}",
                    f"{c['normal'][0]:.6f}", f"{c['normal'][1]:.6f}",
                    f"{c['normal'][2]:.6f}",
                    f"{c['radius_mm']:.4f}",
                    c["num_inliers"], c["num_total_in_slice"],
                    f"{c['inlier_ratio']:.4f}",
                    f"{c['fit_residual_mm']:.4f}",
                ])
        print(f"    💾 Per-slice circles (centers + radii) saved: {circles_path}")

        # Optional: also dump the inlier points of every fitted circle
        # (useful for downstream analysis / plotting)
        pts_path = os.path.join(out_dir, "slice_circles_points.csv")
        with open(pts_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "slice_idx", "circle_idx", "is_articular",
                "x_mm", "y_mm", "z_mm",
            ])
            for c in slice_circles:
                for pt in c["inliers_world"]:
                    writer.writerow([
                        c["slice_idx"], c["circle_idx"],
                        int(c["is_articular"]),
                        f"{pt[0]:.4f}", f"{pt[1]:.4f}", f"{pt[2]:.4f}",
                    ])

        # Render the 5-panel PNG for every articular slice
        circles_by_slice = {}
        for c in slice_circles:
            circles_by_slice.setdefault(c["slice_idx"], []).append(c)

        n_rendered = 0
        for d in per_slice_data:
            slice_idx = d["idx"]
            if slice_idx not in circles_by_slice:
                continue
            sc = circles_by_slice[slice_idx]
            # Render only when at least one circle is articular — that's where
            # the methodology actually needs visual confirmation
            if not any(c["is_articular"] for c in sc):
                continue
            mask = d["mask"]
            if mask is None:
                # mask was loaded via fallback path — try once more
                mask = load_mask(out_dir, os.path.splitext(d["filename"])[0])
            # mask may legitimately be None (older runs without _mask.npy);
            # the renderer falls back to a 3-panel layout in that case.
            if d["dcm"] is None:
                continue
            info = (f"Dice from segmentation (see _seg.png)  |  "
                    f"{len(sc)} circle(s), articular")
            try:
                out_png = render_slice_with_circles(
                    in_dir, out_dir, d["filename"], mask, d["dcm"],
                    sc, info=info, boundary_px=d.get("boundary_px"),
                )
                if out_png:
                    n_rendered += 1
            except Exception as e:
                print(f"    ⚠️  Could not render circle PNG for "
                      f"{d['filename']}: {e}")
        if n_rendered:
            print(f"    🖼  {n_rendered} per-slice *_circle.png saved (articular slices)")

    # --- Step 1.5 (PASS 2): confirm internal outliers using per-slice circles ---
    if area_validation_enabled and validation is not None and not validation.get("skipped"):
        print(f"  Step 1.5 (Pass 2): confirming internal outliers with circle inlier_ratio...")
        circle_lookup = {c["slice_idx"]: c for c in slice_circles
                         if c.get("circle_idx", 0) == 0}
        validation_pass2 = validate_area_curve(
            slices_info, direction_info,
            slice_circles_lookup=circle_lookup,
            head_trim_ratio=_CLI_OVERRIDES.get("head_trim_ratio"),
            shaft_trim_ratio=_CLI_OVERRIDES.get("shaft_trim_ratio"),
        )
        # Pass 2 may add NEW strong outliers (because circles confirmed them).
        new_strong = {o["idx"] for o in validation_pass2.get("internal_outliers", [])
                      if o["confidence"] == "strong"}
        prev_strong = {o["idx"] for o in validation.get("internal_outliers", [])
                       if o["confidence"] == "strong"}
        newly_excluded = sorted(new_strong - prev_strong)
        # Adopt Pass 2 as the authoritative validation
        validation = validation_pass2

        if newly_excluded:
            print(f"    🔻 {len(newly_excluded)} additional outlier(s) confirmed by circle "
                  f"inlier_ratio: {newly_excluded}")
            # Retroactively rewrite affected CSVs (boundary_world_coords +
            # slice_circles + slice_circles_points + circular_arcs).
            kept_now = validation["kept_indices"]

            # boundary_world_coords
            mask_keep = np.array([int(s) in kept_now for s in all_slice_labels],
                                 dtype=bool) if len(all_slice_labels) > 0 else np.array([], dtype=bool)
            kept_pts = all_world_points[mask_keep] if len(all_world_points) > 0 else all_world_points
            kept_labels = all_slice_labels[mask_keep] if len(all_slice_labels) > 0 else all_slice_labels
            csv_path = os.path.join(out_dir, "boundary_world_coords.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["x_mm", "y_mm", "z_mm", "slice_idx"])
                for pt, sl in zip(kept_pts, kept_labels):
                    writer.writerow([f"{pt[0]:.4f}", f"{pt[1]:.4f}",
                                     f"{pt[2]:.4f}", int(sl)])
            np.save(os.path.join(out_dir, "boundary_world_coords.npy"), kept_pts)

            # slice_circles & slice_circles_points
            slice_circles = [c for c in slice_circles
                             if c["slice_idx"] in kept_now]
            if slice_circles:
                circles_path = os.path.join(out_dir, "slice_circles.csv")
                with open(circles_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "slice_idx", "circle_idx", "is_articular", "area_px",
                        "center_x_mm", "center_y_mm", "center_z_mm",
                        "normal_x", "normal_y", "normal_z",
                        "radius_mm", "num_inliers", "num_total_in_slice",
                        "inlier_ratio", "fit_residual_mm",
                    ])
                    for c in slice_circles:
                        writer.writerow([
                            c["slice_idx"], c["circle_idx"],
                            int(c["is_articular"]), f"{c['area_px']:.0f}",
                            f"{c['center3d'][0]:.4f}", f"{c['center3d'][1]:.4f}",
                            f"{c['center3d'][2]:.4f}",
                            f"{c['normal'][0]:.6f}", f"{c['normal'][1]:.6f}",
                            f"{c['normal'][2]:.6f}",
                            f"{c['radius_mm']:.4f}",
                            c["num_inliers"], c["num_total_in_slice"],
                            f"{c['inlier_ratio']:.4f}",
                            f"{c['fit_residual_mm']:.4f}",
                        ])
                pts_path = os.path.join(out_dir, "slice_circles_points.csv")
                with open(pts_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "slice_idx", "circle_idx", "is_articular",
                        "x_mm", "y_mm", "z_mm",
                    ])
                    for c in slice_circles:
                        for pt in c["inliers_world"]:
                            writer.writerow([
                                c["slice_idx"], c["circle_idx"],
                                int(c["is_articular"]),
                                f"{pt[0]:.4f}", f"{pt[1]:.4f}", f"{pt[2]:.4f}",
                            ])
            arcs = [a for a in arcs if int(a["slice"]) in kept_now]
            if arcs:
                arcs_path = os.path.join(out_dir, "circular_arcs.csv")
                with open(arcs_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["slice_idx", "num_points", "x_mm", "y_mm", "z_mm"])
                    for a in arcs:
                        for pt in a["points"]:
                            writer.writerow([int(a["slice"]), a["num_points"],
                                             f"{pt[0]:.4f}", f"{pt[1]:.4f}", f"{pt[2]:.4f}"])

    # Persist the validation report (always, when enabled)
    if area_validation_enabled and validation is not None:
        txt_p, csv_p = write_validation_report(
            out_dir, short_name, validation, direction_info, slices_info,
        )
        print(f"    💾 Area-curve validation report : {txt_p}")
        print(f"    💾 Area-curve validation CSV    : {csv_p}")

    # --- Step 6: Summary diagram ---
    print(f"  Step 6: Generating summary diagram...")
    generate_summary_diagram(
        dataset_name=short_name,
        slices_info=slices_info,
        all_world_pts=all_world_points if len(all_world_points) > 0 else None,
        sphere_center=sphere_center,
        sphere_radius=sphere_radius,
        inlier_mask=inlier_mask,
        arcs=arcs,
        slice_circles=slice_circles,
        output_dir=out_dir,
        validation=validation,
        direction_info=direction_info,
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Humerus boundary analysis pipeline (Steps 1–6 + 1.5 validation)."
    )
    parser.add_argument("--no-area-validation", action="store_true",
                        help="Disable Step 1.5 (direction-aware area-curve validation).")
    parser.add_argument("--head-trim-ratio", type=float, default=None,
                        help=f"Override BELL_BOUNDARY_HEAD_RATIO "
                             f"(default {BELL_BOUNDARY_HEAD_RATIO}).")
    parser.add_argument("--shaft-trim-ratio", type=float, default=None,
                        help=f"Override BELL_BOUNDARY_SHAFT_RATIO "
                             f"(default {BELL_BOUNDARY_SHAFT_RATIO}).")
    parser.add_argument("--enable-resam-fixup", action="store_true",
                        help="Enable Step 1.6 (SAM re-segmentation of outliers). "
                             "Requires torch + segment_anything + the SAM checkpoint.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Process only this dataset (name as in DATASET_MAP).")
    args = parser.parse_args()

    _CLI_OVERRIDES["no_area_validation"] = args.no_area_validation
    _CLI_OVERRIDES["head_trim_ratio"] = args.head_trim_ratio
    _CLI_OVERRIDES["shaft_trim_ratio"] = args.shaft_trim_ratio
    _CLI_OVERRIDES["enable_resam_fixup"] = args.enable_resam_fixup
    _CLI_OVERRIDES["dataset"] = args.dataset

    print("=" * 70)
    print("  HUMERUS BOUNDARY ANALYSIS PIPELINE")
    print("=" * 70)
    if args.no_area_validation:
        print("  ⚙️  Step 1.5 area validation DISABLED via CLI flag")
    if args.enable_resam_fixup:
        print("  ⚙️  Step 1.6 SAM re-segmentation ENABLED via CLI flag")

    items = list(DATASET_MAP.items())
    if args.dataset:
        items = [(k, v) for k, v in items if args.dataset in k or args.dataset in v]
        if not items:
            print(f"  ❌ No dataset matched --dataset {args.dataset!r}")
            return

    for in_name, out_name in items:
        try:
            process_dataset(in_name, out_name)
        except Exception as e:
            print(f"  ❌ Error processing {out_name}: {e}")
            import traceback
            traceback.print_exc()

        # Optional Step 1.6 — runs only if user explicitly opted in
        if args.enable_resam_fixup:
            try:
                from humerus_outlier_resam import resegment_outliers_for_dataset
                resegment_outliers_for_dataset(in_name, out_name)
            except ImportError as e:
                print(f"  ⚠️  Step 1.6 unavailable: {e}")
                print(f"     (requires torch + segment_anything + the SAM checkpoint; "
                      f"this is the heavy install, not venv-analysis/)")
            except Exception as e:
                print(f"  ❌ Step 1.6 failed for {out_name}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("  ALL DATASETS PROCESSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
