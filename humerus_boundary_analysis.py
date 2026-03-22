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

# Minimum points to consider a candidate circular arc
MIN_ARC_POINTS = 6

# RANSAC parameters for circular-arc detection
RANSAC_ITERATIONS = 2000
RANSAC_INLIER_DISTANCE = 2.0  # mm

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
    Among the inlier points (those on the sphere), group by slice and check
    that each slice contributes a contiguous arc of >= MIN_ARC_POINTS points.
    Returns list of dicts with arc info.
    """
    arcs = []
    unique_slices = np.unique(slice_labels[inlier_mask])
    for sl in unique_slices:
        mask_sl = (slice_labels == sl) & inlier_mask
        pts = world_points[mask_sl]
        if len(pts) >= MIN_ARC_POINTS:
            arcs.append({
                "slice": sl,
                "num_points": len(pts),
                "points": pts,
            })
    return arcs


# ===================================================================
# Step 6 – Summary diagram per dataset
# ===================================================================

def generate_summary_diagram(dataset_name, slices_info, all_world_pts,
                             sphere_center, sphere_radius, inlier_mask,
                             arcs, output_dir):
    """Create a single summary PNG for the dataset."""
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(f"Humerus Boundary Analysis – {os.path.basename(dataset_name)}",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

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

    # ---- Panel 3: Area curve ----
    ax3 = fig.add_subplot(gs[0, 2])
    if sorted_sl:
        areas = [s.get("area", 0) for s in sorted_sl]
        ax3.plot(xs, areas, "o-", color="#3498db", markersize=4)
        ax3.set_xlabel("Slice index")
        ax3.set_ylabel("Mask area (px)")
        ax3.set_title("3. Mask Area per Slice")
        ax3.grid(True, alpha=0.3)

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

    # ---- Panel 6: Circular arcs info text ----
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    text_lines = [f"Sphere center: ({sphere_center[0]:.1f}, {sphere_center[1]:.1f}, {sphere_center[2]:.1f}) mm" if sphere_center is not None else "Sphere: N/A",
                  f"Sphere radius: {sphere_radius:.1f} mm" if sphere_radius else "",
                  f"Inlier points: {int(np.sum(inlier_mask))}" if inlier_mask is not None else "",
                  f"Total boundary points: {len(all_world_pts)}" if all_world_pts is not None else "",
                  "",
                  f"Circular arcs detected: {len(arcs)}",
                  ""]
    for a in arcs[:15]:  # show up to 15
        text_lines.append(f"  Slice {a['slice']}: {a['num_points']} pts on sphere")

    text_lines += ["",
                   f"Correct slices: {len(correct)}",
                   f"Ambiguous slices: {len(ambiguous)}",
                   f"Failed slices: {len(failed)}"]

    ax6.text(0.05, 0.95, "\n".join(text_lines), transform=ax6.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax6.set_title("6. Analysis Summary")

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
        output_dir=out_dir,
    )


def main():
    print("=" * 70)
    print("  HUMERUS BOUNDARY ANALYSIS PIPELINE")
    print("=" * 70)

    for in_name, out_name in DATASET_MAP.items():
        try:
            process_dataset(in_name, out_name)
        except Exception as e:
            print(f"  ❌ Error processing {out_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("  ALL DATASETS PROCESSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
