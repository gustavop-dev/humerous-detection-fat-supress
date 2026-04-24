# Humerus Detection in Fat-Suppressed MRI

Segmentation and geometric analysis of the **humerus** from axial fat-suppressed shoulder MRI using **SAM** (Segment Anything Model). The pipeline segments the humerus across DICOM slices, extracts boundary points in real-world 3D coordinates, and detects circular arcs on the humeral head (spherical cap).

## Context

### Clinical / academic motivation

In axial **fat-suppressed** shoulder MRI (PD-FS, T1-FS, PDW-SPAIR), subcutaneous and bone-marrow fat appears dark, which makes the cortical boundary of the **humerus** stand out — but it also makes intensity-based segmentation (Otsu, region growing) brittle, because muscle, fluid, and bright artefacts inside the marrow look very similar to the humeral cortex.

This project tackles that problem by combining a **prompt-based foundation model** (SAM, `vit_b` checkpoint) with **slice-to-slice propagation** and **geometric post-processing**, instead of training a U-Net from scratch (we only have ~8 patient datasets across SIEMENS, GE and Philips scanners — far too few to fine-tune).

### What the pipeline actually does

The flow is broken into four stages, each with a dedicated script:

1. **Seed segmentation** (`one_segmentation.py` for debug, or the first step inside `segment_sam_propagation.py`):
   the user clicks one positive point on the **middle slice** of the volume; SAM proposes 3 masks and the highest-scoring one is refined morphologically (`remove_small_objects` ≥ 500 px → `binary_fill_holes` → opening/closing with disk r=2). In batch mode (`SAM_BATCH_MODE=1`) the seed is placed automatically at `(0.40·W, 0.40·H)` — a position empirically close to the humerus in our axial datasets.

2. **Bidirectional propagation** (`Segmentation/propagation.py`):
   from the central slice we walk both up and down, using the **center of mass of the previous mask** as the SAM positive point on the next slice. Each new mask is compared to the previous one with **Dice / IoU**; if the difference exceeds a **dynamic threshold** (stricter near the center, more permissive at the volume extremes), the algorithm:
   - retries with small offsets `(±10, ±20)` in x/y if the mask comes out empty,
   - then retries adding a **negative point** computed by `negative_points.calculate_negative_point()` (placed ~30% beyond the mask radius in the first free direction), and
   - if it still fails, the slice is logged as **failed** and skipped — it does not block the rest of the volume.

3. **Mask sanity post-processing** (the second half of `segment_sam_propagation.main()`):
   - **Neighbour-overlap repair**: any slice whose mask has more than 25% of its area outside the union of its two neighbours is re-segmented using the closest neighbour's center as positive and the centroid of the "extra" region as negative.
   - **Bright-artefact repair**: in the dilated ring around each mask we look for connected components above the 95th-intensity percentile that cover >8% of the ring. Their centroid becomes a negative point and the slice is re-segmented.
   - All edits are tracked in `propagation_summary.txt` with markers (⚠️ leve, 🚨 severo, ⭕ outlier vecinal).

4. **3D reconstruction + geometric analysis**:
   - `Graphics/grafication.py` resamples each contour to a fixed number of points, triangulates between consecutive contours and writes an **STL** mesh (good enough for 3D printing or import into CAD).
   - `humerus_boundary_analysis.py` (which **does not need SAM or a GPU**) parses `propagation_summary.txt`, classifies each slice as CORRECT / AMBIGUOUS / FAILED based on Dice (≥ 0.80 / ≥ 0.55 / <0.55), converts boundary pixels to **patient coordinates in mm** using the DICOM `ImagePositionPatient` + `ImageOrientationPatient` + `PixelSpacing` affine, and computes the **two geometric features** described below.

### Two-feature geometric representation of the humerus

This is the core methodology — and the part presented at **UEMS**. The humerus is summarised by two complementary geometric primitives:

**Feature 1 — Global sphere (the humeral head as a cap).** A RANSAC sphere fit (2000 iter, 2 mm inlier distance) is run over **all** boundary points across all CORRECT slices. The inliers correspond to the spherical articular cap of the head; the rest of the bone (shaft, tuberosities) falls outside the inlier band. Per-slice arcs that explain ≥ 30 % of a slice's boundary AND have ≥ 10 inliers are exported to `circular_arcs.csv`. The double filter is what suppresses the **single-point spurious arcs** that used to appear in the 3D viz.

**Feature 2 — Per-slice circles lying ON the articular surface.** For each slice with enough boundary points, an iterative 2D RANSAC (Kåsa algebraic fit + refit on inliers, up to 3 circles per slice) is run on the points projected to the slice's best-fit plane via SVD. The resulting circles **do not approximate the head** — they **lie on** the cartilage / glenoid contact zone. Slices whose mask area is ≥ 70 % of the dataset's max area are flagged as **articular** (they are the equatorial slices of the head, identifiable from the area-vs-slice curve in `mask_area_curve.png`). For every articular slice we save a 5-panel `*_circle.png` (Original | Overlay | Mask | Contour | **Fitted circle drawn on the slice**) so the user can visually validate each fit. All centers, normals, radii, inlier counts and residuals go into `slice_circles.csv`; the inlier point clouds go into `slice_circles_points.csv`.

**Why two features and not just one?** The sphere captures the bulk geometry of the head — useful as a coarse anatomical reference. The per-slice circles capture the **local curvature of the articular surface** at the level where rotator-cuff insertions and labral contact actually happen. With the per-slice circles in hand, deviations from a smooth arc on the articular slices become a geometric signature for **localising a tear**: a tear breaks the continuity of the cartilage profile, so the inlier mask of the per-slice circle will drop sharply on the affected angular sector. This is the novel detection mechanism we are building toward; it also enables the **muscular-structure incorporation trick** (using the circle plane + normal as a local frame to project surrounding muscle masks).

**Why direction-aware bell validation matters for the two features**: both Feature 1 and Feature 2 fit shapes to the boundary point cloud. Garbage slices at the extremes (Pattern A / C) bias the global sphere fit toward the volume centre and inflate the articular-max-area calculation. Without Step 1.5, the sphere RANSAC was being dragged toward early-slice noise on the GE dataset (slices 1–7 were spilling muscle/fat points into the cap fit) and the articular threshold was being computed off an over-counted peak. With Step 1.5, both features see only the validated bell, and the articular slices selected by Feature 2 correspond cleanly to the head equator on every dataset.

### How complex is this, really?

Honestly? Not very — it's a research script, not a production system. There are ~2k lines across 9 modules, no class hierarchy, no tests, hardcoded paths, and the configuration lives at the top of each script as module-level globals. The pieces with non-trivial logic are: the **propagation loop with dynamic Dice thresholds**, the **two re-segmentation heuristics** (neighbour overlap and bright artefacts), the **RANSAC sphere fit** (Feature 1), and the **iterative per-slice circle fitter with articular-surface tagging** (Feature 2). Everything else is glue: load DICOM, run SAM, save PNG.

### Datasets included

8 anonymised shoulder MRI volumes from three vendors, with different field-of-view, matrix size and protocols — enough to expose how brittle a fixed-threshold pipeline would be, and why we needed the per-slice repair steps:

| Acquisition | Vendor | Slices | Side |
|---|---|---|---|
| `pd_tse_fs_tra_320_fov150_4` | SIEMENS | 31 | — |
| `t1_tse_fs_tra_320_fov150_5` | SIEMENS | 30 | — |
| `t1_tse_fs_tra_320_fov150_11` | SIEMENS | 30 | — |
| `AXIAL fs PD FSE 256x256` | GE Optima MR360 | 24 | Right |
| `AXIAL-P5SE1 ... pd_tse_fs_tra_320` | SIEMENS | 19 | Right |
| `AXIAL(creo)ePDW_SPAIR` | Philips Ingenia | 25 | Left |
| `axial_A_to_T_fatSupressed` | — | 20 | — |
| `dp_tse_cor_320_fs_8` | SIEMENS | 22 | — |

## Quick Start

```bash
# 1. Clone
git clone https://github.com/gustavop-dev/humerous-detection-fat-supress.git
cd humerous-detection-fat-supress

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git

# 4. Download SAM checkpoint (~375 MB)
mkdir -p Checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P Checkpoints/
```

## Scripts

### 1. `one_segmentation.py` — Segment a single image

Interactive segmentation of one DICOM/PNG image. Edit line 35 to set your image path:

```python
image_path = "Datasets/In/pd_tse_fs_tra_320_fov150_4/IM-0001-0016.dcm"
```

```bash
python one_segmentation.py
```

A window opens with two panels. Mark the humerus and see the mask in real time.

### 2. `segment_sam_propagation.py` — Segment a full dataset

Segments all slices in a DICOM folder using bidirectional propagation from the central slice. Edit lines 41-42:

```python
data_dir = "Datasets/In/pd_tse_fs_tra_320_fov150_4"
output_dir = "Datasets/Out/pd_tse_fs_tra_320_fov150_4"
```

```bash
python segment_sam_propagation.py
```

Generates per-slice segmentation overlays, 3D reconstructions, STL meshes, and a propagation summary.

### 3. `batch_segment_sam_propagation.py` — Batch all datasets

Runs `segment_sam_propagation` on all 8 datasets defined in `DATASET_NAMES`. Uses an automatic seed point (no GUI).

```bash
python batch_segment_sam_propagation.py
```

### 4. `humerus_boundary_analysis.py` — Boundary analysis pipeline

Post-processing analysis on already-segmented datasets. **Does not require SAM or GPU.**

```bash
python humerus_boundary_analysis.py
```

This script runs the full geometric analysis (Features 1 + 2):

| Step | Description | Output |
|------|-------------|--------|
| 1   | Classify slices as **correct** / **ambiguous** / **failed** (Dice-based) | `slice_classification.txt` |
| 2   | List ambiguous slices that need manual review | (printed + in report) |
| 1.5 | **Direction-aware area-curve validation** — detect acquisition direction from DICOM, trim noise outside the bell, flag/exclude internal SAM-grabbed-extra outliers | `area_curve_validation.txt`, `area_curve_validation.csv` |
| 3   | Extract boundary pixels from validated slices only | — |
| 4   | Convert pixels to real-world 3D coords (mm) using DICOM headers | `boundary_world_coords.csv` |
| 5   | **Feature 1** — RANSAC sphere fit, arcs lying on the head sphere | `circular_arcs.csv` |
| 5b  | **Feature 2** — per-slice circle fitter (Kåsa + iterative RANSAC), articular slices auto-detected from the validated area curve | `slice_circles.csv`, `slice_circles_points.csv`, `*_circle.png` |
| 1.5 (Pass 2) | Re-confirm internal outliers using `circle.inlier_ratio`; rewrite affected CSVs if anything new is excluded | (rewrites the CSVs above) |
| 6   | 7-panel summary diagram (Feature 1 + Feature 2 in 3D, area-curve validation overlay) | `boundary_analysis_summary.png` |
| 1.6 (opt-in) | Re-segment confirmed outliers with SAM using neighbour circles as priors. Requires `--enable-resam-fixup` and the heavy deps (torch + segment_anything + SAM checkpoint). | `resam_fixup_log.txt`, `*_mask.bak.npy`, replaced `*_mask.npy` |

The `slice_circles.csv` columns are: `slice_idx, circle_idx, is_articular, area_px, center_{x,y,z}_mm, normal_{x,y,z}, radius_mm, num_inliers, num_total_in_slice, inlier_ratio, fit_residual_mm`. Up to 3 circles per slice are extracted (most slices yield only 1). Each `*_circle.png` is a 5-panel render that adds a fifth panel with the fitted circle drawn back on the original DICOM slice (using the inverse DICOM affine), with one colour per circle and the inlier points overlaid.

### Direction-aware area-curve validation (Step 1.5)

The mask-area-vs-slice curve has three failure patterns the raw segmentation pipeline doesn't catch on its own:
- **Pattern A** — slices that don't actually contain humerus but were still segmented to *something* (e.g. GE dataset slices 1–7);
- **Pattern B** — slices inside the bell where SAM grabbed extra muscle/glenoid (e.g. P5SE1 slices 11–12 spike above the local trend);
- **Pattern C** — slices after the humerus has truly ended (tail noise).

**Anatomical asymmetry** drives the heuristic: the humeral head is roughly spherical, so it closes **abruptly** at the superior end (steep area drop), whereas the diaphysis is roughly cylindrical, so it tapers **gradually** at the inferior end (slow area decrease). A direction-blind heuristic over-trims one or the other side. Step 1.5 reads `ImagePositionPatient` (projected on the slice normal `cross(row_cosine, col_cosine)` from `ImageOrientationPatient`) to decide which file-order direction maps to "head" and applies tighter thresholds to that side.

The trim is the union of two checks:
1. **Static floor from edge** — walk inward from each volume edge; trim contiguous runs of slices below `BELL_BOUNDARY_HEAD_RATIO=0.30` (head) / `BELL_BOUNDARY_SHAFT_RATIO=0.10` (shaft) of the smoothed peak.
2. **Reversal from peak** — walk outward from the peak; trim from the first slice where the smoothed area grows by ≥ `BELL_REVERSAL_RATIO_HEAD=1.10` (head) / `BELL_REVERSAL_RATIO_SHAFT=1.20` (shaft) compared to the previous slice.

Internal Pattern B outliers are detected on the kept slices with a robust residual (rolling-median trend + MAD with a 50 px floor). A flag becomes a STRONG exclusion only if cross-checked by either a `circle.inlier_ratio` drop ≥ 15 % vs neighbours OR a Dice drop ≥ 10 % vs neighbours; otherwise it's logged as `weak` (visible in the report and panel 3 marker, but not excluded). Pattern B without geometric confirmation is hard to detect from analysis alone — the proper fix is to re-segment with SAM using `--enable-resam-fixup` (Step 1.6).

CLI flags:
- `--no-area-validation` — disable Step 1.5 entirely (back to prior behaviour).
- `--head-trim-ratio FLOAT` / `--shaft-trim-ratio FLOAT` — override the static-floor ratios.
- `--enable-resam-fixup` — opt into Step 1.6 (requires SAM/torch).
- `--dataset NAME` — process only one dataset (handy when tuning).

## Interactive Controls

| Action | Control |
|--------|---------|
| Positive point (object) | **Right click** |
| Negative point (exclude) | **Left click** |
| Undo last point | Key `z` |
| Clear all points | Key `c` |
| Skip image (batch) | Key `s` |
| Finish | Close window |

## Project Structure

```
humerous-detection-fat-supress/
├── one_segmentation.py                 # Segment a single image
├── segment_sam_propagation.py          # Segment a full dataset
├── batch_segment_sam_propagation.py    # Batch all datasets
├── humerus_boundary_analysis.py        # Boundary analysis pipeline
├── requirements.txt
├── README.md
│
├── DCM/
│   └── load_dicom_as_image.py          # DICOM loading + dataset info
│
├── Graphics/
│   ├── grafication.py                  # 3D reconstruction, STL export
│   └── interface.py                    # Interactive point selection GUI
│
├── Segmentation/
│   ├── Masks.py                        # Mask operations, contour extraction
│   ├── Metrics.py                      # Dice coefficient, IoU
│   ├── propagation.py                  # Bidirectional propagation logic
│   ├── segment_image.py                # SAM segmentation wrapper
│   └── negative_points.py             # Negative point calculation
│
├── Datasets/
│   ├── In/                             # Input DICOM datasets (8 patients)
│   │   ├── pd_tse_fs_tra_320_fov150_4/
│   │   ├── t1_tse_fs_tra_320_fov150_5/
│   │   ├── t1_tse_fs_tra_320_fov150_11/
│   │   ├── AXIAL fs PD FSE ... hombroDerecho/
│   │   ├── AXIAL-P5SE1 ... SIEMENS/
│   │   ├── AXIAL(creo)ePDW_SPAIR ... hombroIzquierdo/
│   │   ├── axial_A_to_T_fatSupressed_dicoms.../
│   │   └── dp_tse_cor_320_fs_8.../
│   └── Out/                            # Segmentation results per dataset
│       └── <dataset_name>/
│           ├── *_seg.png               # Per-slice segmentation overlay
│           ├── *_mask.npy              # Binary mask (numpy)
│           ├── contour_points_3d.csv   # Contour points (pixel space)
│           ├── propagation_summary.txt # Per-slice metrics
│           ├── slice_classification.txt
│           ├── boundary_world_coords.csv  # 3D coords in mm
│           ├── circular_arcs.csv       # Points on the sphere
│           ├── boundary_analysis_summary.png  # 6-panel diagram
│           ├── reconstruction_3d_*.png
│           ├── solid_mesh_3d_*.png
│           └── modelo_3d.stl
│
└── Checkpoints/                        # SAM weights (not tracked)
    └── sam_vit_b_01ec64.pth
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA GPU, Apple Silicon (MPS), or CPU

Key libraries: `torch`, `torchvision`, `numpy`, `matplotlib`, `opencv-python`, `scikit-image`, `scipy`, `Pillow`, `pydicom`.

## Device Support

Device is auto-detected: CUDA > MPS > CPU.

## Troubleshooting

- **"No module named 'segment_anything'"** → `pip install git+https://github.com/facebookresearch/segment-anything.git`
- **"Checkpoint not found"** → Download it: `wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P Checkpoints/`
- **"No image files found"** → Edit `data_dir` in `segment_sam_propagation.py` (line 41) to point to a valid `Datasets/In/` subfolder
- **Poor segmentation** → Add more positive/negative points interactively

## References

- **SAM**: Kirillov, A., et al. (2023). "Segment Anything" [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- **MedSAM**: Ma, J., et al. (2023). "Segment Anything in Medical Images" [arXiv:2304.12306](https://arxiv.org/abs/2304.12306)

## Authors

**Thomas Molina Molina**
Universidad Nacional de Colombia — Topicos en Geometria Computacional

## License

Open-source project for educational and research use.
