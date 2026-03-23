# Humerus Detection in Fat-Suppressed MRI

Segmentation and geometric analysis of the **humerus** from axial fat-suppressed shoulder MRI using **SAM** (Segment Anything Model). The pipeline segments the humerus across DICOM slices, extracts boundary points in real-world 3D coordinates, and detects circular arcs on the humeral head (spherical cap).

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

This script runs steps 1–5 of the geometric analysis:

| Step | Description | Output |
|------|-------------|--------|
| 1 | Classify slices as **correct** / **ambiguous** / **failed** (Dice-based) | `slice_classification.txt` |
| 2 | List ambiguous slices that need manual review | (printed + in report) |
| 3 | Extract boundary pixels from correctly segmented slices | — |
| 4 | Convert pixels to real-world 3D coords (mm) using DICOM headers | `boundary_world_coords.csv` |
| 5 | RANSAC sphere fit to detect circular arcs (humeral head cap) | `circular_arcs.csv` |
| 6 | Generate 6-panel summary diagram per dataset | `boundary_analysis_summary.png` |

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

## Datasets

| Dataset | Equipment | Slices | Side |
|---------|-----------|--------|------|
| `pd_tse_fs_tra_320_fov150_4` | SIEMENS | 31 | — |
| `t1_tse_fs_tra_320_fov150_5` | SIEMENS | 30 | — |
| `t1_tse_fs_tra_320_fov150_11` | SIEMENS | 30 | — |
| `AXIAL fs PD FSE 256x256` | GE Optima MR360 | 24 | Right |
| `AXIAL-P5SE1 ... SIEMENS` | SIEMENS | 19 | Right |
| `AXIAL(creo)ePDW_SPAIR ...` | Philips Ingenia | 25 | Left |
| `axial_A_to_T_fatSupressed` | — | 20 | — |
| `dp_tse_cor_320_fs_8` | SIEMENS | 22 | — |

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
