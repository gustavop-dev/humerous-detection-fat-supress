# CLAUDE.md

Guía para Claude Code cuando trabaje en este repositorio.

## Qué es este proyecto

Pipeline académico (Universidad Nacional de Colombia — Tópicos en Geometría Computacional) para **segmentar el húmero en MRI axial con saturación de grasa (fat-suppressed)** usando **SAM (Segment Anything Model)** y luego representarlo geométricamente con **dos features**:

1. **Una esfera global** ajustada a la cabeza humeral (la cabeza ≈ casquete esférico).
2. **Círculos per-slice que YACEN sobre la superficie articular** (no la aproximan — son arcos que viven literalmente sobre el cartílago / interfaz con la glenoides). Estos aparecen en las slices con área máxima (la "ecuatorial" del húmero).

Estas dos features alimentan una metodología novedosa para (a) incorporar la estructura muscular vía un truco geométrico y (b) **localizar la posición de un desgarro** del manguito rotador. Es parte de una presentación para **UEMS**.

No es un proyecto productivo: es código de investigación / tarea. Las rutas a datasets están hardcodeadas, hay `print` con emojis por todos lados, y los comentarios están en español. Mantén ese estilo si vas a editar — no "profesionalices" por gusto.

## Pipeline (4 etapas)

1. **Segmentación de una imagen del medio (seed)** — el usuario marca puntos con clic (o se usa un punto automático en `(0.40·W, 0.40·H)` en modo batch).
2. **Propagación bidireccional** — desde la slice central hacia arriba y hacia abajo, usando el centro de masa de la máscara anterior como punto positivo en SAM. Si Dice cae mucho, intenta con un punto negativo calculado fuera de la máscara.
3. **Post-proceso de máscaras**:
   - **Solapamiento vecinal**: si una slice tiene >25% de área fuera de la “sombra” unión de sus vecinas, se re-segmenta con el centro de la vecina más parecida + punto negativo en la zona extra.
   - **Artefacto brillante**: si hay un brote brillante en el anillo externo de la máscara (>p95 de intensidad y >8% del anillo), se re-segmenta usando ese centroide como punto negativo.
4. **Reconstrucción 3D + STL** y, en un script separado (`humerus_boundary_analysis.py`), conversión a coordenadas DICOM (mm) y dos análisis geométricos:
   - **Feature 1 — esfera global** (RANSAC, 2000 iter, 2 mm inlier dist) sobre todos los puntos del contorno: aproxima la cabeza humeral.
   - **Feature 2 — círculos per-slice** (Kåsa + RANSAC iterativo, hasta 3 círculos por slice) sobre los puntos proyectados al plano de la slice via SVD: identifican los arcos que **yacen sobre la superficie articular**. Las slices articulares se detectan automáticamente como aquellas con área de máscara ≥ 70 % del máximo de la curva de áreas.

**Step 1.5 — validación de la curva de áreas (direction-aware)**: antes de ejecutar Steps 3–5b, el pipeline:
   1. Detecta la dirección de adquisición leyendo `ImagePositionPatient` proyectado sobre el normal de la slice (`cross(row_cosine, col_cosine)` de `ImageOrientationPatient`). Determina si los índices alfabéticos van superior→inferior o inferior→superior, y por tanto cuál extremo del bell es la **cabeza** (cierre abrupto) y cuál el **diáfisis** (taper gradual).
   2. Trimea slices fuera del bell aplicando un híbrido: (a) static floor desde el borde — cabeza 30%, diáfisis 10% del peak, mín. 2 slices consecutivas; (b) reversal-from-peak — caminar desde el peak hacia afuera y trimear cuando la curva sube ≥10% (head) / ≥20% (shaft). Trim final = unión.
   3. Detecta outliers internos (Pattern B: SAM agarró extra) usando residual robusto (rolling median + MAD con piso de 50 px) cruzado con `inlier_ratio` del círculo per-slice y/o caída de Dice. Solo outliers fuertes (con cross-check) son excluidos del análisis; débiles solo se reportan.
   4. Pass 2 después de Step 5b: re-ejecuta detección de outliers con los círculos disponibles y reescribe los CSVs afectados (`boundary_world_coords`, `slice_circles`, `circular_arcs`).
   5. Outputs: `area_curve_validation.txt` + `.csv`. El panel 3 del summary muestra status per-slice (verde=kept, gris=trimmed, rojo=outlier) con líneas verticales del bell extent y caret H→/←H del lado de la cabeza.

**Step 1.6 — re-segmentación con SAM (opt-in, separado)**: módulo `humerus_outlier_resam.py` que se importa solo bajo `--enable-resam-fixup`. Re-segmenta los outliers usando el círculo esperado (interpolado de vecinos sanos) como prompt positivo y el centroide de la región extra como negativo. Hace backup `_mask.bak.npy` antes de overwrite. Requiere torch + segment_anything + checkpoint SAM (NO está en `venv-analysis/`).

## Layout del repo

```
one_segmentation.py              # GUI para una sola imagen (debug visual)
segment_sam_propagation.py       # Pipeline completo sobre 1 dataset
batch_segment_sam_propagation.py # Corre el pipeline sobre los 8 datasets
humerus_boundary_analysis.py     # Análisis posterior (NO necesita SAM ni GPU)

DCM/load_dicom_as_image.py       # Carga DICOM con windowing → RGB; get_dataset_files() devuelve DatasetInfo
Graphics/
  interface.py                   # GUI matplotlib de selección de puntos (clic der=+, izq=-)
  grafication.py                 # Reconstrucción 3D, malla sólida, export STL
Segmentation/
  segment_image.py               # Wrappers SAM: segment_image / segment_with_point / segment_first_image
  propagation.py                 # propagate_segmentation() bidireccional con umbral Dice dinámico
  Masks.py                       # refine, calculate_mask_center, find_contours, save_segmentation_result
  Metrics.py                     # Dice, IoU
  negative_points.py             # calculate_negative_point() — busca punto fuera de la máscara

Datasets/In/<dataset>/*.dcm      # 8 datasets (SIEMENS, GE, Philips)
Datasets/Out/<dataset>/          # *_seg.png, *_mask.npy, propagation_summary.txt,
                                 # contour_points_3d.{npy,csv}, modelo_3d.stl,
                                 # boundary_world_coords.csv,
                                 # circular_arcs.csv          (Feature 1: arcs on the global sphere)
                                 # slice_circles.csv          (Feature 2: per-slice circle centers + radii)
                                 # slice_circles_points.csv   (Feature 2: inlier points per circle)
                                 # *_circle.png               (5-panel render for each articular slice)
                                 # area_curve_validation.txt  (Step 1.5: bell extent + outliers)
                                 # area_curve_validation.csv  (Step 1.5: machine-readable per-slice status)
                                 # resam_fixup_log.txt        (Step 1.6, only if --enable-resam-fixup ran)
                                 # *_mask.bak.npy             (Step 1.6 backups, only if anything was overwritten)
                                 # slice_classification.txt, boundary_analysis_summary.png

humerus_outlier_resam.py         # OPT-IN module (Step 1.6). Imports torch + SAM lazily;
                                 # the analysis pipeline catches ImportError gracefully.
                                 # Run via: python humerus_boundary_analysis.py --enable-resam-fixup
                                 # Or standalone: python humerus_outlier_resam.py --dataset <name>
venv-analysis/                   # Lightweight venv for ANALYSIS only (numpy/scipy/cv2/pydicom/matplotlib).
                                 # Does NOT include torch or SAM. Use it for re-running
                                 # humerus_boundary_analysis.py without GPU.
Checkpoints/sam_vit_b_01ec64.pth # NO está en git (375 MB)
```

## Convenciones

- **Idioma**: docstrings, comentarios y mensajes de `print` en **español**. Nombres de funciones/variables en inglés. Mantén ambos.
- **Emojis en los `print`**: son intencionales (es la única UI). No los quites.
- **`[x, y]` vs `(row, col)`**: los puntos para SAM y los centros de máscara son `[x, y]` = `[col, row]`. Las máscaras de numpy se indexan `mask[y, x]`. Cuidado al pasar entre los dos.
- **Máscaras**: siempre booleanas/uint8 binarias. `Masks.refine_medical_mask()` aplica `remove_small_objects(min_size=500)` + `binary_fill_holes` + opening/closing con disco r=2.
- **Z spacing**: hardcodeado a `12` unidades arbitrarias en la reconstrucción 3D (`segment_sam_propagation.py:447`). El análisis en `humerus_boundary_analysis.py` sí usa `PixelSpacing` real del header DICOM para coords en mm.

## Gotchas frecuentes

- **`data_dir` y `output_dir`** están hardcodeados en `segment_sam_propagation.py:41-42`. `batch_segment_sam_propagation.py` los sobrescribe vía atributos de módulo (`ssp.data_dir = ...`) antes de cada `main()`. Si vas a refactorizar, no rompas ese contrato.
- **Modo batch sin GUI**: `batch_segment_sam_propagation.py` exporta `SAM_BATCH_MODE=1` y `segment_first_image()` usa el punto automático en `(0.40·W, 0.40·H)`. Cualquier `plt.show()` en path batch va a colgar.
- **Device**: `one_segmentation.py` detecta CUDA→MPS→CPU; `segment_sam_propagation.py` solo detecta MPS→CPU (no CUDA). Si corres en NVIDIA, esto es un bug a notar antes de cambiarlo.
- **Checkpoint**: `Checkpoints/sam_vit_b_01ec64.pth` (~375 MB). No está versionado. Descárgalo desde `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`.
- **Datasets `In` vs `Out` no coinciden 1:1**: en `Datasets/In/` hay 8 carpetas pero en `Out/` solo 6 procesadas — `AXIAL(creo)ePDW_SPAIR ...` y `dp_tse_cor_320_fs_8...` aún no se han corrido (al menos no estaba la salida cuando se escribió esto — verifica con `ls Datasets/Out/`).
- **`humerus_boundary_analysis.py` usa `DATASET_MAP` propio** (no `DATASET_NAMES` de batch). Si agregas un dataset, actualiza ambos.

## Cómo correr

```bash
# Activar venv y verificar que está el checkpoint
source venv/bin/activate
ls Checkpoints/sam_vit_b_01ec64.pth

# Una imagen (interactivo, requiere display) — edita line 35 con la ruta
python one_segmentation.py

# Un dataset completo — edita lines 41-42 con el dataset
python segment_sam_propagation.py

# Los 8 datasets en batch (sin GUI)
python batch_segment_sam_propagation.py

# Análisis post-segmentación (no necesita SAM)
python humerus_boundary_analysis.py
```

## Cuando edites código

- **No introduzcas dependencias nuevas** sin revisar `requirements.txt`. El proyecto evita pesar más de lo necesario.
- **No mockees DICOM**: la lógica de `pixel_to_world` depende de `ImagePositionPatient`, `ImageOrientationPatient`, `PixelSpacing`. Si vas a probarla, usa un `.dcm` real de `Datasets/In/`.
- **No toques los umbrales** (`SIMILARITY_THRESHOLD`, `WARNING_THRESHOLD`, `NEIGHBOR_EXTRA_RATIO_THRESHOLD`, `BRIGHT_ARTIFACT_*`, `DICE_CORRECT_THRESHOLD`, `DICE_AMBIGUOUS_THRESHOLD`, `RANSAC_*`, `MIN_CIRCLE_*`, `CIRCLE_INLIER_DISTANCE_MM`, `ARTICULAR_AREA_RATIO`, `MIN_ARC_INLIER_RATIO`, `BELL_BOUNDARY_HEAD_RATIO`, `BELL_BOUNDARY_SHAFT_RATIO`, `BELL_REVERSAL_RATIO_*`, `BELL_REVERSAL_MIN_ABS`, `BELL_MIN_RUN_LENGTH`, `INTERNAL_OUTLIER_*`) sin avisar — son parámetros del experimento.
- **Filtro anti-arcos-de-1-punto**: `find_circular_arcs_per_slice` exige `>= MIN_ARC_POINTS=10` Y razón de inliers `>= MIN_ARC_INLIER_RATIO=0.30`. Si bajas estos números van a reaparecer puntos sueltos espurios en la viz 3D — verificado con el dataset `pd_tse_fs_tra_320_fov150_4`.
- **Direction-aware bell extent (Step 1.5)**: el head trim es más estricto (30% floor) que el shaft (10%) porque la cabeza humeral cierra abruptamente y la diáfisis tapera gradualmente. Si trimea slices que NO querías, baja `BELL_BOUNDARY_HEAD_RATIO` (default 0.30); si NO trimea ruido obvio en el extremo opuesto, sube `BELL_BOUNDARY_SHAFT_RATIO` (default 0.10). Hay flags CLI para ambos.
- **Pattern B sin SAM**: la detección de "SAM agarró extra" en una slice del bell es best-effort sin re-segmentar. Cuando los datos no muestran caída en `circle_inlier_ratio` (caso P5SE1 11-12), solo se reporta como `weak`, no se excluye. Si quieres excluirlo de verdad, corre Step 1.6 con `--enable-resam-fixup` en una máquina con GPU+SAM.
- **STL y `.npy` están en `.gitignore`**: no intentes commitearlos.

## Referencias

- SAM: Kirillov et al. 2023, [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- MedSAM: Ma et al. 2023, [arXiv:2304.12306](https://arxiv.org/abs/2304.12306)
