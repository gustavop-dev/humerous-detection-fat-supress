# Guion de la Presentación — UEMS Congress 2026

**Título:** Interpretable Geometric Analysis of the Humerus in Fat-Suppressed Shoulder MRI
**Autores:** Marco Paluszny Kluczynsky, Gustavo Adolfo Pérez
**Evento:** 1st UEMS Congress 2026 — Leuven, Belgium, 27–30 May 2026
**Duración estimada:** ~20–25 min + preguntas

---

## Diapositiva 1 — Title (pág. 1)

> "Good morning / afternoon. My name is [presenter], from Universidad Nacional de Colombia. This work, joint with [co-author], bridges computational geometry and musculoskeletal radiology. We present an interpretable pipeline for geometric analysis of the humerus in fat-suppressed shoulder MRI, toward localization of rotator cuff tears."

**Clave:** Establecer que es un puente entre geometría computacional y radiología, no un modelo de deep learning puro.

---

## Diapositiva 2 — The clinical problem: rotator cuff tears (pág. 2)

> "Rotator cuff disease affects more than 30% of adults over 60 — it is the leading cause of shoulder pain and disability. Tears can be acute or degenerative, and accurate localization — which tendon, where along its insertion — is what drives surgical planning."

**Énfasis:** La palabra clave es *localización*, no solo detección.

---

## Diapositiva 3 — Anatomy of the glenohumeral joint (pág. 3)

> "Here is a schematic of the glenohumeral joint in the axial plane. The humeral head is roughly spherical, the glenoid is the concave socket. The rotator cuff tendons — supraspinatus, infraspinatus, subscapularis — insert around the head. A tear disrupts the articular contour focally, and that focal disruption is what we aim to detect geometrically."

**Señalar en el diagrama TikZ:** los tendones y el marcador de tear.

---

## Diapositiva 4 — Why is this hard to read on MRI? (pág. 4)

> "MRI reading for rotator cuff tears is expert-dependent and time-consuming. Inter-vendor variability across SIEMENS, GE, and Philips makes automation harder. And subtle partial tears can be missed on a single slice — you need 3D context."

> "There is also an equity-of-access gap. In Colombia, regions like Quibdó in Chocó have modern MRI scanners but lack dedicated musculoskeletal radiologists. An interpretable, automated tool could support generalist radiologists in these settings."

---

## Diapositiva 5 — Why fat-suppressed MRI? (pág. 5)

> "We work specifically with fat-suppressed sequences — PD-fs and T1-fs. A selective RF pulse nulls the fat signal, so subcutaneous and bone-marrow fat appear dark. This makes cortical bone, tendons, edema and joint fluid conspicuous. These are the standard sequences for shoulder pathology."

---

## Diapositiva 6 — Why the axial plane? (pág. 6)

> "In the axial plane, the humeral head appears as nested circles that grow to a maximum — the equator of the head — and then shrink. This gives a clean bell-shaped area-versus-slice curve that we can validate geometrically. The axial plane also gives a direct view of the glenohumeral contact surface, where most rotator cuff tears manifest."

> "This bell-shaped curve becomes the anchor for our downstream validation: trimming non-humeral tails, detecting outlier slices, and selecting articular slices."

---

## Diapositiva 7 — Axial PD-fs slice with humeral segmentation (pág. 7)

> "Here is an actual axial PD-fs slice from our data, with the humeral segmentation overlaid. The green contour is the mask boundary and the red dot is the positive prompt centroid used by the segmentation model. The pixel spacing is approximately 0.47 mm, matrix 320×320."

**Señalar:** la forma casi circular de la cabeza humeral en esta slice.

---

## Diapositiva 8 — From anatomy to geometry: two natural primitives (pág. 8)

> "Two anatomical observations motivate our geometric approach. First, the articular cartilage of the humeral head covers a near-perfect spherical surface — radius approximately 23 to 26 mm in adults. This motivates fitting a global sphere as a patient-specific reference."

> "Second, in the equatorial axial slices, the contact contour with the glenoid is locally circular. These per-slice circles lie on the articular surface — they do not approximate the sphere, they live on the cartilage interface."

---

## Diapositiva 9 — Anatomical asymmetry: head vs shaft (pág. 9)

> "The humerus is anatomically asymmetric. The head closes abruptly at the anatomical neck, while the shaft tapers gradually. This is not an arbitrary observation — it directly determines our pipeline parameters: 30% of peak area as threshold on the head side, 10% on the shaft side. These are anatomy-driven thresholds, not heuristics."

---

## Diapositiva 10 — Anatomy and asymmetric trim: visual reference (pág. 10)

> "On the left, a sagittal sketch of the proximal humerus showing the abrupt closure at the head and the gradual taper at the shaft. On the right, the area curve with the two asymmetric trim thresholds. The head side is trimmed at 30% of peak, the shaft side at 10%."

**Señalar ambos diagramas TikZ** y cómo se complementan.

---

## Diapositiva 11 — Hypothesis and objective (pág. 11)

> "Our hypothesis: a rotator cuff tear produces a focal, geometrically detectable discontinuity of the humeral articular contour. If we extract clinically interpretable geometric features from MRI, we can localize the lesion to a specific tendon insertion sector — without training a new neural network."

> "The objective: build an end-to-end pipeline that segments the humerus from a single seed slice, extracts two interpretable features — first, circular arcs on individual slices to identify those that lie on the articular surface; second, the approximating sphere that contains the articular surface — and uses these two mathematical objects to place the tear in the muscular complex of the rotator cuff."

**Énfasis:** El orden es arcos primero, esfera después. Los arcos identifican la superficie articular; la esfera la contiene.

---

## Diapositiva 12 — Design principles (pág. 12)

> "Three design principles. Interpretable: every output is geometrically meaningful and visually verifiable — masks, CSVs, fitted circles, summary figures. Vendor-agnostic: works on SIEMENS, GE and Philips without retraining, because thresholds are geometric, not intensity-based. And anatomy-aware: parameters reflect humeral anatomy, not arbitrary heuristics."

---

## Diapositiva 13 — Pipeline overview (pág. 13)

> "Here is the full pipeline. We start from DICOM axial fat-suppressed MRI. SAM — Segment Anything Model, frozen, no training — segments a central seed slice. Bidirectional propagation with dynamic Dice extends the segmentation to the full volume. Post-processing repairs neighbor-overlap and bright-ring artifacts."

> "Then, direction-aware area-curve validation trims the tails and flags outliers. Optionally, SAM can re-segment confirmed outliers. From the validated data, we extract per-slice circular arcs on the articular surface, then construct the approximating sphere plus the shaft axis. These two mathematical objects are what we use to place the tear in the rotator cuff complex."

**Señalar el diagrama TikZ** siguiendo el flujo de flechas.

---

## Diapositiva 14 — Stage 1: SAM segmentation + bidirectional propagation (pág. 14)

> "Stage 1: we use SAM with the ViT-B backbone as a frozen segmenter — no training, no labels needed. The user clicks one seed point on the central slice, or in batch mode we place it automatically at 40% of width and height."

> "The mask is refined morphologically — small-object removal, hole filling, opening and closing. Then we propagate bidirectionally: the centroid of slice i becomes the positive prompt for slice i±1. The Dice threshold is adaptive — strict near the seed, more permissive at the volume extremes — with a negative-prompt fallback when the mask degrades."

**Fórmula clave:** τᵢ = 0.45 + 0.25 · |i − i_seed| / Δ_max

---

## Diapositiva 15 — Per-slice segmentation example (pág. 15)

> "Here is the per-slice segmentation overlay in practice. Green boundary, red centroid dot. You can see the mask tracking the humerus consistently across slices."

---

## Diapositiva 16 — Stage 2: direction-aware validation of the area curve (pág. 16)

> "Stage 2: before fitting any geometry, we validate the area curve. We compute the slice normal from ImageOrientationPatient and project ImagePositionPatient onto it to determine which end of the volume is the head. This is read from the DICOM geometric headers — not vendor conventions — making the pipeline vendor-agnostic by construction."

---

## Diapositiva 17 — Schematic: per-slice mask area (pág. 17)

> "Here is the schematic of the bell curve. The mask area peaks at the equator of the head. We trim the tails — slices outside the bell that were segmented to something but don't actually contain humerus. We also flag internal strong outliers where SAM grabbed extra tissue. The arrow shows which side is the head."

---

## Diapositiva 18 — Asymmetric trim and outlier flagging (pág. 18)

> "The trim is asymmetric and anatomy-driven. 30% of peak on the head side — because the head closes abruptly at the anatomical neck. 10% on the shaft side — because the shaft tapers gradually."

> "Reversal-from-peak: we walk outward from the peak and trim when the curve genuinely rises again — at least 10% on the head side, 20% on the shaft side, with a minimum of 50 pixels absolute."

> "Internal outliers are detected with a robust MAD residual at 2.5 sigma, cross-checked with either a circle inlier-ratio drop of at least 15% or a Dice drop of at least 10%. Only strong outliers — those with geometric confirmation — are excluded. Weak ones are reported but kept."

---

## Diapositiva 19 — Feature 1: global humeral-head sphere (pág. 19)

> "Feature 1: a single sphere fitted to the 3D cloud of points on the articular surface of the humerus head, in millimeter units. We convert pixel coordinates to world coordinates using ImagePositionPatient, ImageOrientationPatient, and PixelSpacing."

> "We use RANSAC with 2000 iterations and 4-point sampling. The base estimator inside RANSAC is the Kåsa algebraic fit — a linear least-squares formulation that is very fast to compute. On its own, Kåsa is not robust to outliers, but inside RANSAC it becomes highly reliable. Inlier distance is 2 mm; the final sphere is refit by least-squares on the inlier set."

**Énfasis:** La sinergia Kåsa + RANSAC — eficiencia + robustez.

---

## Diapositiva 20 — Sphere fit: inliers and outliers (pág. 20)

> "In this schematic, teal points are inliers — they define the spherical articular surface. Orange points are outliers: the greater tubercle, the shaft, artifacts. RANSAC excludes them automatically. The center and radius characterize the patient-specific humeral head."

---

## Diapositiva 21 — Feature 1: clinical interpretation (pág. 21)

> "Clinically, this sphere recovers the expected curvature of the humeral head — typical radius 23 to 26 mm in adults. The centroid and radius characterize the 'ideal' head for this patient. It enables normalized comparison across patients, vendors, and acquisitions. And it provides a baseline geometry against which local deviations — such as those caused by a tear — stand out."

---

## Diapositiva 22 — Feature 2: per-slice articular circles (pág. 22)

> "Feature 2: per-slice circles computed by Kåsa within RANSAC. These circles contain arcs that should lie on the articular surface, so they provide a validation method of the approximating sphere, and also a useful reference for the glenohumeral space."

> "We select articular slices — those with mask area at least 70% of the peak. Boundary points are projected to the slice's best-fit plane via SVD. Then we run iterative 2D RANSAC with Kåsa fit — up to 3 circles per slice. Inlier distance 1.5 mm, ratio at least 35%, minimum 10 inliers."

---

## Diapositiva 23 — Five-panel render for one articular slice (pág. 23)

> "Here is the five-panel render for one articular slice: the original image, the mask, the contour, the fitted circle, and the overlay. You can see the circle lying precisely on the articular contour. This is what the radiologist reviews to validate each fit."

---

## Diapositiva 24 — Feature 2: stored information and clinical meaning (pág. 24)

> "Each circle stores: center in 3D and radius in mm units — not pixels. An in-plane orthonormal basis (u, v) and slice normal n. The inlier point cloud and residual statistics. And a re-projection on the original axial image for slice-by-slice clinician review."

> "Clinically: a focal drop in inlier ratio on a circle indicates loss of articular continuity — a potential tear. A sudden shift of the center or radius compared to neighbors flags asymmetric pathology. Each circle is rendered as a PNG so the radiologist can verify."

---

## Diapositiva 25 — The geometric trick: parameterizing the articular circle (pág. 25)

> "Now the geometric trick that ties everything together. Each per-slice circle has an in-plane orthonormal basis (u, v) and center c. Any inlier point p has an angular coordinate theta = atan2 of its projections onto v and u. This turns the articular circle into a 1D angular map — each theta corresponds to a known anatomical direction at the humeral head, the same for every patient and vendor."

---

## Diapositiva 26 — Angular parameterization of the circle (pág. 26)

> "Here is the diagram. The circle with its basis vectors u and v. A point p on the circle at angle theta. This is the parameterization that converts the 2D circle into a 1D coordinate system aligned with anatomy."

---

## Diapositiva 27 — Angular sectors → rotator-cuff tendons (pág. 27)

> "The tendon insertions fall in predictable angular sectors on the articular circle. Subscapularis maps to the lesser tubercle, roughly 0 to 45 degrees. Supraspinatus to the superior greater tubercle, 45 to 135. Infraspinatus to the posterior greater tubercle, 135 to 225. And teres minor to the inferior-posterior portion, 225 to 270."

> "The localization rule: a focal drop in inlier ratio in a specific sector identifies a candidate tear in the corresponding tendon, at the slice height where the drop occurs."

---

## Diapositiva 28 — Sector diagram: tendon insertions and a focal drop (pág. 28)

> "Here you can see the sector diagram. Each colored sector corresponds to a tendon. The orange marks show a focal drop — in this case in the supraspinatus sector. The articular circle has become a 1D tendon map."

---

## Diapositiva 29 — Multi-vendor robustness (pág. 29)

> "We tested across three vendors and multiple sites: SIEMENS from Quibdó with PD-fs TSE, GE Optima from Bogotá with PD-fs FSE, and Philips Ingenia with ePDW SPAIR. Matrix sizes range from 256 to 320. The pipeline generalizes because SAM is frozen — no domain-specific weights to overfit — thresholds are geometric, and direction handling reads DICOM headers."

---

## Diapositiva 30 — Per-dataset coverage (pág. 30)

> "Here is the per-dataset coverage table. 'Kept' shows how many slices were retained by the direction-aware validation out of the total. 'Articular' is the number of slices where per-slice circles were fitted — those with at least 70% of peak area. Across all datasets, the pipeline retains a substantial portion and identifies 4 to 5 articular slices per volume."

---

## Diapositiva 31 — Equity of access (pág. 31)

> "Our datasets include Quibdó, Chocó — a region with limited subspecialty radiology despite having modern MRI. A vendor-agnostic, interpretable tool can directly support diagnostic capacity there. Geometric outputs are auditable by generalist radiologists, lowering the barrier compared to opaque classifiers. This aligns with the UEMS pillar of impacting medical practice across Europe and beyond."

---

## Diapositiva 32 — Qualitative results: per-dataset summary (pág. 32)

> "Here is our seven-panel summary for one dataset. Slice classification, Dice per slice, the area curve with validation overlay, 3D boundary point cloud, sphere fit, and per-slice articular circles. Each panel tells part of the story; together they give a complete audit of the pipeline's output."

---

## Diapositiva 33 — How to read the summary figure (pág. 33)

> "How to read the figure. In the area curve: green means kept, gray means trimmed tails, red means strong outlier excluded. Vertical lines mark the bell extent. The caret shows which side is the head."

> "Every slice exports a five-panel circle PNG for slice-by-slice clinician review. And every run writes a CSV for downstream audit. Thresholds are CLI flags — fully reproducible."

---

## Diapositiva 34 — Limitations and clinical considerations (pág. 34)

> "Limitations. The validation cohort is 6 datasets across 3 vendors — proof of concept, not clinical validation. The automatic seed point assumes typical centering. Pattern B outliers without GPU can only be flagged, not re-segmented. And the angular-sector tear-localization rule is still a hypothesis that needs validation against arthroscopic ground truth."

> "For clinical translation we need a larger multi-center cohort with surgical confirmation, an inter-reader study, PACS integration, and a containerized reproducible pipeline. But a key property: because outputs are geometric, visualized, and auditable, the system fails loudly rather than silently — important for clinical deployment."

---

## Diapositiva 35 — Take-home messages (pág. 35)

> "Take-home messages. A frozen foundation model plus anatomy-aware classical geometry produces interpretable features without training. DICOM-header reading makes the pipeline direction-aware with anatomy-driven thresholds. Two features — per-slice articular circles and the approximating sphere — enable tear localization by angular sector. And it works robustly across SIEMENS, GE and Philips, relevant for low-resource settings."

---

## Diapositiva 36 — Next steps and acknowledgements (pág. 36)

> "Next steps: implement the angular-sector to tendon mapping end-to-end, prospective validation against arthroscopy, inter-reader reproducibility study, and integration into a clinical viewer."

> "We thank Universidad Nacional de Colombia, the clinical sites — Clínica Especialistas Reina Virgen María in Quibdó, Clínica Santa Ana in Bogotá, SOMA Radiology — and the open-source tools that made this possible: SAM from Meta AI, pydicom, scikit-image, OpenCV, NumPy and SciPy."

> "Thank you — questions are welcome."

---

## Notas generales para el presentador

- **Ritmo:** ~40–50 segundos por diapositiva de texto, ~20–30 segundos por diapositiva de imagen/diagrama.
- **Señalar diagramas TikZ:** usar puntero láser o cursor en las diapositivas con esquemas (págs. 3, 10, 17, 20, 26, 28).
- **Fórmulas:** no leer la fórmula de Dice dinámico literalmente — explicar el concepto ("estricto cerca de la semilla, permisivo en los extremos").
- **Kåsa + RANSAC:** enfatizar que Kåsa es el estimador base (rápido, lineal), RANSAC provee la robustez. Esta sinergia es clave.
- **Arcos → esfera (no al revés):** mantener el orden conceptual — los arcos per-slice identifican la superficie articular, la esfera la contiene y provee la referencia global.
- **Honestidad:** la localización de desgarros por sector angular es una hipótesis, no un resultado validado. Dejarlo claro en la diap. 34.
