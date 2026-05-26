# Guion de la Presentación — UEMS Congress 2026

**Título:** Interpretable Geometric Analysis of the Humerus in Fat-Suppressed Shoulder MRI
**Autores:** Marco Paluszny Kluczynsky, Gustavo Adolfo Pérez
**Evento:** 1st UEMS Congress 2026 — Leuven, Belgium, 27–30 May 2026
**Duración objetivo:** ~8 minutos + preguntas
**Slides totales:** 16 frames (~30 segundos por slide en promedio)

---

## Slide 1 — Título (≈20 s)

> "Good morning. My name is [presenter], from Universidad Nacional de Colombia. Together with [co-author] we developed a simple geometric pipeline to analyze the humerus in shoulder MRI — with the long-term goal of helping locate rotator cuff tears. The key idea: we do not train a new neural network; we combine a public foundation model with classical geometry."

**Tono:** cálido y directo. Una sola idea clave: *no entrenamos nada nuevo, combinamos lo que ya existe con geometría clásica*.

---

## Slide 2 — El problema clínico (≈30 s)

> "Rotator cuff tears affect more than 30% of adults over sixty. The surgical plan depends not only on *whether* there is a tear, but on *where* it is. Reading shoulder MRI is expert-dependent, and access to specialist radiologists is unequal — for example, in Chocó, Colombia, modern scanners exist but specialists do not."

> "On the right you can see the axial view of the shoulder. The humeral head is round, the glenoid is the socket, and tendons insert around the head. A tear shows up as a *focal* break in the articular contour. That focal trace is what we want to detect geometrically."

**Énfasis:** *localización*, no solo detección. Y la motivación de equidad de acceso (Chocó).

---

## Slide 3 — Por qué MRI axial con saturación de grasa (≈30 s)

> "We work with fat-suppressed axial sequences for two reasons. First, fat suppression makes fat appear dark — so tendons, fluid and edema become visible. Second, in the axial plane the humeral head appears as nested circles that grow until the equator and then shrink. That gives us a clean bell-shaped curve of area against slice, which becomes the anchor of the whole pipeline."

> "On the right is one real axial slice from our data, with our segmentation overlaid in green."

**Idea clave:** las dos elecciones (fat-sup + axial) no son arbitrarias, dan estructura geométrica útil.

---

## Slide 4 — De la anatomía a la geometría (≈35 s)

> "Two anatomical facts drive everything. First, the humeral head is almost a perfect sphere, about 23 to 26 millimeters in radius. Second, the articular contour, slice by slice, is a circle that *lies on* the cartilage — it does not approximate the surface, it lives on it. These give us two natural geometric features: one global sphere, and many local circles."

> "There is also an important asymmetry: the head closes abruptly at the anatomical neck, while the shaft tapers gradually. We use that to set two different thresholds on the area curve — 30% on the head side, 10% on the shaft side. Anatomy, not arbitrary numbers."

**Tip:** señalar las dos curvas en el gráfico de la derecha.

---

## Slide 5 — Hipótesis y objetivo (≈30 s)

> "Our hypothesis is simple: a tear leaves a focal geometric trace on the articular contour. If we extract clinically meaningful geometric features, we can localize the lesion to a specific tendon — without training a new neural network."

> "What we built has three properties on purpose: it is *interpretable* — every output is a mask, a circle, or a sphere that a radiologist can verify. It is *vendor-agnostic* — works on SIEMENS, GE and Philips without retraining. And it is *anatomy-aware* — its thresholds come from anatomy, not from heuristics."

---

## Slide 6 — Vista general del pipeline (≈40 s)

> "This is the full pipeline at a glance. We start from the DICOM axial MRI. SAM — the Segment Anything Model, used as-is, no training — segments one central slice. Then we propagate the mask to neighbouring slices. A post-processing step cleans common artifacts."

> "Next, the area curve is validated: we trim the non-humeral tails and flag suspicious slices. Finally, the geometry: per-slice circles on the articular surface, the global sphere, and the rule that places the tear in the rotator cuff."

> "The important property is that every block produces *auditable* outputs — masks, CSVs, figures — that a radiologist can inspect."

**Tip:** seguir el flujo con el puntero por las flechas.

---

## Slide 7 — Etapa 1: segmentación con SAM (≈30 s)

> "Stage one: segmentation. We use SAM as a frozen segmenter — no labels, no fine-tuning. The user clicks once on a central slice, and from there the mask of slice *i* is used as the prompt for slices *i plus one* and *i minus one*. We propagate up and down through the volume."

> "We add an adaptive quality control: the acceptance threshold is stricter near the seed and looser at the extremes, where the anatomy is naturally harder. If a slice fails, we re-segment it with a corrective negative prompt."

**Tip:** señalar el ejemplo de la derecha — la máscara verde sigue al húmero a lo largo de las slices.

---

## Slide 8 — Etapa 2: validar la curva de áreas (≈30 s)

> "Stage two: before we fit any geometry, we validate the curve. We read the DICOM header to detect which end of the volume is the head and which is the shaft — this is more reliable than trusting vendor conventions."

> "Then two simple jobs: trim the tails that fall outside the bell of the humerus, and flag internal outliers — slices where the segmentation grabbed extra tissue. Only strong outliers, confirmed by a second geometric check, are excluded."

**Énfasis:** "limpiamos los datos *antes* de hacer geometría — porque la geometría hereda la calidad de la máscara."

---

## Slide 9 — Feature 1: la esfera global (≈35 s)

> "The first feature is one sphere, fitted in millimeters, to all the points on the articular surface of the head. We use Kåsa's algebraic fit — a closed-form, linear method that is extremely fast. But Kåsa alone is not robust to outliers, so we wrap it inside RANSAC: it tries thousands of candidate spheres on random samples and keeps the one with most inliers."

> "Why we care: this sphere gives us a *patient-specific reference geometry*. Once we know the ideal humeral head for this patient, local deviations from it — caused by tears or artifacts — really stand out."

**Tip:** mencionar que el típico radio en adultos cae entre 23 y 26 mm — un *sanity check* para el lector clínico.

---

## Slide 10 — Feature 2: los círculos per-slice (≈35 s)

> "The second feature is one circle per articular slice. We only fit circles on slices that are near the equator of the head — at least 70% of the peak area. On each one we run a 2D version of RANSAC plus Kåsa to find the arcs that lie on the cartilage."

> "Clinically this matters because each circle stores the inlier ratio of its arc. A focal *drop* in that ratio means the cartilage there has lost continuity — which is exactly what a tear looks like geometrically. And every circle is exported as a small PNG so the radiologist can verify it, slice by slice."

---

## Slide 11 — El truco geométrico (≈40 s)

> "Now the key idea that ties everything together. Each circle has its own pair of in-plane axes — *u* and *v* — and a center. Any point on the circle is described by a single number: its angle *theta* around the center."

> "Why is this powerful? Because the same angle *theta* corresponds to the same anatomical direction on the humeral head — for every patient and every vendor. We have turned a 2D circle into a 1D anatomical coordinate."

**Tip:** mostrar el diagrama de la derecha — un solo ángulo es todo lo que se necesita.

---

## Slide 12 — Sectores angulares → tendones (≈40 s)

> "And here is the pay-off. The four main rotator cuff tendons insert in predictable angular sectors on the articular circle. Subscapularis between roughly 0 and 45 degrees, supraspinatus between 45 and 135, infraspinatus between 135 and 225, and teres minor between 225 and 270."

> "So the localization rule becomes simple: if the inlier ratio drops focally inside one of those sectors, that is a *candidate tear in the corresponding tendon*, at the height of the slice where the drop occurs. The articular circle has become a one-dimensional *tendon map*."

**Importante:** este es el corazón del trabajo. Pausa breve después de "tendon map".

---

## Slide 13 — Robustez multivendor y equidad (≈30 s)

> "We tested across three vendors and three sites: SIEMENS in Quibdó (Chocó), GE Optima in Bogotá, and Philips Ingenia at SOMA Radiology. Six datasets in total, with matrix sizes from 256 up to 320."

> "The pipeline generalizes for three concrete reasons: SAM is frozen so there are no weights that can overfit, all thresholds are geometric — millimeters or percentages, not pixel intensities — and the direction is read from the DICOM header instead of trusting vendor conventions."

> "This matters in Colombia: places like Chocó have the scanners but not the specialists. A tool that is auditable by a generalist radiologist directly supports diagnostic capacity — which aligns with the UEMS pillar of *impacting* medical practice."

---

## Slide 14 — Resultados cualitativos (≈25 s)

> "Here is one summary figure per dataset, with six panels: slice classification, Dice per slice, the area curve, the 3D boundary point cloud, the sphere fit, and the per-slice articular circles."

> "In the area curve: green slices are kept, gray are trimmed tails, red are strong outliers excluded. The whole pipeline runs end-to-end and exports this figure automatically, so a radiologist can audit any single case in one glance."

**Tip:** no detallar cada panel — mencionar el código de colores y seguir.

---

## Slide 15 — Limitaciones (≈25 s)

> "Honest limitations. The cohort is small — six datasets — so this is a proof of concept, not a clinical validation. The localization rule, the angular-sector to tendon mapping, is a *hypothesis* and still needs validation against arthroscopy. And the automatic seed assumes typical centering of the patient."

> "But the design has one property that matters clinically: because every output is geometric, visualized and auditable, the system fails *loudly*, not silently. That is the property a radiologist needs to actually trust it."

---

## Slide 16 — Take-home y cierre (≈25 s)

> "Three ideas to take home. First, a frozen foundation model plus classical geometry can produce interpretable shoulder MRI features — with no training. Second, two geometric features — a global sphere and per-slice articular circles — let us turn the articular contour into a 1D tendon map. Third, the pipeline is robust across vendors, which makes it especially relevant in low-resource radiology settings."

> "Thank you. Questions are very welcome."

---

## Notas generales para el presentador

- **Tiempo objetivo:** 8 minutos. Promedio de 30 seg por slide; los diagramas (slides 6, 11, 12) pueden costar un poco más, así que los slides de texto deben ir rápidos.
- **Velocidad:** no leer las viñetas en voz alta. Hablar como si estuviera explicando a un colega clínico curioso, usando las viñetas solo como guía visual.
- **Tres palabras a destacar siempre:** *interpretable*, *auditable*, *vendor-agnostic*. Son los tres mensajes que el público debe recordar.
- **Tres slides que NO se pueden saltar:** Slide 6 (pipeline), Slide 11 (el truco geométrico), Slide 12 (sectores → tendones). Es el arco narrativo.
- **Honestidad:** dejar claro en el slide 15 que la regla de localización por sectores es una hipótesis, no un resultado validado. Esto genera confianza, no resta credibilidad.
- **Si se queda corto de tiempo:** se puede acortar el slide 14 (resultados) a "this is the auto-generated audit figure, color-coded as we just described" y pasar al siguiente.
- **Si sobra tiempo:** detenerse un poco más en slide 11 — el paso de circle 2D a coordenada angular 1D es la idea más original del trabajo.
