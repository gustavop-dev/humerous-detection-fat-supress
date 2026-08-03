# Demostraciones — ejercicios del pptx de Marco Paluszny

Los ejercicios de papel del archivo `Proyecto – Gustavo A. Pérez.pptx`. Cada uno indica
qué función y qué test del paquete [`Geometry/`](../Geometry) lo acompaña.

Notación: `η = Q = diag(1, 1, 1, −1)`. Un punto `p` de la esfera de radio `ρ` centrada en el
origen se sube a `P³` como el 4-vector `q = [p, ρ]`, que cumple `qᵀ η q = |p|² − ρ² = 0`.

---

## Ej-4 (diapositiva 4) — `O(3)` es subgrupo de `O(3,1)`

**Enunciado.** Si `R ∈ O(3)`, verificar que `M = [[R, 0], [0, 1]] ∈ O(3,1)`.

**Demostración.** Escribimos `M` y `η` por bloques:

```
M = [ R  0 ]        η = [ I₃   0 ]
    [ 0  1 ]            [ 0   −1 ]
```

Entonces

```
Mᵀ η M = [ Rᵀ  0 ] [ I₃   0 ] [ R  0 ] = [ Rᵀ  0 ] [  R  0 ] = [ RᵀR   0 ] = [ I₃   0 ] = η
         [ 0   1 ] [ 0   −1 ] [ 0  1 ]   [ 0  −1 ] [  0  1 ]   [  0   −1 ]   [ 0   −1 ]
```

usando `RᵀR = I₃`, que es la definición de `O(3)`. ∎

**Interpretación.** Las rotaciones de `R³` son exactamente los elementos de `O(3,1)` que
además **fijan el centro de la esfera**, el punto `[(0,0,0,1)]` de `P³`. Los boosts son los
elementos que preservan la esfera pero **mueven su centro**: por eso pueden alterar la razón
`R/d` y las similaridades no.

**Código.** `Geometry.lorentz.embed_rotation` · test `test_embed_rotation`
(200 rotaciones aleatorias, `‖MᵀηM − η‖ ≤ 7.8e-16`).

---

## Ej-13 (diapositiva 13) — `R/d` es un invariante de las similaridades

**Enunciado.** Sean `d₁`, `d₂` las distancias de los centros de las esferas a las rectas.
Si no se cumple `R₁/d₁ = R₂/d₂`, la `T` construida podría no llevar exactamente un par en el otro.

**Demostración.** Sea `T(x) = (sR)x + b` con `R` ortogonal y `s > 0`. Para cualesquiera `x, y`:

```
‖T(x) − T(y)‖ = ‖sR(x − y)‖ = s‖x − y‖
```

o sea `T` multiplica **todas** las distancias por el mismo factor `s`. En consecuencia:

1. `T` manda la esfera `S₁ = S(C₁, R₁)` en la esfera `S(T(C₁), s·R₁)`. Para que sea `S₂` hace
   falta `T(C₁) = C₂` y `s = R₂/R₁`.
2. La distancia de un punto a una recta es una distancia, luego
   `dist(T(C₁), T(L₁)) = s · dist(C₁, L₁) = s·d₁`.

Si además `T(L₁) = L₂`, entonces `dist(C₂, L₂) = s·d₁`, es decir `d₂ = s·d₁ = (R₂/R₁)·d₁`, y
reordenando:

```
R₁/d₁ = R₂/d₂
```

Por tanto la igualdad de razones es **necesaria**. Contrarrecíproco: si `R₁/d₁ ≠ R₂/d₂`, ninguna
similaridad lleva el par 1 exactamente en el par 2. ∎

El hueco residual es exactamente `|s·d₁ − d₂|`, que el código reporta como `gap_mm`.

**Código.** `Geometry.similarity.pair_similarity_invariant` · test `test_ratio_invariant`.

---

## Ej-17a (diapositiva 17) — siempre existe `T` con `S₁ → S₂` y `L₁ → paralela a L₂`

**Enunciado.** Dados dos pares esfera/recta, verificar que siempre se puede construir
`T(x) = (sR)x + b` que envía `S₁` en `S₂` y `L₁` en una recta paralela a `L₂`.

**Demostración (constructiva).** Sean `v₁`, `v₂` las direcciones unitarias de `L₁`, `L₂`. Tomamos:

- `s = R₂ / R₁` (positivo, pues los radios lo son);
- `R` = la rotación de Rodrigues que lleva `v₁` en `v₂`. Existe siempre: si `v₁ ≠ ±v₂` se toma el
  eje `v₁ × v₂` y el ángulo `arccos(v₁·v₂)`; si `v₁ = v₂` se toma `R = I`; si `v₁ = −v₂` se toma
  media vuelta alrededor de cualquier eje ortogonal a `v₁`;
- `b = C₂ − sR·C₁`.

Entonces:

1. `T(C₁) = sR·C₁ + b = C₂`, y como `T` escala distancias por `s = R₂/R₁`, la imagen de `S₁` es la
   esfera de centro `C₂` y radio `s·R₁ = R₂`, o sea `S₂`. ✔
2. La imagen de `L₁` es una recta (las similaridades mandan rectas en rectas) con dirección
   `R·v₁ = v₂`, luego es paralela a `L₂`. ✔ ∎

Nótese que la construcción **no usa** ninguna hipótesis sobre `d₁` y `d₂`: por eso funciona siempre.
Lo que no puede garantizar es la coincidencia, que es el Ej-17b.

**Código.** `Geometry.similarity.similarity_axis_parallel` · test
`test_axis_parallel_always_exists` (50 pares aleatorios, paralelismo ≤ 1.2e-15).

---

## Ej-17b (diapositiva 17) — por qué no se puede hacer coincidir `T(L₁)` con `L₂`

**Enunciado.** Sugiera razones por las que no es posible hacer coincidir el transformado de `L₁`
con `L₂`. Sugerencia: `T` preserva razones de distancias/longitudes.

**Demostración.** Es el contrarrecíproco del Ej-13. La construcción del Ej-17a ya fija `s` y `b`
sin ningún grado de libertad (los determinan `S₁ → S₂`), y `R` queda determinada salvo una rotación
alrededor de `v₂`. Ninguna de esas rotaciones cambia la distancia del centro a la recta imagen, que
vale `s·d₁` sea cual sea la elección. Así que la recta imagen vive en el cilindro de radio `s·d₁`
alrededor de `C₂`, mientras que `L₂` vive en el de radio `d₂`.

Ambos cilindros coinciden si y solo si `s·d₁ = d₂`, es decir `R₁/d₁ = R₂/d₂`. Si las razones
difieren, los cilindros son disjuntos y **ninguna** elección de `R` puede hacer coincidir las rectas:
el hueco mínimo es `|s·d₁ − d₂|`. ∎

**La raíz del asunto.** `R/d` es adimensional y las similaridades preservan razones de longitudes,
luego preservan cualquier cantidad adimensional construida con distancias. Para cambiar `R/d` hay
que salir del grupo de similaridades: eso es lo que hacen los boosts de `O(3,1)`, que preservan la
esfera pero mueven su centro y sí alteran `R/d`.

---

## Ej-24 (diapositiva 24, vale doble) — contraejemplo

**Enunciado.** Busque dos pares intersectantes esfera/recta que no se puedan transformar uno en el
otro con una similaridad. Verifique que sí existe una similaridad que envía una esfera en la otra y
la recta del primer par en una recta paralela a la del segundo.

**Ejemplo.**

| | Centro | Radio | Punto de la recta | Dirección | `d` | `R/d` |
|---|---|---|---|---|---|---|
| Par 1 | `(0,0,0)` | 5 | `(3,0,0)` | `e_z` | 3 | 5/3 |
| Par 2 | `(10,0,0)` | 5 | `(11,0,0)` | `e_y` | 1 | 5 |

**Ambas rectas son secantes**, que es lo que pide el enunciado: `d₁ = 3 < 5 = R₁` y
`d₂ = 1 < 5 = R₂`. Los cortes son `(3, 0, ±4)` para la primera y `(11, ±√24, 0)` para la segunda.

**No hay similaridad.** `R₁/d₁ = 5/3 ≈ 1.667` y `R₂/d₂ = 5`. Por el Ej-13 no existe. El hueco
irreducible es `|s·d₁ − d₂| = |1·3 − 1| = 2` mm (aquí `s = R₂/R₁ = 1`).

**Pero sí existe la versión paralela.** Aplicando el Ej-17a con `s = 1`, `R` = rotación de 90°
que lleva `e_z` en `e_y`, y `b = (10,0,0)`: la esfera 1 va exactamente a la esfera 2 y la recta
imagen es paralela a `L₂`, a 3 mm del centro en vez de a 1 mm.

**Contraejemplo con datos reales.** El propio proyecto lo produce: el húmero standard tiene
`R/d = 2.18` y el paciente `pd_tse_fs_tra_320_fov150_4` lee `R/d = 5.15`. Ver el informe: buena
parte de esa diferencia resultó ser un artefacto de la cobertura del MRI, no anatomía.

**Código.** `Geometry.similarity.build_non_similar_example` · test `test_non_similar_example`.

---

## Ej-25a (diapositiva 25) — secuencia estructurada para detectar los arcos circulares

**Enunciado.** Investigue cómo detectar estos segmentos de círculo y proponga una secuencia
estructurada de pasos.

**Ya estaba implementado** en este repositorio antes de estos ejercicios; aquí solo se documenta la
secuencia, en [`humerus_boundary_analysis.py`](../humerus_boundary_analysis.py):

1. **Clasificar las slices** por Dice de la propagación (`parse_summary`, `classify_slices`):
   correcta ≥ 0.80, ambigua ≥ 0.55, fallida por debajo.
2. **Validar la curva de áreas** (`validate_area_curve`, "Step 1.5"): se lee la dirección de
   adquisición del DICOM (`ImagePositionPatient` proyectado sobre `cross(row_cosine, col_cosine)`)
   para saber qué extremo es la cabeza, y se recorta lo que queda fuera de la campana. El recorte de
   la cabeza es más estricto (30 %) que el de la diáfisis (10 %) porque la cabeza cierra
   abruptamente y la diáfisis tapera.
3. **Extraer el borde** de cada máscara válida (`extract_boundary_pixels`, contornos externos).
4. **Pasar a coordenadas de paciente en mm** (`pixel_to_world`) con la afín del DICOM.
5. **Proyectar los puntos de cada slice a su plano** por SVD (`project_to_slice_plane`), obteniendo
   una base ortonormal `(u, v)` y coordenadas 2D.
6. **Ajustar círculos en 2D** con Kåsa algebraico (`fit_circle_2d_kasa`) dentro de un RANSAC
   iterativo (`ransac_circle_2d`, `fit_circles_in_slice`): hasta 3 círculos por slice, quitando los
   inliers entre iteraciones, con distancia de inlier 1.5 mm.
7. **Marcar las slices articulares** (`fit_circles_per_slice`): las que tienen área ≥ 70 % del
   máximo de la curva validada (`ARTICULAR_AREA_RATIO = 0.70`). Son las del ecuador de la cabeza.
8. **Filtrar arcos espurios**: se exige `≥ 10` inliers y razón de inliers `≥ 0.30`, lo que elimina
   los "arcos" de un punto suelto.
9. **Persistir y validar visualmente**: `slice_circles.csv`, `slice_circles_points.csv` y un render
   de 5 paneles por slice articular con el círculo dibujado sobre el DICOM original.

Los resultados están en `Datasets/Out/<dataset>/slice_circles.csv` y en los `*_circle.png`.

**Medido en este trabajo:** la cobertura angular de esos arcos es del 33 % al 83 % del círculo, con
arco contiguo máximo de 40° a 160°. Hay huecos reales, así que cualquier análisis por sector tendrá
zonas sin soporte.

---

## Ej-25b (diapositiva 25) — estrategia para detectar el eje del húmero

**Enunciado.** Proponga una estrategia para detectar el eje del húmero a partir de cortes del brazo,
empezando por debajo del cuello y terminando antes del codo.

**Estrategia implementada.**

1. **Separar cabeza de diáfisis.** Los círculos por slice ya vienen etiquetados: los articulares son
   la cabeza (radios de 20–25 mm), los no articulares son el brazo (radios de 9–16 mm).
2. **El eje pasa por los centros de las secciones del brazo.** Cada corte axial de la diáfisis es
   cuasi circular, y el centro de ese círculo es un punto del eje. Se ajusta una recta por PCA a esos
   centros.
3. **Sobre malla (el standard), tres precauciones** que resultaron imprescindibles:
   - muestrear la **superficie** ponderando por área, no los vértices (el 70 % de los vértices de
     `Humero.stl` están en la cabeza);
   - cortar en losas perpendiculares al eje y usar el centroide **robusto** de cada losa (mediana y
     recorte), porque la malla tiene geometría espuria;
   - **no** reseleccionar la región diafisaria iterando sobre el eje refinado: se realimenta y
     diverge (medido: 8.5° → 8.4° → 11.1°).
4. **Cuantificar la incertidumbre**: `σ` angular por bootstrap y una curva de convergencia que
   recalcula el eje con los `k` centros más distales.
5. **Calibrar el sesgo**: truncar el standard a la cobertura del MRI y re-estimar con el mismo
   estimador, para medir cuánto se desvía.

**El resultado principal y su limitación.** Con volúmenes axiales de ~68 mm, los cortes que el
pipeline llama "diáfisis" son en realidad cuello quirúrgico y metáfisis. El eje sale muy bien
*determinado* (`σ` de 0.36°–1.21°) pero con **10° de error sistemático** y una excentricidad `d`
subestimada por un factor de ~0.43. Ver el informe para la calibración y su validación cruzada.

**Lo que haría falta de verdad.** Cobertura distal: o bien un stack axial que llegue a la mitad del
brazo, o bien fusionar la esfera del axial con el eje de un stack **coronal**, donde la diáfisis se
ve a lo largo dentro de cada imagen. En `Datasets/In/dp_tse_cor_320_fs_8…` hay un stack coronal sin
procesar que serviría; requiere segmentarlo con SAM primero.

**Código.** `Geometry.sphere_axis.estimate_shaft_axis`, `fit_axis_from_shaft_circles`,
`axis_convergence_curve`, `calibrate_axis_bias_from_mesh`.

---

## Sobre la receta de `O(3,1)` de las diapositivas 20-21

Se portó a Python y **es correcta y general** (`boost_fixing_sphere_pptx_recipe`). Tres notas de
implementación que la diapositiva no cubre:

1. Los cuatro vectores de completación escritos a mano (`s1 = s20 = [0,1,0,0]`,
   `t1 = [0,0,5,2]/√21`, `t20 = [0,0,5,−3]/4`) **son la completación η-ortonormal de cada marco**;
   una implementación general tiene que generarlos con Gram-Schmidt en la métrica de Minkowski.
2. El paso `s2 = M0·s1`, `t2 = M0·t1` es **redundante**: sale `M0 == M` (comprobado a 9.5e-14).
   Si `s20, t20` son la completación del marco 2, entonces `F2 = M0·F1` y por tanto `M = M0`.
   La ruta corta equivalente es `M = F₂ · (η F₁ᵀ η)`, que además evita invertir numéricamente.
3. `common = (nA + nB)/2` es **timelike**: su norma de Minkowski es negativa (−21 en el ejemplo) y
   `sqrt` de un negativo da `NaN`. Hay que normalizar con `sqrt(|·|)` y arrastrar el signo aparte.

**Trampa detectada.** Si en `F20` se usan por descuido los vectores del marco 1 en vez de los del
marco 2, la matriz resultante **sigue mandando `A₁ → A₂` y `B₁ → B₂` con error 4.4e-16 pero ya no
está en `O(3,1)`** (`‖MᵀηM − η‖ = 1.88`). Es un fallo silencioso: mapea bien y aun así está mal. Por
eso `boost_fixing_sphere` verifica la pertenencia al grupo antes de devolver nada, y hay un test
dedicado (`test_wrong_completion_is_a_silent_failure`).
