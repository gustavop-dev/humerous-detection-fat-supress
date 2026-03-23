#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Segmentación completa de carpeta con SAM usando propagación de centros.
Proceso:
1. Usuario clickea la imagen del medio
2. Propaga hacia arriba y abajo usando centros calculados
3. Procesa TODAS las imágenes (no se detiene por umbral)
4. Registra advertencias cuando hay cambios grandes
"""

# Standard library
import os

# Third-party libraries
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
import cv2
import matplotlib.pyplot as plt

# Local modules
from DCM.load_dicom_as_image import read_image_file, get_dataset_files
from Graphics.grafication import (
    extract_contour_points_3d,
    plot_3d_contours,
    plot_3d_contours_by_slice,
    plot_3d_solid_mesh,
    export_mesh_to_stl,
)
import Segmentation.Masks as Masks
from Segmentation.propagation import propagate_segmentation
from Segmentation.segment_image import segment_first_image, segment_image


device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🖥️  Using device: {device}")

# Paths - MODIFICAR SEGÚN TUS NECESIDADES
ckpt = "Checkpoints/sam_vit_b_01ec64.pth"  # Ruta al checkpoint de SAM
data_dir = "Datasets/In/pd_tse_fs_tra_320_fov150_4"  # Carpeta con imágenes DICOM
output_dir = "Datasets/Out/pd_tse_fs_tra_320_fov150_4"  # Carpeta de resultados



# Parámetros
SIMILARITY_THRESHOLD = 0.25  # 20% - Solo para advertencias, NO detiene la propagación
WARNING_THRESHOLD = 0.35     # 30% - Advertencia severa pero continúa
NEIGHBOR_EXTRA_RATIO_THRESHOLD = 0.25
BRIGHT_ARTIFACT_PERCENTILE = 95.0
BRIGHT_ARTIFACT_MIN_RATIO = 0.08
BRIGHT_RING_RADIUS = 6

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load SAM model
print("🔄 Loading SAM model...")
# Cargar modelo sin checkpoint primero, luego cargar pesos con map_location
sam = sam_model_registry["vit_b"]()
checkpoint_data = torch.load(ckpt, map_location=device)
sam.load_state_dict(checkpoint_data)
sam = sam.to(device)
predictor = SamPredictor(sam)
print("✅ SAM model loaded!")


def main():
    
    # Obtener información del dataset (sin cargar imagen aún)
    dataset = get_dataset_files(data_dir)
    if dataset is None:
        return
    
    # Cargar imagen del medio solo cuando se necesita
    middle_img = read_image_file(dataset.middle_file)
    if middle_img is None:
        print("❌ Error reading middle image!")
        return
    
    print(f"✅ Loaded middle image: {middle_img.shape}")
    
    
    files = dataset.files
    middle_idx = dataset.middle_idx
    middle_name = dataset.middle_name
    middle_file = dataset.middle_file
    
    # STEP 1: Usuario segmenta la imagen del medio usando segment_sam_points.py
    print("\n" + "="*70)
    print("🖱️  PASO 1: SEGMENTACIÓN DE IMAGEN DEL MEDIO")
    print("="*70)
    print("Se abrirá segment_sam_points.py para segmentar la imagen del medio.")
    print("Instrucciones:")
    print("  • Click DERECHO: Punto positivo (dentro del objeto)")
    print("  • Click IZQUIERDO: Punto negativo (fuera del objeto)")
    print("  • Tecla 'z': Deshacer último punto")
    print("  • Tecla 'c': Limpiar todos los puntos")
    print("  • Cierra la ventana para guardar y continuar")
    print("="*70)
    
    # Ejecutar segment_sam_points.py con la imagen del medio
    middle_mask, middle_score, middle_seed = segment_first_image(predictor, middle_img , middle_name)
    
    if middle_mask is None or np.sum(middle_mask) == 0:
        print("❌ No se pudo segmentar la imagen del medio!")
        return
    
    
    
    print(f"\n📌 Segmentación completada usando segment_sam_points.py")
    
    # La máscara ya está refinada desde segment_sam_points.py
    # No necesitamos volver a segmentar
    
    # Calculate center of middle segmentation
    middle_center = Masks.calculate_mask_center(middle_mask)
    
    if middle_center is None:
        print("❌ Failed to calculate center of middle segmentation!")
        return
    
    print(f"✅ Centro calculado: ({middle_center[0]:.0f}, {middle_center[1]:.0f})")
    print(f"   Score: {middle_score:.3f}, Area: {np.sum(middle_mask)} px")

    # Save middle segmentation (seg_point = seed realmente usado)
    Masks.save_segmentation_result(middle_img, middle_mask, middle_name, output_dir, 
                            center=middle_center, seg_point=middle_seed, 
                            info=f"Score: {middle_score:.3f}")
    
    
    # STEP 2: Segmentar TODAS las imágenes
    print("\n" + "="*70)
    print("🔄 PASO 2: SEGMENTACIÓN DE TODAS LAS IMÁGENES")
    print("="*70)
    print("⚡ Procesará TODAS las imágenes sin detenerse")
    print(f"⚠️  Advertencias cuando diferencia > {SIMILARITY_THRESHOLD*100:.0f}%")
    
    # Initialize results storage (guardar también área y centro originales)
    middle_area = float(np.sum(middle_mask))
    segmentations = {middle_idx: {
        'mask': middle_mask,
        'center': middle_center,
        'seg_point': middle_seed,  # Punto con el que realmente se segmentó
        'score': middle_score,
        'area': middle_area,
        'orig_area': middle_area,
        'orig_center': list(middle_center),
        'dice': 1.0,  # referencia consigo misma
        'iou': 1.0
    }}
    
    # Lista de slices fallidas (diferencia > 30%)
    failed_slices = []
    
    # STEP 3: Propagar a todas las imágenes
    print("\n" + "="*70)
    print("🔄 PASO 3: PROPAGACIÓN COMPLETA")
    print("="*70)
    print(f"📊 Umbrales de advertencia: {SIMILARITY_THRESHOLD*100:.0f}% (leve) / {WARNING_THRESHOLD*100:.0f}% (severa)")
    print("✅ Se procesarán TODAS las imágenes")
    print("="*70)
    
    # Propagate backwards (hacia arriba)
    segmentations, failed_slices = propagate_segmentation(
        predictor, files, middle_idx, middle_mask, middle_center,
        segmentations, failed_slices, output_dir, direction="backward"
    )
    
    # Propagate forwards (hacia abajo)
    segmentations, failed_slices = propagate_segmentation(
        predictor, files, middle_idx, middle_mask, middle_center,
        segmentations, failed_slices, output_dir, direction="forward"
    )
    
    # Post-proceso: analizar solapamiento con slices vecinas para detectar outliers locales
    neighbor_suspicious = []
    sorted_indices = sorted(segmentations.keys())
    for idx in sorted_indices:
        if (idx - 1) in segmentations and (idx + 1) in segmentations:
            mask_prev = segmentations[idx - 1]['mask']
            mask_curr = segmentations[idx]['mask']
            mask_next = segmentations[idx + 1]['mask']

            # "Sombra" esperada: unión de máscaras vecinas
            union_neighbors = np.logical_or(mask_prev, mask_next)
            extra_region = np.logical_and(mask_curr, np.logical_not(union_neighbors))

            curr_area = float(np.sum(mask_curr))
            if curr_area == 0.0:
                continue

            extra_area = float(np.sum(extra_region))
            extra_ratio = extra_area / curr_area

            if extra_ratio > NEIGHBOR_EXTRA_RATIO_THRESHOLD:
                neighbor_suspicious.append({
                    'idx': idx,
                    'filename': os.path.basename(files[idx]),
                    'extra_ratio': extra_ratio,
                    'extra_area': extra_area,
                    'area': curr_area
                })

    # Intentar reparar slices sospechosas re-segmentando con puntos de vecinas y punto negativo en la región extra
    if neighbor_suspicious:
        print("\n🛠 Re-segmentando slices sospechosas usando centros de vecinas y zona extra como punto negativo...")
        for s in neighbor_suspicious:
            idx = s['idx']
            if (idx - 1) not in segmentations or (idx + 1) not in segmentations:
                continue
            filename = s['filename']

            mask_prev = segmentations[idx - 1]['mask']
            mask_curr = segmentations[idx]['mask']
            mask_next = segmentations[idx + 1]['mask']

            img = read_image_file(files[idx])
            if img is None:
                print(f"  ⚠️ No se pudo leer imagen para {filename}, se omite reparación.")
                continue

            # Región extra original
            union_neighbors = np.logical_or(mask_prev, mask_next)
            extra_region = np.logical_and(mask_curr, np.logical_not(union_neighbors))

            neg_point = Masks.calculate_mask_center(extra_region)
            if neg_point is None:
                continue

            # Elegir vecina cuya área sea más parecida a la de la slice actual
            area_prev = float(np.sum(mask_prev))
            area_next = float(np.sum(mask_next))
            curr_area = float(np.sum(mask_curr))

            if abs(area_prev - curr_area) <= abs(area_next - curr_area):
                pos_center = segmentations[idx - 1]['center']
            else:
                pos_center = segmentations[idx + 1]['center']

            if pos_center is None:
                continue

            pos_point = [float(pos_center[0]), float(pos_center[1])]

            input_points = np.array([pos_point, neg_point], dtype=np.float32)
            input_labels = np.array([1, 0], dtype=np.int32)

            img_enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
            predictor.set_image(img_enhanced)
            try:
                new_mask, _, new_score, _ = segment_image(predictor, input_points, input_labels, refine=True)
            except Exception as e:
                print(f"  ⚠️ Error re-segmentando {filename}: {e}")
                continue

            if new_mask is None or np.sum(new_mask) == 0:
                continue

            # Calcular nueva área extra con la máscara re-segmentada
            new_union_neighbors = union_neighbors  # vecinas no cambian
            new_extra_region = np.logical_and(new_mask, np.logical_not(new_union_neighbors))
            new_curr_area = float(np.sum(new_mask))
            if new_curr_area == 0.0:
                continue
            new_extra_area = float(np.sum(new_extra_region))
            new_extra_ratio = new_extra_area / new_curr_area

            if new_extra_ratio < s['extra_ratio']:
                print(
                    f"  ✅ {filename}: área extra reducida de {s['extra_ratio']*100:.1f}% a {new_extra_ratio*100:.1f}%. Actualizando máscara."
                )
                new_center = Masks.calculate_mask_center(new_mask) or segmentations[idx]['center']
                segmentations[idx]['mask'] = new_mask
                segmentations[idx]['center'] = new_center
                segmentations[idx]['seg_point'] = pos_point
                segmentations[idx]['score'] = float(new_score)
                segmentations[idx]['area'] = new_curr_area

                # Actualizar métricas de solapamiento almacenadas
                s['extra_ratio'] = new_extra_ratio
                s['extra_area'] = new_extra_area
                s['area'] = new_curr_area
                s['repaired'] = True

                # Guardar visualización actualizada de la segmentación reparada
                base_name = os.path.splitext(filename)[0]
                Masks.save_segmentation_result(
                    img,
                    new_mask,
                    base_name,
                    output_dir,
                    center=new_center,
                    seg_point=pos_point,
                    neg_point=neg_point,
                    info=f"Repaired extra: {new_extra_ratio*100:.1f}% | Score: {float(new_score):.3f}",
                )
            else:
                print(
                    f"  ⚠️ {filename}: resegmentación no mejoró (extra {s['extra_ratio']*100:.1f}% -> {new_extra_ratio*100:.1f}%), se mantiene máscara original."
                )

    # Filtrar lista final de slices sospechosas tras posible reparación
    final_neighbor_suspicious = [
        s for s in neighbor_suspicious if s['extra_ratio'] > NEIGHBOR_EXTRA_RATIO_THRESHOLD
    ]

    print("\n" + "="*70)
    print("🔎 ANÁLISIS DE SOLAPAMIENTO ENTRE SLICES VECINAS")
    print("="*70)
    print(f"Umbral de área extra relativa: {NEIGHBOR_EXTRA_RATIO_THRESHOLD*100:.1f}% del área de la slice")

    if final_neighbor_suspicious:
        print(f"\n🚨 Slices con región extra fuera de la 'sombra' de vecinas: {len(final_neighbor_suspicious)}")
        for s in final_neighbor_suspicious:
            print(f"  - [{s['idx']+1}] {s['filename']}: extra={s['extra_ratio']*100:.1f}% (extra {s['extra_area']:.0f} / total {s['area']:.0f} px)")
    else:
        print("\n✅ No se detectaron outliers locales por solapamiento vecinal.")

    # Post-proceso adicional: detectar regiones muy brillantes dentro de la máscara y usar su centro como punto negativo
    print("\n" + "="*70)
    print("🔦 ANÁLISIS DE ARTEFACTOS BRILLANTES DENTRO DE LA MÁSCARA")
    print("="*70)

    for idx in sorted(segmentations.keys()):
        s = segmentations[idx]
        mask = s['mask']
        if mask is None or np.sum(mask) == 0:
            continue

        img = read_image_file(files[idx])
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Construir un anillo externo alrededor de la máscara actual
        mask_uint8 = (mask.astype(np.uint8) * 255)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * BRIGHT_RING_RADIUS + 1, 2 * BRIGHT_RING_RADIUS + 1)
        )
        dilated = cv2.dilate(mask_uint8, kernel)
        dilated_bool = dilated > 0
        ring = np.logical_and(dilated_bool, np.logical_not(mask))

        ring_area = float(np.sum(ring))
        if ring_area == 0.0:
            continue

        ring_intensity = gray[ring]
        if ring_intensity.size == 0:
            continue

        high_thr = np.percentile(ring_intensity, BRIGHT_ARTIFACT_PERCENTILE)
        bright_ring = np.logical_and(ring, gray >= high_thr)
        bright_area = float(np.sum(bright_ring))
        if bright_area == 0.0:
            continue

        bright_ratio = bright_area / ring_area
        if bright_ratio < BRIGHT_ARTIFACT_MIN_RATIO:
            continue

        bright_uint8 = (bright_ring.astype(np.uint8) * 255)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_uint8, connectivity=8)
        if num_labels <= 1:
            continue

        best_label = -1
        best_area = 0.0
        for label in range(1, num_labels):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area > best_area:
                best_area = area
                best_label = label

        if best_label <= 0:
            continue

        component_mask = labels == best_label
        cy, cx = np.where(component_mask)
        if cx.size == 0 or cy.size == 0:
            continue

        # Centroide de la región brillante EN EL ANILLO EXTERNO
        neg_point = [float(np.mean(cx)), float(np.mean(cy))]

        # Punto positivo: centro de la máscara (región de húmero "oscuro")
        pos_center = s.get('center', None)
        if pos_center is None:
            continue

        pos_point = [float(pos_center[0]), float(pos_center[1])]

        input_points = np.array([pos_point, neg_point], dtype=np.float32)
        input_labels = np.array([1, 0], dtype=np.int32)

        img_enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        predictor.set_image(img_enhanced)
        try:
            new_mask, _, new_score, _ = segment_image(predictor, input_points, input_labels, refine=True)
        except Exception:
            continue

        if new_mask is None or np.sum(new_mask) == 0:
            continue

        new_mask_area = float(np.sum(new_mask))
        if new_mask_area == 0.0 or new_mask_area < 0.4 * mask_area:
            continue

        new_bright_region = np.logical_and(new_mask, gray >= high_thr)
        new_bright_area = float(np.sum(new_bright_region))
        new_bright_ratio = new_bright_area / new_mask_area if new_mask_area > 0 else bright_ratio

        if new_bright_ratio < bright_ratio:
            filename = os.path.basename(files[idx])
            print(
                f"  ✅ {filename}: región brillante reducida de {bright_ratio*100:.1f}% a {new_bright_ratio*100:.1f}%. Actualizando máscara."
            )
            new_center = Masks.calculate_mask_center(new_mask) or pos_center
            s['mask'] = new_mask
            s['center'] = new_center
            s['seg_point'] = pos_point
            s['score'] = float(new_score)
            s['area'] = new_mask_area

            base_name = os.path.splitext(filename)[0]
            Masks.save_segmentation_result(
                img,
                new_mask,
                base_name,
                output_dir,
                center=new_center,
                seg_point=pos_point,
                neg_point=neg_point,
                info=f"Bright fixed: {new_bright_ratio*100:.1f}% | Score: {float(new_score):.3f}",
            )

    # STEP 4: Reconstrucción 3D
    print("\n" + "="*70)
    print("🎨 PASO 4: RECONSTRUCCIÓN 3D")
    print("="*70)
    
    # Extraer puntos 3D de los contornos
    print("📍 Extrayendo puntos de contornos...")
    z_spacing = 12  # Separación entre slices
    points_3d, slice_info = extract_contour_points_3d(segmentations, files, middle_idx, z_spacing=z_spacing)
    
    print(f"\n📊 Total: {len(points_3d)} puntos 3D de {len(segmentations)} slices")
    print(f"   Separación Z entre slices: {z_spacing} unidades")
    
    if len(points_3d) > 0:
        # Guardar puntos en archivo numpy (para usar en otros programas)
        points_path = os.path.join(output_dir, "contour_points_3d.npy")
        np.save(points_path, points_3d)
        print(f"💾 Puntos guardados en: {points_path}")
        
        # También guardar como CSV para compatibilidad
        csv_path = os.path.join(output_dir, "contour_points_3d.csv")
        np.savetxt(csv_path, points_3d, delimiter=',', header='x,y,z', comments='')
        print(f"💾 CSV guardado en: {csv_path}")
        
        # Visualización 3D - Nube de puntos (guardar imágenes)
        print("\n🎨 Generando visualización 3D (nube de puntos)...")
        plot_3d_contours(points_3d, output_dir, title=f"Reconstrucción 3D - {len(segmentations)} slices")
        
        # Visualización 3D - Contornos por slice (guardar imágenes)
        print("\n🎨 Generando visualización 3D (contornos por slice)...")
        plot_3d_contours_by_slice(segmentations, files, middle_idx, output_dir, z_spacing=z_spacing)
        
        # Visualización 3D - Malla sólida (guardar imágenes)
        print("\n🔨 Generando malla sólida 3D...")
        triangles = plot_3d_solid_mesh(
            segmentations, files, middle_idx, output_dir,
            z_spacing=z_spacing,
            num_points_per_contour=100,
            color='skyblue',
            alpha=0.7,
            interactive=False,
            with_caps=True
        )
        
        # Exportar a STL para impresión 3D o software CAD
        if triangles:
            stl_path = os.path.join(output_dir, "modelo_3d.stl")
            export_mesh_to_stl(triangles, stl_path)
        
        # Mostrar visualización interactiva de malla sólida
        print("\n🖱️  Abriendo malla sólida 3D interactiva...")
        plot_3d_solid_mesh(
            segmentations, files, middle_idx, output_dir,
            z_spacing=z_spacing,
            num_points_per_contour=100,
            color='skyblue',
            alpha=0.7,
            interactive=True,
            with_caps=True
        )
    else:
        print("⚠️ No hay puntos 3D para visualizar")
    
    print("="*70)
    
    # Summary
    print("\n" + "="*70)
    print("🎉 PROCESAMIENTO COMPLETADO - CARPETA COMPLETA")
    print("="*70)
    print(f"📁 Resultados guardados en: {output_dir}")
    print(f"✅ Segmentaciones exitosas: {len(segmentations)}/{len(files)} imágenes")
    
    # Contar advertencias
    warnings_count = sum(1 for s in segmentations.values() if 'dice' in s and (1.0 - s['dice']) > SIMILARITY_THRESHOLD)
    
    if warnings_count > 0:
        print(f"⚠️  Imágenes con advertencias leves: {warnings_count}")
    
    # Mostrar imágenes fallidas
    if len(failed_slices) > 0:
        print(f"\n🚨 IMÁGENES FALLIDAS ({len(failed_slices)} total):")
        for failed in failed_slices:
            dice_str = f" (Dice: {failed['dice']:.3f})" if 'dice' in failed else ""
            print(f"   - [{failed['idx']+1}] {failed['filename']}: {failed['reason']}{dice_str}")
    
    # Calculate statistics
    if len(segmentations) > 1:
        dice_scores = [s['dice'] for s in segmentations.values() if 'dice' in s]
        if dice_scores:
            print(f"📊 Estadísticas de similitud:")
            print(f"   - Dice promedio: {np.mean(dice_scores):.3f}")
            print(f"   - Dice mínimo: {np.min(dice_scores):.3f}")
            print(f"   - Dice máximo: {np.max(dice_scores):.3f}")
    
    print("="*70)
    
    # Save summary
    summary_path = os.path.join(output_dir, "propagation_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("SAM Complete Folder Segmentation Summary\n")
        f.write("="*70 + "\n")
        f.write(f"Dataset: {data_dir}\n")
        f.write(f"Middle image: {os.path.basename(middle_file)} (index {middle_idx+1})\n")
        f.write(f"Initial segmentation: segment_sam_points.py\n")
        f.write(f"Warning threshold: {SIMILARITY_THRESHOLD*100:.0f}%\n")
        f.write(f"Severe warning threshold: {WARNING_THRESHOLD*100:.0f}%\n")
        f.write(f"Neighbor extra-area threshold: {NEIGHBOR_EXTRA_RATIO_THRESHOLD*100:.1f}%\n")
        f.write(f"Total images: {len(files)}\n")
        f.write(f"Successfully segmented: {len(segmentations)}\n")
        f.write(f"Images with warnings: {warnings_count}\n")
        f.write(f"Failed images (skipped): {len(failed_slices)}\n")
        
        if len(failed_slices) > 0:
            f.write("\nFailed slices:\n")
            for failed in failed_slices:
                dice_str = f" (Dice: {failed['dice']:.3f})" if 'dice' in failed else ""
                f.write(f"  - [{failed['idx']+1}] {failed['filename']}: {failed['reason']}{dice_str}\n")
        if len(final_neighbor_suspicious) > 0:
            f.write("\nSlices sospechosas por solapamiento con vecinas:\n")
            for s in final_neighbor_suspicious:
                f.write(
                    f"  - [{s['idx']+1}] {s['filename']}: extra={s['extra_ratio']*100:.1f}% (extra {s['extra_area']:.0f} / total {s['area']:.0f} px)\n"
                )
        f.write("\n" + "="*70 + "\n")
        f.write("Results per image:\n")
        f.write("-"*70 + "\n")
        
        suspicious_idx_set = {s['idx'] for s in final_neighbor_suspicious}
        for idx in sorted(segmentations.keys()):
            s = segmentations[idx]
            filename = os.path.basename(files[idx])
            dice_str = f"{s['dice']:.3f}" if 'dice' in s else "REF"
            
            # Marcar advertencias
            warning_marker = ""
            if 'dice' in s:
                diff = 1.0 - s['dice']
                if diff > WARNING_THRESHOLD:
                    warning_marker = " 🚨"
                elif diff > SIMILARITY_THRESHOLD:
                    warning_marker = " ⚠️"
            neighbor_marker = " ⭕" if idx in suspicious_idx_set else ""
            
            # Formatear puntos
            seg_pt_str = f"({s['seg_point'][0]:.0f},{s['seg_point'][1]:.0f})" if 'seg_point' in s else "N/A"
            center_str = f"({s['center'][0]:.0f},{s['center'][1]:.0f})"
            
            f.write(f"{idx+1:3d}. {filename:<15} | Area: {s['area']:>7.0f} px | Dice: {dice_str} | Score: {s['score']:.3f} | SegPt: {seg_pt_str} | Centro: {center_str}{warning_marker}{neighbor_marker}\n")

    print(f"💾 Resumen guardado en: {summary_path}")

    indices = sorted(segmentations.keys())
    if indices:
        # ----- Curva de áreas (usando área original) -----
        x_vals = [i + 1 for i in indices]
        # Usar el área FINAL (ya corregida) de cada slice
        y_vals = [float(segmentations[i]['area']) for i in indices]
        plt.figure(figsize=(8, 4))
        plt.plot(x_vals, y_vals, marker='o')
        plt.xlabel('Slice index')
        plt.ylabel('Mask area (pixels)')
        plt.title(f'Curva de área de máscara - {data_dir}')
        plt.grid(True)
        # Un número en el eje X por cada slice procesada
        plt.xticks(x_vals)
        plt.tight_layout()
        curve_path = os.path.join(output_dir, 'mask_area_curve.png')
        plt.savefig(curve_path, dpi=150)
        plt.close()
        print(f"📈 Curva de área de máscara guardada en: {curve_path}")

        # ----- Dispersión de centros (usando centro original) -----
        centers_x = []
        centers_y = []
        for i in indices:
            info = segmentations[i]
            # Usar el centro FINAL (ya corregido)
            c = info.get('center')
            if c is None:
                continue
            centers_x.append(float(c[0]))  # columna (x)
            centers_y.append(float(c[1]))  # fila (y)

        centers_path = None
        if centers_x and centers_y:
            plt.figure(figsize=(5, 5))
            plt.scatter(centers_x, centers_y, c='blue', s=20)
            plt.xlabel('Center X (columna)')
            plt.ylabel('Center Y (fila)')
            plt.title(f'Centros de máscaras - {data_dir}')
            plt.gca().invert_yaxis()  # sistema de coordenadas tipo imagen
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            centers_path = os.path.join(output_dir, 'mask_centers_scatter.png')
            plt.savefig(centers_path, dpi=150)
            plt.close()
            print(f"📍 Dispersión de centros guardada en: {centers_path}")

        # Combinar cada render de segmentación con las gráficas (si existen)
        if centers_path is not None and os.path.exists(curve_path) and os.path.exists(centers_path):
            seg_files = [f for f in os.listdir(output_dir) if f.endswith('_seg.png')]
            if seg_files:
                # Imagen base de dispersión de centros (misma para todas las slices)
                centers_img = plt.imread(centers_path)

                # Mapa de nombre base de archivo DICOM -> índice de slice
                dicom_basenames = [os.path.splitext(os.path.basename(fp))[0] for fp in files]
                basename_to_idx = {bn: idx for idx, bn in enumerate(dicom_basenames)}

                # Ruta temporal para guardar la curva resaltada por slice
                tmp_curve_path = os.path.join(output_dir, "_tmp_mask_area_curve_highlight.png")

                for seg_name in sorted(seg_files):
                    seg_path = os.path.join(output_dir, seg_name)
                    seg_img = plt.imread(seg_path)

                    # Determinar qué punto de la curva corresponde a este *_seg.png
                    base_name = seg_name[:-8] if seg_name.endswith('_seg.png') else os.path.splitext(seg_name)[0]
                    slice_idx = basename_to_idx.get(base_name, None)

                    # Regenerar la curva de áreas destacando el punto de esta slice
                    highlight_x = None
                    highlight_y = None
                    if slice_idx is not None and slice_idx in indices:
                        pos = indices.index(slice_idx)
                        highlight_x = x_vals[pos]
                        highlight_y = y_vals[pos]

                    # Crear figura de curva específica para esta slice
                    fig_curve, axc = plt.subplots(figsize=(4, 2))
                    axc.plot(x_vals, y_vals, marker='o', color='tab:blue')
                    if highlight_x is not None and highlight_y is not None:
                        axc.scatter([highlight_x], [highlight_y], c='red', s=40, zorder=3)
                    axc.set_xlabel('Slice index')
                    axc.set_ylabel('Mask area (pixels)')
                    axc.set_title(f'Curva de área de máscara - {data_dir}')
                    axc.grid(True)
                    axc.set_xticks(x_vals)
                    fig_curve.tight_layout()
                    fig_curve.savefig(tmp_curve_path, dpi=150, bbox_inches='tight')
                    plt.close(fig_curve)

                    curve_img = plt.imread(tmp_curve_path)

                    # Componer figura final: segmentación arriba, curvas abajo
                    fig = plt.figure(figsize=(10, 10))
                    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

                    # Arriba: render de segmentación completo
                    ax_top = fig.add_subplot(gs[0, :])
                    ax_top.imshow(seg_img)
                    ax_top.axis('off')

                    # Abajo izquierda: curva de áreas con punto resaltado
                    ax_curve = fig.add_subplot(gs[1, 0])
                    ax_curve.imshow(curve_img)
                    ax_curve.axis('off')

                    # Abajo derecha: dispersión de centros
                    ax_cent = fig.add_subplot(gs[1, 1])
                    ax_cent.imshow(centers_img)
                    ax_cent.axis('off')

                    plt.tight_layout()
                    plt.savefig(seg_path, dpi=150, bbox_inches='tight')
                    plt.close(fig)

                # Borrar archivo temporal si existe
                if os.path.exists(tmp_curve_path):
                    try:
                        os.remove(tmp_curve_path)
                    except OSError:
                        pass

                print(f"🖼️ Renders de segmentación actualizados con gráficas en: {output_dir}")


if __name__ == "__main__":
    main()
