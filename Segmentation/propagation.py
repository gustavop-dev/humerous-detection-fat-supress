import os
import numpy as np
from Segmentation import Masks
from Segmentation.segment_image import segment_with_point, segment_image
from DCM.load_dicom_as_image import read_image_file
import Segmentation.Metrics as Metrics
from Segmentation.negative_points import calculate_negative_point
import cv2

SIMILARITY_THRESHOLD = 0.35  # 30% diferencia aceptable (advertencia leve fija)
WARNING_THRESHOLD = 0.45     # Diferencia base (1 - Dice) para saltar imagen cerca del centro
MAX_WARNING_EXTRA = 0.25     # Aumento máximo permitido del umbral severo en los extremos


def propagate_segmentation(predictor, files, start_idx, start_mask, start_center, 
                           segmentations, failed_slices, output_dir, direction="forward"):
    """
    Propaga la segmentación en una dirección (hacia arriba o hacia abajo).
    
    Args:
        predictor: SamPredictor inicializado
        files: Lista de archivos de imágenes
        start_idx: Índice inicial (imagen del medio)
        start_mask: Máscara inicial de referencia
        start_center: Centro inicial de referencia
        segmentations: Diccionario donde guardar resultados (se modifica in-place)
        failed_slices: Lista donde guardar slices fallidas (se modifica in-place)
        output_dir: Directorio de salida
        direction: "forward" (hacia abajo) o "backward" (hacia arriba)
    
    Returns:
        tuple: (segmentations, failed_slices) actualizados
    """
    current_idx = start_idx
    reference_mask = start_mask
    reference_center = start_center
    last_successful_idx = start_idx
    
    # Configurar dirección
    if direction == "backward":
        step = -1
        condition = lambda idx: idx > 0
        get_next_idx = lambda idx: idx - 1
        emoji = "📤"
        desc = "HACIA ARRIBA (imágenes anteriores)"
    else:  # forward
        step = 1
        condition = lambda idx: idx < len(files) - 1
        get_next_idx = lambda idx: idx + 1
        emoji = "📥"
        desc = "HACIA ABAJO (imágenes posteriores)"
    
    print(f"\n{emoji} PROPAGACIÓN {desc}...")
    
    while condition(current_idx):
        next_idx = get_next_idx(current_idx)
        next_file = files[next_idx]
        next_name = os.path.basename(next_file).split('.')[0]
        
        print(f"\n  [{next_idx+1}/{len(files)}] Procesando {os.path.basename(next_file)}...")
        print(f"    📍 Usando centro de imagen {last_successful_idx+1} (última exitosa)")
        
        # Read image
        next_img = read_image_file(next_file)
        if next_img is None:
            print(f"    ❌ Error leyendo imagen. Agregando a fallidas...")
            failed_slices.append({
                'idx': next_idx,
                'filename': os.path.basename(next_file),
                'reason': 'Error leyendo imagen'
            })
            current_idx = next_idx
            continue
        
        # Segment using reference center
        next_mask, next_score = segment_with_point(predictor, next_img, reference_center, label=1, verbose=True)
        
        # Si falla, intentar con offsets
        if next_mask is None or np.sum(next_mask) == 0:
            print(f"    🔄 Intentando con centro ajustado...")
            found_valid = False
            offsets = [(-10, 0), (10, 0), (0, -10), (0, 10), (-20, 0), (20, 0), (0, -20), (0, 20)]
            
            for dx, dy in offsets:
                adjusted_point = [reference_center[0] + dx, reference_center[1] + dy]
                h, w = next_img.shape[:2]
                if 0 <= adjusted_point[0] < w and 0 <= adjusted_point[1] < h:
                    next_mask, next_score = segment_with_point(predictor, next_img, adjusted_point, label=1, verbose=False)
                    if next_mask is not None and np.sum(next_mask) > 0:
                        print(f"    ✅ Encontrada segmentación con offset ({dx}, {dy})")
                        found_valid = True
                        break
            
            if not found_valid:
                print(f"    ❌ Segmentación falló. Agregando a lista de fallidas...")
                failed_slices.append({
                    'idx': next_idx,
                    'filename': os.path.basename(next_file),
                    'reason': 'Segmentación vacía'
                })
                current_idx = next_idx
                continue
        
        # Calculate similarity with reference
        dice = Metrics.calculate_dice_coefficient(reference_mask, next_mask)
        iou = Metrics.calculate_iou(reference_mask, next_mask)
        difference = 1.0 - dice  # Diferencia en términos de (1 - Dice)

        # Umbral severo DINÁMICO según posición relativa de la slice respecto a la central
        # - En el centro del volumen: WARNING_THRESHOLD (más estricto)
        # - En los extremos: WARNING_THRESHOLD + MAX_WARNING_EXTRA (más permisivo)
        max_distance = max(start_idx, len(files) - 1 - start_idx)
        if max_distance > 0:
            dist_from_center = abs(next_idx - start_idx)
            relative_pos = dist_from_center / max_distance  # 0.0 en el centro, ~1.0 en extremos
        else:
            relative_pos = 0.0

        dynamic_warning_threshold = WARNING_THRESHOLD + MAX_WARNING_EXTRA * relative_pos

        print(f"    📊 Dice: {dice:.3f}, IoU: {iou:.3f}, Diferencia: {difference*100:.1f}%")
        print(f"    📏 Umbral severo dinámico: {dynamic_warning_threshold*100:.1f}% (pos_rel={relative_pos:.2f})")
        
        # Si la diferencia es mayor al umbral severo dinámico, intentar con punto negativo
        if difference > dynamic_warning_threshold:
            print(f"    ⚠️ Diferencia alta ({difference*100:.1f}%). Intentando con punto negativo...")
            
            # Calcular punto negativo basado en la máscara de referencia
            neg_point = calculate_negative_point(reference_mask, reference_center, distance_factor=0.30)
            
            if neg_point is not None:
                print(f"    🔵 Punto negativo calculado: ({neg_point[0]:.0f}, {neg_point[1]:.0f})")
                
                # Preparar puntos y etiquetas
                input_points = np.array([reference_center, neg_point])
                input_labels = np.array([1, 0])  # 1=positivo, 0=negativo
                
                # Mejorar contraste y segmentar con ambos puntos
                img_enhanced = cv2.convertScaleAbs(next_img, alpha=1.2, beta=10)
                predictor.set_image(img_enhanced)
                
                try:
                    new_mask, _, new_score, _ = segment_image(predictor, input_points, input_labels, refine=True)
                    
                    if new_mask is not None and np.sum(new_mask) > 0:
                        # Calcular nuevo dice
                        new_dice = Metrics.calculate_dice_coefficient(reference_mask, new_mask)
                        new_difference = 1.0 - new_dice
                        
                        print(f"    📊 Con punto negativo - Dice: {new_dice:.3f}, Diferencia: {new_difference*100:.1f}%")
                        
                        # Si mejoró o está dentro del umbral dinámico, usar esta segmentación
                        if new_difference <= dynamic_warning_threshold or new_dice > dice:
                            print(f"    ✅ Punto negativo mejoró la segmentación!")
                            next_mask = new_mask
                            next_score = new_score
                            dice = new_dice
                            iou = Metrics.calculate_iou(reference_mask, new_mask)
                            difference = new_difference
                        else:
                            print(f"    ❌ Punto negativo no mejoró suficiente")
                except Exception as e:
                    print(f"    ⚠️ Error con punto negativo: {e}")
            else:
                print(f"    ⚠️ No se pudo calcular punto negativo")
        
        # Si aún la diferencia es mayor al umbral dinámico, SALTAR esta imagen
        if difference > dynamic_warning_threshold:
            failed_slices.append({
                'idx': next_idx,
                'filename': os.path.basename(next_file),
                'reason': f'Diferencia {difference*100:.1f}% > {dynamic_warning_threshold*100:.0f}% (umbral dinámico)',
                'dice': dice
            })
            current_idx = next_idx
            continue
        
        # Advertencia leve (pero se acepta)
        if difference > SIMILARITY_THRESHOLD:
            print(f"    ⚠️  Advertencia leve: Diferencia ({difference*100:.1f}%) > {SIMILARITY_THRESHOLD*100:.0f}%")
        
        # Calculate center for next iteration
        next_center = Masks.calculate_mask_center(next_mask)
        
        if next_center is None:
            print(f"    ⚠️  No se pudo calcular centro. Usando centro anterior.")
            next_center = reference_center
        
        print(f"    ✅ EXITOSA. Nuevo centro: ({next_center[0]:.0f}, {next_center[1]:.0f})")
        
        # Guardar resultado
        used_seg_point = list(reference_center)
        area = np.sum(next_mask)
        orig_center = list(next_center) if next_center is not None else None
        segmentations[next_idx] = {
            'mask': next_mask,
            'center': next_center,
            'seg_point': used_seg_point,
            'score': next_score,
            'area': area,
            'orig_area': area,
            'orig_center': orig_center,
            'dice': dice,
            'iou': iou
        }
        
        Masks.save_segmentation_result(next_img, next_mask, next_name, output_dir, 
                                center=next_center, seg_point=used_seg_point, neg_point=None,
                                info=f"Dice: {dice:.3f} | Score: {next_score:.3f}")
        
        # Actualizar referencia
        reference_mask = next_mask
        reference_center = next_center
        last_successful_idx = next_idx
        current_idx = next_idx
    
    return segmentations, failed_slices

