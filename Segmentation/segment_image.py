import sys
sys.path.append('path/to/segment-anything')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamPredictor
from scipy import ndimage
from skimage import morphology
import cv2
from Graphics.interface import interactive_sam_point_selector


def refine_medical_mask(mask):
    """Clean up the segmentation mask for medical images"""
    # Remove small objects
    mask_clean = morphology.remove_small_objects(mask, min_size=500)
    
    # Fill holes
    mask_filled = ndimage.binary_fill_holes(mask_clean)
    
    # Smooth with morphological operations
    kernel = morphology.disk(2)
    mask_smooth = morphology.binary_opening(mask_filled, kernel)
    mask_smooth = morphology.binary_closing(mask_smooth, kernel)
    
    return mask_smooth


def segment_image(predictor, input_points, input_labels, refine=True):
    """Segment the image using SAM with given points and labels
    
    Args:
        predictor: SamPredictor con imagen ya configurada
        input_points: Array de puntos [[x, y], ...]
        input_labels: Array de etiquetas [1, 0, ...] (1=positivo, 0=negativo)
        refine: Si True, aplica refinamiento médico a la máscara
        
    Returns:
        tuple: (refined_mask, best_mask, best_score, all_masks)
    """
    # Generate masks using the selected points
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True
    )

    # Select best mask
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]
    best_score = scores[best_idx]

    # Refine mask for medical uses
    if refine:
        refined_mask = refine_medical_mask(best_mask)
        
        # Si el refinamiento eliminó todo, usar máscara con solo fill holes
        if np.sum(refined_mask) == 0 and np.sum(best_mask) > 0:
            refined_mask = ndimage.binary_fill_holes(best_mask)
    else:
        refined_mask = best_mask
        
    return refined_mask, best_mask, best_score, masks


def segment_with_point(predictor, img, point, label=1, verbose=False):
    """Segment an image using a single point
    
    Args:
        predictor: SamPredictor inicializado
        img: Imagen a segmentar (numpy array RGB)
        point: Punto [x, y]
        label: Etiqueta del punto (1=positivo, 0=negativo)
        verbose: Mostrar información detallada
        
    Returns:
        tuple: (refined_mask, score) o (None, 0.0) si falla
    """
    # Enhance contrast
    img_enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    
    # Set image for SAM
    predictor.set_image(img_enhanced)
    
    # Preparar punto
    input_points = np.array([point])
    input_labels = np.array([label])
    
    if verbose:
        print(f"    📍 Usando punto: ({point[0]:.0f}, {point[1]:.0f})")
    
    try:
        refined_mask, best_mask, best_score, _ = segment_image(
            predictor, input_points, input_labels, refine=True
        )
        
        raw_area = np.sum(best_mask)
        refined_area = np.sum(refined_mask)
        
        if verbose:
            print(f"    🔍 SAM raw mask area: {raw_area} px, score: {best_score:.3f}")
        
        # Si la máscara raw está vacía, SAM no encontró nada
        if raw_area == 0:
            if verbose:
                print(f"    ⚠️ SAM no generó máscara (punto fuera del objeto?)")
            return None, 0.0
        
        if verbose and refined_area < raw_area * 0.1:
            print(f"    ⚠️ Refinamiento redujo mucho la máscara ({raw_area}px -> {refined_area}px)")
        
        return refined_mask, best_score
        
    except Exception as e:
        print(f"    ⚠️ Error in segmentation: {e}")
        return None, 0.0


def segment_first_image(predictor, img, filename):
    """Main function to segment the first image using SAM with interactive point selection"""

    # Enhance contrast for medical images
    img_enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    predictor.set_image(img_enhanced)

    # Use the interactive point selector (if GUI available)
    print("🎯 Selección de puntos iniciando...")
    print("   - Click DERECHO: Marca puntos POSITIVOS (objeto de interés)")
    print("   - Click IZQUIERDO: Marca puntos NEGATIVOS (para omitir contornos)")
    print("   - Tecla 'z': Deshacer último punto")
    print("   - Tecla 'c': Limpiar todos los puntos")
    positive_points, negative_points = interactive_sam_point_selector(img, predictor, filename)

    # Si no hay puntos (por ejemplo, en entorno sin GUI), usar punto central automático
    if len(positive_points) == 0 and len(negative_points) == 0:
        h, w = img.shape[:2]
        auto_point = [w // 2, h // 2]
        positive_points = [auto_point]
        print(f"⚠️ No se seleccionaron puntos en la interfaz. Usando punto central automático: ({auto_point[0]}, {auto_point[1]})")

    # Prepare points and labels for SAM
    input_points = []
    input_labels = []

    # Add positive points (label = 1)
    for point in positive_points:
        input_points.append(point)
        input_labels.append(1)

    # Add negative points (label = 0)
    for point in negative_points:
        input_points.append(point)
        input_labels.append(0)

    input_points = np.array(input_points)
    input_labels = np.array(input_labels)

    print(f"✅ Total de puntos: {len(input_points)}")
    print(f"   - Positivos: {len(positive_points)}")
    print(f"   - Negativos: {len(negative_points)}")

    # Usar segment_image centralizado
    refined_mask, best_mask, best_score, masks = segment_image(
        predictor, input_points, input_labels, refine=True
    )

    # Enhanced visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1: Original results
    axes[0,0].imshow(img)
    axes[0,0].set_title("Original Image")
    axes[0,0].axis('off')

    axes[0,1].imshow(img)
    axes[0,1].imshow(best_mask, alpha=0.5, cmap='Reds')
    for point in positive_points:
        axes[0,1].plot(point[0], point[1], 'g*', markersize=15, markeredgewidth=2)
    for point in negative_points:
        axes[0,1].plot(point[0], point[1], 'rx', markersize=12, markeredgewidth=3)
    axes[0,1].set_title("Raw SAM Output with Points")
    axes[0,1].axis('off')

    axes[0,2].imshow(best_mask, cmap='gray')
    axes[0,2].set_title("Raw Mask")
    axes[0,2].axis('off')

    # Row 2: Enhanced results
    axes[1,0].imshow(img_enhanced)
    axes[1,0].set_title("Enhanced Image")
    axes[1,0].axis('off')

    axes[1,1].imshow(img)
    axes[1,1].imshow(refined_mask, alpha=0.5, cmap='Blues')
    for point in positive_points:
        axes[1,1].plot(point[0], point[1], 'g*', markersize=15, markeredgewidth=2)
    for point in negative_points:
        axes[1,1].plot(point[0], point[1], 'rx', markersize=12, markeredgewidth=3)
    axes[1,1].set_title("Refined Segmentation")
    axes[1,1].axis('off')

    axes[1,2].imshow(refined_mask, cmap='gray')
    axes[1,2].set_title("Refined Mask")
    axes[1,2].axis('off')

    plt.tight_layout()
    plt.show()

    # Results summary
    print(f"\n{'='*50}")
    print(f"🟢 Puntos positivos: {len(positive_points)}")
    print(f"🔴 Puntos negativos: {len(negative_points)}")
    print(f"📏 Mask area: {np.sum(refined_mask)} pixels")
    print(f"⭐ Best mask score: {best_score:.4f}")
    print(f"🎭 Total masks generated: {len(masks)}")
    print(f"{'='*50}")

    return refined_mask, best_score