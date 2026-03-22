import numpy as np
from scipy import ndimage
from skimage import morphology
import cv2
import os
import matplotlib.pyplot as plt

def refine_medical_mask(mask):
    # 1. Si la máscara está vacía, retornarla sin cambios
    if np.sum(mask) == 0:
        return mask
    
    # 2. Eliminar objetos pequeños (< 500 píxeles)
    #    Quita "ruido" o fragmentos pequeños que no son el objeto principal
    mask_clean = morphology.remove_small_objects(mask, min_size=500)
    
    # 3. Rellenar huecos internos
    #    Si hay "agujeros" dentro de la máscara, los rellena
    mask_filled = ndimage.binary_fill_holes(mask_clean)
    
    # 4. Suavizar bordes con operaciones morfológicas
    kernel = morphology.disk(2)  # Kernel circular de radio 2
    
    # Opening: Erosión + Dilatación → elimina protuberancias pequeñas
    mask_smooth = morphology.binary_opening(mask_filled, kernel)
    
    # Closing: Dilatación + Erosión → cierra pequeños gaps en el borde
    mask_smooth = morphology.binary_closing(mask_smooth, kernel)
    
    return mask_smooth

def calculate_mask_center(mask):
    # 1. Si la máscara está vacía (sin píxeles blancos), retorna None
    if np.sum(mask) == 0:
        return None
    
    # 2. Encontrar las coordenadas de todos los píxeles "activos" (valor > 0)
    #    np.where retorna (filas, columnas) = (y, x)
    y_coords, x_coords = np.where(mask > 0)
    
    # 3. Verificar que haya coordenadas válidas
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    
    # 4. Calcular el promedio de las coordenadas X e Y
    #    Esto da el "centro de masa" de la región
    center_x = np.mean(x_coords)  # Promedio de todas las X
    center_y = np.mean(y_coords)  # Promedio de todas las Y
    
    # 5. Retornar como [x, y]
    return [center_x, center_y]

def find_mask_contours(mask):
    """
    Encuentra los contornos/bordes de una máscara binaria.
    
    Args:
        mask: Máscara binaria (numpy array)
    
    Returns:
        Lista de contornos encontrados por OpenCV
    """
    # Asegurar que la máscara sea uint8
    mask_uint8 = (mask.astype(np.uint8) * 255)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def draw_contours_on_image(img, mask, color=(0, 255, 0), thickness=2):
    """
    Dibuja los contornos de la máscara sobre una imagen.
    
    Args:
        img: Imagen RGB (numpy array)
        mask: Máscara binaria
        color: Color del contorno en BGR (default: verde)
        thickness: Grosor de la línea
    
    Returns:
        Imagen con contornos dibujados
    """
    # Copiar imagen para no modificar la original
    img_with_contours = img.copy()
    
    # Encontrar contornos
    contours = find_mask_contours(mask)
    
    # Dibujar contornos
    cv2.drawContours(img_with_contours, contours, -1, color, thickness)
    
    return img_with_contours

def save_segmentation_result(img, mask, filename, output_dir, center=None, seg_point=None, neg_point=None, info=""):
    """Save segmentation visualization
    
    Args:
        img: Imagen original
        mask: Máscara de segmentación
        filename: Nombre del archivo
        output_dir: Directorio de salida
        center: Centro calculado de la máscara resultante (verde)
        seg_point: Punto usado para segmentar (rojo)
        neg_point: Punto negativo usado (azul con X)
        info: Información adicional para el título
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original
    axes[0].imshow(img)
    axes[0].set_title(f"{filename}\nOriginal")
    axes[0].axis('off')
    
    # Overlay
    axes[1].imshow(img)
    axes[1].imshow(mask, alpha=0.5, cmap='Blues')
    
    # Punto usado para segmentar (ROJO)
    if seg_point is not None:
        axes[1].plot(seg_point[0], seg_point[1], 'r*', markersize=18, markeredgewidth=2, label='Pto positivo')
    
    # Punto negativo (AZUL con X)
    if neg_point is not None:
        axes[1].plot(neg_point[0], neg_point[1], 'bX', markersize=16, markeredgewidth=3, label='Pto negativo')
    
    # Centro calculado de la máscara (VERDE)
    if center is not None:
        axes[1].plot(center[0], center[1], 'g*', markersize=14, markeredgewidth=2, label='Centro máscara')
    
    # Leyenda
    if seg_point is not None or center is not None or neg_point is not None:
        axes[1].legend(loc='upper right', fontsize=8)
    
    axes[1].set_title(f"Overlay\n{info}")
    axes[1].axis('off')
    
    # Mask
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title(f"Mask\nArea: {np.sum(mask)} px")
    axes[2].axis('off')
    
    # Contornos/Bordes sobre la imagen original
    img_with_contours = draw_contours_on_image(img, mask, color=(0, 255, 0), thickness=2)
    axes[3].imshow(img_with_contours)
    
    # También mostrar puntos en el panel de contornos
    if seg_point is not None:
        axes[3].plot(seg_point[0], seg_point[1], 'r*', markersize=14, markeredgewidth=2)
    if center is not None:
        axes[3].plot(center[0], center[1], 'g*', markersize=10, markeredgewidth=2)
    
    # Contar contornos encontrados
    contours = find_mask_contours(mask)
    axes[3].set_title(f"Contornos\n{len(contours)} contorno(s)")
    axes[3].axis('off')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{filename}_seg.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Also save binary mask as .npy for downstream analysis
    mask_npy_path = os.path.join(output_dir, f"{filename}_mask.npy")
    np.save(mask_npy_path, mask.astype(np.uint8))

