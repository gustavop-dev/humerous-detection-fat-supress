import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamPredictor
from scipy import ndimage
from skimage import morphology
import cv2
from DCM.load_dicom_as_image import load_dicom_as_image


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")


# Path to SAM model checkpoint
ckpt = "Checkpoints/sam_vit_b_01ec64.pth"

# Load model
sam = sam_model_registry["vit_b"]()
checkpoint_data = torch.load(ckpt, map_location=device)
sam.load_state_dict(checkpoint_data)
sam = sam.to(device)
predictor = SamPredictor(sam)

# Path to the image to segment (DICOM or PNG/JPG)
# Change this to your image path
image_path = "Datasets/In/pd_tse_fs_tra_320_fov150_4/IM-0001-0016.dcm"

ext = os.path.splitext(image_path)[1].lower()
if ext == ".dcm":
    img, _ = load_dicom_as_image(image_path)
else:
    img = np.array(Image.open(image_path).convert("RGB"))

# Enhance contrast for medical images
img_enhanced = cv2.convertScaleAbs(img, alpha=1.2, beta=10)

predictor.set_image(img_enhanced)

H, W = img.shape[:2]

def interactive_point_selector(img, predictor):
    """
    Interactive point selection for SAM segmentation with REAL-TIME preview.
    - Click derecho (botón derecho): Punto POSITIVO (sí es el objeto)
    - Click izquierdo (botón izquierdo): Punto NEGATIVO (omitir este contorno)
    - Tecla 'z': DESHACER último punto
    - Tecla 'c': LIMPIAR todos los puntos
    - Presiona ENTER o cierra la ventana cuando termines
    """
    
    class PointSelector:
        def __init__(self, ax_img, ax_mask):
            self.positive_points = []
            self.negative_points = []
            self.ax_img = ax_img
            self.ax_mask = ax_mask
            self.point_markers = []
            self.mask_display = None
            
        def update_segmentation(self):
            """Update segmentation in real-time"""
            # Clear previous mask
            if self.mask_display is not None:
                self.mask_display.remove()
                self.mask_display = None
            
            # If no points, return
            if len(self.positive_points) == 0 and len(self.negative_points) == 0:
                self.ax_mask.clear()
                self.ax_mask.imshow(img)
                self.ax_mask.set_title("Máscara (agrega puntos para ver)")
                self.ax_mask.axis('off')
                fig.canvas.draw()
                return
            
            # Prepare points and labels
            input_points = []
            input_labels = []
            
            for point in self.positive_points:
                input_points.append(point)
                input_labels.append(1)
            
            for point in self.negative_points:
                input_points.append(point)
                input_labels.append(0)
            
            input_points = np.array(input_points)
            input_labels = np.array(input_labels)
            
            # Generate mask
            try:
                masks, scores, _ = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True
                )
                
                best_mask = masks[np.argmax(scores)]
                
                # Display mask on right subplot
                self.ax_mask.clear()
                self.ax_mask.imshow(img)
                self.mask_display = self.ax_mask.imshow(best_mask, alpha=0.6, cmap='Blues')
                
                # Show points on mask view too
                for point in self.positive_points:
                    self.ax_mask.plot(point[0], point[1], 'g*', markersize=15, markeredgewidth=2)
                for point in self.negative_points:
                    self.ax_mask.plot(point[0], point[1], 'rx', markersize=12, markeredgewidth=3)
                
                score = scores[np.argmax(scores)]
                area = np.sum(best_mask)
                self.ax_mask.set_title(f"Segmentación | Score: {score:.3f} | Área: {area} px")
                self.ax_mask.axis('off')
                
            except Exception as e:
                print(f"⚠️ Error en segmentación: {e}")
            
            fig.canvas.draw()
            
        def onclick(self, event):
            if event.inaxes != self.ax_img:
                return
            if event.xdata is None or event.ydata is None:
                return
                
            x, y = event.xdata, event.ydata
            
            # Botón izquierdo (1) = Punto NEGATIVO (rojo)
            if event.button == 1:
                self.negative_points.append([x, y])
                marker = self.ax_img.plot(x, y, 'rx', markersize=15, markeredgewidth=3)[0]
                self.point_markers.append(('neg', marker))
                print(f"❌ Punto NEGATIVO agregado: ({x:.0f}, {y:.0f})")
                
            # Botón derecho (3) = Punto POSITIVO (verde)
            elif event.button == 3:
                self.positive_points.append([x, y])
                marker = self.ax_img.plot(x, y, 'g*', markersize=20, markeredgewidth=2)[0]
                self.point_markers.append(('pos', marker))
                print(f"✅ Punto POSITIVO agregado: ({x:.0f}, {y:.0f})")
            
            # Update title with counts
            self.ax_img.set_title(f"✅ Positivos: {len(self.positive_points)} | ❌ Negativos: {len(self.negative_points)} | 'z': deshacer | 'c': limpiar")
            
            # Update segmentation in real-time
            self.update_segmentation()
            
        def onkey(self, event):
            """Handle keyboard events"""
            # Z = Undo last point
            if event.key == 'z':
                if len(self.point_markers) > 0:
                    point_type, marker = self.point_markers.pop()
                    marker.remove()
                    
                    if point_type == 'pos' and len(self.positive_points) > 0:
                        removed = self.positive_points.pop()
                        print(f"↩️  Deshecho punto POSITIVO: ({removed[0]:.0f}, {removed[1]:.0f})")
                    elif point_type == 'neg' and len(self.negative_points) > 0:
                        removed = self.negative_points.pop()
                        print(f"↩️  Deshecho punto NEGATIVO: ({removed[0]:.0f}, {removed[1]:.0f})")
                    
                    self.ax_img.set_title(f"✅ Positivos: {len(self.positive_points)} | ❌ Negativos: {len(self.negative_points)} | 'z': deshacer | 'c': limpiar")
                    self.update_segmentation()
            
            # C = Clear all points
            elif event.key == 'c':
                for _, marker in self.point_markers:
                    marker.remove()
                self.point_markers.clear()
                self.positive_points.clear()
                self.negative_points.clear()
                print("🧹 Todos los puntos limpiados")
                self.ax_img.set_title(f"✅ Positivos: 0 | ❌ Negativos: 0 | 'z': deshacer | 'c': limpiar")
                self.update_segmentation()
    
    # Create the selector object with 2 subplots
    fig, (ax_img, ax_mask) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left: Image with points
    ax_img.imshow(img)
    ax_img.set_title("🎯 Imagen Original | Click derecho = POSITIVO | Click izquierdo = NEGATIVO")
    ax_img.axis('off')
    
    # Right: Real-time mask
    ax_mask.imshow(img)
    ax_mask.set_title("Segmentación (agrega puntos para ver)")
    ax_mask.axis('off')
    
    selector_obj = PointSelector(ax_img, ax_mask)
    
    # Connect events
    fig.canvas.mpl_connect('button_press_event', selector_obj.onclick)
    fig.canvas.mpl_connect('key_press_event', selector_obj.onkey)
    
    # Instructions
    plt.figtext(0.5, 0.02, 
                "🟢 Click DERECHO: Punto positivo | 🔴 Click IZQUIERDO: Punto negativo | ⌨️ 'z': Deshacer | 'c': Limpiar | ENTER/ESC: Terminar", 
                ha='center', fontsize=11, weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", edgecolor="blue", linewidth=2))
    
    plt.tight_layout()
    plt.show()
    
    return selector_obj.positive_points, selector_obj.negative_points

# Use the interactive point selector
print("🎯 Selección de puntos iniciando...")
print("   - Click DERECHO: Marca puntos POSITIVOS (objeto de interés)")
print("   - Click IZQUIERDO: Marca puntos NEGATIVOS (para omitir contornos)")
print("   - Tecla 'z': Deshacer último punto")
print("   - Tecla 'c': Limpiar todos los puntos")
positive_points, negative_points = interactive_point_selector(img, predictor)

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

if len(input_points) == 0:
    print("⚠️ No se seleccionaron puntos. Saliendo...")
    sys.exit(0)

input_points = np.array(input_points)
input_labels = np.array(input_labels)

print(f"✅ Total de puntos: {len(input_points)}")
print(f"   - Positivos: {len(positive_points)}")
print(f"   - Negativos: {len(negative_points)}")

# Generate masks using the selected points
masks, scores, _ = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True
)

# Select best mask
best_mask = masks[np.argmax(scores)]

# Post-process mask
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

refined_mask = refine_medical_mask(best_mask)

# Enhanced visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Original results
axes[0,0].imshow(img)
axes[0,0].set_title("Original Image")
axes[0,0].axis('off')

axes[0,1].imshow(img)
axes[0,1].imshow(best_mask, alpha=0.5, cmap='Reds')
# Show selected points
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
# Show selected points on refined view too
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
print(f"🎯 Segmentation completed on {device}")
print(f"🟢 Puntos positivos: {len(positive_points)}")
print(f"🔴 Puntos negativos: {len(negative_points)}")
print(f"📏 Mask area: {np.sum(refined_mask)} pixels")
print(f"⭐ Best mask score: {scores[np.argmax(scores)]:.4f}")
print(f"🎭 Total masks generated: {len(masks)}")
print(f"{'='*50}")

# Save results if needed
# refined_mask_pil = Image.fromarray((refined_mask * 255).astype(np.uint8))
# refined_mask_pil.save("segmentation_result_points.png")
# print("💾 Mask saved as 'segmentation_result_points.png'")