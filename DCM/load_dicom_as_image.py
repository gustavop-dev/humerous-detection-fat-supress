import pydicom
import numpy as np
import os
from PIL import Image
import glob
from dataclasses import dataclass
from typing import List, Optional

def load_dicom_as_image(dicom_path):
    """
    Load DICOM file and convert to RGB image array
    """
    try:
        # Read DICOM file
        dicom_data = pydicom.dcmread(dicom_path)
        
        # Extract pixel array
        pixel_array = dicom_data.pixel_array
        
        # Normalize to 0-255 range
        if pixel_array.dtype != np.uint8:
            # Handle different bit depths
            if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                # Use DICOM windowing if available
                center = dicom_data.WindowCenter
                width = dicom_data.WindowWidth
                if isinstance(center, pydicom.multival.MultiValue):
                    center = center[0]
                if isinstance(width, pydicom.multival.MultiValue):
                    width = width[0]
                
                min_val = center - width // 2
                max_val = center + width // 2
                pixel_array = np.clip(pixel_array, min_val, max_val)
                pixel_array = ((pixel_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                # Simple normalization
                pixel_array = ((pixel_array - pixel_array.min()) / 
                              (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        
        # Convert to RGB (duplicate grayscale to 3 channels)
        if len(pixel_array.shape) == 2:
            rgb_image = np.stack([pixel_array] * 3, axis=-1)
        else:
            rgb_image = pixel_array
            
        return rgb_image, dicom_data
        
    except Exception as e:
        print(f"⚠️ Error loading DICOM {dicom_path}: {e}")
        return None, None
    
def read_image_file(filepath):
    """Read JPG or PNG file and return RGB array"""
    try:
        ext = os.path.splitext(filepath)[1].lower()

        # Handle DICOM files
        if ext == ".dcm":
            img, _ = load_dicom_as_image(filepath)
            return img

        # Fallback to standard image loading (JPG/PNG, etc.)
        img = np.array(Image.open(filepath).convert("RGB"))
        return img
    except Exception as e:
        print(f"⚠️  Error reading {filepath}: {e}")
        return None
    
@dataclass
class DatasetInfo:
    """Información del dataset de imágenes"""
    files: List[str]
    middle_idx: int
    middle_file: str
    middle_name: str
    data_dir: str


def get_dataset_files(data_dir: str) -> Optional[DatasetInfo]:
    """
    Obtiene lista de archivos de imagen y encuentra el del medio.
    NO carga la imagen, solo retorna la información del dataset.
    
    Args:
        data_dir: Directorio con las imágenes
        
    Returns:
        DatasetInfo con la información del dataset, o None si hay error
    """
    # Get all image files (JPG, PNG or DICOM)
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG', '*.dcm', '*.DCM']
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(data_dir, ext)))
    
    # Sort files by name
    files = sorted(files, key=lambda x: os.path.basename(x))
    
    if len(files) == 0:
        print(f"❌ No image files found in {data_dir}")
        return None
    
    print(f"📁 Found {len(files)} image files in {data_dir}")
    print(f"   Files: {os.path.basename(files[0])} ... {os.path.basename(files[-1])}")
    
    # Find middle image
    middle_idx = len(files) // 2
    middle_file = files[middle_idx]
    middle_name = os.path.basename(middle_file).split('.')[0]
    
    print(f"\n🎯 Middle image: {os.path.basename(middle_file)} (index {middle_idx+1}/{len(files)})")
    
    return DatasetInfo(
        files=files,
        middle_idx=middle_idx,
        middle_file=middle_file,
        middle_name=middle_name,
        data_dir=data_dir
    )


# Mantener compatibilidad con código existente
def middle_image(data_dir):
    """DEPRECATED: Usa get_dataset_files() en su lugar"""
    dataset = get_dataset_files(data_dir)
    if dataset is None:
        return None
    
    middle_img = read_image_file(dataset.middle_file)
    if middle_img is None:
        print("❌ Error reading middle image!")
        return None
    
    print(f"✅ Loaded middle image: {middle_img.shape}")
    return middle_img, dataset.middle_name, dataset.middle_idx, dataset.files, dataset.middle_file