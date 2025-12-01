"""Data models for the balloon panorama processor."""
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
import json


@dataclass
class Panorama:
    """Represents a processed panoramic image with metadata."""
    image: Optional[np.ndarray] = None
    width: int = 0
    height: int = 0
    focal_length: float = 0.0
    camera_matrix: Optional[np.ndarray] = None
    distortion_coeffs: Optional[np.ndarray] = None
    features_count: int = 0
    matches_count: int = 0
    processing_time: float = 0.0
    source_images: List[str] = None
    settings_used: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.source_images is None:
            self.source_images = []
        if self.settings_used is None:
            self.settings_used = {}
    
    def load_image(self, image_path: str):
        """Load an image from a file path."""
        import cv2
        self.image = cv2.imread(image_path)
        if self.image is not None:
            self.height, self.width = self.image.shape[:2]
    
    def save_image(self, output_path: str):
        """Save the panorama image to a file."""
        import cv2
        if self.image is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Set compression parameters for higher quality output
            if str(output_path).lower().endswith(('.jpg', '.jpeg')):
                # For JPEG files, use high quality (95%)
                compression_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
                cv2.imwrite(str(output_path), self.image, compression_params)
            elif str(output_path).lower().endswith('.png'):
                # For PNG files, use high compression level (1-9, where 9 is highest compression)
                compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # Less compression = higher quality
                cv2.imwrite(str(output_path), self.image, compression_params)
            else:
                # For other formats or default
                cv2.imwrite(str(output_path), self.image)
    
    def save_metadata(self, metadata_path: str):
        """Save panorama metadata to a JSON file."""
        metadata = {
            'width': self.width,
            'height': self.height,
            'focal_length': self.focal_length,
            'features_count': self.features_count,
            'matches_count': self.matches_count,
            'processing_time': self.processing_time,
            'source_images': self.source_images,
            'settings_used': self.settings_used
        }
        
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


@dataclass
class CameraParams:
    """Camera parameters for a single image."""
    intrinsic_matrix: Optional[np.ndarray] = None
    distortion_coeffs: Optional[np.ndarray] = None
    rotation_matrix: Optional[np.ndarray] = None
    translation_vector: Optional[np.ndarray] = None
    image_size: tuple = (0, 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for storage."""
        result = {
            'image_size': self.image_size
        }
        
        if self.intrinsic_matrix is not None:
            result['intrinsic_matrix'] = self.intrinsic_matrix.tolist()
        
        if self.distortion_coeffs is not None:
            result['distortion_coeffs'] = self.distortion_coeffs.tolist()
            
        if self.rotation_matrix is not None:
            result['rotation_matrix'] = self.rotation_matrix.tolist()
            
        if self.translation_vector is not None:
            result['translation_vector'] = self.translation_vector.tolist()
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraParams':
        """Create CameraParams from dictionary."""
        params = cls()
        params.image_size = tuple(data.get('image_size', (0, 0)))
        
        if 'intrinsic_matrix' in data:
            params.intrinsic_matrix = np.array(data['intrinsic_matrix'])
        
        if 'distortion_coeffs' in data:
            params.distortion_coeffs = np.array(data['distortion_coeffs'])
            
        if 'rotation_matrix' in data:
            params.rotation_matrix = np.array(data['rotation_matrix'])
            
        if 'translation_vector' in data:
            params.translation_vector = np.array(data['translation_vector'])
            
        return params