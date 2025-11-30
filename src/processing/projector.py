"""Module for projecting panoramic images to equirectangular format."""
import cv2
import numpy as np
from typing import Tuple
import logging


class EquirectangularProjector:
    """Handles projection of panoramic images to equirectangular format."""
    
    def __init__(self, settings):
        """Initialize the projector with settings."""
        self.settings = settings
        self.logger = logging.getLogger(__name__)
    
    def to_equirectangular(self, image: np.ndarray, width: int = 2048, height: int = 1024) -> np.ndarray:
        """
        Convert a panoramic image to equirectangular format (2:1 aspect ratio).
        
        Args:
            image: Input panoramic image (assumed to be already stitched)
            width: Output width for equirectangular image (default 2048)
            height: Output height for equirectangular image (default 1024, 2:1 ratio)
            
        Returns:
            Equirectangular projection of the input image
        """
        # Ensure output dimensions maintain 2:1 aspect ratio
        if width / height != 2:
            height = width // 2
            self.logger.warning(f"Adjusted height to {height} to maintain 2:1 aspect ratio")
        
        # Create coordinate maps for the transformation
        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)
        
        # Calculate the transformation
        for i in range(height):
            for j in range(width):
                # Calculate theta and phi for equirectangular projection
                theta = (j / width) * 2 * np.pi - np.pi  # -π to π
                phi = (i / height) * np.pi - np.pi/2    # -π/2 to π/2
                
                # Map to input image coordinates
                # Assuming the input image is already a panoramic image
                # where width corresponds to the full 360° view
                src_x = int((theta + np.pi) * image.shape[1] / (2 * np.pi))
                src_y = int((phi + np.pi/2) * image.shape[0] / np.pi)
                
                # Handle wrap-around for theta
                src_x = src_x % image.shape[1]
                
                # Clamp to valid range
                src_x = max(0, min(src_x, image.shape[1] - 1))
                src_y = max(0, min(src_y, image.shape[0] - 1))
                
                map_x[i, j] = src_x
                map_y[i, j] = src_y
        
        # Apply the remapping to transform the image
        equirectangular = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_WRAP)
        
        return equirectangular
    
    def project_spherical(self, image: np.ndarray, width: int = 2048, height: int = 1024) -> np.ndarray:
        """
        Project a panoramic image onto a spherical surface and return as equirectangular.
        
        Args:
            image: Input panoramic image
            width: Output width for equirectangular image
            height: Output height for equirectangular image
            
        Returns:
            Spherically projected equirectangular image
        """
        # Check if input image is already a panorama or if we need to handle it differently
        # For now, implementing a basic projection assuming the input is a linear panoramic image
        
        # Create spherical coordinate mappings
        theta_range = np.linspace(-np.pi, np.pi, width)  # Azimuthal angle (longitude)
        phi_range = np.linspace(-np.pi/2, np.pi/2, height)  # Polar angle (latitude)
        
        # Create meshgrid of spherical coordinates
        theta, phi = np.meshgrid(theta_range, phi_range)
        
        # Convert spherical coordinates to texture coordinates on the input image
        # Assuming input image spans 360 degrees horizontally
        x = (theta + np.pi) * image.shape[1] / (2 * np.pi)
        y = (phi + np.pi/2) * image.shape[0] / np.pi
        
        # Use OpenCV's remap function to apply the transformation
        map_x = x.astype(np.float32)
        map_y = y.astype(np.float32)
        
        spherical_projection = cv2.remap(
            image, 
            map_x, 
            map_y, 
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP
        )
        
        return spherical_projection
    
    def validate_equirectangular(self, image: np.ndarray) -> bool:
        """
        Validate if an image is in proper equirectangular format.
        
        Args:
            image: Image to validate
            
        Returns:
            True if image is in proper equirectangular format, False otherwise
        """
        height, width = image.shape[:2]
        
        # Check if aspect ratio is close to 2:1 (allowing for some tolerance)
        expected_ratio = 2.0
        actual_ratio = width / height
        
        if abs(actual_ratio - expected_ratio) > 0.1:  # 10% tolerance
            self.logger.warning(f"Image aspect ratio is {actual_ratio:.2f}, expected ~2.0")
            return False
        
        return True