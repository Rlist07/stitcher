"""Utility functions for the balloon panorama processor."""
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image


def validate_image_path(image_path: str) -> bool:
    """Validate that an image path is accessible and is a valid image file."""
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except:
        return False