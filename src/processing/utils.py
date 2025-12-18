import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image


def validate_image_path(image_path: str) -> bool:
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except:
        return False