import cv2
import numpy as np
from typing import Tuple
import logging


class EquirectangularProjector:

    def __init__(self, settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)

    def to_equirectangular(self, image: np.ndarray, width: int = 2048, height: int = 1024) -> np.ndarray:
        if width / height != 2:
            height = width // 2
            self.logger.warning(f"Adjusted height to {height} to maintain 2:1 aspect ratio")

        map_x = np.zeros((height, width), dtype=np.float32)
        map_y = np.zeros((height, width), dtype=np.float32)

        for i in range(height):
            for j in range(width):
                theta = (j / width) * 2 * np.pi - np.pi
                phi = (i / height) * np.pi - np.pi/2

                src_x = int((theta + np.pi) * image.shape[1] / (2 * np.pi))
                src_y = int((phi + np.pi/2) * image.shape[0] / np.pi)

                src_x = src_x % image.shape[1]

                src_x = max(0, min(src_x, image.shape[1] - 1))
                src_y = max(0, min(src_y, image.shape[0] - 1))

                map_x[i, j] = src_x
                map_y[i, j] = src_y

        equirectangular = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_WRAP)

        return equirectangular

    def project_spherical(self, image: np.ndarray, width: int = 2048, height: int = 1024) -> np.ndarray:
        theta_range = np.linspace(-np.pi, np.pi, width)
        phi_range = np.linspace(-np.pi/2, np.pi/2, height)

        theta, phi = np.meshgrid(theta_range, phi_range)

        x = (theta + np.pi) * image.shape[1] / (2 * np.pi)
        y = (phi + np.pi/2) * image.shape[0] / np.pi

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
        height, width = image.shape[:2]

        expected_ratio = 2.0
        actual_ratio = width / height

        if abs(actual_ratio - expected_ratio) > 0.1:
            self.logger.warning(f"Image aspect ratio is {actual_ratio:.2f}, expected ~2.0")
            return False

        return True