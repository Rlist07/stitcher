import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import logging
from .utils import validate_image_path


class ImagePreprocessor:

    def __init__(self, settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)

    def detect_blur(self, image_path: str) -> float:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        except cv2.error as e:
            self.logger.error(f"Error in blur detection for {image_path}: {str(e)}")
            return 0.0

        return laplacian_var

    def is_blurry(self, image_path: str) -> bool:
        blur_score = self.detect_blur(image_path)
        return blur_score < self.settings.blur_threshold

    def resize_image(self, image_path: str, output_path: str = None) -> str:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        h, w = image.shape[:2]

        if w > h:
            new_w = min(self.settings.resize_width, w)
            new_h = int((new_w / w) * h)
        else:
            new_h = min(self.settings.resize_height, h)
            new_w = int((new_h / h) * w)

        try:
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except cv2.error as e:
            self.logger.error(f"Error resizing image {image_path}: {str(e)}")
            raise

        if output_path is None:
            path = Path(image_path)
            output_path = f"temp/{path.stem}_resized{path.suffix}"

        try:
            if str(output_path).lower().endswith(('.jpg', '.jpeg')):
                compression_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
                success = cv2.imwrite(output_path, resized, compression_params)
            elif str(output_path).lower().endswith('.png'):
                compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
                success = cv2.imwrite(output_path, resized, compression_params)
            else:
                success = cv2.imwrite(output_path, resized)

            if not success:
                self.logger.error(f"Failed to write resized image to {output_path}")
                raise IOError(f"Could not write image to {output_path}")
        except cv2.error as e:
            self.logger.error(f"Error writing resized image to {output_path}: {str(e)}")
            raise

        return output_path

    def process_images(self, input_dir: str, output_dir: str = None) -> List[str]:
        input_path = Path(input_dir)
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path("temp")
            output_path.mkdir(parents=True, exist_ok=True)

        valid_images = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp'}

        for img_path in input_path.iterdir():
            if img_path.suffix.lower() not in image_extensions:
                continue

            try:
                if self.settings.enable_blur_filter and self.is_blurry(str(img_path)):
                    self.logger.info(f"Skipping blurry image: {img_path.name}")
                    continue

                if self.settings.enable_resize:
                    output_img_path = output_path / img_path.name
                    resized_path = self.resize_image(str(img_path), str(output_img_path))
                    valid_images.append(resized_path)
                else:
                    valid_images.append(str(img_path))

            except Exception as e:
                self.logger.error(f"Error processing image {img_path.name}: {str(e)}")
                continue

        self.logger.info(f"Processed {len(valid_images)} valid images out of {len(list(input_path.iterdir()))} total images")
        return valid_images