import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from tqdm import tqdm
import time

from .preprocessor import ImagePreprocessor
from .stitcher import PanoramaStitcher
from .projector import EquirectangularProjector
from ..config.settings import Settings


class ProcessingPipeline:

    def __init__(self, settings: Settings = None):
        self.settings = settings or Settings()
        self.logger = logging.getLogger(__name__)

        self.preprocessor = ImagePreprocessor(self.settings)
        self.stitcher = PanoramaStitcher(self.settings)
        self.projector = EquirectangularProjector(self.settings)

        logging.basicConfig(level=logging.INFO)

    def process_images(self, input_dir: str, output_path: str) -> bool:
        start_time = time.time()

        try:
            self.logger.info(f"Starting processing pipeline for directory: {input_dir}")

            self.logger.info("Step 1: Preprocessing images...")
            valid_images = self.preprocessor.process_images(input_dir)

            if len(valid_images) < 2:
                self.logger.error(f"Not enough valid images for stitching: {len(valid_images)}")
                return False

            self.logger.info(f"Found {len(valid_images)} valid images after preprocessing")

            self.logger.info("Step 2: Stitching images...")
            stitched_panorama, stitch_success = self.stitcher.stitch_images(valid_images)

            if not stitch_success or stitched_panorama is None:
                self.logger.error("Stitching failed")
                return False

            self.logger.info("Stitching completed successfully")

            self.logger.info("Step 3: Projecting to equirectangular format...")
            equirectangular = self.projector.to_equirectangular(
                stitched_panorama,
                width=16384,
                height=8192
            )

            if not self.projector.validate_equirectangular(equirectangular):
                self.logger.warning("Output image may not be in proper equirectangular format")

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if str(output_path).lower().endswith(('.jpg', '.jpeg')):
                compression_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
                success = cv2.imwrite(str(output_path), equirectangular, compression_params)
            elif str(output_path).lower().endswith('.png'):
                compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
                success = cv2.imwrite(str(output_path), equirectangular, compression_params)
            else:
                success = cv2.imwrite(str(output_path), equirectangular)

            if success:
                self.logger.info(f"Final panorama saved to: {output_path}")
            else:
                self.logger.error(f"Failed to save panorama to: {output_path}")
                return False

            total_time = time.time() - start_time
            self.logger.info(f"Processing completed successfully in {total_time:.2f} seconds")

            return True

        except Exception as e:
            self.logger.error(f"Error in processing pipeline: {str(e)}")
            return False

    def process_single_image(self, image_path: str, output_path: str) -> bool:
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            if self.settings.enable_resize:
                h, w = image.shape[:2]

                if w > h:
                    new_w = min(self.settings.resize_width, w)
                    new_h = int((new_w / w) * h)
                else:
                    new_h = min(self.settings.resize_height, h)
                    new_w = int((new_h / h) * w)

                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            equirectangular = self.projector.to_equirectangular(
                image,
                width=2048,
                height=1024
            )

            if not self.projector.validate_equirectangular(equirectangular):
                self.logger.warning("Output image may not be in proper equirectangular format")

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if str(output_path).lower().endswith(('.jpg', '.jpeg')):
                compression_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
                success = cv2.imwrite(str(output_path), equirectangular, compression_params)
            elif str(output_path).lower().endswith('.png'):
                compression_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
                success = cv2.imwrite(str(output_path), equirectangular, compression_params)
            else:
                success = cv2.imwrite(str(output_path), equirectangular)

            if success:
                self.logger.info(f"Processed image saved to: {output_path}")
                return True
            else:
                self.logger.error(f"Failed to save image to: {output_path}")
                return False

        except Exception as e:
            self.logger.error(f"Error processing single image: {str(e)}")
            return False