import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from tqdm import tqdm
import sys
import os
from contextlib import redirect_stderr
import io


class PanoramaStitcher:

    def __init__(self, settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)

    def _suppress_stderr(self, func, *args, **kwargs):
        with redirect_stderr(io.StringIO()):
            return func(*args, **kwargs)

    def stitch_images(self, image_paths: List[str]) -> Tuple[Optional[np.ndarray], bool]:
        if len(image_paths) < 2:
            self.logger.error("Need at least 2 images to stitch")
            return None, False

        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                self.logger.error(f"Could not load image: {path}")
                return None, False
            images.append(img)

        if self.settings.panorama_mode == "scans":
            stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        else:
            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

        try:
            status, stitched = self._suppress_stderr(stitcher.stitch, images)
        except Exception as e:
            self.logger.warning(f"High-level stitcher failed with exception: {str(e)}. Trying fallback method.")
            status = -1

        if status == cv2.Stitcher_OK:
            self.logger.info("Successfully stitched images using high-level OpenCV stitcher")
            return stitched, True
        else:
            self.logger.warning(f"High-level stitcher failed with status: {status}. Trying fallback method.")

            if self.settings.enable_fallback:
                return self._fallback_stitching(images)
            else:
                self.logger.error("Stitching failed and fallback is disabled")
                return None, False

    def _fallback_stitching(self, images: List[np.ndarray]) -> Tuple[Optional[np.ndarray], bool]:
        from .feature_matcher import FeatureMatcher

        self.logger.info("Using fallback stitching with SIFT feature matching")
        self.logger.info(f"Attempting to stitch {len(images)} images using SIFT-based approach")
        matcher = FeatureMatcher(self.settings)

        panorama = images[0].copy()

        remaining_images = images[1:]

        pbar = tqdm(total=len(remaining_images), desc="Stitching images", unit="image")
        self.logger.info(f"Starting to stitch {len(remaining_images)} remaining images...")

        images_processed = 0
        total_images = len(images) - 1

        processed_images_indices = set()

        while remaining_images and images_processed < total_images:
            best_match_idx = -1
            best_homography = None
            best_matches = 0

            self.logger.debug(f"Looking for matches for image {images_processed + 1}/{total_images}")

            for i, img in enumerate(remaining_images):
                homography, matches = matcher.match_and_compute_homography(panorama, img)

                if homography is not None and len(matches) > best_matches:
                    best_matches = len(matches)
                    best_homography = homography
                    best_match_idx = i

            if best_match_idx != -1 and best_homography is not None:
                try:
                    panorama = self._warp_and_blend_image(panorama, remaining_images[best_match_idx], best_homography)
                    remaining_images.pop(best_match_idx)
                    images_processed += 1
                    pbar.update(1)
                    self.logger.debug(f"Successfully stitched image {images_processed}, {len(remaining_images)} remaining")
                except Exception as e:
                    self.logger.error(f"Error warping image: {str(e)}")
                    remaining_images.pop(best_match_idx)
                    images_processed += 1
                    pbar.update(1)
            else:
                if remaining_images:
                    self.logger.debug(f"No good matches found for any remaining images, adding first available")
                    added_image = False
                    for i, img in enumerate(remaining_images):
                        if i < len(remaining_images):
                            try:
                                temp_img = remaining_images[i]
                                import numpy as np
                                h, w = temp_img.shape[:2]
                                translation_matrix = np.array([[1, 0, panorama.shape[1]], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
                                new_width = panorama.shape[1] + w
                                new_height = max(panorama.shape[0], h)
                                expanded_panorama = np.zeros((new_height, new_width, 3), dtype=panorama.dtype)
                                expanded_panorama[:panorama.shape[0], :panorama.shape[1]] = panorama
                                expanded_panorama[:h, panorama.shape[1]:new_width] = temp_img
                                panorama = expanded_panorama

                                remaining_images.pop(i)
                                images_processed += 1
                                pbar.update(1)
                                added_image = True
                                self.logger.info(f"Added image with simple translation (gap may exist)")
                                break
                            except Exception as e:
                                self.logger.debug(f"Could not add image with simple translation: {str(e)}")
                                continue

                    if not added_image:
                        if remaining_images:
                            remaining_images.pop(0)
                            images_processed += 1
                            pbar.update(1)
                        break

        pbar.close()
        self.logger.info(f"Fallback stitching completed. Processed {images_processed} images.")

        if panorama is not None:
            self.logger.info("Successfully stitched images using fallback method")
            return panorama, True
        else:
            self.logger.error("Fallback stitching failed")
            return None, False

    def _warp_and_blend_image(self, base_img: np.ndarray, new_img: np.ndarray, homography: np.ndarray) -> np.ndarray:
        homography = homography.astype(np.float32)

        h, w = new_img.shape[:2]

        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

        with redirect_stderr(io.StringIO()):
            transformed_corners = cv2.perspectiveTransform(corners, homography)

        all_corners = np.concatenate([transformed_corners,
                                      np.float32([[0, 0], [base_img.shape[1], 0],
                                                 [base_img.shape[1], base_img.shape[0]],
                                                 [0, base_img.shape[0]]]).reshape(-1, 1, 2)])

        x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

        width = x_max - x_min
        height = y_max - y_min

        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

        translated_h = translation @ homography

        warped_new = self._suppress_stderr(cv2.warpPerspective, new_img, translated_h, (width, height))

        warped_base = np.zeros((height, width, 3), dtype=np.uint8)

        base_translation = translation.copy()
        base_translation[0, 2] = -x_min
        base_translation[1, 2] = -y_min

        if base_img is not None:
            warped_base = self._suppress_stderr(cv2.warpPerspective, base_img, base_translation.astype(np.float32), (width, height))

        result = self._blend_images(warped_base, warped_new)

        return result

    def _blend_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        mask = np.any(img2 != 0, axis=2)

        result = img1.copy()

        overlap = np.logical_and(mask, np.any(img1 != 0, axis=2))
        result[overlap] = (img1[overlap] + img2[overlap]) / 2

        background = np.logical_and(mask, np.logical_not(np.any(img1 != 0, axis=2)))
        result[background] = img2[background]

        return result.astype(np.uint8)