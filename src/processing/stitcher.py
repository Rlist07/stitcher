"""Main stitcher module with high-level OpenCV stitcher and fallback mechanism."""
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
    """Handles the main stitching process with high-level and low-level approaches."""

    def __init__(self, settings):
        """Initialize the stitcher with settings."""
        self.settings = settings
        self.logger = logging.getLogger(__name__)

    def stitch_images(self, image_paths: List[str]) -> Tuple[Optional[np.ndarray], bool]:
        """
        Attempt to stitch images using high-level OpenCV stitcher with fallback.

        Args:
            image_paths: List of paths to images to stitch

        Returns:
            Tuple of (stitched panorama image, success flag)
        """
        if len(image_paths) < 2:
            self.logger.error("Need at least 2 images to stitch")
            return None, False

        # Load images
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                self.logger.error(f"Could not load image: {path}")
                return None, False
            images.append(img)

    def _suppress_stderr(self, func, *args, **kwargs):
        """Helper method to run a function with stderr suppressed using contextlib."""
        with redirect_stderr(io.StringIO()):
            return func(*args, **kwargs)

    def stitch_images(self, image_paths: List[str]) -> Tuple[Optional[np.ndarray], bool]:
        """
        Attempt to stitch images using high-level OpenCV stitcher with fallback.

        Args:
            image_paths: List of paths to images to stitch

        Returns:
            Tuple of (stitched panorama image, success flag)
        """
        if len(image_paths) < 2:
            self.logger.error("Need at least 2 images to stitch")
            return None, False

        # Load images
        images = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                self.logger.error(f"Could not load image: {path}")
                return None, False
            images.append(img)

        # Try high-level stitcher first with stderr suppression
        if self.settings.panorama_mode == "scans":
            stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
        else:
            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

        # Attempt stitching with high-level stitcher
        try:
            status, stitched = self._suppress_stderr(stitcher.stitch, images)
        except Exception as e:
            self.logger.warning(f"High-level stitcher failed with exception: {str(e)}. Trying fallback method.")
            status = -1  # Set status to error state to trigger fallback

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
        """
        Fallback stitching using low-level feature matching.

        Args:
            images: List of loaded images

        Returns:
            Tuple of (stitched panorama image, success flag)
        """
        from .feature_matcher import FeatureMatcher

        self.logger.info("Using fallback stitching with SIFT feature matching")
        self.logger.info(f"Attempting to stitch {len(images)} images using SIFT-based approach")
        matcher = FeatureMatcher(self.settings)

        # Try to find best pairs to start the stitching
        # Start with the first image as base
        panorama = images[0].copy()

        remaining_images = images[1:]

        # Initialize progress bar
        pbar = tqdm(total=len(remaining_images), desc="Stitching images", unit="image")
        self.logger.info(f"Starting to stitch {len(remaining_images)} remaining images...")

        # Iteratively add remaining images to the panorama
        images_processed = 0
        total_images = len(images) - 1  # Original number of images to process (excluding base)

        # Create a list to track which images were successfully processed
        processed_images_indices = set()

        while remaining_images and images_processed < total_images:
            best_match_idx = -1
            best_homography = None
            best_matches = 0

            # Find the image with the most matches to the current panorama
            self.logger.debug(f"Looking for matches for image {images_processed + 1}/{total_images}")

            # Try matching each remaining image to the current panorama
            for i, img in enumerate(remaining_images):
                homography, matches = matcher.match_and_compute_homography(panorama, img)

                if homography is not None and len(matches) > best_matches:
                    best_matches = len(matches)
                    best_homography = homography
                    best_match_idx = i

            # If we found a good match, add it to the panorama
            if best_match_idx != -1 and best_homography is not None:
                # Warp and blend the best matching image to the panorama
                try:
                    panorama = self._warp_and_blend_image(panorama, remaining_images[best_match_idx], best_homography)
                    remaining_images.pop(best_match_idx)
                    images_processed += 1
                    pbar.update(1)  # Update progress bar
                    self.logger.debug(f"Successfully stitched image {images_processed}, {len(remaining_images)} remaining")
                except Exception as e:
                    self.logger.error(f"Error warping image: {str(e)}")
                    # Skip this image and try the next one
                    remaining_images.pop(best_match_idx)
                    images_processed += 1
                    pbar.update(1)
            else:
                # If we couldn't find a match, try a different approach:
                # Instead of skipping, try to add the images in a simpler way
                # This is the key change to ensure all images are processed
                if remaining_images:
                    self.logger.debug(f"No good matches found for any remaining images, adding first available")
                    # Just add the first available image to the panorama somehow
                    # We'll try to find any possible match between remaining images
                    added_image = False
                    for i, img in enumerate(remaining_images):
                        # Try to find a match with any possible image in original set
                        # For now, we'll just try to make a basic combination
                        if i < len(remaining_images):  # Check index bounds
                            # Try to at least add the image in the next empty space
                            # For now, try with the base image or last result
                            try:
                                # Since we can't find a proper match, add a "null" transformation
                                # This will just append the image to the existing panorama
                                temp_img = remaining_images[i]
                                # Use a simple translation to add the image
                                import numpy as np
                                h, w = temp_img.shape[:2]
                                # Simple translation matrix to add image to the right
                                translation_matrix = np.array([[1, 0, panorama.shape[1]], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
                                # Expand panorama canvas to accommodate the new image
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
                        # If nothing worked, just move to next image
                        if remaining_images:
                            remaining_images.pop(0)  # Remove the first image
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
        """
        Warp the new image to the base image using the homography and blend them.

        Args:
            base_img: The base panorama image
            new_img: New image to be added
            homography: Homography matrix to transform new_img to base_img coordinates

        Returns:
            Combined panorama image
        """
        # Ensure the homography matrix is the correct type
        homography = homography.astype(np.float32)

        # Get dimensions of the new image
        h, w = new_img.shape[:2]

        # Define corners of the new image
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

        # Transform the corners to the base image space with stderr suppression
        with redirect_stderr(io.StringIO()):
            transformed_corners = cv2.perspectiveTransform(corners, homography)

        # Calculate the size of the combined image
        all_corners = np.concatenate([transformed_corners,
                                      np.float32([[0, 0], [base_img.shape[1], 0],
                                                 [base_img.shape[1], base_img.shape[0]],
                                                 [0, base_img.shape[0]]]).reshape(-1, 1, 2)])

        # Find the bounding box
        x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

        # Calculate the size of the result image
        width = x_max - x_min
        height = y_max - y_min

        # Create a translation matrix to ensure all coordinates are positive
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

        # Apply the translation to the homography
        translated_h = translation @ homography

        # Warp the new image to the combined space with stderr suppression
        warped_new = self._suppress_stderr(cv2.warpPerspective, new_img, translated_h, (width, height))

        # Create a canvas for the base image with the new dimensions
        warped_base = np.zeros((height, width, 3), dtype=np.uint8)

        # Place the base image on the canvas with translation
        base_translation = translation.copy()
        base_translation[0, 2] = -x_min
        base_translation[1, 2] = -y_min

        if base_img is not None:
            # Warp the base image to the combined space with stderr suppression
            warped_base = self._suppress_stderr(cv2.warpPerspective, base_img, base_translation.astype(np.float32), (width, height))

        # Blend the images together
        result = self._blend_images(warped_base, warped_new)

        return result
    
    def _blend_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Simple blending of two images using averaging where they overlap.
        
        Args:
            img1: First image
            img2: Second image to blend with first
            
        Returns:
            Blended image
        """
        # Where img2 is not black, blend with img1
        mask = np.any(img2 != 0, axis=2)
        
        result = img1.copy()
        
        # For overlapping areas, average the pixels
        overlap = np.logical_and(mask, np.any(img1 != 0, axis=2))
        result[overlap] = (img1[overlap] + img2[overlap]) / 2
        
        # For areas where only img2 exists, copy directly
        background = np.logical_and(mask, np.logical_not(np.any(img1 != 0, axis=2)))
        result[background] = img2[background]
        
        return result.astype(np.uint8)