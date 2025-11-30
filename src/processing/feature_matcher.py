"""Feature matcher module using SIFT and FLANN for low-level image matching."""
import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging
import os
import sys
import warnings
from contextlib import redirect_stderr
import io


class FeatureMatcher:
    """Handles low-level feature matching using SIFT and FLANN."""
    
    def __init__(self, settings):
        """Initialize the feature matcher with settings."""
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Initialize SIFT detector
        self.sift = cv2.SIFT_create(nfeatures=self.settings.sift_features)
        
        # Initialize FLANN matcher
        index_params = dict(algorithm=1, trees=5)  # KDTree
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    def detect_and_compute_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and compute SIFT features in an image.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match features between two sets of descriptors using FLANN.

        Args:
            desc1: First set of descriptors
            desc2: Second set of descriptors

        Returns:
            List of matches
        """
        if desc1 is None or desc2 is None:
            return []

        try:
            with redirect_stderr(io.StringIO()):
                matches = self.flann.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            self.logger.error("Feature matching failed due to OpenCV error")
            return []

        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.settings.matcher_ratio * n.distance:
                    good_matches.append(m)

        # Perform additional geometric verification but be more permissive
        # Only do this if we have enough matches to work with
        if len(good_matches) > 5:  # Lowered threshold to work with fewer matches
            # Convert to keypoint format for geometric check
            src_pts = np.float32([desc1[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([desc2[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

            # Perform geometric verification but be more permissive
            try:
                # Use findHomography with looser parameters to keep more matches
                _, mask = cv2.findHomography(
                    src_pts, dst_pts,
                    cv2.RANSAC,
                    ransacReprojThreshold=min(self.settings.ransac_threshold * 2, 20.0),  # Looser threshold
                    maxIters=500  # Fewer iterations for performance
                )

                if mask is not None and len(mask) == len(good_matches):
                    # Keep more matches by being less strict about outliers
                    refined_matches = []
                    for i, match in enumerate(good_matches):
                        if mask[i] == 1:  # Still use RANSAC inliers but more permissive
                            refined_matches.append(match)

                    # If we kept too few matches, use the original matches instead
                    if len(refined_matches) >= max(3, len(good_matches) // 3):  # Keep at least 1/3rd
                        return refined_matches
                    else:
                        # Use original matches if filtering was too aggressive
                        pass
            except cv2.error:
                # If homography computation fails, return the original good matches
                pass

        return good_matches
    
    def compute_homography(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint],
                          matches: List[cv2.DMatch]) -> Tuple[Optional[np.ndarray], List[cv2.DMatch]]:
        """
        Compute homography matrix from matched keypoints using RANSAC.

        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: List of good matches

        Returns:
            Tuple of (homography matrix, list of inlier matches)
        """
        if len(matches) < 4:
            return None, []

        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Additional check: ensure we have enough points to compute homography
        if len(src_pts) < 4 or len(dst_pts) < 4:
            return None, []

        # Check for geometric consistency before computing homography
        # Calculate the range of points to ensure they're not collinear
        src_pts_reshaped = src_pts.reshape(-1, 2)
        dst_pts_reshaped = dst_pts.reshape(-1, 2)

        # Check if points are distinct enough (not all the same or nearly the same)
        src_unique_points = len(np.unique(src_pts_reshaped, axis=0))
        dst_unique_points = len(np.unique(dst_pts_reshaped, axis=0))

        if src_unique_points < 3 or dst_unique_points < 3:
            self.logger.warning("Not enough unique points for reliable homography estimation")
            return None, []

        # Use the more robust stderr suppression approach
        try:
            # Find homography using RANSAC with more strict parameters
            with redirect_stderr(io.StringIO()):
                homography, mask = cv2.findHomography(
                    src_pts, dst_pts,
                    cv2.RANSAC,
                    ransacReprojThreshold=self.settings.ransac_threshold,  # Use configured threshold (now 5.0)
                    maxIters=2000,  # More iterations for better accuracy
                    confidence=0.99  # Higher confidence for more strictness
                )
        except cv2.error as e:
            self.logger.error(f"OpenCV error during homography computation: {str(e)}")
            return None, []
        except Exception as e:
            self.logger.error(f"Error during homography computation: {str(e)}")
            return None, []

        # If homography computation fails or returns an invalid matrix
        if homography is None or homography.shape != (3, 3):
            return None, []

        # Additional check: ensure the computed homography is reasonable
        # Check if the homography matrix contains any NaN or infinite values
        if np.any(np.isnan(homography)) or np.any(np.isinf(homography)):
            return None, []

        # Additional validation: check if the homography is well-conditioned
        # A badly conditioned homography might cause numerical issues
        try:
            # Calculate the condition number (ratio of largest to smallest singular value)
            # If it's too large, the matrix is ill-conditioned
            _, s, _ = np.linalg.svd(homography)
            cond_num = s[0] / s[-1] if s[-1] != 0 else float('inf')

            # If condition number is very high, the matrix is ill-conditioned
            if cond_num > 1e6:  # Threshold for ill-conditioned matrix
                self.logger.warning(f"Homography matrix is ill-conditioned (condition number: {cond_num})")
                return None, []
        except:
            # If SVD fails, the homography is problematic
            return None, []

        # Extract inlier matches
        inlier_matches = []
        if mask is not None:
            inlier_matches = [matches[i] for i in range(len(matches)) if mask[i] == 1]

        return homography, inlier_matches
    
    def match_and_compute_homography(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[Optional[np.ndarray], List[cv2.DMatch]]:
        """
        Full pipeline: detect features, match them, and compute homography.

        Args:
            img1: First image
            img2: Second image

        Returns:
            Tuple of (homography matrix, list of inlier matches)
        """
        # Detect and compute features
        kp1, desc1 = self.detect_and_compute_features(img1)
        kp2, desc2 = self.detect_and_compute_features(img2)

        if desc1 is None or desc2 is None:
            self.logger.error("Could not compute descriptors for one of the images")
            return None, []

        # Match features
        matches = self.match_features(desc1, desc2)

        if len(matches) < 10:  # Increased requirement for better matching
            self.logger.debug(f"Not enough matches found: {len(matches)} (required at least 10)")
            return None, []

        # Before computing homography, do a quick geometric check
        # Use the first 10 matches to check if they have reasonable geometric consistency
        sample_matches = matches[:min(10, len(matches))]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in sample_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in sample_matches]).reshape(-1, 1, 2)

        # Quick geometric check: calculate the distribution of distances and angles
        # If the keypoints don't have consistent geometric relationships, skip homography computation
        if len(src_pts) >= 4:
            # Calculate distances between keypoints to check geometric consistency
            src_center = np.mean(src_pts.reshape(-1, 2), axis=0)
            dst_center = np.mean(dst_pts.reshape(-1, 2), axis=0)

            src_distances = np.linalg.norm(src_pts.reshape(-1, 2) - src_center, axis=1)
            dst_distances = np.linalg.norm(dst_pts.reshape(-1, 2) - dst_center, axis=1)

            # Check if distance ratios are reasonable (not too variable)
            if len(src_distances) > 0 and np.any(src_distances > 0):
                distance_ratios = dst_distances / src_distances
                distance_ratios = distance_ratios[~np.isnan(distance_ratios)]  # Remove NaN values

                if len(distance_ratios) > 0:
                    ratio_std = np.std(distance_ratios)
                    ratio_mean = np.mean(distance_ratios)

                    # If the scale varies too much, it's likely not a good match
                    # Tightened threshold to be more strict: was 2.0, now 0.8
                    if ratio_mean > 0 and ratio_std / ratio_mean > 0.8:  # Lower threshold for more strictness
                        self.logger.debug(f"Geometric inconsistency detected, skipping homography. Scale variation: {ratio_std/ratio_mean:.2f}")
                        return None, []

        # Compute homography
        homography, inlier_matches = self.compute_homography(kp1, kp2, matches)

        if homography is None:
            self.logger.debug("Could not compute homography matrix")
            return None, []

        # Additional check: verify homography quality
        if len(inlier_matches) < 10:  # Increased requirement for more strictness
            self.logger.debug(f"Homography has too few inliers: {len(inlier_matches)}")
            return None, []

        self.logger.debug(f"Found {len(inlier_matches)} inlier matches after RANSAC")
        return homography, inlier_matches