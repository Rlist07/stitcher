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

    def __init__(self, settings):
        self.settings = settings
        self.logger = logging.getLogger(__name__)

        self.sift = cv2.SIFT_create(nfeatures=self.settings.sift_features)

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def detect_and_compute_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)

        return keypoints, descriptors

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        if desc1 is None or desc2 is None:
            return []

        try:
            with redirect_stderr(io.StringIO()):
                matches = self.flann.knnMatch(desc1, desc2, k=2)
        except cv2.error:
            self.logger.error("Feature matching failed due to OpenCV error")
            return []

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.settings.matcher_ratio * n.distance:
                    good_matches.append(m)

        if len(good_matches) > 5:
            src_pts = np.float32([desc1[m.queryIdx] for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([desc2[m.trainIdx] for m in good_matches]).reshape(-1, 1, 2)

            try:
                _, mask = cv2.findHomography(
                    src_pts, dst_pts,
                    cv2.RANSAC,
                    ransacReprojThreshold=min(self.settings.ransac_threshold * 2, 20.0),
                    maxIters=500
                )

                if mask is not None and len(mask) == len(good_matches):
                    refined_matches = []
                    for i, match in enumerate(good_matches):
                        if mask[i] == 1:
                            refined_matches.append(match)

                    if len(refined_matches) >= max(3, len(good_matches) // 3):
                        return refined_matches
                    else:
                        pass
            except cv2.error:
                pass

        return good_matches

    def compute_homography(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint],
                          matches: List[cv2.DMatch]) -> Tuple[Optional[np.ndarray], List[cv2.DMatch]]:
        if len(matches) < 4:
            return None, []

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        if len(src_pts) < 4 or len(dst_pts) < 4:
            return None, []

        src_pts_reshaped = src_pts.reshape(-1, 2)
        dst_pts_reshaped = dst_pts.reshape(-1, 2)

        src_unique_points = len(np.unique(src_pts_reshaped, axis=0))
        dst_unique_points = len(np.unique(dst_pts_reshaped, axis=0))

        if src_unique_points < 3 or dst_unique_points < 3:
            self.logger.warning("Not enough unique points for reliable homography estimation")
            return None, []

        try:
            with redirect_stderr(io.StringIO()):
                homography, mask = cv2.findHomography(
                    src_pts, dst_pts,
                    cv2.RANSAC,
                    ransacReprojThreshold=self.settings.ransac_threshold,
                    maxIters=2000,
                    confidence=0.99
                )
        except cv2.error as e:
            self.logger.error(f"OpenCV error during homography computation: {str(e)}")
            return None, []
        except Exception as e:
            self.logger.error(f"Error during homography computation: {str(e)}")
            return None, []

        if homography is None or homography.shape != (3, 3):
            return None, []

        if np.any(np.isnan(homography)) or np.any(np.isinf(homography)):
            return None, []

        try:
            _, s, _ = np.linalg.svd(homography)
            cond_num = s[0] / s[-1] if s[-1] != 0 else float('inf')

            if cond_num > 1e6:
                self.logger.warning(f"Homography matrix is ill-conditioned (condition number: {cond_num})")
                return None, []
        except:
            return None, []

        inlier_matches = []
        if mask is not None:
            inlier_matches = [matches[i] for i in range(len(matches)) if mask[i] == 1]

        return homography, inlier_matches

    def match_and_compute_homography(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[Optional[np.ndarray], List[cv2.DMatch]]:
        kp1, desc1 = self.detect_and_compute_features(img1)
        kp2, desc2 = self.detect_and_compute_features(img2)

        if desc1 is None or desc2 is None:
            self.logger.error("Could not compute descriptors for one of the images")
            return None, []

        matches = self.match_features(desc1, desc2)

        if len(matches) < 10:
            self.logger.debug(f"Not enough matches found: {len(matches)} (required at least 10)")
            return None, []

        sample_matches = matches[:min(10, len(matches))]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in sample_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in sample_matches]).reshape(-1, 1, 2)

        if len(src_pts) >= 4:
            src_center = np.mean(src_pts.reshape(-1, 2), axis=0)
            dst_center = np.mean(dst_pts.reshape(-1, 2), axis=0)

            src_distances = np.linalg.norm(src_pts.reshape(-1, 2) - src_center, axis=1)
            dst_distances = np.linalg.norm(dst_pts.reshape(-1, 2) - dst_center, axis=1)

            if len(src_distances) > 0 and np.any(src_distances > 0):
                distance_ratios = dst_distances / src_distances
                distance_ratios = distance_ratios[~np.isnan(distance_ratios)]

                if len(distance_ratios) > 0:
                    ratio_std = np.std(distance_ratios)
                    ratio_mean = np.mean(distance_ratios)

                    if ratio_mean > 0 and ratio_std / ratio_mean > 0.8:
                        self.logger.debug(f"Geometric inconsistency detected, skipping homography. Scale variation: {ratio_std/ratio_mean:.2f}")
                        return None, []

        homography, inlier_matches = self.compute_homography(kp1, kp2, matches)

        if homography is None:
            self.logger.debug("Could not compute homography matrix")
            return None, []

        if len(inlier_matches) < 10:
            self.logger.debug(f"Homography has too few inliers: {len(inlier_matches)}")
            return None, []

        self.logger.debug(f"Found {len(inlier_matches)} inlier matches after RANSAC")
        return homography, inlier_matches