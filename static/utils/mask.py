"""
Mask Processing Utilities for SAM3 Tracker

Provides adaptive erosion, fast DBSCAN clustering, and mask merging.
"""

import cv2
import logging
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Dict
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def adaptive_erode_mask_area(
    mask: np.ndarray, target_area_ratio: float = 0.85
) -> np.ndarray:
    """
    Erode mask to achieve target area ratio while preserving shape.

    Args:
        mask: Binary mask [H, W].
        target_area_ratio: Target area fraction after erosion (0.85 = keep 85%).

    Returns:
        Eroded mask [H, W] with area >= target_area_ratio * original.
    """
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)

    original_area = cv2.countNonZero(mask)
    if original_area == 0:
        return mask

    # Iteratively increase kernel size until target ratio reached
    kernel_size = 3
    while kernel_size <= 7:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        eroded = cv2.erode(mask, kernel, iterations=1)
        current_area = cv2.countNonZero(eroded)

        if current_area / original_area >= target_area_ratio:
            return eroded

        kernel_size += 2

    # Fallback: minimal erosion if target not achievable
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.erode(mask, kernel, iterations=1)


def safe_erode(mask: np.ndarray, min_area_after: int = 32) -> np.ndarray:
    """
    Safely erode mask with rollback if it becomes too small.

    Args:
        mask: Binary mask [H, W].
        min_area_after: Minimum area threshold after erosion.

    Returns:
        Eroded mask or original if erosion would make it too small.
    """
    original_area = cv2.countNonZero(mask)

    # Skip erosion for very small objects
    if original_area < min_area_after * 2:
        return mask

    eroded = adaptive_erode_mask_area(mask)

    # Rollback if erosion made mask too small
    if cv2.countNonZero(eroded) < min_area_after:
        return mask

    return eroded


class FastPixelDBSCAN:
    """
    Fast DBSCAN clustering via image downsampling.

    Reduces resolution before clustering for 10-15x speedup.
    """

    def __init__(
        self,
        target_size: int = 300,
        eps_pixels: int = 20,
        min_samples: int = 10,
        upscale_method: str = "nearest",
    ):
        """
        Args:
            target_size: Long side after downsample (300 = fast, 500 = accurate).
            eps_pixels: DBSCAN eps in downsample coordinates.
            min_samples: Minimum pixels per cluster.
            upscale_method: Upscale method ('nearest' or 'bilinear').
        """
        self.target_size = target_size
        self.eps_pixels = eps_pixels
        self.min_samples = min_samples
        self.upscale_method = upscale_method

        # Debug info
        self.original_shape = None
        self.downsample_shape = None
        self.scale_factor = 1.0

    def cluster_pixels(
        self, mask_2d: np.ndarray, verbose: bool = False
    ) -> Tuple[List[np.ndarray], int, int]:
        """
        Cluster mask pixels via downsample + DBSCAN + upsample.

        Args:
            mask_2d: Binary mask [H, W].
            verbose: Print debug information.

        Returns:
            masks: List of separated masks.
            n_pixels: Number of pixels clustered.
            n_clusters: Number of clusters found.
        """
        # Store original dimensions
        self.original_shape = mask_2d.shape
        H, W = mask_2d.shape

        # Compute downsample dimensions
        scale = self.target_size / max(H, W)
        self.scale_factor = 1.0 / scale
        new_H = int(H * scale)
        new_W = int(W * scale)
        self.downsample_shape = (new_H, new_W)

        if verbose:
            print(f"Downsample: {H}×{W} → {new_H}×{new_W} (scale={scale:.2f})")

        # Downsample mask
        mask_uint8 = (mask_2d > 0).astype(np.uint8) * 255
        mask_small = cv2.resize(
            mask_uint8, (new_W, new_H), interpolation=cv2.INTER_AREA
        )
        mask_small = mask_small > 0

        # Find pixel coordinates
        y_coords, x_coords = np.where(mask_small > 0)
        n_pixels = len(x_coords)

        if n_pixels == 0:
            return [mask_2d], 0, 0

        if verbose:
            print(f"Pixels for clustering: {n_pixels}")

        # DBSCAN in downsample space
        X = np.column_stack([x_coords, y_coords])
        clustering = DBSCAN(
            eps=self.eps_pixels,
            min_samples=self.min_samples,
            metric="euclidean",
            n_jobs=-1,
        )
        labels = clustering.fit_predict(X)

        # Get cluster statistics
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        n_clusters = len(unique_labels)
        n_noise = np.sum(labels == -1)

        if verbose:
            print(f"DBSCAN: eps={self.eps_pixels}, min_samples={self.min_samples}")
            print(
                f"Clusters: {n_clusters}, Noise: {n_noise} pixels ({100*n_noise/n_pixels:.1f}%)"
            )

        # Create masks for each cluster
        masks_small = []
        for label in unique_labels:
            cluster_mask = np.zeros((new_H, new_W), dtype=bool)
            cluster_indices = np.where(labels == label)[0]
            for idx in cluster_indices:
                x = x_coords[idx]
                y = y_coords[idx]
                cluster_mask[y, x] = True
            masks_small.append(cluster_mask)

        # Upsample to original resolution
        masks = []
        for mask_small in masks_small:
            mask_small_uint8 = mask_small.astype(np.uint8) * 255
            if self.upscale_method == "nearest":
                mask_large = cv2.resize(
                    mask_small_uint8, (W, H), interpolation=cv2.INTER_NEAREST
                )
            else:
                mask_large = cv2.resize(
                    mask_small_uint8, (W, H), interpolation=cv2.INTER_LINEAR
                )
            masks.append(mask_large > 0)

        # Fallback if no clusters found
        if len(masks) == 0:
            return [mask_2d], n_pixels, 0

        if verbose:
            print(f"Result: {n_pixels} pixels → {len(masks)} objects")

        return masks, n_pixels, n_clusters


def merge_masks(tracks: Dict[int, Dict], area_size: int = 50) -> Dict[int, Dict]:
    """
    Merge fragmented masks within tracks and split into new tracks if needed.

    Uses DBSCAN to separate disconnected mask components. Large components
    become new tracks, small components are merged into the largest.

    Args:
        tracks: Dict of track_id -> track_data with 'masks' and 'cls' fields.
        area_size: Minimum area threshold for creating new tracks.

    Returns:
        Updated tracks dict with merged/split masks.

    Notes:
        - Tracks with multiple mask components are split if components > area_size
        - Small components (< area_size) are merged into the largest component
    """
    mask_merger = FastPixelDBSCAN()
    n = max(tracks.keys()) + 1
    all_tracks = {}
    total_masks = sum(len(track["masks"]) for track in tracks.values())

    logger.info(
        "Merging masks for %d tracks and %d frame masks", len(tracks), total_masks
    )
    merge_progress = tqdm(
        total=total_masks,
        desc="Merging masks",
        unit="mask",
        dynamic_ncols=True,
    )

    for key in tracks.keys():
        track = tracks[key]

        for i in track["masks"].keys():
            mask = track["masks"][i]

            # Separate disconnected components
            merged_masks, _, _ = mask_merger.cluster_pixels(mask)
            merged_masks = np.array(merged_masks)

            if len(merged_masks) == 1:
                # Single component - keep as is
                track["masks"][i] = merged_masks[0]
            else:
                # Multiple components - split by area
                merged_mask_area = np.sum(
                    merged_masks.reshape(len(merged_masks), -1), axis=1
                )
                area_idx = np.argsort(-merged_mask_area)

                # Largest component stays with original track
                max_mask = merged_masks[area_idx[0]]
                remaining_masks = merged_masks[area_idx[1:]]
                remaining_area = merged_mask_area[area_idx[1:]]

                # Merge small components into largest
                small_masks = remaining_masks[remaining_area < area_size]
                max_mask += np.sum(small_masks, axis=0).astype(bool)
                track["masks"][i] = max_mask

                # Create new tracks for large components
                big_masks = remaining_masks[remaining_area >= area_size]
                for mask in big_masks:
                    all_tracks[n] = {}
                    all_tracks[n]["cls"] = track["cls"]
                    all_tracks[n]["masks"] = {i: mask}
                    n += 1

            merge_progress.update(1)
            merge_progress.set_postfix(
                current_track=key,
                new_tracks=max(0, n - len(tracks)),
            )

        all_tracks[key] = track

    merge_progress.close()
    logger.info("Mask merge complete: %d tracks -> %d tracks", len(tracks), len(all_tracks))

    return all_tracks
