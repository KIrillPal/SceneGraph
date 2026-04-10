from __future__ import annotations

from typing import Dict

import numpy as np
import open3d as o3d

from .point_cloud import normalize_mask, split_point_cloud


def filter_tracking_mask_statistical(
    pc: np.ndarray,
    tracking_mask: Dict[str, np.ndarray],
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> Dict[str, np.ndarray]:
    """Filter each tracking mask with Open3D statistical outlier removal.

    Args:
        pc: Point cloud with shape `(N, 3)` or `(N, 6)`.
        tracking_mask: Mapping from `'classname.id'` to a boolean mask of shape `(N,)`.
        nb_neighbors: Number of neighbors used by `remove_statistical_outlier`.
        std_ratio: Standard deviation threshold used by `remove_statistical_outlier`.

    Returns:
        A dictionary with the same keys as `tracking_mask`, where each value is a filtered
        boolean mask of shape `(N,)`.
    """
    points, _ = split_point_cloud(pc)
    filtered_tracking_mask: Dict[str, np.ndarray] = {}

    for mask_key, mask in tracking_mask.items():
        point_mask = normalize_mask(mask, len(points), mask_key)
        point_indices = np.flatnonzero(point_mask)

        filtered_mask = np.zeros(len(points), dtype=bool)
        if point_indices.size == 0:
            filtered_tracking_mask[mask_key] = filtered_mask
            continue

        masked_points = points[point_indices]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(masked_points)

        _, inlier_indices = point_cloud.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )

        if len(inlier_indices) == 0:
            filtered_tracking_mask[mask_key] = filtered_mask
            continue

        filtered_mask[point_indices[np.asarray(inlier_indices, dtype=np.int64)]] = True
        filtered_tracking_mask[mask_key] = filtered_mask

    return filtered_tracking_mask


def filter_tracking_mask_dbscan(
    pc: np.ndarray,
    tracking_mask: Dict[str, np.ndarray],
    eps: float = 0.02,
    min_points: int = 10,
) -> Dict[str, np.ndarray]:
    """Filter each tracking mask by keeping the largest DBSCAN cluster.

    Args:
        pc: Point cloud with shape `(N, 3)` or `(N, 6)`.
        tracking_mask: Mapping from `'classname.id'` to a boolean mask of shape `(N,)`.
        eps: DBSCAN neighborhood radius.
        min_points: Minimum number of points required to form a cluster.

    Returns:
        A dictionary with the same keys as `tracking_mask`, where each value is a filtered
        boolean mask of shape `(N,)`.
    """
    points, _ = split_point_cloud(pc)
    filtered_tracking_mask: Dict[str, np.ndarray] = {}

    for mask_key, mask in tracking_mask.items():
        point_mask = normalize_mask(mask, len(points), mask_key)
        point_indices = np.flatnonzero(point_mask)

        filtered_mask = np.zeros(len(points), dtype=bool)
        if point_indices.size == 0:
            filtered_tracking_mask[mask_key] = filtered_mask
            continue

        masked_points = points[point_indices]
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(masked_points)

        cluster_labels = np.asarray(
            point_cloud.cluster_dbscan(eps=eps, min_points=min_points),
            dtype=np.int32,
        )

        valid_labels = cluster_labels[cluster_labels >= 0]
        if valid_labels.size == 0:
            filtered_tracking_mask[mask_key] = point_mask.copy()
            continue

        largest_label = np.bincount(valid_labels).argmax()
        inlier_indices = np.flatnonzero(cluster_labels == largest_label)
        filtered_mask[point_indices[inlier_indices]] = True
        filtered_tracking_mask[mask_key] = filtered_mask

    return filtered_tracking_mask
