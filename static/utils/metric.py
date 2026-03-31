"""
Similarity Metrics for SAM3 Tracker

Provides Gaussian-IoA, embedding similarity, and voxel-based matching.
"""

import cv2
import numpy as np
import torch
import open3d as o3d
from typing import Tuple, Optional


def adaptive_sigma(
    points_track: np.ndarray, points_det: np.ndarray, scale_factor: float = 0.2
) -> float:
    """
    Compute adaptive sigma based on object size for Gaussian IoA.

    Larger objects get larger sigma to account for boundary uncertainty.

    Args:
        points_track: Track 3D points [N, 3].
        points_det: Detection 3D points [M, 3].
        scale_factor: Multiplier for object size (default: 0.2).

    Returns:
        sigma: Adaptive sigma value clipped to [0.05, 0.3].
    """
    if len(points_track) == 0 or len(points_det) == 0:
        return 0.1

    # Compute object dimensions (bounding box diagonal)
    size_t = points_track.max(axis=0) - points_track.min(axis=0)
    size_d = points_det.max(axis=0) - points_det.min(axis=0)

    # Diagonal lengths
    diag_t = np.linalg.norm(size_t)
    diag_d = np.linalg.norm(size_d)
    avg_diag = (diag_t + diag_d) / 2.0

    # Sigma proportional to object size
    sigma = avg_diag * scale_factor

    # Clip to reasonable range
    sigma = np.clip(sigma, 0.05, 0.3)

    return sigma


def compute_bidirectional_gaussian_ioa(
    pcd_track: o3d.geometry.PointCloud,
    pcd_det: o3d.geometry.PointCloud,
    voxel_size: float = 0.01,
    sigma: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    Compute bidirectional Gaussian IoA between two point clouds.

    Averages IoA(track→det) and IoA(det→track) to handle density asymmetry.
    Uses Gaussian weighting based on point-to-point distances.

    Args:
        pcd_track: Track point cloud.
        pcd_det: Detection point cloud.
        voxel_size: Voxel size for downsampling detection (default: 0.01m).
        sigma: Gaussian width. If None, computed adaptively from object size.

    Returns:
        ioa_from_t: IoA from track perspective [0, 1].
        ioa_from_d: IoA from detection perspective [0, 1].
        ioa_bidirectional: Averaged bidirectional IoA [0, 1].

    Notes:
        - Detection is voxelized to match track density
        - Adaptive sigma used if not provided
        - Returns 0.0 if either cloud is empty
    """
    # Voxelized detection for fair density comparison
    pcd_det_voxelized = pcd_det.voxel_down_sample(voxel_size)

    points_t = np.asarray(pcd_track.points)
    points_d = np.asarray(pcd_det_voxelized.points)

    if len(points_t) == 0 or len(points_d) == 0:
        return 0.0, 0.0, 0.0

    # Compute adaptive sigma if not provided
    if sigma is None:
        sigma = adaptive_sigma(points_t, points_d)

    # Compute distances in both directions
    dist_t_to_d = np.array(pcd_track.compute_point_cloud_distance(pcd_det_voxelized))
    dist_d_to_t = np.array(pcd_det_voxelized.compute_point_cloud_distance(pcd_track))

    # Gaussian weighting
    match_t_to_d = np.exp(-(dist_t_to_d**2) / (2 * sigma**2))
    match_d_to_t = np.exp(-(dist_d_to_t**2) / (2 * sigma**2))

    # IoA from track perspective
    intersection_from_t = np.sum(match_t_to_d)
    ioa_from_t = intersection_from_t / max(len(points_t), 1e-6)

    # IoA from detection perspective
    intersection_from_d = np.sum(match_d_to_t)
    ioa_from_d = intersection_from_d / max(len(points_d), 1e-6)

    # Bidirectional average (handles density asymmetry)
    ioa_bidirectional = (ioa_from_t + ioa_from_d) / 2.0

    return ioa_from_t, ioa_from_d, np.clip(ioa_bidirectional, 0.0, 1.0)


def normalize(vec: np.ndarray) -> np.ndarray:
    """
    L2-normalize vectors along axis 1 if not already normalized.

    Args:
        vec: Input vectors [N, D].

    Returns:
        Normalized vectors [N, D] with unit norm.
    """
    # Check if already normalized (within tolerance)
    if not np.allclose(np.linalg.norm(vec, axis=1), 1.0, atol=1e-3):
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8)
    return vec


def calc_emb_similarity(
    tracks_text_emb: np.ndarray, dets_text_emb: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between track and detection text embeddings.

    Args:
        tracks_text_emb: Track embeddings [N, D].
        dets_text_emb: Detection embeddings [M, D].

    Returns:
        text_cos_sim: Cosine similarity matrix [N, M].
    """
    tracks_text_emb = normalize(tracks_text_emb)
    dets_text_emb = normalize(dets_text_emb)
    text_cos_sim = tracks_text_emb @ dets_text_emb.T
    return text_cos_sim


def calc_voxel_similarity(
    track, det: dict, w_iou: float = 0.6, w_dist: float = 0.2
) -> float:
    """
    Compute combined similarity between track and detection.

    Combines Gaussian-IoA (geometry), embedding similarity (appearance),
    and center distance (spatial proximity).

    Args:
        track: Track object with 'voxels', 'embedding' attributes.
        det: Detection dict with 'points', 'embedding' keys.
        w_iou: Weight for IoA component (default: 0.6).
        w_dist: Weight for distance component (default: 0.2).

    Returns:
        similarity: Combined similarity score [0, 1].

    Notes:
        - Remaining weight (1 - w_iou - w_dist) goes to embedding similarity
        - Distance similarity uses Gaussian kernel with sigma=1.0
    """
    # Convert detection points to Open3D point cloud
    pcd_det = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(det["points"]))
    pcd_track = track.voxels.get_pcd()

    # Compute bidirectional Gaussian-IoA
    ioa_from_t, ioa_from_d, ioa = compute_bidirectional_gaussian_ioa(pcd_track, pcd_det)

    # Compute embedding similarity
    emb_sim = calc_emb_similarity(track.embedding, det["embedding"])

    # Compute center distance similarity
    dist = np.linalg.norm(pcd_det.get_center() - pcd_track.get_center())
    dist_similarity = np.exp(-(dist**2) / 2.0)

    # Combine all components
    similarity = w_iou * ioa + (1 - w_iou - w_dist) * emb_sim + w_dist * dist_similarity

    return similarity
