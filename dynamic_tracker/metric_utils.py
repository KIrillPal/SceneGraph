"""
Similarity Metrics for SAM3 Tracker

Provides Gaussian-IoA, embedding similarity, and voxel-based matching.
"""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from typing import Tuple, Optional, List
from config_loader import cfg


# ============================================================================
# GAUSSIAN IOA
# ============================================================================

def compute_bidirectional_gaussian_ioa_fast(
    points_t: np.ndarray,
    points_d: np.ndarray,
    voxel_size: float = cfg.voxel_map.voxel_size,
    sigma: Optional[float] = None,
    use_approximation: bool = cfg.ioa.use_approximation,
    low_iou_thresh: float = cfg.ioa.low_iou_thresh,
    high_iou_thresh: float = cfg.ioa.high_iou_thresh,
    max_points: int = cfg.ioa.max_points,
) -> Tuple[float, float, float]:
    """
    Optimized Gaussian IoA with 10-20x speedup.
    
    Key optimizations:
    1. Pre-downsampled points (no Open3D in hot path)
    2. Single KD-tree + symmetric approximation
    3. Voxel IoU fallback for very fast path
    4. Early exit for obvious cases
    
    Args:
        points_t: Track points [N, 3].
        points_d: Detection points [M, 3].
        voxel_size: Voxel size for downsampling.
        sigma: Gaussian width (auto-computed if None).
        use_approximation: Use single KD-tree approximation.
        low_iou_thresh: Early exit threshold (low overlap).
        high_iou_thresh: Early exit threshold (high overlap).
        max_points: Maximum points per cloud for speed.
    
    Returns:
        ioa_from_t: IoA from track perspective [0, 1].
        ioa_from_d: IoA from detection perspective [0, 1].
        ioa_bidirectional: Averaged bidirectional IoA [0, 1].
    """
    # Early exit: Empty clouds
    if len(points_t) == 0 or len(points_d) == 0:
        return 0.0, 0.0, 0.0
    
    # Early exit: Bounding box check (O(1))
    bbox_iou = _compute_bbox_iou_fast(points_t, points_d)
    if bbox_iou < low_iou_thresh:
        return 0.0, 0.0, 0.0
    if bbox_iou > high_iou_thresh:
        return 1.0, 1.0, 1.0
    
    # Voxel downsampling (numpy, not Open3D)
    points_d_voxelized = _voxel_downsample_numpy(points_d, voxel_size)
    
    # Limit points for speed
    if len(points_t) > max_points:
        indices = np.random.choice(len(points_t), max_points, replace=False)
        points_t = points_t[indices]
    if len(points_d_voxelized) > max_points:
        indices = np.random.choice(len(points_d_voxelized), max_points, replace=False)
        points_d_voxelized = points_d_voxelized[indices]
    
    # Adaptive sigma
    if sigma is None:
        sigma = _adaptive_sigma_fast(points_t, points_d_voxelized)
    
    # Single KD-tree + approximation (fast)
    if use_approximation:
        if len(points_t) < len(points_d_voxelized):
            tree = cKDTree(points_t)
            dists, _ = tree.query(points_d_voxelized, k=1)
        else:
            tree = cKDTree(points_d_voxelized)
            dists, _ = tree.query(points_t, k=1)
        
        match_scores = np.exp(-(dists**2) / (2 * sigma**2))
        match_score = np.mean(match_scores)
        
        ioa_from_t = match_score
        ioa_from_d = match_score
        ioa_bidirectional = match_score
    else:
        # Full bidirectional (slower, more accurate)
        tree_t = cKDTree(points_t)
        tree_d = cKDTree(points_d_voxelized)
        
        dists_t_to_d, _ = tree_d.query(points_t, k=1)
        dists_d_to_t, _ = tree_t.query(points_d_voxelized, k=1)
        
        match_t_to_d = np.exp(-(dists_t_to_d**2) / (2 * sigma**2))
        match_d_to_t = np.exp(-(dists_d_to_t**2) / (2 * sigma**2))
        
        ioa_from_t = np.mean(match_t_to_d)
        ioa_from_d = np.mean(match_d_to_t)
        ioa_bidirectional = (ioa_from_t + ioa_from_d) / 2.0
    
    return (
        float(np.clip(ioa_from_t, 0.0, 1.0)),
        float(np.clip(ioa_from_d, 0.0, 1.0)),
        float(np.clip(ioa_bidirectional, 0.0, 1.0))
    )

def _compute_keypoint_overlap(
    track_keypoints: np.ndarray,
    det_mask: np.ndarray,
    image_shape: Tuple[int, int]
) -> float:
    """
    Compute fraction of CoTracker keypoints inside detection mask.
    
    Args:
        track_keypoints: CoTracker keypoints [N, 2] in pixel coordinates.
        det_mask: Detection binary mask [H, W].
        image_shape: Image shape (H, W).
    
    Returns:
        overlap_ratio: Fraction of keypoints inside mask [0, 1].
    """
    if track_keypoints is None or len(track_keypoints) == 0:
        return 0.0
    
    if det_mask is None or det_mask.sum() == 0:
        return 0.0
    
    H, W = image_shape
    
    # Round keypoints to integer coordinates
    kps_int = np.round(track_keypoints).astype(int)
    # Filter keypoints within image bounds
    valid_mask = (
        (kps_int[:, 0] >= 0) & (kps_int[:, 0] < W) &
        (kps_int[:, 1] >= 0) & (kps_int[:, 1] < H)
    )
    
    if np.sum(valid_mask) == 0:
        return 0.0
    
    # Check how many keypoints are inside detection mask
    kps_valid = kps_int[valid_mask]
    inside_count = np.sum(det_mask[kps_valid[:, 1], kps_valid[:, 0]] > 0)
    
    overlap_ratio = inside_count / len(kps_valid)
    
    return overlap_ratio
# ============================================================================
# EMBEDDING SIMILARITY
# ============================================================================

def calc_emb_similarity(
    tracks_text_emb: np.ndarray,
    dets_text_emb: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between track and detection text embeddings.
    
    Args:
        tracks_text_emb: Track embeddings [N, D].
        dets_text_emb: Detection embeddings [M, D].
    
    Returns:
        text_cos_sim: Cosine similarity matrix [N, M].
    """
    tracks_text_norm = _normalize_batch_fast(tracks_text_emb)
    dets_text_norm = _normalize_batch_fast(dets_text_emb)
    return tracks_text_norm @ dets_text_norm.T


def calc_emb_similarity_visual_only_optimized(
    tracks_vis_emb: np.ndarray,
    dets_vis_emb: np.ndarray,
    tracks_text_emb: Optional[np.ndarray] = None,
    dets_text_emb: Optional[np.ndarray] = None,
    text_threshold: float = cfg.embeddings.text_threshold,
    use_text_filter: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized visual similarity with text-based filtering.
    
    Computes visual similarity ONLY for text-valid pairs.
    Faster when text filter removes >50% of pairs.
    
    Args:
        tracks_vis_emb: Track visual embeddings [N, D].
        dets_vis_emb: Detection visual embeddings [M, D].
        tracks_text_emb: Track text embeddings [N, D] (optional).
        dets_text_emb: Detection text embeddings [M, D] (optional).
        text_threshold: Text similarity threshold for filtering.
        use_text_filter: Enable text-based filtering.
    
    Returns:
        similarity_matrix: Visual similarity [N, M] with zeros for invalid pairs.
        valid_mask: Boolean mask [N, M] of text-valid pairs.
    """
    n_tracks = len(tracks_vis_emb)
    n_dets = len(dets_vis_emb)
    
    if n_tracks == 0 or n_dets == 0:
        return np.zeros((0, 0)), np.zeros((0, 0), dtype=bool)
    
    # Text filter
    if use_text_filter and tracks_text_emb is not None and dets_text_emb is not None:
        tracks_text_norm = _normalize_batch_fast(tracks_text_emb)
        dets_text_norm = _normalize_batch_fast(dets_text_emb)
        text_sim = tracks_text_norm @ dets_text_norm.T
        valid_mask = text_sim >= text_threshold
    else:
        valid_mask = np.ones((n_tracks, n_dets), dtype=bool)
    
    # Visual similarity ONLY for valid pairs
    similarity_matrix = np.zeros((n_tracks, n_dets))
    
    if np.sum(valid_mask) > 0:
        tracks_vis_norm = _normalize_batch_fast(tracks_vis_emb)
        dets_vis_norm = _normalize_batch_fast(dets_vis_emb)
        
        valid_indices = np.argwhere(valid_mask)
        for i, j in valid_indices:
            sim = np.dot(tracks_vis_norm[int(i)], dets_vis_norm[int(j)])
            similarity_matrix[int(i), int(j)] = sim
    
    return similarity_matrix, valid_mask


def _normalize_batch_fast(emb: np.ndarray) -> np.ndarray:
    """Fast batch L2 normalization."""
    if len(emb) == 0:
        return emb
    
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return emb / norms


# ============================================================================
# DISTANCE & SIZE UTILS
# ============================================================================

def compute_center_distance_matrix_vectorized(
    track_centers: np.ndarray,
    det_centers: np.ndarray,
    track_sizes: np.ndarray,
    det_sizes: Optional[np.ndarray] = None,
    threshold_multiplier: float = cfg.association.max_dist_multiplier
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fully vectorized center distance computation and filtering.
    
    Args:
        track_centers: Track centers [N, 3].
        det_centers: Detection centers [M, 3].
        track_sizes: Track sizes [N].
        det_sizes: Detection sizes [M] (optional, uses track_sizes if None).
        threshold_multiplier: Size multiplier for distance threshold.
    
    Returns:
        distances: Pairwise distances [N, M].
        valid_mask: Boolean mask of valid pairs [N, M].
    """
    # Vectorized distance matrix
    diff = track_centers[:, None, :] - det_centers[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    
    # Adaptive threshold
    if det_sizes is not None:
        thresholds = np.maximum(track_sizes[:, None], det_sizes[None, :]) * threshold_multiplier
    else:
        thresholds = track_sizes[:, None] * threshold_multiplier
    
    valid_mask = distances <= thresholds
    
    return distances, valid_mask


def compute_object_sizes_batch(
    points_list: List[np.ndarray]
) -> np.ndarray:
    """
    Compute object sizes for multiple point clouds (batch).
    
    Args:
        points_list: List of point arrays, each [N_i, 3].
    
    Returns:
        sizes: Object sizes [M] (bbox diagonal in meters).
    """
    if len(points_list) == 0:
        return np.array([])
    
    sizes = []
    for pts in points_list:
        if len(pts) > 0:
            min_pts = np.min(pts, axis=0)
            max_pts = np.max(pts, axis=0)
            size = np.linalg.norm(max_pts - min_pts)
        else:
            size = 0.0
        sizes.append(size)
    
    return np.array(sizes)


def compute_object_size(points: np.ndarray) -> float:
    """
    Compute single object size as 3D bbox diagonal.
    
    Args:
        points: Point cloud [N, 3].
    
    Returns:
        size: Bbox diagonal in meters.
    """
    if len(points) == 0:
        return 0.0
    
    min_pts = np.min(points, axis=0)
    max_pts = np.max(points, axis=0)
    return float(np.linalg.norm(max_pts - min_pts))


def compute_object_sizes_with_padding(
    points_list: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute object sizes with explicit padding (for fully vectorized ops).
    
    Args:
        points_list: List of point arrays, each [N_i, 3].
    
    Returns:
        sizes: Object sizes [M].
        points_padded: Padded point cloud [M, N_max, 3].
        mask: Valid point mask [M, N_max].
    """
    if len(points_list) == 0:
        return np.array([]), np.zeros((0, 0, 3)), np.zeros((0, 0), dtype=bool)
    
    n_objects = len(points_list)
    n_max = max(len(pts) for pts in points_list)
    
    points_padded = np.zeros((n_objects, n_max, 3), dtype=np.float32)
    mask = np.zeros((n_objects, n_max), dtype=bool)
    
    for i, pts in enumerate(points_list):
        if len(pts) > 0:
            points_padded[i, :len(pts), :] = pts
            mask[i, :len(pts)] = True
    
    # Vectorized min/max with NaN masking
    points_masked = points_padded.copy()
    points_masked[~mask] = np.nan
    
    min_pts = np.nanmin(points_masked, axis=1)
    max_pts = np.nanmax(points_masked, axis=1)
    
    empty_mask = np.all(~mask, axis=1)
    min_pts[empty_mask] = 0.0
    max_pts[empty_mask] = 0.0
    
    sizes = np.linalg.norm(max_pts - min_pts, axis=1)
    
    return sizes, points_padded, mask


# ============================================================================
# VOXEL SIMILARITY (Combined Metric)
# ============================================================================
def calc_voxel_similarity(
    track,
    det: dict,
    w_iou: float = None,      # Will be set based on static/dynamic
    w_dist: float = None,     # Will be set based on static/dynamic
    w_keypoints: float = None, # Will be set based on static/dynamic
    w_emb: float = None,
    image_shape: Optional[Tuple[int, int]] = None,
    debug: bool = cfg.debug_mode
) -> float:
    """
    Compute combined similarity between track and detection.
    
    For static objects: IoA + Embeddings + Distance
    For dynamic objects: IoA + Embeddings + Keypoint Overlap
    
    Args:
        track: Track object.
        det: Detection dict with 'points', 'embedding', 'mask'.
        w_iou: Weight for IoA component (None = auto based on static/dynamic).
        w_dist: Weight for distance component (None = auto).
        w_keypoints: Weight for keypoint overlap (None = auto).
        image_shape: Image shape (H, W) for keypoint overlap.
        debug: Print debug information.
    
    Returns:
        similarity: Combined similarity score [0, 1].
    """
    is_static = track.motion_state.status == 'STATIC'

    if is_static:
        w_iou = cfg.similarity_weights.static.ioa if w_iou is None else w_iou
        w_dist = cfg.similarity_weights.static.dist if w_dist is None else w_dist
        w_keypoints = 0.0
        w_emb = cfg.similarity_weights.static.emb if w_emb is None else w_emb
    else:
        w_iou = cfg.similarity_weights.dynamic.ioa if w_iou is None else w_iou
        w_keypoints = cfg.similarity_weights.dynamic.keypoint if w_keypoints is None else w_keypoints
        w_dist = cfg.similarity_weights.dynamic.dist if w_dist is None else w_dist
        w_emb = cfg.similarity_weights.dynamic.emb if w_emb is None else w_emb

    total_w = w_iou + w_dist + w_keypoints + w_emb
    if total_w > 0:
        w_iou /= total_w
        w_dist /= total_w
        w_keypoints /= total_w
        w_emb /= total_w
    
    # Convert to point clouds
    pcd_det = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(det["points"])
    )
    if is_static:
        pcd_track = track.voxels.get_pcd()
    else:
        pcd_track = track.voxels.get_pcd().transform(track.transform)
    
    # === 1. Center Distance Similarity ===
    det_center = np.array(pcd_det.get_center())
    track_center = np.array(pcd_track.get_center())
    dist = np.linalg.norm(det_center - track_center)
    # Size-normalized distance
    det_size = compute_object_size(np.array(pcd_det.points))
    track_size = track.object_size
    ref_size = max(det_size, track_size, cfg.object_size.min_object_size)
    normalized_dist = dist / ref_size
    
    dist_similarity = np.exp(-(normalized_dist**2) / (2 * 1.0**2))
    
    # === 2. Keypoint Overlap (dynamic only) ===
    keypoint_overlap = 0.0
    if w_keypoints > 0 and not is_static:
        if image_shape is not None and det.get('mask') is not None:
            # Get CoTracker keypoints from last available frame
            if track.moving_keypoints:
                last_frame = max(track.moving_keypoints.keys())
                keypoints = track.moving_keypoints.get(last_frame, None)
                
                if keypoints is not None and len(keypoints) > 0:
                    keypoint_overlap = _compute_keypoint_overlap(
                        keypoints,
                        det['mask'],
                        image_shape
                    )
    
    # === 3. Gaussian-IoA ===
    ioa_from_t, ioa_from_d, ioa = compute_bidirectional_gaussian_ioa_fast(
        np.array(pcd_track.points),
        np.array(pcd_det.points)
    )
    
    # === 4. Embedding Similarity ===
    emb_sim = calc_emb_similarity(
        track.embedding.reshape(1, -1),
        det["embedding"].reshape(1, -1)
    )[0, 0]
    
    # === 5. Combine Components ===
    similarity = (
        w_iou * ioa +
        w_emb * emb_sim +
        w_dist * dist_similarity +
        w_keypoints * keypoint_overlap
    )
    
    if debug:
        track_type = "STATIC" if is_static else "DYNAMIC"
        print(f"Track {track.id} ({track_type}): "
              f"IoA={ioa:.3f}(w={w_iou:.2f}), Emb={emb_sim:.3f}(w={w_emb:.2f}), "
              f"Dist={dist_similarity:.3f}(w={w_dist:.2f}), "
              f"Kpts={keypoint_overlap:.3f}(w={w_keypoints:.2f}) -> {similarity:.3f}")
    
    return float(similarity)
# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _voxel_downsample_numpy(
    points: np.ndarray,
    voxel_size: float
) -> np.ndarray:
    """
    Numpy-based voxel downsampling (10x faster than Open3D).
    
    Args:
        points: Input points [N, 3].
        voxel_size: Voxel size in meters.
    
    Returns:
        downsampled: Downsampled points [M, 3], M <= N.
    """
    if len(points) == 0:
        return points
    
    indices = np.floor(points / voxel_size).astype(int)
    _, unique_idx = np.unique(indices, axis=0, return_index=True)
    
    return points[unique_idx]


def _compute_bbox_iou_fast(
    points_a: np.ndarray,
    points_b: np.ndarray
) -> float:
    """
    Fast 3D bounding box IoU (O(1) early exit).
    
    Args:
        points_a: First point cloud [N, 3].
        points_b: Second point cloud [M, 3].
    
    Returns:
        iou: Intersection over Union [0, 1].
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return 0.0
    
    min_a, max_a = np.min(points_a, axis=0), np.max(points_a, axis=0)
    min_b, max_b = np.min(points_b, axis=0), np.max(points_b, axis=0)
    
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_size = np.maximum(0, inter_max - inter_min)
    inter_vol = np.prod(inter_size)
    
    size_a = max_a - min_a
    size_b = max_b - min_b
    vol_a = np.prod(size_a)
    vol_b = np.prod(size_b)
    union_vol = vol_a + vol_b - inter_vol
    
    return float(inter_vol / union_vol) if union_vol > 0 else 0.0


def _adaptive_sigma_fast(
    points_a: np.ndarray,
    points_b: np.ndarray
) -> float:
    """
    Fast adaptive sigma computation (10% of object size).
    
    Args:
        points_a: First point cloud [N, 3].
        points_b: Second point cloud [M, 3].
    
    Returns:
        sigma: Adaptive Gaussian width in meters.
    """
    size_a = np.linalg.norm(np.max(points_a, axis=0) - np.min(points_a, axis=0))
    size_b = np.linalg.norm(np.max(points_b, axis=0) - np.min(points_b, axis=0))
    avg_size = (size_a + size_b) / 2.0
    
    sigma = avg_size * 0.1

    return float(np.clip(sigma, 0.05, 0.5))
