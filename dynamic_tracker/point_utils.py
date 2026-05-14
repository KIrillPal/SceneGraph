"""
Point Cloud Utilities for SAM3 Tracker

Provides outlier removal, depth-to-point-cloud conversion, and DA3 loading.
"""

import cv2
import torch
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm.auto import tqdm
from typing import Tuple, List, Optional, Union
from config_loader import cfg


# ============================================================================
# OUTLIER REMOVAL
# ============================================================================

def statistical_outlier_removal(
    points: np.ndarray,
    k: int = None,
    std_ratio: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers using statistical analysis of nearest neighbor distances.
    
    Points whose average distance to k nearest neighbors exceeds
    (global_mean + std_ratio * global_std) are removed.
    
    Args:
        points: Input point cloud [N, 3].
        k: Number of nearest neighbors (recommended: 8-15).
        std_ratio: Threshold in standard deviations (recommended: 1.0-2.0).
    
    Returns:
        cleaned_points: Filtered point cloud [M, 3], M <= N.
        mask: Boolean array [N] indicating kept points.
    """
    k = cfg.point_cloud.sor_k if k is None else k
    std_ratio = cfg.point_cloud.sor_std_ratio if std_ratio is None else std_ratio

    # Not enough points for k neighbors
    if len(points) < k + 1:
        return points, np.ones(len(points), dtype=bool)
    
    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(points)
    
    # Query k+1 neighbors (including self)
    distances, _ = tree.query(points, k=k + 1)
    
    # Mean distance to k nearest neighbors (excluding self at index 0)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    
    # Global statistics
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    
    # Mask: keep points with distance below threshold
    threshold = global_mean + std_ratio * global_std
    mask = mean_distances < threshold
    
    return points[mask], mask


# ============================================================================
# DEPTH TO POINT CLOUD
# ============================================================================

def depth_to_point_cloud_vectorized(
    depth: Union[np.ndarray, torch.Tensor],
    intrinsics: Union[np.ndarray, torch.Tensor],
    extrinsics: Union[np.ndarray, torch.Tensor],
    device: Optional[str] = None
) -> Tuple[Union[np.ndarray, torch.Tensor], np.ndarray]:
    """
    Convert depth maps to 3D point clouds in world coordinates.
    
    Batch-processes multiple depth frames and projects them to 3D points
    using camera intrinsics and extrinsics. Supports both numpy and torch.
    
    Args:
        depth: Depth maps [N, H, W] in meters.
        intrinsics: Camera intrinsic matrices [N, 3, 3].
        extrinsics: World-to-camera poses [N, 3, 4].
        device: Target device ('cuda', 'cpu'). If None, uses input device.
    
    Returns:
        point_cloud_world: 3D points [N, H, W, 3] in world coordinates.
        pixel_coords: 2D pixel coordinates [N, H, W, 2] (u, v).
    """
    input_is_numpy = isinstance(depth, np.ndarray)
    
    # Convert to torch tensors if needed
    if input_is_numpy:
        depth_tensor = torch.tensor(depth, dtype=torch.float32)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
        extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float32)
    else:
        depth_tensor = depth.float()
        intrinsics_tensor = intrinsics.float()
        extrinsics_tensor = extrinsics.float()
    
    # Move to device
    if device is not None:
        depth_tensor = depth_tensor.to(device)
        intrinsics_tensor = intrinsics_tensor.to(device)
        extrinsics_tensor = extrinsics_tensor.to(device)
    
    # Get dimensions
    N, H, W = depth_tensor.shape
    device = depth_tensor.device
    
    # Generate pixel coordinate grid [N, H, W, 2]
    u = torch.arange(W, device=device).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v = torch.arange(H, device=device).float().view(1, H, 1, 1).expand(N, H, W, 1)
    pixel_coords = torch.cat([u, v], dim=-1)  # [N, H, W, 2]
    
    # Homogeneous pixel coordinates [N, H, W, 3]
    ones = torch.ones((N, H, W, 1), device=device)
    pixel_coords_homo = torch.cat([u, v, ones], dim=-1)
    
    # Project to camera space: K^(-1) * [u, v, 1]^T
    intrinsics_inv = torch.inverse(intrinsics_tensor)
    camera_coords = torch.einsum("nij,nhwj->nhwi", intrinsics_inv, pixel_coords_homo)
    
    # Scale by depth to get 3D camera coordinates
    camera_coords = camera_coords * depth_tensor.unsqueeze(-1)
    
    # Convert to homogeneous coordinates
    camera_coords_homo = torch.cat([camera_coords, ones], dim=-1)
    
    # Convert extrinsics from 3x4 to 4x4 homogeneous transformation
    extrinsics_4x4 = torch.zeros(N, 4, 4, device=device)
    extrinsics_4x4[:, :3, :4] = extrinsics_tensor
    extrinsics_4x4[:, 3, 3] = 1.0
    
    # Invert w2c to get c2w (camera-to-world)
    c2w = torch.inverse(extrinsics_4x4)
    
    # Transform to world coordinates
    world_coords_homo = torch.einsum("nij,nhwj->nhwi", c2w, camera_coords_homo)
    point_cloud_world = world_coords_homo[..., :3]
    
    # Convert back to numpy if input was numpy
    if input_is_numpy:
        point_cloud_world = point_cloud_world.cpu().numpy()
        pixel_coords = pixel_coords.cpu().numpy()
    
    return point_cloud_world, pixel_coords


# ============================================================================
# DA3 POINT CLOUD LOADING
# ============================================================================

def get_da3_pointclouds(
    depth_dir: str,
    extrinsics_file: str,
    N: int,
    conf_thresh_mul: float = None
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], int, int]:
    """
    Load DA3 depth frames and convert to filtered point clouds.
    
    Args:
        depth_dir: Directory with 'frame_0.npz', 'frame_1.npz', etc.
        extrinsics_file: Text file with one 4x4 pose per line.
        N: Number of frames to load.
        conf_thresh_mul: Confidence threshold multiplier (0-1).
    
    Returns:
        points_per_frame: List of N arrays [M_i, 3] with valid points.
        points_per_frame_masks: List of N boolean masks [H*W].
        pixels_per_frame: List of N arrays [M_i, 2] with pixel coordinates.
        h: Frame height (pixels).
        w: Frame width (pixels).
    
    Notes:
        - Each NPZ should contain: 'image', 'depth', 'conf', 'intrinsics'
        - Confidence threshold = conf_thresh_mul * mean(conf)
        - Statistical outlier removal applied after projection
    """
    conf_thresh_mul = (
        cfg.point_cloud.da3_conf_thresh_mul
        if conf_thresh_mul is None
        else conf_thresh_mul
    )

    # Load extrinsics (poses)
    extrinsics = []
    with open(extrinsics_file, "r") as f:
        all_poses = f.read().split("\n")
    
    for i in range(N):
        pose = all_poses[i]
        pose_np = np.array(pose.split(" ")).astype(float).reshape((4, 4))
        extrinsics.append(pose_np)
    
    points_per_frame = []
    points_per_frame_masks = []
    pixels_per_frame = []
    frame_intrinsics = []
    # Process each frame
    for i in tqdm(
        range(N),
        desc="Filtering DA3 point clouds",
        unit="frame",
        dynamic_ncols=True,
    ):
        data = np.load(f"{depth_dir}frame_{i}.npz")
        
        image = data["image"]      # [H, W, 3] uint8
        depth = data["depth"]      # [H, W] float32
        conf = data["conf"]        # [H, W] float32
        intrinsics = data["intrinsics"]  # [3, 3] float32
        frame_intrinsics.append(intrinsics)
        H_img, W_img, _ = image.shape
        h, w = depth.shape
        
        # Confidence threshold
        conf_thresh = conf.mean() * conf_thresh_mul
        
        # Reshape for batch processing (batch size = 1)
        depth_reshaped = depth[np.newaxis, :, :]        # [1, H, W]
        intrinsics_reshaped = intrinsics[np.newaxis, :, :]  # [1, 3, 3]
        
        # Convert c2w to w2c
        c2w = extrinsics[i]
        w2c = np.linalg.inv(c2w)
        extrinsics_ = w2c[:3, :]                        # [3, 4]
        extrinsics_reshaped = extrinsics_[np.newaxis, :, :]  # [1, 3, 4]
        
        # Depth to point cloud
        points_world, pixel_coords = depth_to_point_cloud_vectorized(
            depth_reshaped,
            intrinsics_reshaped,
            extrinsics_reshaped
        )
        
        # Reshape to [H*W, 3] and [H*W, 2]
        points_world = points_world.reshape(-1, 3).astype(np.float32, copy=False)
        pixel_coords = pixel_coords.reshape(-1, 2).astype(np.float32, copy=False)
        
        # Scale pixel coordinates to original image size
        pixel_coords[:, 0] *= W_img / w
        pixel_coords[:, 1] *= H_img / h
        
        # Statistical outlier removal
        _, points_mask = statistical_outlier_removal(points_world)
        
        # Confidence-based filtering
        conf_mask = conf > conf_thresh
        conf_mask_flat = conf_mask.flatten()
        
        # Combined mask
        mask = points_mask & conf_mask_flat
        
        # Store filtered data
        points_per_frame.append(points_world[mask])
        points_per_frame_masks.append(mask)
        pixels_per_frame.append(pixel_coords[mask])
    
    return points_per_frame, points_per_frame_masks, pixels_per_frame, h, w, extrinsics, frame_intrinsics
