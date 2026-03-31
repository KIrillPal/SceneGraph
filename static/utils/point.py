import cv2
import torch
import numpy as np
import os
import open3d as o3d
from scipy.spatial import cKDTree
import numpy as np


def statistical_outlier_removal(
    points: np.ndarray, k: int = 10, std_ratio: float = 1.5
):
    """
    Remove outliers using statistical analysis of nearest neighbor distances.

    Args:
        points: Input point cloud [N, 3].
        k: Number of nearest neighbors (recommended: 8-15).
        std_ratio: Threshold in standard deviations (recommended: 1.0-2.0).

    Returns:
        cleaned_points: Filtered point cloud [M, 3], M <= N.
        mask: Boolean array [N] indicating kept points.
    """
    # not enough points
    if len(points) < k + 1:
        return points, np.ones(len(points)).astype(bool)

    # Строим KD-дерево для быстрого поиска соседей
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k + 1)  # +1 потому что точка сама себе сосед

    # Среднее расстояние до k ближайших (исключая саму точку)
    mean_distances = np.mean(distances[:, 1:], axis=1)

    # Глобальные статистики
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)

    # Маска: оставить точки с расстоянием < порога
    mask = mean_distances < (global_mean + std_ratio * global_std)
    return points[mask], mask


def depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics, device=None):
    """
    depth: [N, H, W] numpy array or torch tensor
    intrinsics: [N, 3, 3] numpy array or torch tensor
    extrinsics: [N, 3, 4] (w2c) numpy array or torch tensor
    Returns: point_cloud_world: [N, H, W, 3] same type as input
    """
    input_is_numpy = False
    if isinstance(depth, np.ndarray):
        input_is_numpy = True

        depth_tensor = torch.tensor(depth, dtype=torch.float32)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
        extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float32)

        if device is not None:
            depth_tensor = depth_tensor.to(device)
            intrinsics_tensor = intrinsics_tensor.to(device)
            extrinsics_tensor = extrinsics_tensor.to(device)
    else:
        depth_tensor = depth
        intrinsics_tensor = intrinsics
        extrinsics_tensor = extrinsics

    if device is not None:
        depth_tensor = depth_tensor.to(device)
        intrinsics_tensor = intrinsics_tensor.to(device)
        extrinsics_tensor = extrinsics_tensor.to(device)

    # main logic

    N, H, W = depth_tensor.shape

    device = depth_tensor.device

    u = torch.arange(W, device=device).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v = torch.arange(H, device=device).float().view(1, H, 1, 1).expand(N, H, W, 1)
    ones = torch.ones((N, H, W, 1), device=device)
    pixel_coords = torch.cat([u, v, ones], dim=-1)
    # depth to camera
    intrinsics_inv = torch.inverse(intrinsics_tensor)  # [N, 3, 3]
    camera_coords = torch.einsum("nij,nhwj->nhwi", intrinsics_inv, pixel_coords)
    camera_coords = camera_coords * depth_tensor.unsqueeze(-1)
    camera_coords_homo = torch.cat([camera_coords, ones], dim=-1)

    extrinsics_4x4 = torch.zeros(N, 4, 4, device=device)
    extrinsics_4x4[:, :3, :4] = extrinsics_tensor
    extrinsics_4x4[:, 3, 3] = 1.0
    # camera to world
    c2w = torch.inverse(extrinsics_4x4)
    world_coords_homo = torch.einsum("nij,nhwj->nhwi", c2w, camera_coords_homo)
    point_cloud_world = world_coords_homo[..., :3]

    if input_is_numpy:
        point_cloud_world = point_cloud_world.cpu().numpy()

    return point_cloud_world


def get_da3_pointclouds(depth_dir, extrinsics_file, N, conf_thresh_mul=0.5):
    """
    Load DA3 depth frames and convert to filtered point clouds.

    Args:
        depth_dir: Directory with 'frame_0.npz', 'frame_1.npz', etc.
        extrinsics_file: Text file with one 4x4 pose per line.
        N: Number of frames to load.
        conf_thresh_mul: confidence threshold (0-1)

    Returns:
        points_per_frame: List of N arrays [M_i, 3] with valid points.
        points_per_frame_masks: List of N boolean masks.
        h: Frame height (pixels).
        w: Frame width (pixels).

    Notes:
        - Each NPZ should contain: 'image', 'depth', 'conf', 'intrinsics'
        - Confidence threshold = conf_thresh_nul * mean(conf)
        - Statistical outlier removal applied after projection
    """
    extrinsics = []
    # pose extraction
    with open(extrinsics_file, "r") as f:
        all_poses = f.read().split("\n")
    for i in range(N):
        pose = all_poses[i]
        pose_np = np.array(pose.split(" ")).astype(float).reshape((4, 4))
        extrinsics.append(pose_np)
    points_per_frame = []
    points_per_frame_masks = []
    for i in range(N):
        data = np.load(depth_dir + "frame_%d.npz" % (i))
        image = data["image"]  # [H, W, 3] uint8
        depth = data["depth"]  # [H, W] float32
        h, w = depth.shape
        conf = data["conf"]  # [H, W] float32
        intrinsics = data["intrinsics"]  # [3, 3] float32
        # conf_thresh
        conf_thresh = conf.mean() * conf_thresh_mul
        depth_reshaped = depth[np.newaxis, :, :]  # [1, H, W]
        intrinsics_reshaped = intrinsics[np.newaxis, :, :]  # [1, 3, 3]
        c2w = extrinsics[i]
        w2c = np.linalg.inv(c2w)
        extrinsics_ = w2c[:3, :]  # [3, 4]
        extrinsics_reshaped = extrinsics_[np.newaxis, :, :]  # [1, 3, 4]
        points_world = depth_to_point_cloud_vectorized(
            depth_reshaped, intrinsics_reshaped, extrinsics_reshaped
        )
        points_world = points_world.reshape(-1, 3).astype(np.float32, copy=False)
        # filtering
        points, points_mask = statistical_outlier_removal(points_world)
        # conf-based filtering
        points_mask2 = conf > conf_thresh
        mask = points_mask * points_mask2.flatten()
        points_per_frame.append(points_world[mask])
        points_per_frame_masks.append(mask)
    return points_per_frame, points_per_frame_masks, h, w
