"""
3D Tracking Core for SAM3 Tracker

Provides VoxelMap, Track, SmartUpdateRules, and Simple3DTracker.
Visibility update EVERY frame with camera motion compensation using EXTRINSICS.
"""

import cv2
import os
import torch
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from point_utils import statistical_outlier_removal
from metric_utils import (
    compute_bidirectional_gaussian_ioa_fast,
    calc_voxel_similarity,
    compute_object_size
)
from config_loader import cfg
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from cotracker.predictor import CoTrackerOnlinePredictor
from scipy.optimize import linear_sum_assignment


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _project_3d_points_to_2d(
    points_3d: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Batch-project 3D world points to 2D image coordinates."""
    if len(points_3d) == 0:
        return np.zeros((0, 2)), np.array([], dtype=bool)

    if extrinsics.shape == (3, 4):
        ext_4x4 = np.eye(4)
        ext_4x4[:3, :] = extrinsics
    else:
        ext_4x4 = extrinsics

    pts_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    pts_cam = (ext_4x4 @ pts_homo.T).T[:, :3]
    valid_front = pts_cam[:, 2] > 1e-6

    final_2d = np.full((len(points_3d), 2), -1.0)
    if np.any(valid_front):
        pts_img = intrinsics @ pts_cam[valid_front].T
        final_2d[valid_front] = (pts_img[:2] / pts_img[2]).T

    return final_2d, valid_front


def _ensure_embedding_2d(emb_array: np.ndarray, expected_dim: int = cfg.embeddings.text_embedding_dim) -> np.ndarray:
    """Ensure embedding array is 2D (N, D)."""
    if len(emb_array) == 0:
        return np.zeros((0, expected_dim))
    
    if emb_array.ndim == 3:
        emb_array = emb_array.squeeze(axis=1)
    elif emb_array.ndim == 1:
        if emb_array.shape[0] == expected_dim:
            emb_array = emb_array.reshape(1, -1)
    
    return emb_array


def _infer_embedding_dim(default_dim: int, *emb_arrays: np.ndarray) -> int:
    for emb_array in emb_arrays:
        emb_array = np.asarray(emb_array)
        if emb_array.size == 0:
            continue
        if emb_array.ndim == 1:
            return emb_array.shape[0]
        return emb_array.shape[-1]
    return default_dim


def _compute_filter_matrices(
    tracks: List,
    detections: List[Dict],
    text_threshold: float,
    size_multiplier: float = 1.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute distance and text filter matrices for association."""
    if len(tracks) == 0 or len(detections) == 0:
        return (
            np.zeros((0, 0), dtype=bool),
            np.zeros((0, 0), dtype=bool),
            np.zeros((0, 0), dtype=bool)
        )
    
    track_centers = np.array([t.center_3d for t in tracks])
    track_sizes = np.array([t.object_size for t in tracks])
    det_centers = np.array([
        np.mean(d['points'], axis=0) if len(d['points']) > 0 else np.zeros(3)
        for d in detections
    ])
    det_sizes = np.array([
        compute_object_size(d['points']) if len(d['points']) > 0 else 0.0
        for d in detections
    ])
    
    diff = track_centers[:, None, :] - det_centers[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    size_thresholds = np.maximum(track_sizes[:, None], det_sizes[None, :]) * size_multiplier
    distance_filter_mask = distances <= size_thresholds
    
    track_text_embs = np.array([t.text_embedding[0] for t in tracks])
    det_text_embs = np.array([d["text_embedding"][0] for d in detections])
    
    track_text_embs = _ensure_embedding_2d(track_text_embs)
    det_text_embs = _ensure_embedding_2d(det_text_embs)
    
    track_text_norm = track_text_embs / (np.linalg.norm(track_text_embs, axis=1, keepdims=True) + 1e-8)
    det_text_norm = det_text_embs / (np.linalg.norm(det_text_embs, axis=1, keepdims=True) + 1e-8)
    text_sim_matrix = track_text_norm @ det_text_norm.T
    text_filter_mask = text_sim_matrix >= text_threshold
    
    combined_filter_mask = distance_filter_mask & text_filter_mask
    
    n_pass = np.sum(combined_filter_mask)
    n_total = len(tracks) * len(detections)
    print(f"Filter: {n_pass}/{n_total} pairs pass ({100*n_pass/(n_total+1e-6):.1f}%)")
    
    return distance_filter_mask, text_filter_mask, combined_filter_mask


def _get_good_points(
    points: np.ndarray,
    visibilities: np.ndarray,
    H: int,
    W: int,
    thresh: float = 0.7
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter points by visibility and image bounds."""
    out_of_border = (
        (points[:, 0] < 0) | (points[:, 0] >= W) |
        (points[:, 1] < 0) | (points[:, 1] >= H)
    )
    is_lost = (visibilities < thresh) | out_of_border
    return points[~is_lost], ~is_lost


def _get_3d_for_keypoints(
    keypoints: np.ndarray,
    points_2d: np.ndarray,
    points_3d: np.ndarray,
    max_distance: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Map 2D keypoints to 3D coordinates using nearest neighbor search."""
    if len(keypoints) == 0 or len(points_2d) == 0:
        return np.zeros((0, 3)), np.array([], dtype=bool)
    
    tree = cKDTree(points_2d)
    distances, indices = tree.query(keypoints, k=1)
    valid_mask = distances < max_distance
    
    keypoints_3d = np.zeros((len(keypoints), 3))
    keypoints_3d[valid_mask] = points_3d[indices[valid_mask]]
    
    return keypoints_3d, valid_mask


def _check_text_filter_only(
    track_text_emb: np.ndarray,
    det_text_emb: np.ndarray,
    threshold: float
) -> bool:
    """Check only text embedding similarity (for visual bypass)."""
    track_norm = track_text_emb / (np.linalg.norm(track_text_emb) + 1e-8)
    det_norm = det_text_emb / (np.linalg.norm(det_text_emb) + 1e-8)
    text_sim = np.dot(track_norm, det_norm.T)[0, 0]
    return text_sim >= threshold


def _project_3d_to_2d(
    point_3d: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray
) -> np.ndarray:
    """
    Project 3D world point to 2D image coordinates.
    
    Args:
        point_3d: 3D point in world coordinates [3].
        intrinsics: Camera intrinsics [3, 3].
        extrinsics: World-to-camera extrinsics [3, 4] or [4, 4].
    
    Returns:
        point_2d: 2D image coordinates [2] or None if behind camera.
    """
    # Convert extrinsics to 4x4 if needed
    if extrinsics.shape == (3, 4):
        extrinsics_4x4 = np.eye(4)
        extrinsics_4x4[:3, :] = extrinsics
    else:
        extrinsics_4x4 = extrinsics
    
    # Transform to camera coordinates
    point_3d_homo = np.append(point_3d, 1.0)
    point_cam = extrinsics_4x4 @ point_3d_homo
    point_cam = point_cam[:3]
    
    # Check if point is behind camera
    if point_cam[2] <= 0:
        return None
    
    # Project to image plane
    point_img_homo = intrinsics @ point_cam
    point_2d = point_img_homo[:2] / point_img_homo[2]
    
    return point_2d


def _is_position_in_frame(
    pos_2d: np.ndarray,
    H: int,
    W: int,
    margin: int = 0
) -> bool:
    """Check if 2D position is within frame bounds."""
    if pos_2d is None:
        return False
    x, y = pos_2d
    return (x >= -margin and x < W + margin and 
            y >= -margin and y < H + margin)


# ============================================================================
# CLASSES
# ============================================================================

class ObjectState:
    """Manages dynamic/static state detection for a track."""
    
    def __init__(self):
        self.status = "STATIC"
        self.v_buffer: List[float] = []
        self.size_buffer: List[float] = []
        self.v_avg = 0.0
        self.size_avg = 0.2
    
    def update(
        self,
        prev_pcd: o3d.geometry.PointCloud,
        curr_pcd: o3d.geometry.PointCloud,
        prev_frame: int,
        curr_frame: int
    ) -> str:
        """Update motion state based on velocity."""
        center_dist = np.linalg.norm(
            np.array(curr_pcd.get_center()) - np.array(prev_pcd.get_center())
        )
        size = compute_object_size(np.array(curr_pcd.points))
        
        frame_delta = max(1, curr_frame - prev_frame)
        velocity = center_dist / frame_delta
        
        self.v_buffer.append(velocity)
        self.size_buffer.append(size)
        
        if len(self.v_buffer) > cfg.motion.velocity_buffer_size:
            self.v_buffer.pop(0)
            self.size_buffer.pop(0)
        
        self.v_avg = np.mean(self.v_buffer)
        self.size_avg = max(0.2, np.mean(self.size_buffer))
        
        normalized_velocity = self.v_avg / self.size_avg
        
        if self.status == "STATIC":
            if normalized_velocity > cfg.motion.t_start:
                self.status = "MOVING"
        else:
            if normalized_velocity < cfg.motion.t_stop:
                self.status = "STATIC"
        
        return self.status


class VoxelMap:
    """Voxel-based 3D map with EMA updates and age-based expiration."""
    
    def __init__(
        self,
        voxel_size: float = cfg.voxel_map.voxel_size,
        alpha: float = cfg.voxel_map.alpha,
        max_age: int = cfg.voxel_map.max_age
    ):
        self.voxel_size = voxel_size
        self.alpha = alpha
        self.max_age = max_age
        self.voxels: Dict[Tuple[int, int, int], List[np.ndarray, int]] = {}
        self.current_frame = -1
    
    def add_points(self, points: np.ndarray) -> None:
        """Add 3D points to voxel map with EMA updates."""
        self.current_frame += 1
        
        if len(points) == 0:
            self._expire()
            return
        
        indices = np.floor(points / self.voxel_size).astype(int)
        
        for i, idx in enumerate(indices):
            idx_tuple = tuple(idx)
            new_pt = points[i]
            
            if idx_tuple in self.voxels:
                current_data = self.voxels[idx_tuple]
                updated_pt = (1 - self.alpha) * current_data[0] + self.alpha * new_pt
                self.voxels[idx_tuple] = [updated_pt, self.current_frame]
            else:
                self.voxels[idx_tuple] = [new_pt, self.current_frame]
        
        self._expire()
    
    def _expire(self) -> None:
        """Remove voxels not updated for more than max_age frames."""
        expired_keys = [
            k for k, v in self.voxels.items()
            if (self.current_frame - v[1]) > self.max_age
        ]
        for k in expired_keys:
            del self.voxels[k]
    
    def get_pcd(self) -> o3d.geometry.PointCloud:
        """Get PointCloud from current voxels."""
        if not self.voxels:
            return o3d.geometry.PointCloud()
        
        points, _ = statistical_outlier_removal(
            np.array([v[0] for v in self.voxels.values()])
        )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd


class MedianEMAFilter:
    """Combined median filter + EMA for robust smoothing."""
    
    def __init__(self, ema_alpha: float = 0.85, median_window: int = 5):
        self.ema_alpha = ema_alpha
        self.median_window = median_window
        self.history = deque(maxlen=median_window)
        self.ema_value: Optional[np.ndarray] = None
    
    def update(self, value: np.ndarray) -> np.ndarray:
        """Update filter with new value."""
        self.history.append(value)
        
        if len(self.history) >= 3:
            median_filtered = np.median(list(self.history), axis=0)
        else:
            median_filtered = value
        
        if self.ema_value is None:
            self.ema_value = median_filtered
        else:
            self.ema_value = (
                self.ema_alpha * median_filtered +
                (1 - self.ema_alpha) * self.ema_value
            )
        
        return self.ema_value


class Track:
    """Single object track with voxels, embeddings, and state management."""
    
    def __init__(
        self,
        track_id: int,
        detection: Dict[str, Any],
        frame_idx: int,
        is_dynamic: bool = False
    ):
        self.id = track_id
        self.sam_id = detection["sam_id"]
        self.voxels = VoxelMap()
        self.is_dynamic = is_dynamic
        
        # Embeddings with smoothing
        self.text_embedding_smoother = MedianEMAFilter(ema_alpha=0.85, median_window=5)
        self.text_embedding = self.text_embedding_smoother.update(
            detection["text_embedding"]
        )
        
        self.embedding_smoother = MedianEMAFilter(ema_alpha=0.95, median_window=10)
        self.embedding = self.embedding_smoother.update(detection["embedding"])
        
        # Initial points
        points = detection["points"]
        self.voxels.add_points(points)
        pcd = self.voxels.get_pcd()
        self.center_3d = np.array(pcd.get_center())
        
        # Object size tracking
        self.detection_size = compute_object_size(points)
        self.voxel_size = self.detection_size
        self.size_history = deque(maxlen=10)
        self.size_history.append(self.detection_size)
        self.use_detection_size = (
            cfg.object_size.use_detection_size_for_moving and is_dynamic
        )
        
        # State machine
        self.state = "tentative"
        self.hits = 1
        self.min_hits_to_activate = cfg.track_lifecycle.min_hits_to_activate
        self.time_since_update = 0
        
        # Per-frame data
        self.masks: Dict[int, np.ndarray] = {frame_idx: detection["mask"]}
        self.cls: Dict[int, str] = {frame_idx: detection["cls"]}
        self.embeddings: Dict[int, np.ndarray] = {
            frame_idx: np.asarray(detection["embedding"], dtype=np.float32).reshape(-1)
        }
        self.keypoints: Dict[int, np.ndarray] = {frame_idx: detection.get("keypoints")}
        self.moving_keypoints: Dict[int, np.ndarray] = {frame_idx: detection.get("keypoints")}
        
        # Motion
        self.motion_state = ObjectState()
        if self.is_dynamic:
            self.motion_state.status = 'MOVING'
        self.transform = np.eye(4)
        self.update_rules = SmartUpdateRules()
        
        # === VISIBILITY (only for dynamic objects) ===
        self.visibility_state = "VISIBLE"
        self.cotracker_is_visible = True
        self.predicted_pos_2d = None
        self.last_predicted_pos_2d = None
        self.frames_without_cotracker = 0
        self.stable_3d_keypoints: Optional[np.ndarray] = None
        self.last_3d_keypoints: Optional[np.ndarray] = None
    
    @property
    def object_size(self) -> float:
        """Get object size based on motion state."""
        if self.use_detection_size and self.motion_state.status == "MOVING":
            return max(self.detection_size, cfg.object_size.min_object_size)
        else:
            return max(self.voxel_size, cfg.object_size.min_object_size)
    
    def update_size(self, detection: Dict[str, Any], frame_idx: int):
        """Update object size from new detection."""
        det_size = compute_object_size(detection["points"])
        
        if self.use_detection_size and self.motion_state.status == "MOVING":
            max_allowed_size = self.detection_size * cfg.object_size.max_size_increase_factor
            det_size = min(det_size, max_allowed_size)
            
            alpha = cfg.object_size.size_smoothing_alpha
            self.detection_size = (
                alpha * det_size + 
                (1 - alpha) * self.detection_size
            )
            
            self.size_history.append(self.detection_size)
            self.detection_size = np.mean(list(self.size_history))
        else:
            pcd = self.voxels.get_pcd()
            if len(pcd.points) > 0:
                self.voxel_size = compute_object_size(np.array(pcd.points))
    
    def update_visibility_from_cotracker(
        self,
        track_points: np.ndarray,
        track_visibilities: np.ndarray,
        frame_id: int,
        H: int,
        W: int
    ):
        """Update visibility from CoTracker output with drift rejection."""
        if not self.is_dynamic:
            self.visibility_state = "VISIBLE"
            self.cotracker_is_visible = True
            return
        
        if len(track_points) == 0:
            self.visibility_state = "OCCLUDED"
            self.cotracker_is_visible = False
            return
        
        in_frame_x = (track_points[:, 0] >= 0) & (track_points[:, 0] < W)
        in_frame_y = (track_points[:, 1] >= 0) & (track_points[:, 1] < H)
        in_frame = in_frame_x & in_frame_y
        visible = track_visibilities > cfg.cotracker.visibility_thresh
        
        valid_points = np.sum(in_frame & visible)
        total_points = len(track_points)
        low_vis_points = np.sum(~visible & in_frame)
        outside_points = np.sum(~in_frame)
        
        if np.sum(in_frame) > 0:
            current_center = np.mean(track_points[in_frame], axis=0)
        else:
            current_center = np.mean(track_points, axis=0)
        
        max_drift = getattr(cfg.cotracker, "max_drift_px", 150.0)
        if self.last_predicted_pos_2d is not None:
            drift = np.linalg.norm(current_center - self.last_predicted_pos_2d)
            if drift > max_drift:
                self.visibility_state = "OCCLUDED"
                self.cotracker_is_visible = True
                self.predicted_pos_2d = current_center
                self.last_predicted_pos_2d = current_center.copy()
                return
        
        self.predicted_pos_2d = current_center
        self.last_predicted_pos_2d = current_center.copy()
        
        if valid_points >= max(cfg.cotracker.min_keypoints_visible, total_points * 0.3):
            self.visibility_state = "VISIBLE"
            self.cotracker_is_visible = True
        elif low_vis_points >= max(2, total_points * 0.3):
            self.visibility_state = "OCCLUDED"
            self.cotracker_is_visible = True
        elif outside_points >= total_points * 0.5:
            self.visibility_state = "OUTSIDE_FRAME"
            self.cotracker_is_visible = False
        else:
            self.visibility_state = "OCCLUDED"
            self.cotracker_is_visible = True
    
    def update_moving_keypoints_with_extrinsics(
        self,
        frame_id: int,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        H: int,
        W: int
    ) -> None:
        """Project last valid 3D CoTracker keypoints into the current frame."""
        if self.last_3d_keypoints is None or len(self.last_3d_keypoints) == 0:
            return

        keypoints_2d, valid_front = _project_3d_points_to_2d(
            self.last_3d_keypoints, intrinsics, extrinsics
        )
        if len(keypoints_2d) == 0:
            return

        in_frame = (
            valid_front &
            (keypoints_2d[:, 0] >= 0) & (keypoints_2d[:, 0] < W) &
            (keypoints_2d[:, 1] >= 0) & (keypoints_2d[:, 1] < H)
        )
        max_drift = getattr(cfg.cotracker, "max_drift_px", 150.0)
        if self.last_predicted_pos_2d is not None:
            drift_ok = np.linalg.norm(keypoints_2d - self.last_predicted_pos_2d, axis=1) < max_drift
            in_frame = in_frame & drift_ok
        elif np.sum(in_frame) > 0:
            self.last_predicted_pos_2d = np.mean(keypoints_2d[in_frame], axis=0)

        if np.sum(in_frame) > 0:
            self.moving_keypoints[frame_id] = keypoints_2d[in_frame]
            self.predicted_pos_2d = np.mean(self.moving_keypoints[frame_id], axis=0)

    def update_visibility_with_extrinsics_compensation(
        self,
        frame_id: int,
        H: int,
        W: int,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray
    ):
        """Update visibility for tracks absent from CoTracker using camera pose."""
        if not self.is_dynamic:
            self.visibility_state = "VISIBLE"
            self.cotracker_is_visible = True
            return

        self.update_moving_keypoints_with_extrinsics(frame_id, intrinsics, extrinsics, H, W)

        if frame_id not in self.moving_keypoints or len(self.moving_keypoints[frame_id]) == 0:
            if self.center_3d is not None and len(self.center_3d) > 0:
                center_keypoint, valid = _project_3d_points_to_2d(
                    self.center_3d.reshape(1, -1), intrinsics, extrinsics
                )
                if len(valid) > 0 and valid[0]:
                    max_drift = getattr(cfg.cotracker, "max_drift_px", 150.0)
                    if self.last_predicted_pos_2d is not None:
                        drift = np.linalg.norm(center_keypoint[0] - self.last_predicted_pos_2d)
                        if drift < max_drift:
                            self.moving_keypoints[frame_id] = center_keypoint[valid]
                    else:
                        self.moving_keypoints[frame_id] = center_keypoint[valid]
                        self.last_predicted_pos_2d = center_keypoint[0].copy()

        if frame_id in self.moving_keypoints and len(self.moving_keypoints[frame_id]) > 0:
            current_keypoints = self.moving_keypoints[frame_id]
            in_frame = (
                (current_keypoints[:, 0] >= 0) & (current_keypoints[:, 0] < W) &
                (current_keypoints[:, 1] >= 0) & (current_keypoints[:, 1] < H)
            )
            inside_count = int(np.sum(in_frame))
            total_count = len(current_keypoints)

            self.predicted_pos_2d = (
                np.mean(current_keypoints[in_frame], axis=0)
                if inside_count > 0
                else np.mean(current_keypoints, axis=0)
            )
            self.last_predicted_pos_2d = self.predicted_pos_2d.copy()

            if inside_count >= max(cfg.cotracker.min_keypoints_visible, total_count * 0.3):
                self.visibility_state = "VISIBLE"
                self.cotracker_is_visible = True
            elif inside_count >= max(2, total_count * 0.15):
                self.visibility_state = "OCCLUDED"
                self.cotracker_is_visible = True
            else:
                self.visibility_state = "OUTSIDE_FRAME"
                self.cotracker_is_visible = False
        else:
            self.visibility_state = "OCCLUDED"
            self.cotracker_is_visible = False
            self.frames_without_cotracker += 1
    
    def should_allow_association(self, visual_sim: float, emb_threshold: float) -> bool:
        """
        Check if track should be allowed for association.
        ONLY applies to dynamic objects - static always allowed.
        """
        if not self.is_dynamic:
            return True
        
        if self.visibility_state == "VISIBLE":
            return True
        elif self.visibility_state == "OCCLUDED":
            return True
        elif self.visibility_state == "OUTSIDE_FRAME":
            if visual_sim > emb_threshold:
                return True
            return False
        
        return True
    
    def update(self, detection: Dict[str, Any], frame_idx: int) -> None:
        """Update track with new detection."""
        print(f"{detection['cls']} track {self.id}")
        
        points = detection["points"]
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(points)
        curr_pcd = self.voxels.get_pcd()
        
        if self.motion_state.status == 'STATIC':
            ioa_from_t, ioa_from_d, ioa = compute_bidirectional_gaussian_ioa_fast(
                np.array(curr_pcd.points), np.array(new_pcd.points)
            )
        else:
            curr_pcd_transformed = curr_pcd.transform(self.transform)
            ioa_from_t, ioa_from_d, ioa = compute_bidirectional_gaussian_ioa_fast(
                np.array(curr_pcd_transformed.points),
                np.array(new_pcd.points)
            )
        
        update_flags = self.update_rules.decide_update(ioa_from_t, ioa_from_d)
        
        prev_frame = max(self.masks.keys())
        if self.is_dynamic:
            self.motion_state.update(curr_pcd, new_pcd, prev_frame, frame_idx)
        
        self.update_size(detection, frame_idx)
        self.center_3d = np.mean(points, axis=0)
        
        if update_flags["update_embeddings"]:
            self.embedding = self.embedding_smoother.update(detection["embedding"])
            self.text_embedding = self.text_embedding_smoother.update(
                detection["text_embedding"]
            )
        
        if update_flags["update_voxels"]:
            if update_flags["use_det_voxels"] and self.motion_state.status == "MOVING":
                self.voxels = VoxelMap(max_age=cfg.voxel_map.max_age_moving)
            self.voxels.add_points(points)
        elif update_flags["use_det_voxels"] and self.motion_state.status == "MOVING":
            self.voxels = VoxelMap(max_age=cfg.voxel_map.max_age_moving)
            self.voxels.add_points(points)
        
        if update_flags["re_init"]:
            print(f"Re-init track {self.id}")
            max_age = (cfg.voxel_map.max_age_moving
                      if self.motion_state.status == "MOVING"
                      else cfg.voxel_map.max_age)
            self.voxels = VoxelMap(max_age=max_age)
            self.voxels.add_points(points)
        
        self.hits += 1
        
        if self.state == "tentative" and self.hits >= self.min_hits_to_activate:
            self.state = "active"
        
        if self.motion_state.status == 'STATIC':
            self.transform = np.eye(4)
        
        self.time_since_update = 0
        self.visibility_state = "VISIBLE"
        self.cotracker_is_visible = True
        
        self.masks[frame_idx] = detection["mask"]
        self.cls[frame_idx] = detection["cls"]
        self.embeddings[frame_idx] = np.asarray(
            detection["embedding"], dtype=np.float32
        ).reshape(-1)
        
        if detection.get('keypoints') is not None:
            self.keypoints[frame_idx] = detection['keypoints']


class SmartUpdateRules:
    """Track update rules based on bidirectional IoA."""
    
    def __init__(self):
        self.high_overlap_threshold = 0.7
        self.medium_overlap_threshold = 0.4
        self.re_init_threshold = 0.1
    
    def decide_update(
        self,
        ioa_track_to_det: float,
        ioa_det_to_track: float
    ) -> Dict[str, Any]:
        """Decide what to update based on bidirectional IoA."""
        update = {
            "update_embeddings": False,
            "update_voxels": False,
            "re_init": False,
            "use_det_voxels": False,
            "reason": "",
        }
        
        if (ioa_track_to_det > self.high_overlap_threshold and
            ioa_det_to_track > self.high_overlap_threshold):
            update.update({
                "update_embeddings": True,
                "update_voxels": True,
                "use_det_voxels": True,
                "reason": "full_overlap"
            })
        elif (ioa_det_to_track > self.high_overlap_threshold and
              ioa_track_to_det < self.medium_overlap_threshold):
            update.update({
                "update_embeddings": True,
                "update_voxels": True,
                "use_det_voxels": True,
                "reason": "det_larger_than_track"
            })
        elif (ioa_track_to_det > self.high_overlap_threshold and
              ioa_det_to_track < self.medium_overlap_threshold):
            update.update({
                "update_embeddings": False,
                "update_voxels": True,
                "reason": "partial_visibility"
            })
        elif (ioa_track_to_det > self.medium_overlap_threshold or
              ioa_det_to_track > self.medium_overlap_threshold):
            update.update({
                "update_embeddings": False,
                "update_voxels": True,
                "reason": "medium_overlap"
            })
        elif (ioa_track_to_det < self.re_init_threshold or
              ioa_det_to_track < self.re_init_threshold):
            update.update({
                "re_init": True,
                "reason": "re_init_needed"
            })
        else:
            update['use_det_voxels'] = True
            update["reason"] = "low_overlap"
        
        return update


class Simple3DTracker:
    """Multi-object 3D tracker with hysteresis state management."""
    
    def __init__(
        self,
        video: torch.Tensor,
        device: torch.device,
        dynamic_classes_list: List[str],
        intrinsics_list: Optional[List[np.ndarray]] = None,
        extrinsics_list: Optional[List[np.ndarray]] = None
    ):
        self.tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.next_id = 0
        
        self.ass_threshold = cfg.association.ass_threshold
        self.lost_ass_threshold = cfg.association.lost_ass_threshold
        self.tentative_ass_threshold = cfg.association.tentative_ass_threshold
        self.text_emb_threshold = cfg.embeddings.text_threshold
        self.emb_threshold = cfg.embeddings.visual_threshold
        self.max_dist_multiplier = cfg.association.max_dist_multiplier
        
        self.max_tentative_age = cfg.track_lifecycle.max_tentative_age
        self.max_active_frames = cfg.track_lifecycle.max_active_frames
        self.dynamic_classes_list = [str(cls).lower() for cls in dynamic_classes_list]
        
        self.device = device
        cotracker_checkpoint = os.environ.get(
            "COTRACKER_CHECKPOINT",
            cfg.cotracker.checkpoint_path,
        )
        self.point_tracker = CoTrackerOnlinePredictor(cotracker_checkpoint).to(device)
        self.cotracker_window_len = cfg.cotracker.window_len
        self.video = video
        _,  _, self.H, self.W, = video.size()
        
        # === CAMERA EXTRINSICS ===
        self.intrinsics_list = intrinsics_list  # List of [3, 3]  intrinsics per frame
        self.extrinsics_list = extrinsics_list  # List of [3, 4] or [4, 4] extrinsics per frame
    
    def _get_current_extrinsics(self, frame_id: int) -> Optional[np.ndarray]:
        """Get camera extrinsics for current frame."""
        if self.extrinsics_list is None or self.intrinsics_list is None:
            return None, None
        if frame_id < 0 or frame_id >= len(self.extrinsics_list):
            return None, None
        return self.extrinsics_list[frame_id], self.intrinsics_list[frame_id]
    
    def _compute_transform_for_moving_tracks(
        self,
        frame_id: int,
        points_2d: np.ndarray,
        points_3d: np.ndarray
    ) -> None:
        """
        Compute rigid transform for moving tracks using CoTracker.
        Source/target 3D points are aligned with the selected query keypoints.
        """
        queries = []
        queries_by_object = []
        start_frame = max(0, frame_id - self.cotracker_window_len + 1)
        fin_frame = frame_id + 1
        
        all_tracks_to_track = self.tracks + self.lost_tracks
        use_moving_keypoints = getattr(cfg.cotracker, "use_moving_keypoints", True)
        
        for track in all_tracks_to_track:
            if not track.is_dynamic:
                continue
            
            if (track.motion_state.status == "MOVING" and
                track.time_since_update < self.cotracker_window_len):
                last_frame = max(track.keypoints.keys()) if track.keypoints else -1
                chosen_keypoints = None

                if (use_moving_keypoints and last_frame >= 0 and
                    last_frame in track.moving_keypoints):
                    moving_points = track.moving_keypoints[last_frame]
                    if moving_points is not None:
                        in_bounds = (
                            (moving_points[:, 0] >= 0) & (moving_points[:, 0] < self.W) &
                            (moving_points[:, 1] >= 0) & (moving_points[:, 1] < self.H)
                        )
                        if np.sum(in_bounds) >= cfg.cotracker.min_keypoints_for_transform:
                            chosen_keypoints = moving_points[in_bounds]

                if chosen_keypoints is None and last_frame >= 0 and last_frame in track.keypoints:
                    static_points = track.keypoints[last_frame]
                    if static_points is not None:
                        chosen_keypoints = static_points

                if chosen_keypoints is not None and len(chosen_keypoints) > 0:
                    local_frame = last_frame - start_frame
                    if 0 <= local_frame < self.cotracker_window_len:
                        src_3d_keypoints, _ = _get_3d_for_keypoints(
                            chosen_keypoints, points_2d, points_3d
                        )
                        track.src_3d_keypoints = src_3d_keypoints

                        obj_queries = torch.tensor([
                            [local_frame, q[0], q[1]]
                            for q in chosen_keypoints
                        ])
                        queries.append(obj_queries)
                        queries_by_object.append(
                            torch.ones(len(obj_queries), dtype=int) * track.id
                        )
                    else:
                        track.src_3d_keypoints = None
                else:
                    track.src_3d_keypoints = None
            else:
                track.src_3d_keypoints = None
        
        current_extrinsics, current_intrinsics = self._get_current_extrinsics(frame_id)
        
        if len(queries) == 0:
            for track in all_tracks_to_track:
                if track.is_dynamic and current_extrinsics is not None:
                    track.update_visibility_with_extrinsics_compensation(
                        frame_id, self.H, self.W,
                        current_intrinsics, current_extrinsics
                    )
                track.src_3d_keypoints = None
                track.current_3d_keypoints = None
            return
        
        queries_tensor = torch.cat(queries)
        obj_tensor = torch.cat(queries_by_object)
        
        self.point_tracker(
            video_chunk=self.video[start_frame:fin_frame].unsqueeze(0).to(self.device),
            is_first_step=True,
            queries=queries_tensor.unsqueeze(0).to(self.device)
        )
        points, visibilities = self.point_tracker(
            video_chunk=self.video[start_frame:fin_frame].unsqueeze(0).to(self.device)
        )
        
        points_ = points[0][-1].cpu().detach().numpy()
        visibilities_ = visibilities[0][-1].cpu().detach().numpy()
        _, visible = _get_good_points(
            points_, visibilities_, self.H, self.W,
            thresh=cfg.cotracker.visibility_thresh
        )
        
        keypoints_3d, valid_mask = _get_3d_for_keypoints(
            points_, points_2d, points_3d
        )
        valid_mask = valid_mask * visible
        
        for track in all_tracks_to_track:
            track_mask = obj_tensor == track.id
            
            if torch.sum(track_mask) == 0:
                if track.is_dynamic and current_extrinsics is not None:
                    track.update_visibility_with_extrinsics_compensation(
                        frame_id, self.H, self.W,
                        current_intrinsics, current_extrinsics
                    )
                track.src_3d_keypoints = None
                track.current_3d_keypoints = None
                continue
            
            track_mask_np = track_mask.cpu().numpy()
            track_points = points_[track_mask_np]
            track_visibilities = visibilities_[track_mask_np]
            
            track.update_visibility_from_cotracker(
                track_points, track_visibilities, frame_id, self.H, self.W
            )
            
            if track.is_dynamic and track.motion_state.status == "MOVING":
                track.moving_keypoints[frame_id] = track_points[visible[track_mask_np]]
                track.current_3d_keypoints = keypoints_3d[track_mask_np]
                valid_3d = track.current_3d_keypoints[valid_mask[track_mask_np]]
                if len(valid_3d) > 0:
                    track.last_3d_keypoints = valid_3d.copy()
            else:
                track.current_3d_keypoints = None
        
        for track in self.tracks:
            if not track.is_dynamic or not track.cotracker_is_visible:
                track.transform = np.eye(4)
                continue
            
            src_pts = getattr(track, "src_3d_keypoints", None)
            tgt_pts = getattr(track, "current_3d_keypoints", None)
            transform_success = False

            if (src_pts is not None and tgt_pts is not None and
                len(src_pts) == len(tgt_pts) and len(src_pts) > 0):
                track_mask = obj_tensor == track.id
                indices = torch.nonzero(track_mask, as_tuple=True)[0].cpu().numpy()
                if len(indices) > 0:
                    track_valid = valid_mask[indices]
                    depth_ok = (src_pts[:, 2] > 0.1) & (tgt_pts[:, 2] > 0.1)
                    good_mask = track_valid & depth_ok

                    if np.sum(good_mask) >= cfg.cotracker.min_keypoints_for_transform:
                        cur_pcd = o3d.geometry.PointCloud()
                        cur_pcd.points = o3d.utility.Vector3dVector(src_pts[good_mask])
                        new_pcd = o3d.geometry.PointCloud()
                        new_pcd.points = o3d.utility.Vector3dVector(tgt_pts[good_mask])
                        corres = o3d.utility.Vector2iVector(
                            np.array([(i, i) for i in range(len(src_pts[good_mask]))])
                        )

                        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
                        transformation = p2p.compute_transformation(cur_pcd, new_pcd, corres)
                        rmse = p2p.compute_rmse(cur_pcd, new_pcd, corres)
                        if rmse <= cfg.cotracker.max_rmse:
                            track.transform = transformation
                            track.last_3d_keypoints = tgt_pts[good_mask].copy()
                            track.stable_3d_keypoints = track.last_3d_keypoints.copy()
                            transform_success = True
                        else:
                            print(f"Track {track.id}: suspicious transform (RMSE={rmse:.3f})")

            if not transform_success:
                track.transform = np.eye(4)
                track.last_3d_keypoints = track.stable_3d_keypoints
                if current_extrinsics is not None:
                    track.update_visibility_with_extrinsics_compensation(
                        frame_id, self.H, self.W,
                        current_intrinsics, current_extrinsics
                    )

            track.src_3d_keypoints = None
            track.current_3d_keypoints = None
    
    def update(
        self,
        frame_detections: List[Dict[str, Any]],
        frame_id: int,
        points_2d: np.ndarray,
        points_3d: np.ndarray
    ) -> List[Track]:
        """Main tracking update step with global assignment per track stage."""
        
        for track in self.tracks + self.lost_tracks:
            track.time_since_update += 1

        self._compute_transform_for_moving_tracks(frame_id, points_2d, points_3d)
        
        matched_detections = set()
        active_tracks = [t for t in self.tracks if t.state == "active"]
        tentative_tracks = [t for t in self.tracks if t.state == "tentative"]

        def visual_similarity_matrix(tracks: List[Track], detections: List[Dict[str, Any]]) -> np.ndarray:
            if len(tracks) == 0 or len(detections) == 0:
                return np.zeros((len(tracks), len(detections)))

            track_vis_embs = np.array([t.embedding for t in tracks])
            det_vis_embs = np.array([d["embedding"] for d in detections])
            vis_dim = _infer_embedding_dim(
                cfg.embeddings.vis_embedding_dim,
                track_vis_embs,
                det_vis_embs,
            )
            track_vis_embs = _ensure_embedding_2d(track_vis_embs, vis_dim)
            det_vis_embs = _ensure_embedding_2d(det_vis_embs, vis_dim)
            track_norm = track_vis_embs / (np.linalg.norm(track_vis_embs, axis=1, keepdims=True) + 1e-8)
            det_norm = det_vis_embs / (np.linalg.norm(det_vis_embs, axis=1, keepdims=True) + 1e-8)
            return track_norm @ det_norm.T

        def assign_tracks(
            tracks: List[Track],
            detections: List[Dict[str, Any]],
            threshold: float,
            prefer_sam_id: bool,
        ) -> Tuple[List[Tuple[int, int, Track, Dict[str, Any]]], set, set]:
            if len(tracks) == 0 or len(detections) == 0:
                return [], set(), set()

            visual_sim = visual_similarity_matrix(tracks, detections)
            _, _, filter_mask = _compute_filter_matrices(
                tracks, detections, self.text_emb_threshold, self.max_dist_multiplier
            )
            cost = np.full((len(tracks), len(detections)), 1e9)
            valid = np.zeros((len(tracks), len(detections)), dtype=bool)

            for track_idx, track in enumerate(tracks):
                for det_idx, det in enumerate(detections):
                    if not track.should_allow_association(visual_sim[track_idx, det_idx], self.emb_threshold):
                        continue
                    if len(filter_mask) > 0 and not filter_mask[track_idx, det_idx]:
                        continue

                    sim = calc_voxel_similarity(track, det, image_shape=(self.H, self.W))
                    if sim <= threshold:
                        continue

                    if prefer_sam_id and det.get("sam_id") == track.sam_id:
                        cost[track_idx, det_idx] = -1.0
                    else:
                        cost[track_idx, det_idx] = 1.0 - sim
                    valid[track_idx, det_idx] = True

            row_ind, col_ind = linear_sum_assignment(cost)
            assignments = []
            matched_track_indices = set()
            matched_det_indices = set()
            for row, col in zip(row_ind, col_ind):
                if valid[row, col]:
                    assignments.append((row, col, tracks[row], detections[col]))
                    matched_track_indices.add(row)
                    matched_det_indices.add(col)
            return assignments, matched_track_indices, matched_det_indices

        active_assignments, _, _ = assign_tracks(
            active_tracks, frame_detections, self.ass_threshold, prefer_sam_id=True
        )
        for _, _, track, det in active_assignments:
            track.update(det, frame_id)
            track.matched = True
            matched_detections.add(id(det))

        revived_tracks = []
        unmatched_detections = [
            det for det in frame_detections if id(det) not in matched_detections
        ]

        lost_assignments, matched_lost_idx, matched_lost_det_idx = assign_tracks(
            self.lost_tracks, unmatched_detections, self.lost_ass_threshold, prefer_sam_id=False
        )
        for _, _, track, det in lost_assignments:
            track.update(det, frame_id)
            track.state = "tentative"
            if track.hits > track.min_hits_to_activate:
                track.state = "active"
            revived_tracks.append(track)
            matched_detections.add(id(det))

        if matched_lost_idx:
            self.lost_tracks = [
                track for idx, track in enumerate(self.lost_tracks)
                if idx not in matched_lost_idx
            ]
        if matched_lost_det_idx:
            unmatched_detections = [
                det for idx, det in enumerate(unmatched_detections)
                if idx not in matched_lost_det_idx
            ]

        tentative_assignments, _, matched_tent_det_idx = assign_tracks(
            tentative_tracks, unmatched_detections,
            self.tentative_ass_threshold,
            prefer_sam_id=True,
        )
        for _, _, track, det in tentative_assignments:
            track.update(det, frame_id)
            track.matched = True
            matched_detections.add(id(det))

        if matched_tent_det_idx:
            unmatched_detections = [
                det for idx, det in enumerate(unmatched_detections)
                if idx not in matched_tent_det_idx
            ]
        
        for det in unmatched_detections:
            is_dynamic = str(det['cls']).lower() in self.dynamic_classes_list
            new_track = Track(
                self.next_id, det, frame_id, is_dynamic=is_dynamic
            )
            new_track.state = "tentative"
            new_track.hits = 1
            new_track.sam_id = det.get("sam_id")
            self.tracks.append(new_track)
            self.next_id += 1
        
        curr_active_tracks = []
        new_lost_tracks = []
        
        for track in self.tracks:
            if track.state == "tentative":
                if (track.time_since_update > self.max_tentative_age or
                    track.hits >= track.min_hits_to_activate):
                    if track.hits >= track.min_hits_to_activate:
                        track.state = "active"
                        track.was_active = True
                    else:
                        curr_active_tracks.append(track)
                else:
                    curr_active_tracks.append(track)
            else:
                if track.time_since_update > self.max_active_frames:
                    track.state = "lost"
                    new_lost_tracks.append(track)
                else:
                    curr_active_tracks.append(track)
        
        curr_active_tracks.extend(revived_tracks)
        self.tracks = curr_active_tracks
        self.lost_tracks = self.lost_tracks + new_lost_tracks
        
        for track in self.lost_tracks:
            track.transform = np.eye(4)
        
        return self._get_current_tracks()
    
    def _get_current_tracks(self) -> List[Track]:
        """Get only active and tentative tracks."""
        return [t for t in self.tracks if t.state in ["active", "tentative"]]
