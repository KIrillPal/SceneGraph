"""
3D Tracking Core for SAM3 Tracker

Provides VoxelMap, Track, SmartUpdateRules, and Simple3DTracker.
Visibility update EVERY frame with camera motion compensation using EXTRINSICS.
"""

import cv2
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
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from cotracker.predictor import CoTrackerOnlinePredictor


# ============================================================================
# CONSTANTS
# ============================================================================

class Config:
    """Tracking configuration parameters."""
    # Motion detection
    T_START = 0.07
    T_STOP = 0.03
    VELOCITY_BUFFER_SIZE = 10
    
    # Voxel map
    VOXEL_SIZE = 0.05
    VOXEL_ALPHA = 0.3
    VOXEL_MAX_AGE = 300
    VOXEL_MAX_AGE_MOVING = 5
    
    # Association thresholds
    ASS_THRESHOLD = 0.45
    LOST_ASS_THRESHOLD = 0.4
    TENTATIVE_ASS_THRESHOLD = 0.55
    TEXT_EMB_THRESHOLD = 0.75
    VISUAL_EMB_THRESHOLD = 0.75
    MAX_DIST_MULTIPLIER = 3
    
    # Track lifecycle
    MAX_TENTATIVE_AGE = 20
    MAX_ACTIVE_FRAMES = 20
    MIN_HITS_TO_ACTIVATE = 10
    
    # CoTracker
    COTRACKER_WINDOW_LEN = 16
    COTRACKER_VISIBILITY_THRESH = 0.7
    MIN_KEYPOINTS_FOR_TRANSFORM = 10
    MIN_KEYPOINTS_VISIBLE = 3
    MAX_RMSE = 0.1

    # Embeddings
    VIS_EMBEDDING_DIM = 384
    TEXT_EMBEDDING_DIM = 384
    
    # Object size
    USE_DETECTION_SIZE_FOR_MOVING = True
    MIN_OBJECT_SIZE = 0.2
    SIZE_SMOOTHING_ALPHA = 0.3
    MAX_SIZE_INCREASE_FACTOR = 2.0
    
    # === VISIBILITY (dynamic objects only) ===
    OUTSIDE_FRAME_MARGIN = 50  # Pixels outside frame to consider "outside"




# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _ensure_embedding_2d(emb_array: np.ndarray, expected_dim: int = Config.TEXT_EMBEDDING_DIM) -> np.ndarray:
    """Ensure embedding array is 2D (N, D)."""
    if len(emb_array) == 0:
        return np.zeros((0, expected_dim))
    
    if emb_array.ndim == 3:
        emb_array = emb_array.squeeze(axis=1)
    elif emb_array.ndim == 1:
        if emb_array.shape[0] == expected_dim:
            emb_array = emb_array.reshape(1, -1)
    
    return emb_array


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
        
        if len(self.v_buffer) > Config.VELOCITY_BUFFER_SIZE:
            self.v_buffer.pop(0)
            self.size_buffer.pop(0)
        
        self.v_avg = np.mean(self.v_buffer)
        self.size_avg = max(0.2, np.mean(self.size_buffer))
        
        normalized_velocity = self.v_avg / self.size_avg
        
        if self.status == "STATIC":
            if normalized_velocity > Config.T_START:
                self.status = "MOVING"
        else:
            if normalized_velocity < Config.T_STOP:
                self.status = "STATIC"
        
        return self.status


class VoxelMap:
    """Voxel-based 3D map with EMA updates and age-based expiration."""
    
    def __init__(
        self,
        voxel_size: float = Config.VOXEL_SIZE,
        alpha: float = Config.VOXEL_ALPHA,
        max_age: int = Config.VOXEL_MAX_AGE
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
            Config.USE_DETECTION_SIZE_FOR_MOVING and is_dynamic
        )
        
        # State machine
        self.state = "tentative"
        self.hits = 1
        self.min_hits_to_activate = Config.MIN_HITS_TO_ACTIVATE
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
    
    @property
    def object_size(self) -> float:
        """Get object size based on motion state."""
        if self.use_detection_size and self.motion_state.status == "MOVING":
            return max(self.detection_size, Config.MIN_OBJECT_SIZE)
        else:
            return max(self.voxel_size, Config.MIN_OBJECT_SIZE)
    
    def update_size(self, detection: Dict[str, Any], frame_idx: int):
        """Update object size from new detection."""
        det_size = compute_object_size(detection["points"])
        
        if self.use_detection_size and self.motion_state.status == "MOVING":
            max_allowed_size = self.detection_size * Config.MAX_SIZE_INCREASE_FACTOR
            det_size = min(det_size, max_allowed_size)
            
            alpha = Config.SIZE_SMOOTHING_ALPHA
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
        """
        Update visibility from CoTracker output (when track IS in CoTracker output).
        """
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
        
        visible = track_visibilities > Config.COTRACKER_VISIBILITY_THRESH
        
        valid_points = np.sum(in_frame & visible)
        total_points = len(track_points)
        low_vis_points = np.sum(~visible & in_frame)
        outside_points = np.sum(~in_frame)
        
        if np.sum(in_frame) > 0:
            self.predicted_pos_2d = np.mean(track_points[in_frame], axis=0)
        else:
            self.predicted_pos_2d = np.mean(track_points, axis=0) if len(track_points) > 0 else None
        
        self.last_predicted_pos_2d = self.predicted_pos_2d.copy() if self.predicted_pos_2d is not None else None
        
        if valid_points >= max(Config.MIN_KEYPOINTS_VISIBLE, total_points * 0.3):
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
    
    def update_visibility_with_extrinsics_compensation(
        self,
        frame_id: int,
        H: int,
        W: int,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray
    ):
        """
        Update visibility for tracks NOT in CoTracker output.
        
        Uses track's 3D position + current camera extrinsics to project
        and check if object might be back in frame.
        
        Args:
            frame_id: Current frame index.
            H: Frame height.
            W: Frame width.
            intrinsics: Camera intrinsics [3, 3].
            extrinsics: Current frame world-to-camera extrinsics [3, 4].
        """
        if not self.is_dynamic:
            self.visibility_state = "VISIBLE"
            self.cotracker_is_visible = True
            return
        
        # No 3D position available
        if self.center_3d is None or len(self.center_3d) == 0:
            self.visibility_state = "OCCLUDED"
            self.cotracker_is_visible = False
            return
        
        # Project 3D position to current frame using extrinsics
        projected_pos_2d = _project_3d_to_2d(
            self.center_3d,
            intrinsics,
            extrinsics
        )
        
        # Update predicted position
        self.predicted_pos_2d = projected_pos_2d
        if projected_pos_2d is not None:
            self.last_predicted_pos_2d = projected_pos_2d.copy()
        
        # Check if projected position is in frame
        if projected_pos_2d is not None and _is_position_in_frame(
            projected_pos_2d, H, W, margin=Config.OUTSIDE_FRAME_MARGIN
        ):
            # Position is in frame - might be occluded but trackable
            self.visibility_state = "OCCLUDED"
            self.cotracker_is_visible = True
        else:
            # Outside frame
            self.visibility_state = "OUTSIDE_FRAME"
            self.cotracker_is_visible = False
    
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
                self.voxels = VoxelMap(max_age=Config.VOXEL_MAX_AGE_MOVING)
            self.voxels.add_points(points)
        elif update_flags["use_det_voxels"] and self.motion_state.status == "MOVING":
            self.voxels = VoxelMap(max_age=Config.VOXEL_MAX_AGE_MOVING)
            self.voxels.add_points(points)
        
        if update_flags["re_init"]:
            print(f"Re-init track {self.id}")
            max_age = (Config.VOXEL_MAX_AGE_MOVING
                      if self.motion_state.status == "MOVING"
                      else Config.VOXEL_MAX_AGE)
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
        
        self.ass_threshold = Config.ASS_THRESHOLD
        self.lost_ass_threshold = Config.LOST_ASS_THRESHOLD
        self.tentative_ass_threshold = Config.TENTATIVE_ASS_THRESHOLD
        self.text_emb_threshold = Config.TEXT_EMB_THRESHOLD
        self.emb_threshold = Config.VISUAL_EMB_THRESHOLD
        self.max_dist_multiplier = Config.MAX_DIST_MULTIPLIER
        
        self.max_tentative_age = Config.MAX_TENTATIVE_AGE
        self.max_active_frames = Config.MAX_ACTIVE_FRAMES
        self.dynamic_classes_list = [str(cls).lower() for cls in dynamic_classes_list]
        
        self.device = device
        self.point_tracker = CoTrackerOnlinePredictor(
            'co-tracker/checkpoints/scaled_online.pth'
        ).to(device)
        self.cotracker_window_len = Config.COTRACKER_WINDOW_LEN
        self.video = video
        _,  _, self.H, self.W, = video.size()
        
        # === CAMERA EXTRINSICS ===
        self.intrinsics_list = intrinsics_list  # List of [3, 3]  intrinsics per frame
        self.extrinsics_list = extrinsics_list  # List of [3, 4] or [4, 4] extrinsics per frame
    
    def _get_current_extrinsics(self, frame_id: int) -> Optional[np.ndarray]:
        """Get camera extrinsics for current frame."""
        if self.extrinsics_list is None:
            return None, None
        if frame_id < 0 or frame_id >= len(self.extrinsics_list):
            return None
        return self.extrinsics_list[frame_id],self.intrinsics_list[frame_id]
    
    def _compute_transform_for_moving_tracks(
        self,
        frame_id: int,
        points_2d: np.ndarray,
        points_3d: np.ndarray
    ) -> None:
        """
        Compute rigid transform for moving tracks using CoTracker.
        Updates visibility for ALL tracks using EXTRINSICS for compensation.
        """
        queries = []
        queries_by_object = []
        start_frame = max(0, frame_id - self.cotracker_window_len + 1)
        fin_frame = frame_id + 1
        
        all_tracks_to_track = self.tracks + self.lost_tracks
        
        for track in all_tracks_to_track:
            if not track.is_dynamic:
                continue
            
            if (track.motion_state.status == "MOVING" and
                track.time_since_update < self.cotracker_window_len):
                last_frame_detected = max(track.keypoints.keys()) if track.keypoints else -1
                if last_frame_detected >= 0 and track.keypoints[last_frame_detected] is not None:
                    local_frame = last_frame_detected - start_frame
                    obj_queries = torch.tensor([
                        [local_frame, q[0], q[1]]
                        for q in track.keypoints[last_frame_detected]
                    ])
                    queries.append(obj_queries)
                    queries_by_object.append(
                        torch.ones(len(obj_queries), dtype=int) * track.id
                    )
        
        # Get current extrinsics
        current_extrinsics,current_intrinsics = self._get_current_extrinsics(frame_id)
        
        if len(queries) == 0:
            # No CoTracker output - update all with extrinsics compensation
            for track in all_tracks_to_track:
                if track.is_dynamic and current_extrinsics is not None:
                    track.update_visibility_with_extrinsics_compensation(
                        frame_id, self.H, self.W,
                        current_intrinsics, current_extrinsics
                    )
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
            thresh=Config.COTRACKER_VISIBILITY_THRESH
        )
        
        keypoints_3d, valid_mask = _get_3d_for_keypoints(
            points_, points_2d, points_3d
        )
        valid_mask = valid_mask * visible
        
        # Update visibility for ALL tracks
        for track in all_tracks_to_track:
            track_mask = obj_tensor == track.id
            
            if torch.sum(track_mask) == 0:
                # Track NOT in CoTracker output - use EXTRINSICS compensation
                if track.is_dynamic and current_extrinsics is not None:
                    track.update_visibility_with_extrinsics_compensation(
                        frame_id, self.H, self.W,
                        current_intrinsics, current_extrinsics
                    )
                continue
            
            # Track IS in CoTracker output - use actual predictions
            track_points = points_[track_mask.cpu().numpy()]
            track_visibilities = visibilities_[track_mask.cpu().numpy()]
            
            track.update_visibility_from_cotracker(
                track_points, track_visibilities, frame_id, self.H, self.W
            )
            
            if track.is_dynamic and track.motion_state.status == "MOVING":
                track.moving_keypoints[frame_id] = track_points[visible[track_mask.cpu().numpy()]]
        
        # Compute transforms
        for track in self.tracks:
            if not track.is_dynamic or not track.cotracker_is_visible:
                track.transform = np.eye(4)
                continue
            
            track_mask = obj_tensor == track.id
            new_keypoints_3d = keypoints_3d[track_mask]
            new_valid_mask = valid_mask[track_mask]
            
            if len(new_valid_mask) == 0:
                track.transform = np.eye(4)
                continue
            
            last_frame_detected = max(track.keypoints.keys()) if track.keypoints else -1
            if (last_frame_detected < 0 or 
                track.keypoints[last_frame_detected] is None or 
                len(track.keypoints[last_frame_detected]) == 0):
                track.transform = np.eye(4)
                continue
            
            cur_keypoints_3d, cur_valid_mask = _get_3d_for_keypoints(
                track.keypoints[last_frame_detected], points_2d, points_3d
            )
            good_point_mask = cur_valid_mask & new_valid_mask
            
            if np.sum(good_point_mask) < Config.MIN_KEYPOINTS_FOR_TRANSFORM:
                track.transform = np.eye(4)
                continue
            
            cur_pcd = o3d.geometry.PointCloud()
            cur_pcd.points = o3d.utility.Vector3dVector(
                cur_keypoints_3d[good_point_mask]
            )
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(
                new_keypoints_3d[good_point_mask]
            )
            
            indices_2d = np.array([
                (i, i) for i in range(len(new_keypoints_3d[good_point_mask]))
            ])
            corres = o3d.utility.Vector2iVector(indices_2d)
            
            p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
            transformation = p2p.compute_transformation(cur_pcd, new_pcd, corres)
            
            if len(corres) > 0:
                rmse = p2p.compute_rmse(cur_pcd, new_pcd, corres)
                if rmse > Config.MAX_RMSE:
                    print(f'Track {track.id}: suspicious transform (RMSE={rmse:.3f})')
                    track.transform = np.eye(4)
                else:
                    track.transform = transformation
            else:
                track.transform = np.eye(4)
    
    def update(
        self,
        frame_detections: List[Dict[str, Any]],
        frame_id: int,
        points_2d: np.ndarray,
        points_3d: np.ndarray
    ) -> List[Track]:
        """Main tracking update step."""
        
        for track in self.tracks + self.lost_tracks:
            track.time_since_update += 1
        
        for track in self.tracks:
            track.matched = False
        
        self._compute_transform_for_moving_tracks(frame_id, points_2d, points_3d)
        
        matched_detections = set()
        active_tracks = [t for t in self.tracks if t.state == "active"]
        tentative_tracks = [t for t in self.tracks if t.state == "tentative"]
        
        active_track_vis_embs = np.array([t.embedding for t in active_tracks])
        det_vis_embs = np.array([d["embedding"] for d in frame_detections])
        active_track_vis_embs = _ensure_embedding_2d(active_track_vis_embs, Config.VIS_EMBEDDING_DIM)
        det_vis_embs = _ensure_embedding_2d(det_vis_embs, Config.VIS_EMBEDDING_DIM)
        
        active_vis_norm = active_track_vis_embs / (np.linalg.norm(active_track_vis_embs, axis=1, keepdims=True) + 1e-8)
        det_vis_norm = det_vis_embs / (np.linalg.norm(det_vis_embs, axis=1, keepdims=True) + 1e-8)
        active_vis_sim_matrix = active_vis_norm @ det_vis_norm.T
        
        (_, _, active_filter_mask) = _compute_filter_matrices(
            active_tracks, frame_detections, self.text_emb_threshold, self.max_dist_multiplier
        )
        
        for det_idx, det in enumerate(frame_detections):
            for track_idx, track in enumerate(active_tracks):
                if (not track.matched and
                    det.get("sam_id") == track.sam_id):
                    if len(active_filter_mask) > 0 and not active_filter_mask[track_idx, det_idx]:
                        continue
                    
                    if  calc_voxel_similarity(track, det,image_shape=(self.H,self.W)) > self.ass_threshold:
                        track.update(det, frame_id)
                        track.matched = True
                        matched_detections.add(id(det))
                        break
        
        for det_idx, det in enumerate(frame_detections):
            if id(det) in matched_detections:
                continue
            
            best_track = None
            best_sim = 0
            
            for track_idx, track in enumerate(active_tracks):
                if track.matched:
                    continue
                
                visual_sim = active_vis_sim_matrix[track_idx, det_idx]
                
                if not track.should_allow_association(visual_sim, self.emb_threshold):
                    continue
                
                if track.is_dynamic and visual_sim > self.emb_threshold:
                    if len(active_filter_mask) > 0:
                        text_pass = _check_text_filter_only(
                            track.text_embedding, det["text_embedding"],
                            self.text_emb_threshold
                        )
                        if not text_pass:
                            continue
                else:
                    if len(active_filter_mask) > 0 and not active_filter_mask[track_idx, det_idx]:
                        continue
                
                sim = calc_voxel_similarity(track, det,image_shape=(self.H,self.W))
                if sim > best_sim and sim > self.ass_threshold:
                    best_sim = sim
                    best_track = track
            
            if best_track:
                best_track.update(det, frame_id)
                best_track.matched = True
                matched_detections.add(id(det))
        
        revived_tracks = []
        unmatched_detections = [
            det for det in frame_detections if id(det) not in matched_detections
        ]
        
        if len(self.lost_tracks) > 0:
            lost_track_vis_embs = np.array([t.embedding for t in self.lost_tracks])
            lost_det_vis_embs = np.array([d["embedding"] for d in unmatched_detections])
            lost_track_vis_embs = _ensure_embedding_2d(lost_track_vis_embs, Config.VIS_EMBEDDING_DIM)
            lost_det_vis_embs = _ensure_embedding_2d(lost_det_vis_embs, Config.VIS_EMBEDDING_DIM)
            
            lost_vis_norm = lost_track_vis_embs / (np.linalg.norm(lost_track_vis_embs, axis=1, keepdims=True) + 1e-8)
            lost_det_vis_norm = lost_det_vis_embs / (np.linalg.norm(lost_det_vis_embs, axis=1, keepdims=True) + 1e-8)
            lost_vis_sim_matrix = lost_vis_norm @ lost_det_vis_norm.T
        else:
            lost_vis_sim_matrix = np.zeros((0, 0))
        
        (_, _, lost_filter_mask) = _compute_filter_matrices(
            self.lost_tracks, unmatched_detections, self.text_emb_threshold, self.max_dist_multiplier
        )
        
        for det_idx, det in enumerate(unmatched_detections[:]):
            best_track = None
            best_sim = 0
            
            for idx, track in enumerate(self.lost_tracks):
                visual_sim = lost_vis_sim_matrix[idx, det_idx] if len(lost_vis_sim_matrix) > 0 else 0.0
                
                if not track.should_allow_association(visual_sim, self.emb_threshold):
                    continue
                
                if track.is_dynamic and visual_sim > self.emb_threshold:
                    if len(lost_filter_mask) > 0:
                        text_pass = _check_text_filter_only(
                            track.text_embedding, det["text_embedding"],
                            self.text_emb_threshold
                        )
                        if not text_pass:
                            continue
                else:
                    if len(lost_filter_mask) > 0 and not lost_filter_mask[idx, det_idx]:
                        continue
                
                is_moving = track.motion_state.status == 'MOVING'
    
                sim =  calc_voxel_similarity(track, det,image_shape=(self.H,self.W))
                
                if sim > best_sim and sim > self.lost_ass_threshold:
                    best_sim = sim
                    best_track = (idx, track)
            
            if best_track:
                idx, track = best_track
                track.update(det, frame_id)
                track.state = "tentative"
                if track.hits > track.min_hits_to_activate:
                    track.state = "active"
                revived_tracks.append(track)
                self.lost_tracks.pop(idx)
                unmatched_detections.remove(det)
                matched_detections.add(id(det))
        
        for det in unmatched_detections[:]:
            for track in tentative_tracks:
                if (not track.matched and
                    det.get("sam_id") == track.sam_id):
                    if  calc_voxel_similarity(track, det,image_shape=(self.H,self.W)) > self.tentative_ass_threshold:
                        track.update(det, frame_id)
                        track.matched = True
                        unmatched_detections.remove(det)
                        break
        
        if len(tentative_tracks) > 0 and len(unmatched_detections) > 0:
            tentative_track_vis_embs = np.array([t.embedding for t in tentative_tracks])
            tentative_det_vis_embs = np.array([d["embedding"] for d in unmatched_detections])
            tentative_track_vis_embs = _ensure_embedding_2d(tentative_track_vis_embs, Config.VIS_EMBEDDING_DIM)
            tentative_det_vis_embs = _ensure_embedding_2d(tentative_det_vis_embs, Config.VIS_EMBEDDING_DIM)
            
            tentative_vis_norm = tentative_track_vis_embs / (np.linalg.norm(tentative_track_vis_embs, axis=1, keepdims=True) + 1e-8)
            tentative_det_vis_norm = tentative_det_vis_embs / (np.linalg.norm(tentative_det_vis_embs, axis=1, keepdims=True) + 1e-8)
            tentative_vis_sim_matrix = tentative_vis_norm @ tentative_det_vis_norm.T
        else:
            tentative_vis_sim_matrix = np.zeros((0, 0))
        
        (_, _, tentative_filter_mask) = _compute_filter_matrices(
            tentative_tracks, unmatched_detections, self.text_emb_threshold, self.max_dist_multiplier
        )
        
        for det_idx, det in enumerate(unmatched_detections[:]):
            best_track = None
            best_sim = 0
            
            for track_idx, track in enumerate(tentative_tracks):
                if track.matched:
                    continue
                
                visual_sim = tentative_vis_sim_matrix[track_idx, det_idx] if len(tentative_vis_sim_matrix) > 0 else 0.0
                
                if not track.should_allow_association(visual_sim, self.emb_threshold):
                    continue
                
                if track.is_dynamic and visual_sim > self.emb_threshold:
                    if len(tentative_filter_mask) > 0:
                        text_pass = _check_text_filter_only(
                            track.text_embedding, det["text_embedding"],
                            self.text_emb_threshold
                        )
                        if not text_pass:
                            continue
                else:
                    if len(tentative_filter_mask) > 0 and not tentative_filter_mask[track_idx, det_idx]:
                        continue
                
                sim =  calc_voxel_similarity(track, det,image_shape=(self.H,self.W))
                if sim > best_sim and sim > self.tentative_ass_threshold:
                    best_sim = sim
                    best_track = track
            
            if best_track:
                best_track.update(det, frame_id)
                best_track.matched = True
                unmatched_detections.remove(det)
                matched_detections.add(id(det))
        
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
