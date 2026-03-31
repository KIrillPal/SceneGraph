"""
3D Tracking Core for SAM3 Tracker

Provides VoxelMap, Track, SmartUpdateRules, and Simple3DTracker.
"""

import cv2
import torch
import numpy as np
import open3d as o3d
from utils.point import *
from utils.metric import *
from collections import deque
from typing import Dict, List, Tuple, Optional, Any


class VoxelMap:
    """
    Voxel-based 3D map with EMA updates and age-based expiration.

    Stores points in a voxel grid, updates coordinates via EMA,
    and removes stale voxels after max_age frames.
    """

    def __init__(
        self, voxel_size: float = 0.01, alpha: float = 0.3, max_age: int = 300
    ):
        """
        Args:
            voxel_size: Voxel size in meters (default: 0.01m).
            alpha: EMA weight for new points (0-1, default: 0.3).
            max_age: Frames before voxel expires (default: 300).
        """
        self.voxel_size = voxel_size
        self.alpha = alpha
        self.max_age = max_age

        # Voxel storage: {(ix, iy, iz): [ [x, y, z], age ]}
        self.voxels: Dict[Tuple[int, int, int], List[np.ndarray, int]] = {}
        self.current_frame = -1

    def add_points(self, points: np.ndarray) -> None:
        """
        Add 3D points to voxel map with EMA updates.

        Args:
            points: 3D points [N, 3] in world coordinates.
        """
        self.current_frame += 1

        if len(points) == 0:
            self._expire()
            return

        # Compute voxel indices
        indices = np.floor(points / self.voxel_size).astype(int)

        for i, idx in enumerate(indices):
            idx_tuple = tuple(idx)
            new_pt = points[i]

            if idx_tuple in self.voxels:
                # EMA update for existing voxel
                current_data = self.voxels[idx_tuple]
                updated_pt = (1 - self.alpha) * current_data[0] + self.alpha * new_pt
                self.voxels[idx_tuple] = [updated_pt, self.current_frame]
            else:
                # Create new voxel
                self.voxels[idx_tuple] = [new_pt, self.current_frame]

        # Remove expired voxels
        self._expire()

    def _expire(self) -> None:
        """Remove voxels not updated for more than max_age frames."""
        expired_keys = [
            k
            for k, v in self.voxels.items()
            if (self.current_frame - v[1]) > self.max_age
        ]
        for k in expired_keys:
            del self.voxels[k]

    def get_pcd(self) -> o3d.geometry.PointCloud:
        """
        Get PointCloud from current voxels.

        Returns:
            Open3D PointCloud with outlier removal applied.
        """
        if not self.voxels:
            return o3d.geometry.PointCloud()

        points, _ = statistical_outlier_removal(
            np.array([v[0] for v in self.voxels.values()])
        )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd


class MedianEMAFilter:
    """
    Combined median filter + EMA for robust smoothing.

    Uses median filtering for outlier rejection, then EMA
    for smooth temporal updates.
    """

    def __init__(self, ema_alpha: float = 0.85, median_window: int = 5):
        """
        Args:
            ema_alpha: EMA weight (0-1, default: 0.85).
            median_window: Median filter window size (default: 5).
        """
        self.ema_alpha = ema_alpha
        self.median_window = median_window
        self.history = deque(maxlen=median_window)
        self.ema_value: Optional[np.ndarray] = None

    def update(self, value: np.ndarray) -> np.ndarray:
        """
        Update filter with new value.

        Args:
            value: New value to filter.

        Returns:
            Filtered value after median + EMA.
        """
        self.history.append(value)

        # Median filtering for outlier rejection
        if len(self.history) >= 3:
            median_filtered = np.median(list(self.history), axis=0)
        else:
            median_filtered = value

        # EMA smoothing
        if self.ema_value is None:
            self.ema_value = median_filtered
        else:
            self.ema_value = (
                self.ema_alpha * median_filtered + (1 - self.ema_alpha) * self.ema_value
            )

        return self.ema_value


class Track:
    """
    Single object track with voxels, embeddings, and state management.

    Maintains 3D voxel map, visual/text embeddings, and hysteresis
    state machine (tentative → active → lost).
    """

    def __init__(self, track_id: int, detection: Dict[str, Any], frame_idx: int):
        """
        Args:
            track_id: Unique track identifier.
            detection: Detection dict with points, embeddings, mask, cls.
            frame_idx: Frame index where track was created.
        """
        self.id = track_id
        self.sam_id = detection["sam_id"]
        self.voxels = VoxelMap()
        self.voxel_size = 0.01

        # Initialize embeddings with smoothing
        self.text_embedding_smoother = MedianEMAFilter(ema_alpha=0.85, median_window=5)
        self.text_embedding = self.text_embedding_smoother.update(
            detection["text_embedding"]
        )

        self.embedding_smoother = MedianEMAFilter(ema_alpha=0.95, median_window=10)
        self.embedding = self.embedding_smoother.update(detection["embedding"])

        # Add initial points to voxel map
        points = detection["points"]
        self.voxels.add_points(points)

        # Hysteresis state machine
        self.state = "tentative"
        self.hits = 1
        self.min_hits_to_activate = 10
        self.time_since_update = 0

        # Store masks and classes per frame
        self.masks: Dict[int, np.ndarray] = {frame_idx: detection["mask"]}
        self.cls: Dict[int, str] = {frame_idx: detection["cls"]}

        # Update rules based on IoA
        self.update_rules = SmartUpdateRules()

    def update(self, detection: Dict[str, Any], frame_idx: int) -> None:
        """
        Update track with new detection.

        Args:
            detection: New detection dict.
            frame_idx: Current frame index.
        """
        print(detection["cls"], self.id)
        points = detection["points"]

        # Compute IoA for update decision
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(points)
        curr_pcd = self.voxels.get_pcd()

        ioa_from_t, ioa_from_d, ioa = compute_bidirectional_gaussian_ioa(
            curr_pcd, new_pcd
        )
        update_flags = self.update_rules.decide_update(ioa_from_t, ioa_from_d)

        # Update embeddings if flag set
        if update_flags["update_embeddings"]:
            self.embedding = self.embedding_smoother.update(detection["embedding"])
            self.text_embedding = self.text_embedding_smoother.update(
                detection["text_embedding"]
            )

        # Update voxels if flag set
        if update_flags["update_voxels"]:
            self.voxels.add_points(points)

        # Re-initialize voxels if flag set
        if update_flags["re_init"]:
            print("re-init track %d" % self.id)
            self.voxels = VoxelMap()
            self.voxels.add_points(points)

        self.hits += 1

        # Hysteresis: transition to active state
        if self.state == "tentative" and self.hits >= self.min_hits_to_activate:
            self.state = "active"

        self.time_since_update = 0
        self.masks[frame_idx] = detection["mask"]
        self.cls[frame_idx] = detection["cls"]


class SmartUpdateRules:
    """
    Track update rules based on bidirectional IoA.

    Decides whether to update embeddings, voxels, or re-initialize
    based on overlap between track and detection.
    """

    def __init__(self):
        """Initialize IoA thresholds for update decisions."""
        self.high_overlap_threshold = 0.7
        self.medium_overlap_threshold = 0.4
        self.re_init_threshold = 0.1

    def decide_update(
        self, ioa_track_to_det: float, ioa_det_to_track: float
    ) -> Dict[str, Any]:
        """
        Decide what to update based on bidirectional IoA.

        Args:
            ioa_track_to_det: Fraction of track covered by detection.
            ioa_det_to_track: Fraction of detection covered by track.

        Returns:
            Dict with update flags: 'update_embeddings', 'update_voxels', 're_init', 'reason'.
        """
        update = {
            "update_embeddings": False,
            "update_voxels": False,
            "re_init": False,
            "reason": "",
        }

        # Scenario 1: Full overlap (both IoA high)
        if (
            ioa_track_to_det > self.high_overlap_threshold
            and ioa_det_to_track > self.high_overlap_threshold
        ):
            update["update_embeddings"] = True
            update["update_voxels"] = True
            update["reason"] = "full_overlap"

        # Scenario 2: Detection covers track (track outdated)
        elif (
            ioa_det_to_track > self.high_overlap_threshold
            and ioa_track_to_det < self.medium_overlap_threshold
        ):
            update["update_embeddings"] = True
            update["update_voxels"] = True
            update["reason"] = "det_larger_than_track"

        # Scenario 3: Track covers detection (partial visibility)
        elif (
            ioa_track_to_det > self.high_overlap_threshold
            and ioa_det_to_track < self.medium_overlap_threshold
        ):
            update["update_embeddings"] = False
            update["update_voxels"] = True
            update["reason"] = "partial_visibility"

        # Scenario 4: Medium overlap
        elif (
            ioa_track_to_det > self.medium_overlap_threshold
            or ioa_det_to_track > self.medium_overlap_threshold
        ):
            update["update_embeddings"] = False
            update["update_voxels"] = True
            update["reason"] = "medium_overlap"

        # Scenario 5: Very low overlap (re-init needed)
        elif (
            ioa_track_to_det < self.re_init_threshold
            or ioa_det_to_track < self.re_init_threshold
        ):
            update["re_init"] = True
            update["reason"] = "re_init_needed"

        else:
            update["reason"] = "nothing_to_update_low_overlap"

        return update


class Simple3DTracker:
    """
    Multi-object 3D tracker with hysteresis state management.

    Manages active, tentative, and lost tracks with multi-cue
    association (SAM-ID, embeddings, voxel similarity).
    """

    def __init__(self):
        """
        Args:
            max_lost_frames: Maximum frames before lost track is removed.
        """
        self.tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.next_id = 0

        # Association thresholds
        self.ass_threshold = 0.45
        self.lost_ass_threshold = 0.4
        self.tentative_ass_threshold = 0.6

        # Embedding similarity thresholds (same object)
        self.text_emb_threshold = 0.75
        self.emb_threshold = 0.5

        # Age limits
        self.max_tentative_age = 20
        self.max_active_frames = 20

    def update(
        self, frame_detections: List[Dict[str, Any]], frame_id: int
    ) -> List[Track]:
        """
        Main tracking update step.

        Args:
            frame_detections: List of detection dicts with bbox_3d, embeddings, sam_id.
            frame_id: Current frame index.

        Returns:
            List of active and tentative tracks.
        """
        # 1. Prediction: increment time_since_update for all tracks
        for track in self.tracks:
            track.time_since_update += 1
        for track in self.lost_tracks:
            track.time_since_update += 1

        # 2. Mark all tracks as unmatched
        for track in self.tracks:
            track.matched = False

        # 3. Association: process active tracks first
        matched_detections = set()
        active_tracks = [t for t in self.tracks if t.state == "active"]
        tentative_tracks = [t for t in self.tracks if t.state == "tentative"]

        # 3.1. Match by SAM-ID (if SAM3 is trusted)
        for det in frame_detections:
            for track in active_tracks:
                if not track.matched and det.get("sam_id") == track.sam_id:
                    if (
                        calc_voxel_similarity(track, det, w_iou=0.4)
                        > self.ass_threshold
                    ):
                        track.update(det, frame_id)
                        track.matched = True
                        matched_detections.add(id(det))
                        break

        # 3.2. Match by similarity for remaining detections
        for det in frame_detections:
            if id(det) in matched_detections:
                continue

            best_track = None
            best_sim = 0

            for track in active_tracks:
                if track.matched:
                    continue

                text_emb_similarity = calc_emb_similarity(
                    track.text_embedding, det["text_embedding"]
                )
                emb_similarity = calc_emb_similarity(track.embedding, det["embedding"])

                if (
                    text_emb_similarity > self.text_emb_threshold
                    or emb_similarity > self.emb_threshold
                ):
                    sim = calc_voxel_similarity(track, det, w_iou=0.4)
                    if sim > best_sim and sim > self.ass_threshold:
                        best_sim = sim
                        best_track = track

            if best_track:
                best_track.update(det, frame_id)
                best_track.matched = True
                matched_detections.add(id(det))
                continue

        # 4. Recover lost tracks
        revived_tracks = []
        unmatched_detections = [
            det for det in frame_detections if id(det) not in matched_detections
        ]

        for det in unmatched_detections[:]:
            best_track = None
            best_sim = 0

            for idx, track in enumerate(self.lost_tracks):
                text_emb_similarity = calc_emb_similarity(
                    track.text_embedding, det["text_embedding"]
                )
                emb_similarity = calc_emb_similarity(track.embedding, det["embedding"])

                if (
                    text_emb_similarity > self.text_emb_threshold
                    or emb_similarity > self.emb_threshold
                ):
                    sim = calc_voxel_similarity(track, det, w_iou=0.4)
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

        # Match tentative tracks by SAM-ID
        for det in unmatched_detections[:]:
            for track in tentative_tracks:
                if not track.matched and det.get("sam_id") == track.sam_id:
                    if (
                        calc_voxel_similarity(track, det, w_iou=0.4)
                        > self.tentative_ass_threshold
                    ):
                        track.update(det, frame_id)
                        track.matched = True
                        unmatched_detections.remove(det)
                        break

        # Match tentative tracks by similarity
        for det in unmatched_detections[:]:
            best_track = None
            best_sim = 0

            for track in tentative_tracks:
                if track.matched:
                    continue

                text_emb_similarity = calc_emb_similarity(
                    track.text_embedding, det["text_embedding"]
                )
                emb_similarity = calc_emb_similarity(track.embedding, det["embedding"])

                if (
                    text_emb_similarity > self.text_emb_threshold
                    or emb_similarity > self.emb_threshold
                ):
                    sim = calc_voxel_similarity(track, det, w_iou=0.4)
                    if sim > best_sim and sim > self.tentative_ass_threshold:
                        best_sim = sim
                        best_track = track

            if best_track:
                best_track.update(det, frame_id)
                best_track.matched = True
                unmatched_detections.remove(det)
                matched_detections.add(id(det))
                continue

        # 5. Create new tracks from unmatched detections
        for det in unmatched_detections:
            new_track = Track(self.next_id, det, frame_id)
            new_track.state = "tentative"
            new_track.hits = 1
            new_track.sam_id = det.get("sam_id")
            self.tracks.append(new_track)
            self.next_id += 1

        # 6. Manage track states
        curr_active_tracks = []
        new_lost_tracks = []

        for track in self.tracks:
            if track.state == "tentative":
                if (
                    track.time_since_update > self.max_tentative_age
                    or track.hits >= track.min_hits_to_activate
                ):
                    if track.hits >= track.min_hits_to_activate:
                        track.state = "active"
                        track.was_active = True
                    else:
                        continue  # Remove candidate
                curr_active_tracks.append(track)
            else:  # active state
                if track.time_since_update > self.max_active_frames:
                    track.state = "lost"
                    new_lost_tracks.append(track)
                else:
                    curr_active_tracks.append(track)

        # Add revived tracks
        curr_active_tracks.extend(revived_tracks)

        # Update track lists
        self.tracks = curr_active_tracks

        # Clean up old lost tracks
        self.lost_tracks = [t for t in self.lost_tracks + new_lost_tracks]

        return self._get_current_tracks()

    def _get_current_tracks(self) -> List[Track]:
        """
        Get only active and tentative tracks.

        Returns:
            List of current tracks (excludes lost).
        """
        return [t for t in self.tracks if t.state in ["active", "tentative"]]
