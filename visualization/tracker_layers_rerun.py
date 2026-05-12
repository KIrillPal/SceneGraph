#!/usr/bin/env python3
"""
Build tracker_layers .rrd from the per-frame tracker export saved by run_tracker.py.

Expected export directory content:
  - frame_000000.npz
  - frame_000001.npz
  - ...

Each frame NPZ contains:
  - frame_id
  - image
  - masks
  - embeddings
  - keypoints (optional)
  - moving_keypoints (optional)
  - keypoint_vis (optional)
  - object_states (optional)
  - point_cloud
  - intrinsic
  - extrinsic
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
from typing import Optional

import cv2
import distinctipy
import numpy as np
import rerun as rr
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from tqdm.auto import tqdm

try:
    import rerun.blueprint as rrb
except ImportError:
    rrb = None

if not hasattr(rr, "script_add_args"):
    raise ImportError(
        "rerun-sdk is required (not the old `rerun` package from PyPI). "
        "Use: pip uninstall rerun -y && pip install rerun-sdk"
    )

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CAMERA_ENTITY = "world/camera"
ENTITY_SCENE = "world/scene_frame"
ENTITY_TRACKS = "world/track_points"
ENTITY_TRACK_VOXELS = "world/track_voxels_merged"
ENTITY_TRACK_LABELS = "world/track_labels"
ENTITY_IMAGE_ROOT = "image"
ENTITY_MASKED_IMAGE = "image/masked"
ENTITY_KEYPOINT_IMAGE = "image/keypoints"
DEFAULT_CONFIG = {
    "fps": 5.0,
    "scene_sub": 2,
    "scene_voxel_size": 0.03,
    "track_sub": 1,
    "track_radius": 0.004,
    "track_voxel_size": 0.02,
    "track_voxels_sub": 1,
    "track_voxels_radius": 0.004,
    "track_accum_voxel_size": 0.05,
    "track_accum_log_stride": 5,
    "track_accum_max_points": 200000,
    "keypoint_history": 10,
    "mask_alpha": 0.3,
    "enable_sor": False,
}
DESCRIPTION = """
# 3D tracker - Rerun

| Entity | Content |
|--------|---------|
| `world/camera` | Camera pose + pinhole |
| `world/scene_frame` | Scene point cloud for the current frame |
| `world/track_points` | Track point clouds reconstructed from masks |
| `world/track_voxels_merged` | Accumulated track geometry, voxelized and logged sparsely |
| `world/track_labels` | Track ids at 3D centroids |
| `image/masked` | Image with track masks overlaid |
| `image/keypoints` | Image with per-track keypoints overlaid |
""".strip()

_INDEX_RE = re.compile(r"(\d+)(?!.*\d)")


def _voxel_downsample(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    voxel_size: float,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if voxel_size <= 0 or len(points) <= 1:
        return points, colors

    points = np.asarray(points, dtype=np.float32)
    finite_mask = np.isfinite(points).all(axis=1)
    if not np.all(finite_mask):
        points = points[finite_mask]
        colors = colors[finite_mask] if colors is not None else None
    if len(points) <= 1:
        return points, colors

    indices = np.floor(points / voxel_size).astype(np.int64)
    _, unique_indices = np.unique(indices, axis=0, return_index=True)
    unique_indices.sort()
    points = points[unique_indices]
    if colors is not None:
        colors = np.asarray(colors)[unique_indices]
    return points, colors


class OnlineVoxelMap:
    """Lightweight voxel accumulator used to rebuild merged track geometry."""

    def __init__(
        self, voxel_size: float = 0.02, alpha: float = 0.3, enable_sor: bool = False
    ):
        self.voxel_size = voxel_size
        self.alpha = alpha
        self.enable_sor = enable_sor
        self.voxels: dict[tuple[int, int, int], np.ndarray] = {}
        self._cached_points: Optional[np.ndarray] = None
        self._dirty = True

    def add_points(self, points: np.ndarray) -> None:
        if len(points) == 0:
            return

        points = np.asarray(points, dtype=np.float32)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            return

        # Reduce repeated raw points before touching the Python dict.
        points, _ = _voxel_downsample(points, None, self.voxel_size)
        indices = np.floor(points / self.voxel_size).astype(np.int32)
        for idx, point in zip(indices, points):
            key = (int(idx[0]), int(idx[1]), int(idx[2]))
            if key in self.voxels:
                self.voxels[key] = (1.0 - self.alpha) * self.voxels[
                    key
                ] + self.alpha * point
            else:
                self.voxels[key] = point.copy()
        self._dirty = True

    def get_points(self) -> np.ndarray:
        if not self._dirty and self._cached_points is not None:
            return self._cached_points

        if not self.voxels:
            self._cached_points = np.zeros((0, 3), dtype=np.float32)
            self._dirty = False
            return self._cached_points

        points = np.stack(list(self.voxels.values()), axis=0).astype(np.float32)
        if self.enable_sor:
            points, _ = statistical_outlier_removal(points)
        self._cached_points = np.asarray(points, dtype=np.float32)
        self._dirty = False
        return self._cached_points


def statistical_outlier_removal(
    points: np.ndarray,
    k: int = 10,
    std_ratio: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    if len(points) < k + 1:
        return points, np.ones(len(points), dtype=bool)
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=k + 1)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    mask = mean_distances < (global_mean + std_ratio * global_std)
    return points[mask], mask


def adaptive_erode_mask_area(
    mask: np.ndarray, target_area_ratio: float = 0.85
) -> np.ndarray:
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    original_area = cv2.countNonZero(mask)
    if original_area == 0:
        return mask
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.erode(mask, kernel, iterations=1)


def safe_erode(mask: np.ndarray, min_area_after: int = 32) -> np.ndarray:
    original_area = cv2.countNonZero(mask)
    if original_area < min_area_after * 2:
        return mask
    eroded = adaptive_erode_mask_area(mask)
    if cv2.countNonZero(eroded) < min_area_after:
        return mask
    return eroded


def _build_blueprint():
    if rrb is None:
        return None
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D", origin="world"),
            rrb.Vertical(
                rrb.Spatial2DView(name="Masked Image", origin=ENTITY_MASKED_IMAGE),
                rrb.Spatial2DView(name="Keypoints", origin=ENTITY_KEYPOINT_IMAGE),
            ),
            rrb.TextDocumentView(name="Info", origin="description"),
        )
    )


def _sort_key(path: Path) -> tuple[int, int | str]:
    match = _INDEX_RE.search(path.stem)
    if match is None:
        return (1, path.stem)
    return (0, int(match.group(1)))


def _resolve_export_dir(path: Path) -> Path:
    if path.is_dir() and list(path.glob("frame_*.npz")):
        return path
    candidate = path / "tracker_outputs"
    if candidate.is_dir() and list(candidate.glob("frame_*.npz")):
        return candidate
    raise FileNotFoundError(f"Could not find frame_*.npz export in {path}")


def _default_save_path(export_dir: Path) -> Path:
    if export_dir.name == "tracker_outputs":
        return export_dir.parent / "tracker_layers.rrd"
    return export_dir / "tracker_layers.rrd"


def _get_frame_paths(export_dir: Path) -> list[Path]:
    frame_paths = sorted(export_dir.glob("frame_*.npz"), key=_sort_key)
    if not frame_paths:
        raise FileNotFoundError(f"No frame_*.npz files found in {export_dir}")
    return frame_paths


def _load_object_dict(
    raw_obj: np.ndarray | dict | None, value_dtype: np.dtype
) -> dict[str, dict[int, np.ndarray]]:
    if raw_obj is None:
        return {}
    if isinstance(raw_obj, np.ndarray) and raw_obj.dtype == object:
        raw_obj = raw_obj.item()
    if raw_obj is None:
        return {}

    out: dict[str, dict[int, np.ndarray]] = {}
    for class_name, track_map in dict(raw_obj).items():
        out[str(class_name)] = {
            int(track_id): np.asarray(value, dtype=value_dtype)
            for track_id, value in dict(track_map).items()
        }
    return out


def _load_object_states(raw_obj: np.ndarray | dict | None) -> dict[str, str]:
    if raw_obj is None:
        return {}
    if isinstance(raw_obj, np.ndarray) and raw_obj.dtype == object:
        raw_obj = raw_obj.item()
    if raw_obj is None:
        return {}
    return {
        str(class_name).lower(): str(state).lower()
        for class_name, state in dict(raw_obj).items()
    }


def _load_keypoint_vis(raw_obj: np.ndarray | dict | None) -> dict[int, dict[str, object]]:
    if raw_obj is None:
        return {}
    if isinstance(raw_obj, np.ndarray) and raw_obj.dtype == object:
        raw_obj = raw_obj.item()
    if raw_obj is None:
        return {}

    out: dict[int, dict[str, object]] = {}
    for track_id, raw_info in dict(raw_obj).items():
        info = dict(raw_info)
        keypoints = info.get("keypoints")
        moving_keypoints = info.get("moving_keypoints")
        out[int(track_id)] = {
            "class_name": str(info.get("class_name", "")),
            "is_dynamic": bool(info.get("is_dynamic", False)),
            "motion_state": str(info.get("motion_state", "UNKNOWN")),
            "keypoints": (
                np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
                if keypoints is not None
                else None
            ),
            "moving_keypoints": (
                np.asarray(moving_keypoints, dtype=np.float32).reshape(-1, 2)
                if moving_keypoints is not None
                else None
            ),
        }
    return out


def _load_frame_payload(frame_path: Path) -> dict[str, object]:
    with np.load(frame_path, allow_pickle=True) as data:
        frame_id = (
            int(data["frame_id"])
            if "frame_id" in data.files
            else int(_sort_key(frame_path)[1])
        )
        return {
            "frame_id": frame_id,
            "image": np.asarray(data["image"]),
            "masks": _load_object_dict(data["masks"], np.bool_),
            "keypoints": _load_object_dict(
                data["keypoints"] if "keypoints" in data.files else None,
                np.float32,
            ),
            "moving_keypoints": _load_object_dict(
                data["moving_keypoints"] if "moving_keypoints" in data.files else None,
                np.float32,
            ),
            "keypoint_vis": _load_keypoint_vis(
                data["keypoint_vis"] if "keypoint_vis" in data.files else None
            ),
            "object_states": _load_object_states(
                data["object_states"] if "object_states" in data.files else None
            ),
            "point_cloud": np.asarray(data["point_cloud"], dtype=np.float32),
            "intrinsic": np.asarray(data["intrinsic"], dtype=np.float32),
            "extrinsic": np.asarray(data["extrinsic"], dtype=np.float32),
        }


def _load_frame_masks(frame_path: Path) -> dict[str, dict[int, np.ndarray]]:
    with np.load(frame_path, allow_pickle=True) as data:
        return _load_object_dict(data["masks"], np.bool_)


def _dynamic_track_ids_from_masks(
    masks: dict[str, dict[int, np.ndarray]],
    object_states: dict[str, str],
) -> set[int]:
    dynamic_track_ids: set[int] = set()
    for class_name, track_map in masks.items():
        if object_states.get(str(class_name).lower()) != "dynamic":
            continue
        dynamic_track_ids.update(int(track_id) for track_id in track_map)
    return dynamic_track_ids


def _collect_track_ids(frame_paths: list[Path]) -> list[int]:
    track_ids: set[int] = set()
    for frame_path in frame_paths:
        masks = _load_frame_masks(frame_path)
        for track_map in masks.values():
            track_ids.update(int(track_id) for track_id in track_map)
        with np.load(frame_path, allow_pickle=True) as data:
            keypoint_vis = _load_keypoint_vis(
                data["keypoint_vis"] if "keypoint_vis" in data.files else None
            )
            track_ids.update(keypoint_vis)
    return sorted(track_ids)


def _make_track_color_map(track_ids: list[int]) -> dict[int, np.ndarray]:
    colors = distinctipy.get_colors(len(track_ids))
    return {
        track_id: (np.asarray(color, dtype=np.float64) * 255.0)
        .clip(0, 255)
        .astype(np.uint8)
        for track_id, color in zip(track_ids, colors)
    }


def _rgb_to_bgr(color: np.ndarray) -> np.ndarray:
    return np.asarray(color, dtype=np.uint8)[::-1]


def _resize_mask(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    if mask.shape[:2] == (height, width):
        return np.asarray(mask, dtype=bool)
    resized = cv2.resize(
        mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    )
    return resized > 0


def _render_masked_image(
    image: np.ndarray,
    masks: dict[str, dict[int, np.ndarray]],
    track_colors: dict[int, np.ndarray],
    alpha: float,
) -> np.ndarray:
    overlay = np.zeros_like(image)
    h, w = image.shape[:2]
    for track_map in masks.values():
        for track_id, mask in track_map.items():
            resized_mask = _resize_mask(mask, (h, w))
            overlay[resized_mask] = _rgb_to_bgr(track_colors[int(track_id)])
    return cv2.addWeighted(image, 1.0 - alpha, overlay, alpha, 0.0)


def _detect_keypoints_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    max_keypoints: int = 50,
) -> np.ndarray:
    resized_mask = _resize_mask(mask, image.shape[:2]).astype(np.uint8)
    if cv2.countNonZero(resized_mask) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_keypoints,
        qualityLevel=0.01,
        minDistance=5,
        mask=resized_mask,
    )
    if keypoints is None:
        return np.zeros((0, 2), dtype=np.float32)
    return keypoints.reshape(-1, 2).astype(np.float32)


def _render_keypoint_image(
    image: np.ndarray,
    masks: dict[str, dict[int, np.ndarray]],
    keypoints: dict[str, dict[int, np.ndarray]],
    moving_keypoints: dict[str, dict[int, np.ndarray]],
    track_colors: dict[int, np.ndarray],
) -> np.ndarray:
    vis = image.copy()
    h, w = image.shape[:2]

    for class_name, track_map in masks.items():
        class_keypoints = keypoints.get(class_name, {})
        class_moving_keypoints = moving_keypoints.get(class_name, {})
        for track_id, mask in track_map.items():
            track_id = int(track_id)
            color = tuple(int(c) for c in _rgb_to_bgr(track_colors[track_id]))

            resized_mask = _resize_mask(mask, (h, w)).astype(np.uint8)
            contours, _ = cv2.findContours(
                resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, color, 1)

            kps = class_keypoints.get(track_id)
            if kps is None or len(kps) == 0:
                kps = _detect_keypoints_from_mask(image, mask)
            kps = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
            for x, y in kps:
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < w and 0 <= yi < h:
                    cv2.circle(vis, (xi, yi), 2, color, -1, lineType=cv2.LINE_AA)

            moving_kps = class_moving_keypoints.get(track_id)
            if moving_kps is not None and len(moving_kps) > 0:
                moving_kps = np.asarray(moving_kps, dtype=np.float32).reshape(-1, 2)
                for x, y in moving_kps:
                    xi, yi = int(round(x)), int(round(y))
                    if 0 <= xi < w and 0 <= yi < h:
                        cv2.drawMarker(
                            vis,
                            (xi, yi),
                            color,
                            markerType=cv2.MARKER_CROSS,
                            markerSize=6,
                            thickness=1,
                            line_type=cv2.LINE_AA,
                        )

            coords = np.column_stack(np.where(resized_mask > 0))
            if len(coords) > 0:
                y_min, x_min = np.min(coords, axis=0)
                cv2.putText(
                    vis,
                    str(track_id),
                    (int(x_min), max(0, int(y_min) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    return vis


def _update_keypoint_history(
    history: dict[int, list[tuple[int, np.ndarray]]],
    frame_id: int,
    keypoint_vis: dict[int, dict[str, object]],
    max_history: int,
) -> None:
    min_frame = frame_id - max_history
    for track_id, info in keypoint_vis.items():
        moving_keypoints = info.get("moving_keypoints")
        if moving_keypoints is None or len(moving_keypoints) == 0:
            continue
        track_history = history.setdefault(int(track_id), [])
        track_history.append(
            (frame_id, np.asarray(moving_keypoints, dtype=np.float32).reshape(-1, 2))
        )

    for track_id in list(history):
        history[track_id] = [item for item in history[track_id] if item[0] >= min_frame]
        if not history[track_id]:
            history.pop(track_id, None)


def _render_keypoint_vis_image(
    image: np.ndarray,
    current_frame: int,
    keypoint_vis: dict[int, dict[str, object]],
    history: dict[int, list[tuple[int, np.ndarray]]],
    track_colors: dict[int, np.ndarray],
    show_history: int,
) -> np.ndarray:
    vis = image.copy()
    h, w = image.shape[:2]
    total_shi_tomasi = 0
    total_cotracker = 0

    for track_id in sorted(keypoint_vis):
        info = keypoint_vis[track_id]
        # Match visualize_all_tracks_keypoints: only moving dynamic tracks are shown.
        if not bool(info.get("is_dynamic", False)):
            continue
        if str(info.get("motion_state", "")).upper() != "MOVING":
            continue

        color_bgr = tuple(int(c) for c in _rgb_to_bgr(track_colors[int(track_id)]))

        # Match notebook trail logic: concatenate the first few CoTracker points
        # from the current frame history window into one polyline.
        trail_points: list[tuple[int, int]] = []
        if show_history >= 0:
            track_history = dict(history.get(int(track_id), []))
            for frame_idx in range(max(0, current_frame - show_history), current_frame + 1):
                moving_keypoints = track_history.get(frame_idx)
                if moving_keypoints is None or len(moving_keypoints) == 0:
                    continue
                moving_keypoints = np.asarray(moving_keypoints, dtype=np.float32).reshape(-1, 2)
                for x, y in moving_keypoints[:5]:
                    xi, yi = int(round(float(x))), int(round(float(y)))
                    if 0 <= xi < w and 0 <= yi < h:
                        trail_points.append((xi, yi))
        for idx in range(len(trail_points) - 1):
            alpha = (idx + 1) / max(1, len(trail_points))
            trail_color = tuple(int(c * alpha) for c in color_bgr)
            cv2.line(
                vis,
                trail_points[idx],
                trail_points[idx + 1],
                trail_color,
                thickness=3,
                lineType=cv2.LINE_AA,
            )

        keypoints = info.get("keypoints")
        if keypoints is not None and len(keypoints) > 0:
            keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
            total_shi_tomasi += len(keypoints)
            for x, y in keypoints:
                xi, yi = int(round(float(x))), int(round(float(y)))
                if 0 <= xi < w and 0 <= yi < h:
                    cv2.circle(vis, (xi, yi), 4, (255, 255, 255), -1)
                    cv2.circle(vis, (xi, yi), 3, color_bgr, -1)

        moving_keypoints = info.get("moving_keypoints")
        if moving_keypoints is not None and len(moving_keypoints) > 0:
            moving_keypoints = np.asarray(moving_keypoints, dtype=np.float32).reshape(-1, 2)
            total_cotracker += len(moving_keypoints)
            for x, y in moving_keypoints:
                xi, yi = int(round(float(x))), int(round(float(y)))
                if 0 <= xi < w and 0 <= yi < h:
                    diamond_size = 4
                    pts = np.array(
                        [
                            [xi, yi - diamond_size],
                            [xi + diamond_size, yi],
                            [xi, yi + diamond_size],
                            [xi - diamond_size, yi],
                        ],
                        np.int32,
                    )
                    cv2.fillPoly(vis, [pts], color_bgr)
                    cv2.polylines(vis, [pts], True, (255, 255, 255), 1)

        label_points = []
        if keypoints is not None and len(keypoints) > 0:
            label_points.append(keypoints)
        if moving_keypoints is not None and len(moving_keypoints) > 0:
            label_points.append(moving_keypoints)
        if label_points:
            center = np.mean(np.vstack(label_points), axis=0)
            xi, yi = int(round(float(center[0]))), int(round(float(center[1])))
            label = f"#{track_id}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_w, text_h = text_size
            cv2.rectangle(
                vis,
                (xi - 5, yi - text_h - 8),
                (xi + text_w + 5, yi + 3),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                vis,
                label,
                (xi, yi - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_bgr,
                1,
            )

    legend_h = 120
    cv2.rectangle(vis, (10, 10), (250, 10 + legend_h), (0, 0, 0), -1)
    cv2.putText(
        vis,
        "Keypoint Visualization",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.line(vis, (20, 45), (240, 45), (255, 255, 255), 1)
    cv2.circle(vis, (30, 65), 3, (200, 200, 200), -1)
    cv2.putText(
        vis,
        "Shi-Tomasi (detected)",
        (45, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
    )
    diamond_pts = np.array([[30, 90], [34, 94], [30, 98], [26, 94]], np.int32)
    cv2.fillPoly(vis, [diamond_pts], (200, 200, 200))
    cv2.polylines(vis, [diamond_pts], True, (255, 255, 255), 1)
    cv2.putText(
        vis,
        "CoTracker (predicted)",
        (45, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 255, 255),
        1,
    )
    cv2.line(vis, (20, 110), (240, 110), (255, 255, 255), 1)
    cv2.putText(
        vis,
        f"Shi-Tomasi: {total_shi_tomasi}",
        (20, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (0, 255, 0),
        1,
    )
    cv2.putText(
        vis,
        f"CoTracker: {total_cotracker}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (255, 0, 0),
        1,
    )
    cv2.putText(
        vis,
        f"Frame {current_frame}",
        (w - 150, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    return vis


def _extract_scene_points(
    image: np.ndarray, point_cloud: np.ndarray, voxel_size: float = 0.0
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    valid_mask = np.isfinite(point_cloud).all(axis=2)
    if not np.any(valid_mask):
        return np.zeros((0, 3), dtype=np.float32), None

    points = point_cloud[valid_mask].astype(np.float32)
    h_pc, w_pc = point_cloud.shape[:2]
    image_resized = cv2.resize(image, (w_pc, h_pc), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    colors = image_rgb[valid_mask].astype(np.uint8)
    points, colors = _voxel_downsample(points, colors, voxel_size)
    return points, colors


def _extract_track_clouds(
    masks: dict[str, dict[int, np.ndarray]],
    point_cloud: np.ndarray,
    voxel_size: float = 0.0,
    enable_sor: bool = False,
) -> list[tuple[int, np.ndarray]]:
    clouds: list[tuple[int, np.ndarray]] = []
    h_pc, w_pc = point_cloud.shape[:2]
    valid_points = np.isfinite(point_cloud).all(axis=2)

    for track_map in masks.values():
        for track_id, mask in track_map.items():
            resized_mask = cv2.resize(
                mask.astype(np.uint8),
                (w_pc, h_pc),
                interpolation=cv2.INTER_NEAREST,
            )
            eroded_mask = safe_erode(resized_mask)
            object_mask = valid_points & (eroded_mask > 0)
            points = point_cloud[object_mask]
            if enable_sor:
                points, _ = statistical_outlier_removal(points)
            points, _ = _voxel_downsample(points, None, voxel_size)
            if len(points) == 0:
                continue
            clouds.append((int(track_id), np.asarray(points, dtype=np.float32)))
    return clouds


def _log_camera_transform(
    extrinsic_c2w: np.ndarray,
    intrinsic: np.ndarray,
    resolution_wh: tuple[int, int],
) -> None:
    extrinsic_w2c = np.linalg.inv(extrinsic_c2w)[:3, :]
    rotation = Rotation.from_matrix(extrinsic_w2c[:3, :3])
    rr.log(
        CAMERA_ENTITY,
        rr.Transform3D(
            translation=np.asarray(extrinsic_w2c[:, 3], dtype=np.float32),
            quaternion=np.asarray(rotation.as_quat(), dtype=np.float32),
            relation=rr.TransformRelation.ChildFromParent,
        ),
    )
    rr.log(
        CAMERA_ENTITY,
        rr.Pinhole(
            image_from_camera=intrinsic,
            resolution=list(resolution_wh),
        ),
    )


def _log_points(
    positions: np.ndarray,
    colors_u8: Optional[np.ndarray],
    entity: str,
    subsample: int = 1,
    static: bool = False,
    radius: float = 0.005,
) -> None:
    if len(positions) == 0:
        rr.log(
            entity,
            rr.Points3D(positions=np.zeros((0, 3), dtype=np.float32)),
            static=static,
        )
        return
    pts = np.asarray(positions, dtype=np.float32)[::subsample]
    cols = (
        np.asarray(colors_u8, dtype=np.uint8)[::subsample]
        if colors_u8 is not None
        else None
    )
    rr.log(entity, rr.Points3D(positions=pts, colors=cols, radii=radius), static=static)


def _log_track_id_labels(
    clouds: list[tuple[int, np.ndarray]],
    track_colors: dict[int, np.ndarray],
    label_radius: float = 0.02,
) -> None:
    if not clouds:
        rr.log(
            ENTITY_TRACK_LABELS,
            rr.Points3D(positions=np.zeros((0, 3), dtype=np.float32)),
            static=False,
        )
        return

    centers: list[np.ndarray] = []
    labels: list[str] = []
    colors_rgb: list[np.ndarray] = []
    for track_id, points in clouds:
        centers.append(np.mean(points, axis=0))
        labels.append(str(track_id))
        colors_rgb.append(track_colors[int(track_id)])
    rr.log(
        ENTITY_TRACK_LABELS,
        rr.Points3D(
            positions=np.stack(centers, axis=0),
            labels=labels,
            show_labels=True,
            radii=label_radius,
            colors=np.stack(colors_rgb, axis=0),
        ),
        static=False,
    )


def _merge_track_clouds(
    clouds: list[tuple[int, np.ndarray]],
    track_colors: dict[int, np.ndarray],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if not clouds:
        return np.zeros((0, 3), dtype=np.float32), None

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    for track_id, points in clouds:
        all_points.append(points)
        all_colors.append(np.tile(track_colors[int(track_id)], (len(points), 1)))
    return np.vstack(all_points), np.vstack(all_colors)


def _update_track_voxel_maps(
    track_clouds: list[tuple[int, np.ndarray]],
    voxel_maps: dict[int, OnlineVoxelMap],
    dynamic_track_ids: set[int],
    voxel_size: float,
    enable_sor: bool,
) -> None:
    for track_id, points in track_clouds:
        track_id = int(track_id)
        if track_id in dynamic_track_ids:
            voxel_maps.pop(track_id, None)
            continue
        voxel_map = voxel_maps.setdefault(
            track_id, OnlineVoxelMap(voxel_size=voxel_size, enable_sor=enable_sor)
        )
        voxel_map.add_points(points)


def _merge_track_voxel_clouds(
    voxel_maps: dict[int, OnlineVoxelMap],
    track_colors: dict[int, np.ndarray],
    final_voxel_size: float = 0.0,
    max_points: int = 0,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    voxel_clouds: list[tuple[int, np.ndarray]] = []
    for track_id in sorted(voxel_maps):
        points = voxel_maps[track_id].get_points()
        if len(points) == 0:
            continue
        voxel_clouds.append((track_id, points))
    points, colors = _merge_track_clouds(voxel_clouds, track_colors)
    points, colors = _voxel_downsample(points, colors, final_voxel_size)
    if max_points > 0 and len(points) > max_points:
        sample_indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points = points[sample_indices]
        colors = colors[sample_indices] if colors is not None else None
    return points, colors


def run_from_export(export_dir: Path, rr_args: argparse.Namespace) -> None:
    export_dir = _resolve_export_dir(export_dir.resolve())
    frame_paths = _get_frame_paths(export_dir)
    track_colors = _make_track_color_map(_collect_track_ids(frame_paths))
    voxel_maps: dict[int, OnlineVoxelMap] = {}
    scene_voxel_size = float(
        getattr(rr_args, "scene_voxel_size", DEFAULT_CONFIG["scene_voxel_size"])
    )
    track_voxel_size = float(
        getattr(rr_args, "track_voxel_size", DEFAULT_CONFIG["track_voxel_size"])
    )
    track_accum_voxel_size = float(
        getattr(
            rr_args,
            "track_accum_voxel_size",
            DEFAULT_CONFIG["track_accum_voxel_size"],
        )
    )
    track_accum_log_stride = max(
        1,
        int(
            getattr(
                rr_args,
                "track_accum_log_stride",
                DEFAULT_CONFIG["track_accum_log_stride"],
            )
        ),
    )
    track_accum_max_points = int(
        getattr(
            rr_args,
            "track_accum_max_points",
            DEFAULT_CONFIG["track_accum_max_points"],
        )
    )
    keypoint_history = max(
        0,
        int(getattr(rr_args, "keypoint_history", DEFAULT_CONFIG["keypoint_history"])),
    )
    enable_sor = bool(getattr(rr_args, "enable_sor", DEFAULT_CONFIG["enable_sor"]))

    rr.script_setup(rr_args, "3d_tracker_rrd")
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    rr.log(
        "description",
        rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN),
        static=True,
    )
    blueprint_sent = False
    keypoint_history_by_track: dict[int, list[tuple[int, np.ndarray]]] = {}

    for frame_offset, frame_path in enumerate(
        tqdm(frame_paths, desc="Writing Rerun frames", unit="frame", dynamic_ncols=True)
    ):
        payload = _load_frame_payload(frame_path)
        frame_id = int(payload["frame_id"])
        image = np.asarray(payload["image"])
        masks = payload["masks"]
        keypoints = payload["keypoints"]
        moving_keypoints = payload["moving_keypoints"]
        keypoint_vis = payload["keypoint_vis"]
        object_states = payload["object_states"]
        point_cloud = np.asarray(payload["point_cloud"], dtype=np.float32)
        intrinsic = np.asarray(payload["intrinsic"], dtype=np.float32)
        extrinsic = np.asarray(payload["extrinsic"], dtype=np.float32)

        rr.set_time_sequence("frame", frame_id)
        rr.set_time_seconds("time", frame_id / float(DEFAULT_CONFIG["fps"]))

        h_pc, w_pc = point_cloud.shape[:2]
        _log_camera_transform(extrinsic, intrinsic, (w_pc, h_pc))

        masked_image = _render_masked_image(
            image,
            masks,
            track_colors,
            alpha=float(DEFAULT_CONFIG["mask_alpha"]),
        )
        rr.log(ENTITY_MASKED_IMAGE, rr.Image(masked_image, color_model="bgr"))

        _update_keypoint_history(
            keypoint_history_by_track, frame_id, keypoint_vis, keypoint_history
        )
        keypoint_image = (
            _render_keypoint_vis_image(
                image,
                frame_id,
                keypoint_vis,
                keypoint_history_by_track,
                track_colors,
                show_history=keypoint_history,
            )
            if keypoint_vis
            else _render_keypoint_image(
                image,
                masks,
                keypoints,
                moving_keypoints,
                track_colors,
            )
        )
        rr.log(ENTITY_KEYPOINT_IMAGE, rr.Image(keypoint_image, color_model="bgr"))
        if not blueprint_sent:
            blueprint = _build_blueprint()
            if blueprint is not None:
                rr.send_blueprint(blueprint, make_active=True, make_default=True)
            blueprint_sent = True

        scene_points, scene_colors = _extract_scene_points(
            image, point_cloud, voxel_size=scene_voxel_size
        )
        _log_points(
            scene_points,
            scene_colors,
            ENTITY_SCENE,
            subsample=int(DEFAULT_CONFIG["scene_sub"]),
            static=False,
        )

        track_clouds = _extract_track_clouds(
            masks,
            point_cloud,
            voxel_size=track_voxel_size,
            enable_sor=enable_sor,
        )
        _log_track_id_labels(track_clouds, track_colors)
        track_points, track_point_colors = _merge_track_clouds(
            track_clouds, track_colors
        )
        _log_points(
            track_points,
            track_point_colors,
            ENTITY_TRACKS,
            subsample=int(DEFAULT_CONFIG["track_sub"]),
            static=False,
            radius=float(DEFAULT_CONFIG["track_radius"]),
        )

        _update_track_voxel_maps(
            track_clouds,
            voxel_maps,
            _dynamic_track_ids_from_masks(masks, object_states),
            voxel_size=track_accum_voxel_size,
            enable_sor=enable_sor,
        )
        should_log_accum = (
            frame_offset % track_accum_log_stride == 0
            or frame_offset == len(frame_paths) - 1
        )
        if should_log_accum:
            merged_voxel_points, merged_voxel_colors = _merge_track_voxel_clouds(
                voxel_maps,
                track_colors,
                final_voxel_size=track_accum_voxel_size,
                max_points=track_accum_max_points,
            )
            _log_points(
                merged_voxel_points,
                merged_voxel_colors,
                ENTITY_TRACK_VOXELS,
                subsample=int(DEFAULT_CONFIG["track_voxels_sub"]),
                static=False,
                radius=float(DEFAULT_CONFIG["track_voxels_radius"]),
            )

    rr.script_teardown(rr_args)
    logger.info("Saved %s", rr_args.save)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build tracker_layers .rrd from run_tracker.py per-frame export."
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="Path to the tracker output directory with frame_*.npz files.",
    )
    parser.add_argument(
        "--scene-voxel-size",
        type=float,
        default=float(DEFAULT_CONFIG["scene_voxel_size"]),
        help="Voxel size for scene point clouds before logging. Use 0 to disable.",
    )
    parser.add_argument(
        "--track-voxel-size",
        type=float,
        default=float(DEFAULT_CONFIG["track_voxel_size"]),
        help="Voxel size for per-frame object clouds before logging. Use 0 to disable.",
    )
    parser.add_argument(
        "--track-accum-voxel-size",
        type=float,
        default=float(DEFAULT_CONFIG["track_accum_voxel_size"]),
        help="Voxel size for accumulated track geometry before logging.",
    )
    parser.add_argument(
        "--track-accum-log-stride",
        type=int,
        default=int(DEFAULT_CONFIG["track_accum_log_stride"]),
        help="Log accumulated track geometry every N frames to reduce .rrd size.",
    )
    parser.add_argument(
        "--track-accum-max-points",
        type=int,
        default=int(DEFAULT_CONFIG["track_accum_max_points"]),
        help="Maximum accumulated track points to log per accumulated frame. Use 0 to disable.",
    )
    parser.add_argument(
        "--keypoint-history",
        type=int,
        default=int(DEFAULT_CONFIG["keypoint_history"]),
        help="Number of previous frames to use for CoTracker keypoint trails.",
    )
    parser.add_argument(
        "--enable-sor",
        action="store_true",
        help="Enable slow statistical outlier removal for visualization clouds.",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    export_dir = _resolve_export_dir(args.export_dir)
    if getattr(args, "save", None) is None:
        args.save = str(_default_save_path(export_dir))

    run_from_export(export_dir=export_dir, rr_args=args)


if __name__ == "__main__":
    main()
