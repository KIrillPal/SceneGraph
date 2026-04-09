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
DEFAULT_CONFIG = {
    "fps": 5.0,
    "scene_sub": 2,
    "track_sub": 1,
    "track_radius": 0.004,
    "track_voxels_sub": 1,
    "track_voxels_radius": 0.004,
    "mask_alpha": 0.3,
}
DESCRIPTION = """
# 3D tracker - Rerun

| Entity | Content |
|--------|---------|
| `world/camera` | Camera pose + pinhole |
| `world/scene_frame` | Scene point cloud for the current frame |
| `world/track_points` | Track point clouds reconstructed from masks |
| `world/track_voxels_merged` | Accumulated track geometry rebuilt over time |
| `world/track_labels` | Track ids at 3D centroids |
| `image/masked` | Image with track masks overlaid |
""".strip()

_INDEX_RE = re.compile(r"(\d+)(?!.*\d)")


class OnlineVoxelMap:
    """Lightweight voxel accumulator used to rebuild merged track geometry."""

    def __init__(self, voxel_size: float = 0.01, alpha: float = 0.3):
        self.voxel_size = voxel_size
        self.alpha = alpha
        self.voxels: dict[tuple[int, int, int], np.ndarray] = {}

    def add_points(self, points: np.ndarray) -> None:
        if len(points) == 0:
            return

        points = np.asarray(points, dtype=np.float32)
        indices = np.floor(points / self.voxel_size).astype(np.int32)
        for idx, point in zip(indices, points):
            key = (int(idx[0]), int(idx[1]), int(idx[2]))
            if key in self.voxels:
                self.voxels[key] = (1.0 - self.alpha) * self.voxels[
                    key
                ] + self.alpha * point
            else:
                self.voxels[key] = point.copy()

    def get_points(self) -> np.ndarray:
        if not self.voxels:
            return np.zeros((0, 3), dtype=np.float32)

        points = np.stack(list(self.voxels.values()), axis=0).astype(np.float32)
        points, _ = statistical_outlier_removal(points)
        return np.asarray(points, dtype=np.float32)


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
            rrb.Spatial2DView(name="Masked Image", origin=ENTITY_IMAGE_ROOT),
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
            "point_cloud": np.asarray(data["point_cloud"], dtype=np.float32),
            "intrinsic": np.asarray(data["intrinsic"], dtype=np.float32),
            "extrinsic": np.asarray(data["extrinsic"], dtype=np.float32),
        }


def _collect_track_ids(frame_paths: list[Path]) -> list[int]:
    track_ids: set[int] = set()
    for frame_path in frame_paths:
        payload = _load_frame_payload(frame_path)
        masks = payload["masks"]
        for track_map in masks.values():
            track_ids.update(int(track_id) for track_id in track_map)
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


def _extract_scene_points(
    image: np.ndarray, point_cloud: np.ndarray
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    valid_mask = np.isfinite(point_cloud).all(axis=2)
    if not np.any(valid_mask):
        return np.zeros((0, 3), dtype=np.float32), None

    points = point_cloud[valid_mask].astype(np.float32)
    h_pc, w_pc = point_cloud.shape[:2]
    image_resized = cv2.resize(image, (w_pc, h_pc), interpolation=cv2.INTER_LINEAR)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    colors = image_rgb[valid_mask].astype(np.uint8)
    return points, colors


def _extract_track_clouds(
    masks: dict[str, dict[int, np.ndarray]],
    point_cloud: np.ndarray,
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
            points, _ = statistical_outlier_removal(points)
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
            translation=extrinsic_w2c[:, 3],
            quaternion=rr.Quaternion(xyzw=rotation.as_quat()),
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
) -> None:
    for track_id, points in track_clouds:
        voxel_map = voxel_maps.setdefault(int(track_id), OnlineVoxelMap())
        voxel_map.add_points(points)


def _merge_track_voxel_clouds(
    voxel_maps: dict[int, OnlineVoxelMap],
    track_colors: dict[int, np.ndarray],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    voxel_clouds: list[tuple[int, np.ndarray]] = []
    for track_id in sorted(voxel_maps):
        points = voxel_maps[track_id].get_points()
        if len(points) == 0:
            continue
        voxel_clouds.append((track_id, points))
    return _merge_track_clouds(voxel_clouds, track_colors)


def run_from_export(export_dir: Path, rr_args: argparse.Namespace) -> None:
    export_dir = _resolve_export_dir(export_dir.resolve())
    frame_paths = _get_frame_paths(export_dir)
    track_colors = _make_track_color_map(_collect_track_ids(frame_paths))
    voxel_maps: dict[int, OnlineVoxelMap] = {}

    rr.script_setup(rr_args, "3d_tracker_rrd")
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    rr.log(
        "description",
        rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN),
        static=True,
    )
    blueprint_sent = False

    for frame_path in tqdm(
        frame_paths, desc="Writing Rerun frames", unit="frame", dynamic_ncols=True
    ):
        payload = _load_frame_payload(frame_path)
        frame_id = int(payload["frame_id"])
        image = np.asarray(payload["image"])
        masks = payload["masks"]
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
        if not blueprint_sent:
            blueprint = _build_blueprint()
            if blueprint is not None:
                rr.send_blueprint(blueprint, make_active=True, make_default=True)
            blueprint_sent = True

        scene_points, scene_colors = _extract_scene_points(image, point_cloud)
        _log_points(
            scene_points,
            scene_colors,
            ENTITY_SCENE,
            subsample=int(DEFAULT_CONFIG["scene_sub"]),
            static=False,
        )

        track_clouds = _extract_track_clouds(masks, point_cloud)
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

        _update_track_voxel_maps(track_clouds, voxel_maps)
        merged_voxel_points, merged_voxel_colors = _merge_track_voxel_clouds(
            voxel_maps,
            track_colors,
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
    rr.script_add_args(parser)
    args = parser.parse_args()

    export_dir = _resolve_export_dir(args.export_dir)
    if getattr(args, "save", None) is None:
        args.save = str(export_dir / "tracker_layers.rrd")

    run_from_export(export_dir=export_dir, rr_args=args)


if __name__ == "__main__":
    main()
