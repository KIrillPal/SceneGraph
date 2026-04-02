#!/usr/bin/env python3
"""
Build tracker_layers .rrd from the export saved by run_tracker.py.

This script follows the same visualization logic as the notebook path:
  - camera pose + pinhole per frame
  - scene points per frame colored from DA3 RGB images
  - track points per frame reconstructed from saved masks
  - merged voxel clouds per track from saved voxel history
  - track id labels in 3D

Expected export directory content:
  - tracks.pkl
  - extrinsics.npy
  - points_per_frame.pkl
  - points_per_frame_masks.pkl
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import cv2
import distinctipy
import numpy as np
import open3d as o3d
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
DEFAULT_CONFIG = {
    "fps": 5.0,
    "scene_sub": 2,
    "track_sub": 1,
    "track_voxels_sub": 1,
    "track_voxels_radius": 0.004,
}

DESCRIPTION = """
# 3D tracker — Rerun

| Entity | Content |
|--------|---------|
| `world/camera` | Camera pose + pinhole |
| `world/scene_frame` | Scene point cloud for the current frame |
| `world/track_points` | Track point clouds reconstructed from masks |
| `world/track_voxels_merged` | Saved merged voxel clouds per track |
| `world/track_labels` | Track ids at 3D centroids |
""".strip()


class TrackLike:
    def __init__(
        self,
        track_id: int,
        masks: dict[int, np.ndarray],
        voxels_by_frame: Optional[dict[int, np.ndarray]] = None,
    ) -> None:
        self.id = track_id
        self.masks = masks
        self.voxels_by_frame = voxels_by_frame or {}


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


def adaptive_erode_mask_area(mask: np.ndarray, target_area_ratio: float = 0.85) -> np.ndarray:
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    original_area = cv2.countNonZero(mask)
    if original_area == 0:
        return mask
    kernel_size = 3
    while kernel_size <= 7:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
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


def get_obj_point_cloud(
    points: np.ndarray,
    clean_mask: np.ndarray,
    mask: np.ndarray,
    h: int = 378,
    w: int = 504,
) -> np.ndarray:
    mask_r = cv2.resize(mask.astype(np.uint8), (w, h))
    eroded_mask = safe_erode(mask_r)
    mask_r = eroded_mask.flatten()[clean_mask]
    pts_mask = points[mask_r > 0]
    pts_mask, _ = statistical_outlier_removal(pts_mask)
    return pts_mask


def track_point_clouds_at_frame(
    frame_idx: int,
    all_tracks: list[TrackLike],
    points_per_frame: list[np.ndarray],
    points_per_frame_masks: list[np.ndarray],
    h: int,
    w: int,
) -> list[tuple[int, np.ndarray]]:
    clouds: list[tuple[int, np.ndarray]] = []
    for track in all_tracks:
        if frame_idx not in track.masks:
            continue
        pts = get_obj_point_cloud(
            points_per_frame[frame_idx],
            points_per_frame_masks[frame_idx],
            track.masks[frame_idx],
            h,
            w,
        )
        if len(pts) == 0:
            continue
        clouds.append((track.id, pts))
    return clouds


def tracks_to_merged_open3d_pcd(
    frame_idx: int,
    all_tracks: list[TrackLike],
    points_per_frame: list[np.ndarray],
    points_per_frame_masks: list[np.ndarray],
    h: int,
    w: int,
    track_colors: np.ndarray,
    clouds: Optional[list[tuple[int, np.ndarray]]] = None,
) -> o3d.geometry.PointCloud:
    id_to_j = {t.id: j for j, t in enumerate(all_tracks)}
    if clouds is None:
        clouds = track_point_clouds_at_frame(
            frame_idx, all_tracks, points_per_frame, points_per_frame_masks, h, w
        )
    merged = o3d.geometry.PointCloud()
    if not clouds:
        return merged
    parts: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    for tid, pts in clouds:
        j = id_to_j.get(tid, 0)
        c = np.asarray(track_colors[j], dtype=np.float64)
        parts.append(pts)
        cols.append(np.tile(c, (len(pts), 1)))
    all_pts = np.vstack(parts)
    all_cols = np.vstack(cols)
    merged.points = o3d.utility.Vector3dVector(all_pts)
    merged.colors = o3d.utility.Vector3dVector(all_cols)
    return merged


def merged_track_voxels_at_frame(
    frame_idx: int,
    all_tracks: list[TrackLike],
) -> list[tuple[int, np.ndarray]]:
    merged_clouds: list[tuple[int, np.ndarray]] = []
    for track in all_tracks:
        pts = track.voxels_by_frame.get(frame_idx)
        if pts is None:
            continue
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) == 0:
            continue
        merged_clouds.append((track.id, pts))
    return merged_clouds


def _build_blueprint():
    if rrb is None:
        return None
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D", origin="world"),
            rrb.TextDocumentView(name="Info", origin="description"),
        )
    )


def _resolve_export_dir(path: Path) -> Path:
    if path.name == "rerun_export":
        return path
    candidate = path / "rerun_export"
    if candidate.is_dir():
        return candidate
    return path


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_tracks_pickle(path: Path) -> list[TrackLike]:
    raw: list[dict[str, Any]] = _load_pickle(path)
    out: list[TrackLike] = []
    for item in raw:
        masks = {int(k): v for k, v in item["masks"].items()}
        voxels_by_frame = {
            int(k): np.asarray(v, dtype=np.float32)
            for k, v in item.get("voxels_by_frame", {}).items()
        }
        out.append(TrackLike(int(item["id"]), masks, voxels_by_frame))
    return out


def _find_dataset_root(export_dir: Path) -> Optional[Path]:
    for parent in export_dir.parents:
        if (parent / "da3_outputs").is_dir():
            return parent
    return None


def _infer_runtime_config(
    export_dir: Path,
    depth_dir: Optional[Path],
) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    dataset_root = _find_dataset_root(export_dir)

    if depth_dir is not None:
        cfg["depth_dir"] = depth_dir
    elif dataset_root is not None:
        cfg["depth_dir"] = dataset_root / "da3_outputs" / "results_output"
    else:
        cfg["depth_dir"] = None
    cfg["dataset_root"] = dataset_root
    return cfg


def _log_camera_transform(
    extrinsic_w2c: np.ndarray,
    intrinsic: np.ndarray,
    resolution_wh: tuple[int, int],
) -> None:
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
        rr.log(entity, rr.Points3D(positions=np.zeros((0, 3), dtype=np.float32)), static=static)
        return
    pts = np.asarray(positions, dtype=np.float32)[::subsample]
    cols = np.asarray(colors_u8, dtype=np.uint8)[::subsample] if colors_u8 is not None else None
    rr.log(entity, rr.Points3D(positions=pts, colors=cols, radii=radius), static=static)


def _log_track_id_labels(
    clouds: list[tuple[int, np.ndarray]],
    track_colors: np.ndarray,
    all_tracks: list[TrackLike],
    label_radius: float = 0.02,
) -> None:
    if not clouds:
        rr.log(
            ENTITY_TRACK_LABELS,
            rr.Points3D(positions=np.zeros((0, 3), dtype=np.float32)),
            static=False,
        )
        return

    id_to_j = {t.id: j for j, t in enumerate(all_tracks)}
    centers: list[np.ndarray] = []
    labels: list[str] = []
    colors_rgb: list[np.ndarray] = []
    for tid, pts in clouds:
        centers.append(np.mean(pts, axis=0))
        labels.append(str(tid))
        j = id_to_j.get(tid, 0)
        c = track_colors[j]
        rgb = (np.asarray(c, dtype=np.float64) * 255.0).clip(0, 255).astype(np.uint8)
        colors_rgb.append(rgb)
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


def run_from_export(
    export_dir: Path,
    rr_args: argparse.Namespace,
    depth_dir: Optional[Path],
    fps: Optional[float],
    scene_sub: Optional[int],
    track_sub: Optional[int],
    track_voxels_sub: Optional[int],
    track_voxels_radius: Optional[float],
) -> None:
    export_dir = _resolve_export_dir(export_dir.resolve())
    required_files = (
        export_dir / "tracks.pkl",
        export_dir / "extrinsics.npy",
        export_dir / "points_per_frame.pkl",
        export_dir / "points_per_frame_masks.pkl",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing export files:\n" + "\n".join(missing))

    cfg = _infer_runtime_config(export_dir, depth_dir=depth_dir)
    if fps is not None:
        cfg["fps"] = fps
    if scene_sub is not None:
        cfg["scene_sub"] = scene_sub
    if track_sub is not None:
        cfg["track_sub"] = track_sub
    if track_voxels_sub is not None:
        cfg["track_voxels_sub"] = track_voxels_sub
    if track_voxels_radius is not None:
        cfg["track_voxels_radius"] = track_voxels_radius

    if cfg["depth_dir"] is None:
        raise FileNotFoundError(
            "Could not infer depth directory from export path. "
            "Pass --depth-dir explicitly."
        )

    depth_dir = Path(cfg["depth_dir"])
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory does not exist: {depth_dir}")

    logger.info("Loading tracker export from %s", export_dir)
    logger.info("Using depth frames from %s", depth_dir)

    extr = np.load(export_dir / "extrinsics.npy")
    with (export_dir / "points_per_frame.pkl").open("rb") as f:
        points_per_frame: list[np.ndarray] = pickle.load(f)
    with (export_dir / "points_per_frame_masks.pkl").open("rb") as f:
        points_per_frame_masks: list[np.ndarray] = pickle.load(f)
    all_tracks = _load_tracks_pickle(export_dir / "tracks.pkl")
    track_colors = np.array(distinctipy.get_colors(len(all_tracks)), dtype=np.float64)

    n_frames = min(len(extr), len(points_per_frame), len(points_per_frame_masks))

    rr.script_setup(rr_args, "3d_tracker_rrd", default_blueprint=_build_blueprint())
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    rr.log(
        "description",
        rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN),
        static=True,
    )

    for i in tqdm(range(n_frames), desc="Writing Rerun frames", unit="frame", dynamic_ncols=True):
        rr.set_time_sequence("frame", i)
        rr.set_time_seconds("time", i / float(cfg["fps"]))

        frame_file = depth_dir / f"frame_{i}.npz"
        data = np.load(frame_file)
        depth_i = data["depth"]
        intrinsics_i = data["intrinsics"]
        image_i = data["image"]
        hi, wi = int(depth_i.shape[0]), int(depth_i.shape[1])
        resolution_wh = (wi, hi)

        c2w = extr[i]
        w2c = np.linalg.inv(c2w)[:3, :]
        _log_camera_transform(w2c, intrinsics_i, resolution_wh)

        mask = points_per_frame_masks[i]
        scene_pts = points_per_frame[i]
        rgb_flat = image_i.reshape(-1, 3)
        scene_cols = rgb_flat[mask].astype(np.uint8)
        _log_points(scene_pts, scene_cols, ENTITY_SCENE, subsample=int(cfg["scene_sub"]), static=False)

        track_clouds = track_point_clouds_at_frame(
            i,
            all_tracks,
            points_per_frame,
            points_per_frame_masks,
            hi,
            wi,
        )
        _log_track_id_labels(track_clouds, track_colors, all_tracks)

        pcd_tr = tracks_to_merged_open3d_pcd(
            i,
            all_tracks,
            points_per_frame,
            points_per_frame_masks,
            hi,
            wi,
            track_colors=track_colors,
            clouds=track_clouds,
        )
        tr_pts = np.asarray(pcd_tr.points)
        tr_cols = np.asarray(pcd_tr.colors)
        tr_cols_u8 = (
            (tr_cols * 255.0).clip(0, 255).astype(np.uint8)
            if len(tr_pts) and tr_cols.size
            else None
        )
        _log_points(tr_pts, tr_cols_u8, ENTITY_TRACKS, subsample=int(cfg["track_sub"]), static=False)

        merged_voxels_clouds = merged_track_voxels_at_frame(i, all_tracks)
        id_to_j = {t.id: j for j, t in enumerate(all_tracks)}
        if merged_voxels_clouds:
            mv_parts: list[np.ndarray] = []
            mv_cols: list[np.ndarray] = []
            for tid, pts in merged_voxels_clouds:
                mv_parts.append(pts)
                j = id_to_j.get(tid, 0)
                c = (np.asarray(track_colors[j], dtype=np.float64) * 255.0).clip(0, 255).astype(
                    np.uint8
                )
                mv_cols.append(np.tile(c, (len(pts), 1)))
            mv_pts = np.vstack(mv_parts)
            mv_cols_u8 = np.vstack(mv_cols)
            _log_points(
                mv_pts,
                mv_cols_u8,
                ENTITY_TRACK_VOXELS,
                subsample=int(cfg["track_voxels_sub"]),
                static=False,
                radius=float(cfg["track_voxels_radius"]),
            )
        else:
            _log_points(
                np.zeros((0, 3), dtype=np.float32),
                None,
                ENTITY_TRACK_VOXELS,
                subsample=1,
                static=False,
                radius=float(cfg["track_voxels_radius"]),
            )

    rr.script_teardown(rr_args)
    logger.info("Saved %s", rr_args.save)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build tracker_layers .rrd from run_tracker.py export."
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="Path to rerun_export or its parent directory.",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=None,
        help="Override DA3 results_output directory.",
    )
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--scene-sub", type=int, default=None)
    parser.add_argument("--track-sub", type=int, default=None)
    parser.add_argument("--track-voxels-sub", type=int, default=None)
    parser.add_argument("--track-voxels-radius", type=float, default=None)
    rr.script_add_args(parser)
    args = parser.parse_args()

    export_dir = _resolve_export_dir(args.export_dir)
    if getattr(args, "save", None) is None:
        args.save = str(export_dir / "tracker_layers.rrd")

    run_from_export(
        export_dir=export_dir,
        rr_args=args,
        depth_dir=args.depth_dir,
        fps=args.fps,
        scene_sub=args.scene_sub,
        track_sub=args.track_sub,
        track_voxels_sub=args.track_voxels_sub,
        track_voxels_radius=args.track_voxels_radius,
    )


if __name__ == "__main__":
    main()
