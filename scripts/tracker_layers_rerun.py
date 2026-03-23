#!/usr/bin/env python3
"""
Собрать .rrd для 3D tracker: камера, сцена по кадрам, треки, combined ply.

Ожидает каталог экспорта из ноутбука (см. ячейку «сохранить rerun_export»):
  - config.json
  - extrinsics.npy       (N, 4, 4) c2w
  - points_per_frame.pkl
  - points_per_frame_masks.pkl
  - tracks.pkl            список {id, masks: {frame_i: HxW}}
  - track_colors.npy      (K, 3) float [0,1]

Пер-кадровые depth/image/intrinsics читаются с диска из depth_dir (как в ноутбуке).

Пример:
  python tracker_layers_rerun.py --export-dir /path/to/rerun_export --save out.rrd
  # --save задаёт rerun-sdk (см. rr.script_add_args); если не указан — <export-dir>/tracker_layers.rrd
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import open3d as o3d
import rerun as rr
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    import rerun.blueprint as rrb
except ImportError:
    rrb = None

if not hasattr(rr, "script_add_args"):
    raise ImportError(
        "Нужен пакет rerun-sdk (не старый `rerun` с PyPI). "
        "pip uninstall rerun -y && pip install rerun-sdk"
    )

# --- геометрия треков (как в 3d_tracker.ipynb) --------------------------------


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
    pts_mask_ = points[mask_r > 0]
    pts_mask, _ = statistical_outlier_removal(pts_mask_)
    return pts_mask


class TrackLike:
    def __init__(self, track_id: int, masks: dict[int, np.ndarray]):
        self.id = track_id
        self.masks = masks


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


# --- Rerun --------------------------------------------------------------------

CAMERA_ENTITY = "world/camera"
ENTITY_SCENE = "world/scene_frame"
ENTITY_TRACKS = "world/track_points"
ENTITY_TRACK_LABELS = "world/track_labels"
ENTITY_COMBINED = "world/scene_combined"

DESCRIPTION = """
# 3D tracker — Rerun

| Entity | Содержимое |
|--------|------------|
| `world/camera` | Поза камеры + pinhole |
| `world/scene_frame` | Облако сцены на кадре |
| `world/track_points` | Треки на кадре |
| `world/track_labels` | Номера треков (центр облака, подписи в 3D) |
| `world/scene_combined` | Статическое облако из combined_pcd.ply (RGBA, см. `combined_alpha` в config) |
""".strip()


def _log_camera_transform(
    extrinsic_w2c: np.ndarray,
    intrinsic: np.ndarray,
    resolution_wh: tuple[int, int],
) -> None:
    R_mat = extrinsic_w2c[:3, :3]
    t_vec = extrinsic_w2c[:, 3]
    rotation = Rotation.from_matrix(R_mat)
    rr.log(
        CAMERA_ENTITY,
        rr.Transform3D(
            translation=t_vec,
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


def _rgb_u8_to_rgba(rgb: np.ndarray, alpha_u8: int) -> np.ndarray:
    """(N, 3) uint8 -> (N, 4) uint8."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 2 or rgb.shape[1] not in (3, 4):
        raise ValueError(f"Expected (N, 3) or (N, 4) colors, got {rgb.shape}")
    a = np.full((len(rgb), 1), int(np.clip(alpha_u8, 0, 255)), dtype=np.uint8)
    if rgb.shape[1] == 4:
        return np.concatenate([rgb[:, :3], a], axis=1)
    return np.concatenate([rgb, a], axis=1)


def _log_points(
    positions: np.ndarray,
    colors_u8: Optional[np.ndarray],
    entity: str,
    subsample: int = 1,
    static: bool = False,
    radius: float = 0.005,
) -> None:
    if len(positions) == 0:
        rr.log(entity, rr.Points3D(positions=np.zeros((0, 3))), static=static)
        return
    pts = positions[::subsample]
    cols = colors_u8[::subsample] if colors_u8 is not None else None
    rr.log(
        entity,
        rr.Points3D(positions=pts, colors=cols, radii=radius),
        static=static,
    )


def _log_track_id_labels(
    clouds: list[tuple[int, np.ndarray]],
    track_colors: np.ndarray,
    all_tracks: list[TrackLike],
    label_radius: float = 0.02,
) -> None:
    """Подписи id треков в центрах масс облаков (тот же `track_point_clouds_at_frame`, что и точки)."""
    if not clouds:
        rr.log(
            ENTITY_TRACK_LABELS,
            rr.Points3D(positions=np.zeros((0, 3))),
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
    pos = np.stack(centers, axis=0)
    cols_u8 = np.stack(colors_rgb, axis=0)
    rr.log(
        ENTITY_TRACK_LABELS,
        rr.Points3D(
            positions=pos,
            labels=labels,
            show_labels=True,
            radii=label_radius,
            colors=cols_u8,
        ),
        static=False,
    )


def _build_blueprint():
    if rrb is None:
        return None
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D", origin="world"),
            rrb.TextDocumentView(name="Info", origin="description"),
        )
    )


def _load_tracks_pickle(path: Path) -> list[TrackLike]:
    with path.open("rb") as f:
        raw: list[dict[str, Any]] = pickle.load(f)
    out: list[TrackLike] = []
    for item in raw:
        masks = {int(k): v for k, v in item["masks"].items()}
        out.append(TrackLike(int(item["id"]), masks))
    return out


def _first_existing(paths: list[str | Path]) -> Optional[Path]:
    for p in paths:
        pp = Path(p)
        if pp.is_file():
            return pp
    return None


def run_from_export(export_dir: Path, rr_args: argparse.Namespace) -> None:
    export_dir = export_dir.resolve()
    rrd_out = Path(rr_args.save)
    cfg_path = export_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Нет {cfg_path}")

    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)

    depth_dir = cfg["depth_dir"]
    if not depth_dir.endswith("/"):
        depth_dir = depth_dir + "/"
    n_frames = int(cfg["n_frames"])
    fps = float(cfg.get("fps", 5.0))
    scene_sub = int(cfg.get("scene_sub", 2))
    track_sub = int(cfg.get("track_sub", 1))
    combined_sub = int(cfg.get("combined_sub", 2))
    # непрозрачность combined: float 0..1 в config (по умолчанию ~полупрозрачные точки)
    combined_alpha = float(cfg.get("combined_alpha", 0.35))
    combined_alpha_u8 = int(np.clip(round(combined_alpha * 255.0), 1, 255))

    extr = np.load(export_dir / "extrinsics.npy")
    with (export_dir / "points_per_frame.pkl").open("rb") as f:
        points_per_frame: list[np.ndarray] = pickle.load(f)
    with (export_dir / "points_per_frame_masks.pkl").open("rb") as f:
        points_per_frame_masks: list[np.ndarray] = pickle.load(f)
    all_tracks = _load_tracks_pickle(export_dir / "tracks.pkl")
    track_colors = np.load(export_dir / "track_colors.npy")

    combined_candidates = cfg.get("combined_pcd_paths") or []
    combined_path = _first_existing(combined_candidates)

    rr.script_setup(rr_args, "3d_tracker_rrd", default_blueprint=_build_blueprint())

    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    rr.log(
        "description",
        rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN),
        static=True,
    )

    if combined_path is not None:
        comb = o3d.io.read_point_cloud(str(combined_path))
        pts_c = np.asarray(comb.points)
        cols_c = np.asarray(comb.colors) if comb.has_colors() else None
        if cols_c is not None and cols_c.size and float(cols_c.max()) <= 1.0 + 1e-6:
            cols_c = (cols_c * 255.0).clip(0, 255).astype(np.uint8)
        elif cols_c is not None:
            cols_c = cols_c.clip(0, 255).astype(np.uint8)
        if cols_c is None:
            cols_c = np.full((len(pts_c), 3), 200, dtype=np.uint8)
        cols_c = _rgb_u8_to_rgba(cols_c, combined_alpha_u8)
        _log_points(pts_c, cols_c, ENTITY_COMBINED, subsample=combined_sub, static=True)
        print(f"Logged combined PCD: {combined_path} ({len(pts_c)} pts)")
    else:
        print("Skip combined PCD (no existing path in combined_pcd_paths)")

    for i in range(n_frames):
        rr.set_time("frame", sequence=i)
        rr.set_time("time", duration=i / fps)

        data = np.load(f"{depth_dir}frame_{i}.npz")
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
        _log_points(scene_pts, scene_cols, ENTITY_SCENE, subsample=scene_sub, static=False)

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
        if len(tr_pts) and tr_cols.size:
            tr_cols_u8 = (tr_cols * 255.0).clip(0, 255).astype(np.uint8)
        else:
            tr_cols_u8 = None
        _log_points(tr_pts, tr_cols_u8, ENTITY_TRACKS, subsample=track_sub, static=False)

    rr.script_teardown(rr_args)
    print("Saved:", rrd_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tracker_layers .rrd from notebook export.")
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="Каталог с config.json, *.pkl, extrinsics.npy, track_colors.npy",
    )
    # --save добавляет rr.script_add_args; не дублировать
    rr.script_add_args(parser)
    args = parser.parse_args()

    export_dir = args.export_dir
    if getattr(args, "save", None) is None:
        args.save = str(export_dir / "tracker_layers.rrd")

    run_from_export(export_dir, args)


if __name__ == "__main__":
    main()
