"""Pair RGB / depth / intrinsics / GT poses for one ARKitScenes validation folder."""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import load_arkit_traj_w2c, traj_interpolate_w2c


def _stem_ts(p: Path) -> float:
    parts = p.stem.split("_", 1)
    if len(parts) < 2:
        raise ValueError(f"Expected name like '<video_id>_<timestamp>.png', got {p.name!r}")
    return float(parts[1])


@dataclass
class FrameSample:
    rgb_ts: float
    rgb_path: Path
    depth_path: Path
    depth_ts: float
    pincam_path: Path


def _build_depth_index(depth_dir: Path) -> tuple[list[float], list[Path]]:
    items = []
    for p in sorted(depth_dir.glob("*.png")):
        ts = _stem_ts(p)
        if not math.isfinite(ts):
            raise ValueError(f"Non-finite timestamp in depth filename: {p.name}")
        items.append((ts, p))
    items.sort(key=lambda x: x[0])
    if not items:
        raise FileNotFoundError(f"No depth PNGs with valid names in {depth_dir}")
    return [x[0] for x in items], [x[1] for x in items]


def _nearest_depth(rgb_ts: float, depth_ts: list[float], depth_paths: list[Path]) -> tuple[Path, float]:
    if not depth_ts:
        raise RuntimeError("depth_ts must be non-empty")
    i = bisect.bisect_left(depth_ts, rgb_ts)
    if i == 0:
        return depth_paths[0], depth_ts[0]
    if i >= len(depth_ts):
        return depth_paths[-1], depth_ts[-1]
    left_dt = abs(rgb_ts - depth_ts[i - 1])
    right_dt = abs(depth_ts[i] - rgb_ts)
    if left_dt <= right_dt:
        return depth_paths[i - 1], depth_ts[i - 1]
    return depth_paths[i], depth_ts[i]


def enumerate_frames(scene_dir: Path) -> list[FrameSample]:
    scene_dir = Path(scene_dir)
    rgb_dir = scene_dir / "lowres_wide"
    depth_dir = scene_dir / "lowres_depth"
    intr_dir = scene_dir / "lowres_wide_intrinsics"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(rgb_dir)
    if not depth_dir.is_dir():
        raise FileNotFoundError(depth_dir)
    if not intr_dir.is_dir():
        raise FileNotFoundError(intr_dir)

    depth_ts, depth_paths = _build_depth_index(depth_dir)
    rgb_files = sorted(rgb_dir.glob("*.png"))
    if not rgb_files:
        raise FileNotFoundError(f"No RGB PNGs in {rgb_dir}")

    out: list[FrameSample] = []
    for rgb in rgb_files:
        ts = _stem_ts(rgb)
        if not math.isfinite(ts):
            raise ValueError(f"Non-finite timestamp in RGB filename: {rgb.name}")
        depth_path, dts = _nearest_depth(ts, depth_ts, depth_paths)
        pin = intr_dir / f"{rgb.stem}.pincam"
        if not pin.is_file():
            raise FileNotFoundError(
                f"Missing intrinsics for frame {rgb.name}: expected {pin}"
            )
        out.append(FrameSample(ts, rgb, depth_path, dts, pin))
    if not out:
        raise ValueError(f"No frames assembled under {scene_dir}")
    out.sort(key=lambda f: f.rgb_ts)
    return out


def read_pincam(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    vals = path.read_text().strip().split()
    if len(vals) < 6:
        raise ValueError(f"Invalid pincam: {path}")
    w, h = int(float(vals[0])), int(float(vals[1]))
    fx, fy, cx, cy = map(float, vals[2:6])
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return k, (w, h)


def load_gt_depth_meters(path: Path, depth_scale: float = 1.0 / 1000.0) -> np.ndarray:
    """ARKitScenes lowres depth: uint16 millimeters → meters by default."""
    if not path.is_file():
        raise FileNotFoundError(path)
    d = np.asarray(Image.open(path))
    if d.dtype == np.uint16:
        g = d.astype(np.float64) * depth_scale
    else:
        g = d.astype(np.float64) * depth_scale
    g[d == 0] = 0.0
    return g


def gt_extrinsics_for_frames(scene_dir: Path, frames: list[FrameSample]) -> np.ndarray:
    traj_path = Path(scene_dir) / "lowres_wide.traj"
    if not traj_path.is_file():
        raise FileNotFoundError(traj_path)
    ts, ext = load_arkit_traj_w2c(str(traj_path))
    qts = np.asarray([f.rgb_ts for f in frames], dtype=np.float64)
    return traj_interpolate_w2c(ts, ext, qts)


def find_mesh_path(scene_dir: Path) -> Path:
    scene_dir = Path(scene_dir)
    vid = scene_dir.name
    cand = scene_dir / f"{vid}_3dod_mesh.ply"
    if cand.is_file():
        return cand
    meshes = list(scene_dir.glob("*_3dod_mesh.ply"))
    if len(meshes) == 1:
        return meshes[0]
    raise FileNotFoundError(f"No *_3dod_mesh.ply in {scene_dir}")
