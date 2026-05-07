"""Load DA3 predictions: either da3.pt or a streaming output directory."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def load_camera_poses(poses_path: Path) -> np.ndarray:
    """camera_poses.txt: one 4x4 c2w matrix per line (row-major). Shape (N, 4, 4)."""
    lines = poses_path.read_text().strip().split("\n")
    poses = []
    for line in lines:
        if not line.strip():
            continue
        vals = np.array(line.split(), dtype=np.float64)
        poses.append(vals.reshape(4, 4))
    return np.stack(poses, axis=0)


def c2w_to_w2c_3x4(T_c2w: np.ndarray) -> np.ndarray:
    """4x4 c2w → 3x4 w2c. p_cam = R @ p_world + t."""
    T_w2c = np.linalg.inv(T_c2w)
    return T_w2c[:3, :].astype(np.float64)


_FRAME_RE = re.compile(r"frame_(\d+)\.npz")


def discover_frame_npz(results_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for f in results_dir.iterdir():
        if not f.is_file():
            continue
        m = _FRAME_RE.fullmatch(f.name)
        if m:
            out.append((int(m.group(1)), f))
    out.sort(key=lambda x: x[0])
    return out


def load_prediction_da3_dir(
    root: Path,
    max_frames: int | None = None,
    *,
    progress: bool = False,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """
    Load from DA3 streaming root (same layout as ``build_da3_pt.py``):

      ROOT/results_output/frame_*.npz  (keys depth, intrinsics)
      ROOT/camera_poses.txt           (c2w 4x4 per line)

    Returns depth (N,H,W), intrinsic (N,3,3), extrinsic (N,3,4) w2c.

    If ``max_frames`` is set, only the first N frames are loaded (after sorting by index).
    """
    root = Path(root)
    results_dir = root / "results_output"
    poses_path = root / "camera_poses.txt"
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Missing results_output: {results_dir}")
    if not poses_path.is_file():
        raise FileNotFoundError(f"Missing camera_poses.txt: {poses_path}")

    frame_files = discover_frame_npz(results_dir)
    if not frame_files:
        raise FileNotFoundError(f"No frame_*.npz under {results_dir}")

    T_c2w_all = load_camera_poses(poses_path)
    if len(T_c2w_all) != len(frame_files):
        raise ValueError(
            f"Frame count mismatch: {len(frame_files)} npz vs {len(T_c2w_all)} poses in {poses_path}"
        )

    if max_frames is not None:
        cap = int(max_frames)
        frame_files = frame_files[:cap]
        T_c2w_all = T_c2w_all[:cap]

    if verbose:
        print(
            f"[recon_metrics] Loading {len(frame_files)} DA3 frames from {results_dir} …",
            flush=True,
        )

    depth_list: list[np.ndarray] = []
    intrinsic_list: list[np.ndarray] = []
    extrinsic_list: list[np.ndarray] = []

    iterable: list[tuple[int, Path]] = list(frame_files)
    if progress:
        from tqdm import tqdm

        iterable = tqdm(iterable, desc="Load DA3 npz", unit="frame", ncols=100)

    for i, (_idx, path) in enumerate(iterable):
        data = np.load(path)
        depth = np.asarray(data["depth"], dtype=np.float64)
        K = np.asarray(data["intrinsics"], dtype=np.float64)
        T_c2w = T_c2w_all[i]
        depth_list.append(depth)
        intrinsic_list.append(K)
        extrinsic_list.append(c2w_to_w2c_3x4(T_c2w))

    depth_arr = np.stack(depth_list, axis=0)
    intrinsic_arr = np.stack(intrinsic_list, axis=0)
    extrinsic_arr = np.stack(extrinsic_list, axis=0)
    return {"depth": depth_arr, "intrinsic": intrinsic_arr, "extrinsic": extrinsic_arr}


def load_prediction_pt(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=False)

    def to_np(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu()
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    depth = to_np(raw["depth"])
    if depth.ndim == 4:
        depth = depth.squeeze(-1)
    intrinsic = to_np(raw["intrinsic"])
    extrinsic = to_np(raw["extrinsic"])
    if intrinsic.ndim == 2:
        intrinsic = np.broadcast_to(intrinsic[np.newaxis, ...], (depth.shape[0], 3, 3))
    if extrinsic.shape[-2:] != (3, 4):
        raise ValueError(f"Expected extrinsic (N,3,4), got {extrinsic.shape}")
    return {"depth": depth.astype(np.float64), "intrinsic": intrinsic.astype(np.float64), "extrinsic": extrinsic.astype(np.float64)}


def load_prediction(
    path: str | Path,
    *,
    prefer_pt_in_dir: bool = True,
    max_frames: int | None = None,
    progress: bool = False,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """
    Load predictions from:

    - A directory: uses ``da3.pt`` inside if present and ``prefer_pt_in_dir``, else
      ``results_output/*.npz`` + ``camera_poses.txt`` (no PyTorch needed).
    - A file: treated as ``da3.pt`` (requires PyTorch).

    ``max_frames`` truncates to the first N frames after sorting (reduces I/O for npz).
    """
    path = Path(path)
    if path.is_dir():
        pt = path / "da3.pt"
        if prefer_pt_in_dir and pt.is_file():
            if verbose:
                print(f"[recon_metrics] Loading torch checkpoint {pt} …", flush=True)
            out = load_prediction_pt(pt)
        else:
            return load_prediction_da3_dir(
                path, max_frames=max_frames, progress=progress, verbose=verbose
            )
    elif not path.is_file():
        raise FileNotFoundError(path)
    else:
        if verbose:
            print(f"[recon_metrics] Loading torch checkpoint {path} …", flush=True)
        out = load_prediction_pt(path)

    if max_frames is not None:
        n = int(max_frames)
        out = {
            "depth": out["depth"][:n],
            "intrinsic": out["intrinsic"][:n],
            "extrinsic": out["extrinsic"][:n],
        }
    return out


def default_combined_ply(da3_root: Path) -> Path | None:
    """``pcd/combined_pcd.ply`` if it exists."""
    cand = Path(da3_root) / "pcd" / "combined_pcd.ply"
    return cand if cand.is_file() else None
