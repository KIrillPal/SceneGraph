from __future__ import annotations

import numpy as np


def split_point_cloud(pc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a point cloud into XYZ coordinates and RGB colors."""
    pc = np.asarray(pc)
    if pc.ndim != 2 or pc.shape[1] not in (3, 6):
        raise ValueError("pc must have shape (N, 3) or (N, 6)")

    points = np.asarray(pc[:, :3], dtype=np.float32)
    if pc.shape[1] == 6:
        colors = np.asarray(pc[:, 3:6], dtype=np.float32)
    else:
        print("Current point cloud has no colors!")
        colors = np.full((len(points), 3), 200.0, dtype=np.float32)

    if colors.max(initial=0.0) <= 1.0:
        colors = colors * 255.0

    return points, np.clip(colors, 0.0, 255.0).astype(np.uint8)


def normalize_mask(mask: np.ndarray, num_points: int, mask_name: str) -> np.ndarray:
    """Validate and normalize a tracking mask to a boolean vector."""
    mask_array = np.asarray(mask).reshape(-1)
    if mask_array.shape[0] != num_points:
        raise ValueError(
            f"Mask '{mask_name}' has {mask_array.shape[0]} entries, expected {num_points}"
        )
    return mask_array.astype(bool)
