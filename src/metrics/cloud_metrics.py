"""Chamfer and (approximate) EMD between point sets."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def random_subsample(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    n = points.shape[0]
    if n <= max_points:
        return points
    idx = rng.choice(n, size=max_points, replace=False)
    return points[idx]


def chamfer_l2(
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_points_a: int | None = 50000,
    max_points_b: int | None = 50000,
    seed: int = 0,
) -> dict[str, float]:
    """
    Symmetric Chamfer: mean_{x in A} min_{y in B} ||x-y||^2 + mean_{y in B} min_{x in A} ||x-y||^2.

    Subsamples independently when counts exceed max_points_* (None = no cap).
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    a = a[np.isfinite(a).all(axis=1)]
    b = b[np.isfinite(b).all(axis=1)]
    rng = np.random.default_rng(seed)
    if max_points_a is not None and len(a) > max_points_a:
        a = random_subsample(a, max_points_a, rng)
    if max_points_b is not None and len(b) > max_points_b:
        b = random_subsample(b, max_points_b, rng)
    if len(a) == 0 or len(b) == 0:
        return {"chamfer_l2": float("nan"), "chamfer_l2_sqrt": float("nan")}

    ta, tb = cKDTree(b), cKDTree(a)
    da, _ = ta.query(a, k=1)
    db, _ = tb.query(b, k=1)
    mean_a = float(np.mean(da**2))
    mean_b = float(np.mean(db**2))
    cham = mean_a + mean_b
    return {"chamfer_l2": cham, "chamfer_l2_sqrt": float(np.sqrt(cham))}


def emd_uniform_l1(
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_points: int = 512,
    seed: int = 0,
) -> float:
    """
    Exact Wasserstein-1 cost for equal-size uniform point masses: min_pi sum ||a_i - b_pi(i)||.

    Uses Hungarian algorithm (cubic in n). Subsamples both clouds to max_points.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 3)
    a = a[np.isfinite(a).all(axis=1)]
    b = b[np.isfinite(b).all(axis=1)]
    rng = np.random.default_rng(seed)
    n = min(max_points, len(a), len(b))
    if n < 3:
        return float("nan")
    if len(a) > n:
        a = random_subsample(a, n, rng)
    if len(b) > n:
        b = random_subsample(b, n, rng)
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    if n == 0:
        return float("nan")
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean())
