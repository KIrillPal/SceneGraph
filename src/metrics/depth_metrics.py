"""Depth map metrics vs GT."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom


def resize_depth_to(depth: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if depth.shape == (h, w):
        return depth
    zh = h / depth.shape[0]
    zw = w / depth.shape[1]
    return zoom(depth, (zh, zw), order=1)


def absrel_map(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, eps: float = 1e-8) -> float:
    pred = pred[valid].astype(np.float64)
    gt = gt[valid].astype(np.float64)
    if gt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - gt) / (gt + eps)))


def delta1_accuracy(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, thresh: float = 1.25) -> float:
    pred = pred[valid].astype(np.float64)
    gt = gt[valid].astype(np.float64)
    if gt.size == 0:
        return float("nan")
    ratio = np.maximum(pred / (gt + 1e-8), gt / (pred + 1e-8))
    return float(np.mean(ratio < thresh))


def aggregate_depth_metrics(
    pred_depths: np.ndarray,
    gt_depths: list[np.ndarray],
    *,
    min_depth: float = 1e-3,
    max_depth: float = 100.0,
    scale_align: str | None = "median",
    delta_thresh: float = 1.25,
    eps: float = 1e-8,
) -> dict[str, float]:
    """
    pred_depths: (T, Hp, Wp), gt_depths: list of (Hg, Wg) per frame.
    Only overlapping frame count is used (min length).
    """
    _idx, absrels, deltas = per_frame_depth_metrics(
        pred_depths,
        gt_depths,
        min_depth=min_depth,
        max_depth=max_depth,
        scale_align=scale_align,
        delta_thresh=delta_thresh,
        eps=eps,
    )
    if not absrels.size:
        return {"absrel": float("nan"), "delta_1": float("nan"), "depth_frames_used": 0.0}
    return {
        "absrel": float(np.mean(absrels)),
        "delta_1": float(np.mean(deltas)),
        "depth_frames_used": float(absrels.size),
    }


def per_frame_depth_metrics(
    pred_depths: np.ndarray,
    gt_depths: list[np.ndarray],
    *,
    min_depth: float = 1e-3,
    max_depth: float = 100.0,
    scale_align: str | None = "median",
    delta_thresh: float = 1.25,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-frame AbsRel and δ1 (same scaling rules as ``aggregate_depth_metrics``).

    Returns:
        frame_indices: (K,) indices into pred/GT lists
        absrel: (K,) finite values
        delta1: (K,) per-frame δ1 accuracy in [0, 1]
    """
    t = min(len(pred_depths), len(gt_depths))
    fidx: list[int] = []
    absrels: list[float] = []
    deltas: list[float] = []
    for i in range(t):
        pr = np.asarray(pred_depths[i], dtype=np.float64)
        gt = np.asarray(gt_depths[i], dtype=np.float64)
        if pr.shape != gt.shape:
            pr = resize_depth_to(pr, gt.shape)
        valid = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt) & np.isfinite(pr) & (pr > min_depth)
        if scale_align == "median" and np.any(valid):
            s = float(np.median(gt[valid] / (pr[valid] + eps)))
            if np.isfinite(s) and s > 1e-6:
                pr = pr * s
        elif scale_align is not None and scale_align.lower() not in ("none", ""):
            raise ValueError(f"Unknown scale_align: {scale_align}")
        if not np.any(valid):
            continue
        fidx.append(i)
        absrels.append(absrel_map(pr, gt, valid, eps=eps))
        deltas.append(delta1_accuracy(pr, gt, valid, thresh=delta_thresh))
    return (
        np.asarray(fidx, dtype=np.int32),
        np.asarray(absrels, dtype=np.float64),
        np.asarray(deltas, dtype=np.float64),
    )
