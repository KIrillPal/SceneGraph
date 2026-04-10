from __future__ import annotations

from typing import Any

import hashlib

import numpy as np

from .point_cloud import normalize_mask, split_point_cloud


def parse_tracking_key(mask_key: str) -> tuple[str, str]:
    """Split a tracking key of the form 'classname.id' into class name and track id."""
    if "." not in mask_key:
        return "unknown", mask_key
    class_name, track_id = mask_key.rsplit(".", 1)
    return class_name, track_id


def track_color(track_id: str) -> np.ndarray:
    """Create a deterministic RGB color for a tracking id."""
    digest = hashlib.md5(track_id.encode("utf-8")).digest()
    color = np.frombuffer(digest[:3], dtype=np.uint8).astype(np.int16)
    color = 96 + (color % 160)
    return color.astype(np.uint8)


def blend_mask_colors(
    base_colors: np.ndarray,
    tracking_mask: dict[str, np.ndarray],
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend per-track colors into point RGB values using boolean point masks."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in the range [0, 1]")

    blended = np.asarray(base_colors, dtype=np.float32).copy()
    num_points = blended.shape[0]

    for mask_key, mask in tracking_mask.items():
        point_mask = normalize_mask(mask, num_points, mask_key)
        if not np.any(point_mask):
            continue
        _, track_id = parse_tracking_key(mask_key)
        mask_color = track_color(track_id).astype(np.float32)
        blended[point_mask] = (1.0 - alpha) * blended[point_mask] + alpha * mask_color

    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def get_mask_overlay(
    points: np.ndarray,
    base_colors: np.ndarray,
    tracking_mask: dict[str, np.ndarray],
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a visible point overlay for masked points only."""
    if not tracking_mask:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
        )

    num_points = len(points)
    union_mask = np.zeros(num_points, dtype=bool)
    for mask_key, mask in tracking_mask.items():
        union_mask |= normalize_mask(mask, num_points, mask_key)

    if not np.any(union_mask):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
        )

    overlay_colors = blend_mask_colors(base_colors, tracking_mask, alpha=alpha)
    return points[union_mask], overlay_colors[union_mask]


def get_bounding_boxes(
    pc: np.ndarray, tracking_mask: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Create axis-aligned boxes from point-level tracking masks."""
    points, _ = split_point_cloud(pc)
    boxes: list[dict[str, Any]] = []

    for mask_key, mask in tracking_mask.items():
        point_mask = normalize_mask(mask, len(points), mask_key)
        point_indices = np.flatnonzero(point_mask)
        if point_indices.size == 0:
            continue

        masked_points = points[point_indices]
        min_corner = masked_points.min(axis=0)
        max_corner = masked_points.max(axis=0)
        class_name, track_id = parse_tracking_key(mask_key)

        boxes.append(
            {
                "mask_key": mask_key,
                "class_name": class_name,
                "track_id": track_id,
                "point_indices": point_indices,
                "center": ((min_corner + max_corner) / 2.0).astype(np.float32),
                "size": (max_corner - min_corner).astype(np.float32),
                "min_corner": min_corner.astype(np.float32),
                "max_corner": max_corner.astype(np.float32),
                "color": track_color(track_id),
                "num_points": int(point_indices.size),
            }
        )

    boxes.sort(key=lambda box: (box["class_name"], box["track_id"]))
    return boxes


def get_edge_costs(
    boxes: list[dict[str, Any]], edges: list[tuple[str, str, str]]
) -> list[dict[str, Any]]:
    """Compute Euclidean center-to-center distances for graph edges."""
    boxes_by_key = {box["mask_key"]: box for box in boxes}
    edge_costs: list[dict[str, Any]] = []

    for source_key, relation, target_key in edges:
        source_box = boxes_by_key.get(source_key)
        target_box = boxes_by_key.get(target_key)
        if source_box is None or target_box is None:
            continue

        source_center = source_box["center"]
        target_center = target_box["center"]
        edge_costs.append(
            {
                "source": source_key,
                "target": target_key,
                "relation": relation,
                "distance": float(np.linalg.norm(target_center - source_center)),
                "source_center": source_center,
                "target_center": target_center,
                "vector": (target_center - source_center).astype(np.float32),
            }
        )

    return edge_costs
