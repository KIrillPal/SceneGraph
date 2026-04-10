from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import rerun as rr

from .point_cloud import split_point_cloud
from .tracking import (
    get_bounding_boxes,
    get_edge_costs,
    get_mask_overlay,
    parse_tracking_key,
    track_color,
)


DEFAULT_POINT_RADIUS = 0.01
DEFAULT_EDGE_RADIUS = 0.01
DEFAULT_EDGE_COLOR = np.array([255, 255, 255], dtype=np.uint8)


def log_graph_rerun(
    pc: np.ndarray,
    tracking_mask: Dict[str, np.ndarray],
    edges: List[Tuple[str, str, str]],
    image: np.ndarray | None = None,
    image_tracking_mask: Dict[str, np.ndarray] | None = None,
    alpha: float = 0.5,
    point_radius: float = DEFAULT_POINT_RADIUS,
    edge_radius: float = DEFAULT_EDGE_RADIUS,
    image_entity: str = "image",
    image_overlay_entity: str = "image/masked",
    point_entity: str = "world/pc",
    mask_entity: str = "world/masks",
    box_entity: str = "world/boxes",
    edge_entity: str = "world/edges",
    image_color_model: str | None = None,
) -> None:
    """Log a point cloud, mask overlay, boxes, and graph edges to Rerun."""
    points, base_colors = split_point_cloud(pc)
    overlay_points, overlay_colors = get_mask_overlay(
        points, base_colors, tracking_mask, alpha=alpha
    )
    boxes = get_bounding_boxes(pc, tracking_mask)
    edge_costs = get_edge_costs(boxes, edges)

    if image is not None:
        image_kwargs = {}
        if image_color_model is not None:
            image_kwargs["color_model"] = image_color_model
        rr.log(image_entity, rr.Image(image, **image_kwargs))
        if image_tracking_mask:
            rr.log(
                image_overlay_entity,
                rr.Image(
                    get_image_mask_overlay(image, image_tracking_mask, alpha=alpha),
                    **image_kwargs,
                ),
            )
        else:
            rr.log(image_overlay_entity, rr.Clear(recursive=True))
    else:
        rr.log(image_entity, rr.Clear(recursive=True))
        rr.log(image_overlay_entity, rr.Clear(recursive=True))

    rr.log(point_entity, rr.Points3D(points, colors=base_colors, radii=point_radius))
    if len(overlay_points) > 0:
        rr.log(
            mask_entity,
            rr.Points3D(
                overlay_points,
                colors=overlay_colors,
                radii=point_radius * 1.25,
            ),
        )
    else:
        rr.log(mask_entity, rr.Clear(recursive=True))

    if boxes:
        rr.log(
            box_entity,
            rr.Boxes3D(
                centers=np.asarray([box["center"] for box in boxes], dtype=np.float32),
                sizes=np.asarray([box["size"] for box in boxes], dtype=np.float32),
                colors=np.asarray([box["color"] for box in boxes], dtype=np.uint8),
                labels=[
                    f"{box['class_name']}.{box['track_id']} ({box['num_points']} pts)"
                    for box in boxes
                ],
                show_labels=True,
            ),
        )
    else:
        rr.log(box_entity, rr.Clear(recursive=True))

    if edge_costs:
        rr.log(
            edge_entity,
            rr.Arrows3D(
                origins=np.asarray(
                    [edge_cost["source_center"] for edge_cost in edge_costs],
                    dtype=np.float32,
                ),
                vectors=np.asarray(
                    [edge_cost["vector"] for edge_cost in edge_costs], dtype=np.float32
                ),
                colors=np.repeat(DEFAULT_EDGE_COLOR[None, :], len(edge_costs), axis=0),
                labels=[
                    (
                        f"{edge_cost['source']} -> {edge_cost['target']}: "
                        f"{edge_cost['relation']}, d={edge_cost['distance']:.3f}"
                    )
                    for edge_cost in edge_costs
                ],
                show_labels=True,
                radii=edge_radius,
            ),
        )
    else:
        rr.log(edge_entity, rr.Clear(recursive=True))


def visualize_single_graph_rerun(
    pc: np.ndarray,
    tracking_mask: Dict[str, np.ndarray],
    edges: List[Tuple[str, str, str]],
    image: np.ndarray | None = None,
    image_tracking_mask: Dict[str, np.ndarray] | None = None,
    alpha: float = 0.5,
    recording_name: str = "single_graph",
    point_radius: float = DEFAULT_POINT_RADIUS,
    edge_radius: float = DEFAULT_EDGE_RADIUS,
) -> None:
    """Log a point cloud, mask overlay, boxes, and graph edges to Rerun."""
    rr.init(recording_name)
    rr.connect_tcp("127.0.0.1:9876")
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    log_graph_rerun(
        pc=pc,
        tracking_mask=tracking_mask,
        edges=edges,
        image=image,
        image_tracking_mask=image_tracking_mask,
        alpha=alpha,
        point_radius=point_radius,
        edge_radius=edge_radius,
    )


def get_image_mask_overlay(
    image: np.ndarray,
    tracking_mask: Dict[str, np.ndarray],
    alpha: float = 0.5,
) -> np.ndarray:
    overlay = np.asarray(image, dtype=np.float32).copy()

    for mask_key, mask in tracking_mask.items():
        if not np.any(mask):
            continue
        _, track_id = parse_tracking_key(mask_key)
        mask_color = track_color(track_id).astype(np.float32)
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * mask_color

    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)
