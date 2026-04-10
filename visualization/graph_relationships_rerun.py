#!/usr/bin/env python3
from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import json
import logging
from pathlib import Path
import re

import numpy as np
import rerun as rr
from tqdm.auto import tqdm

from visualization.tracker_layers_rerun import (
    DEFAULT_CONFIG,
    _build_blueprint,
    _extract_scene_points,
    _get_frame_paths,
    _load_frame_payload,
    _log_camera_transform,
    _resolve_export_dir,
    _resize_mask,
)
from visualization.utils.visualization import log_graph_rerun


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ENTITY_IMAGE_RAW = "image/raw"
ENTITY_IMAGE_MASKED = "image/masked"
ENTITY_SCENE = "world/scene_frame"
ENTITY_MASKS = "world/object_masks"
ENTITY_BOXES = "world/object_boxes"
ENTITY_EDGES = "world/relationship_edges"

DESCRIPTION = """
# Scene Graph Relationships - Rerun

| Entity | Content |
|--------|---------|
| `world/camera` | Camera pose + pinhole |
| `world/scene_frame` | Current frame point cloud |
| `world/object_masks` | Current visible object points |
| `world/object_boxes` | Current object boxes |
| `world/relationship_edges` | Merged relationship edges for the current frame |
| `image/raw` | Raw frame image |
| `image/masked` | Frame image with track masks |
""".strip()

_FRAME_INDEX_RE = re.compile(r"(\d+)(?!.*\d)")


def _load_relationships(path: Path) -> list[list[object]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        relationships = _recover_partial_relationships(text, path)
        logger.warning(
            "Recovered %d complete relationships from malformed JSON in %s",
            len(relationships),
            path,
        )
        return relationships

    relationships = payload.get("relationships", [])
    if not isinstance(relationships, list):
        raise ValueError(f"Expected 'relationships' list in {path}")
    return relationships


def _recover_partial_relationships(text: str, path: Path) -> list[list[object]]:
    match = re.search(r'"relationships"\s*:\s*\[', text)
    if match is None:
        raise ValueError(
            f"Could not find 'relationships' array in malformed JSON: {path}"
        )

    decoder = json.JSONDecoder()
    relationships: list[list[object]] = []
    pos = match.end()

    while pos < len(text):
        while pos < len(text) and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(text) or text[pos] == "]":
            break

        try:
            relationship, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            break

        if isinstance(relationship, list):
            relationships.append(relationship)
        pos = end

    if not relationships:
        raise ValueError(f"Could not recover any complete relationships from {path}")
    return relationships


def _validate_intervals(path: Path, intervals: object) -> list[tuple[int, int]]:
    if not isinstance(intervals, list):
        raise ValueError(f"Expected intervals list in {path}")

    out: list[tuple[int, int]] = []
    for interval in intervals:
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"Invalid interval {interval!r} in {path}")
        start_frame = int(interval[0])
        end_frame = int(interval[1])
        if end_frame < start_frame:
            raise ValueError(f"Invalid interval [{start_frame}, {end_frame}] in {path}")
        out.append((start_frame, end_frame))
    return out


def _frame_id_from_path(frame_path: Path) -> int:
    match = _FRAME_INDEX_RE.search(frame_path.stem)
    if match is None:
        raise ValueError(f"Could not infer frame id from {frame_path}")
    return int(match.group(1))


def _merge_relationship_jsons(
    relationship_paths: list[Path],
    valid_frame_ids: list[int],
) -> dict[int, list[tuple[int, str, int]]]:
    relations_by_frame: dict[int, dict[tuple[int, int], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    max_frame_id = max(valid_frame_ids, default=-1)
    warned_legacy_paths: set[Path] = set()

    for relationship_path in relationship_paths:
        for relationship in _load_relationships(relationship_path):
            if not isinstance(relationship, list) or len(relationship) != 4:
                raise ValueError(
                    f"Expected [subject_id, predicate, object_id, intervals] in {relationship_path}"
                )

            subject_id = int(relationship[0])
            predicate = str(relationship[1])
            object_id = int(relationship[2])

            if _looks_like_interval_list(relationship[3], max_frame_id):
                intervals = _validate_intervals(relationship_path, relationship[3])

                for start_frame, end_frame in intervals:
                    start_idx = bisect_left(valid_frame_ids, start_frame)
                    end_idx = bisect_right(valid_frame_ids, end_frame)
                    for frame_id in valid_frame_ids[start_idx:end_idx]:
                        relations_by_frame[frame_id][(subject_id, object_id)].add(
                            predicate
                        )
                continue

            if relationship_path not in warned_legacy_paths:
                logger.warning(
                    "Treating legacy point-based relationships in %s as sequence-level relations",
                    relationship_path,
                )
                warned_legacy_paths.add(relationship_path)
            for frame_id in valid_frame_ids:
                relations_by_frame[frame_id][(subject_id, object_id)].add(predicate)

    merged_relations: dict[int, list[tuple[int, str, int]]] = {}
    for frame_id, pair_map in relations_by_frame.items():
        merged_relations[frame_id] = [
            (subject_id, "|".join(sorted(predicates)), object_id)
            for (subject_id, object_id), predicates in sorted(pair_map.items())
        ]
    return merged_relations


def _looks_like_interval_list(raw_intervals: object, max_frame_id: int) -> bool:
    if not isinstance(raw_intervals, list) or not raw_intervals:
        return False

    values: list[int] = []
    for interval in raw_intervals:
        if not isinstance(interval, list) or len(interval) != 2:
            return False
        try:
            start_value = int(interval[0])
            end_value = int(interval[1])
        except (TypeError, ValueError):
            return False
        values.extend([start_value, end_value])

    return all(0 <= value <= max_frame_id for value in values)


def _build_tracking_masks(
    masks: dict[str, dict[int, np.ndarray]],
    point_cloud: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[int, str]]:
    valid_points = np.isfinite(point_cloud).all(axis=2)
    h_pc, w_pc = point_cloud.shape[:2]
    tracking_masks: dict[str, np.ndarray] = {}
    image_tracking_masks: dict[str, np.ndarray] = {}
    track_key_by_id: dict[int, str] = {}

    for class_name in sorted(masks):
        for track_id in sorted(masks[class_name]):
            mask = np.asarray(masks[class_name][track_id], dtype=bool)
            mask_key = f"{class_name}.{int(track_id)}"

            image_mask = _resize_mask(mask, image_shape)
            point_mask = valid_points & _resize_mask(mask, (h_pc, w_pc))
            flat_point_mask = point_mask[valid_points]
            if not np.any(flat_point_mask):
                continue

            tracking_masks[mask_key] = flat_point_mask
            image_tracking_masks[mask_key] = image_mask
            track_key_by_id[int(track_id)] = mask_key

    return tracking_masks, image_tracking_masks, track_key_by_id


def _resolve_frame_edges(
    frame_relations: list[tuple[int, str, int]],
    track_key_by_id: dict[int, str],
) -> list[tuple[str, str, str]]:
    edges: list[tuple[str, str, str]] = []
    for subject_id, predicate, object_id in frame_relations:
        source_key = track_key_by_id.get(int(subject_id))
        target_key = track_key_by_id.get(int(object_id))
        if source_key is None or target_key is None:
            continue
        edges.append((source_key, predicate, target_key))
    return edges


def _make_point_cloud(points: np.ndarray, colors: np.ndarray | None) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    if colors is None:
        return np.asarray(points, dtype=np.float32)
    return np.concatenate(
        [
            np.asarray(points, dtype=np.float32),
            np.asarray(colors, dtype=np.float32),
        ],
        axis=1,
    )


def run_from_export(
    export_dir: Path,
    relationship_paths: list[Path],
    rr_args: argparse.Namespace,
) -> None:
    export_dir = _resolve_export_dir(export_dir.resolve())
    frame_paths = _get_frame_paths(export_dir)
    valid_frame_ids = [_frame_id_from_path(path) for path in frame_paths]
    relations_by_frame = _merge_relationship_jsons(relationship_paths, valid_frame_ids)

    rr.script_setup(rr_args, "scene_graph_relationships_rrd")
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    rr.log(
        "description",
        rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN),
        static=True,
    )
    blueprint_sent = False

    for frame_path in tqdm(
        frame_paths, desc="Writing graph frames", unit="frame", dynamic_ncols=True
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

        scene_points, scene_colors = _extract_scene_points(image, point_cloud)
        pc = _make_point_cloud(scene_points, scene_colors)
        tracking_masks, image_tracking_masks, track_key_by_id = _build_tracking_masks(
            masks,
            point_cloud,
            image.shape[:2],
        )
        frame_edges = _resolve_frame_edges(
            relations_by_frame.get(frame_id, []),
            track_key_by_id,
        )

        log_graph_rerun(
            pc=pc,
            tracking_mask=tracking_masks,
            edges=frame_edges,
            image=image,
            image_tracking_mask=image_tracking_masks,
            alpha=float(DEFAULT_CONFIG["mask_alpha"]),
            point_radius=float(DEFAULT_CONFIG["track_radius"]),
            edge_radius=float(DEFAULT_CONFIG["track_radius"]),
            image_entity=ENTITY_IMAGE_RAW,
            image_overlay_entity=ENTITY_IMAGE_MASKED,
            point_entity=ENTITY_SCENE,
            mask_entity=ENTITY_MASKS,
            box_entity=ENTITY_BOXES,
            edge_entity=ENTITY_EDGES,
            image_color_model="bgr",
        )

        if not blueprint_sent:
            blueprint = _build_blueprint()
            if blueprint is not None:
                rr.send_blueprint(blueprint, make_active=True, make_default=True)
            blueprint_sent = True

    rr.script_teardown(rr_args)
    logger.info("Saved %s", rr_args.save)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build relationship-graph .rrd from tracker export and relationship JSONs."
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="Path to the tracker output directory with frame_*.npz files.",
    )
    parser.add_argument(
        "--relationships-json",
        type=Path,
        nargs="+",
        required=True,
        help="One or more relationship JSON files in Qwen-like interval format.",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    export_dir = _resolve_export_dir(args.export_dir)
    if getattr(args, "save", None) is None:
        args.save = str(export_dir / "graph_relationships.rrd")

    run_from_export(
        export_dir=export_dir,
        relationship_paths=[path.resolve() for path in args.relationships_json],
        rr_args=args,
    )


if __name__ == "__main__":
    main()
