#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    from .extract_relationships import (
        _build_user_content,
        _extract_assistant_text,
        _post_json,
        _read_frames_json,
    )
    from .prompts import get_system_prompt
except ImportError:
    from extract_relationships import (
        _build_user_content,
        _extract_assistant_text,
        _post_json,
        _read_frames_json,
    )
    from prompts import get_system_prompt


Pair = tuple[int, int]
Relationship = list[Any]


def _objects_by_frame(frames: list[dict[str, Any]]) -> dict[int, dict[int, str]]:
    objects_by_frame: dict[int, dict[int, str]] = {}
    for frame in frames:
        frame_id = int(frame["frame_id"])
        objects_by_frame[frame_id] = {
            int(obj["id"]): str(obj.get("label", "object"))
            for obj in frame.get("detected_objects", [])
        }
    return objects_by_frame


def _pair_frame_ids(
    objects_by_frame: dict[int, dict[int, str]],
    min_co_visible_frames: int,
) -> dict[Pair, list[int]]:
    pair_frames: dict[Pair, list[int]] = {}
    for frame_id, object_map in objects_by_frame.items():
        for pair in combinations(sorted(object_map), 2):
            pair_frames.setdefault(pair, []).append(frame_id)
    return {
        pair: frame_ids
        for pair, frame_ids in sorted(pair_frames.items())
        if len(frame_ids) >= min_co_visible_frames
    }


def _parse_pair(raw_pair: str) -> Pair:
    parts = raw_pair.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Pairs must use the format '<id>,<id>'")
    left, right = sorted((int(parts[0]), int(parts[1])))
    if left == right:
        raise argparse.ArgumentTypeError("Pair ids must be different")
    return left, right


def _labels_for_pair(objects_by_frame: dict[int, dict[int, str]], pair: Pair) -> dict[int, str]:
    labels: dict[int, str] = {}
    for object_map in objects_by_frame.values():
        for object_id in pair:
            if object_id in object_map and object_id not in labels:
                labels[object_id] = object_map[object_id]
    return labels


def _build_pair_instruction(
    pair: Pair,
    labels: dict[int, str],
    co_visible_frame_ids: list[int],
) -> dict[str, str]:
    left_id, right_id = pair
    left_label = labels.get(left_id, "object")
    right_label = labels.get(right_id, "object")
    pair_payload = {
        "pair": [left_id, right_id],
        "labels": {str(left_id): left_label, str(right_id): right_label},
        "co_visible_frame_ids": co_visible_frame_ids,
    }
    return {
        "type": "text",
        "text": (
            "Pairwise request:\n"
            f"{json.dumps(pair_payload, ensure_ascii=True)}\n"
            "Analyze only this object pair using the full frame sequence above. "
            "Return relationships only between these two ids, in either supported direction. "
            "Return exactly one JSON object and no extra text."
        ),
    }


def _build_pair_payload(
    model: str,
    max_tokens: int,
    metadata_format: str,
    shared_content: list[dict[str, Any]],
    pair_instruction: dict[str, str],
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": get_system_prompt(metadata_format, pairwise=True)},
            {"role": "user", "content": [*shared_content, pair_instruction]},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def _parse_relationship_json(text: str) -> list[Relationship]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        payload = json.loads(text[start : end + 1])

    relationships = payload.get("relationships", [])
    if not isinstance(relationships, list):
        raise ValueError("Expected 'relationships' to be a list")
    return relationships


def _valid_pair_relationship(relationship: Relationship, pair: Pair) -> Relationship | None:
    if not isinstance(relationship, list) or len(relationship) != 4:
        return None
    subject_id = int(relationship[0])
    predicate = str(relationship[1])
    object_id = int(relationship[2])
    if {subject_id, object_id} != set(pair):
        return None

    intervals: list[list[int]] = []
    for interval in relationship[3]:
        if not isinstance(interval, list) or len(interval) != 2:
            continue
        start_frame = int(interval[0])
        end_frame = int(interval[1])
        if end_frame < start_frame:
            continue
        intervals.append([start_frame, end_frame])

    if not intervals:
        return None
    return [subject_id, predicate, object_id, intervals]


def _merge_relationships(relationships: list[Relationship]) -> list[Relationship]:
    intervals_by_relation: dict[tuple[int, str, int], list[list[int]]] = {}
    for subject_id, predicate, object_id, intervals in relationships:
        key = (int(subject_id), str(predicate), int(object_id))
        intervals_by_relation.setdefault(key, []).extend(intervals)

    merged: list[Relationship] = []
    for (subject_id, predicate, object_id), intervals in sorted(intervals_by_relation.items()):
        normalized = sorted([list(map(int, interval)) for interval in intervals])
        coalesced: list[list[int]] = []
        for start_frame, end_frame in normalized:
            if coalesced and start_frame <= coalesced[-1][1] + 1:
                coalesced[-1][1] = max(coalesced[-1][1], end_frame)
            else:
                coalesced.append([start_frame, end_frame])
        merged.append([subject_id, predicate, object_id, coalesced])
    return merged


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pairwise Qwen spatial-relationship extraction on selected frames."
    )
    parser.add_argument("--selected-dir", type=Path, required=True)
    parser.add_argument(
        "--metadata-format",
        choices=["bbox", "center"],
        default="bbox",
        help="Frame metadata format: 'bbox' reads frames.json, 'center' reads frames_centers.json.",
    )
    parser.add_argument(
        "--image-source",
        choices=["unmarked_frames", "marked_frames"],
        default="unmarked_frames",
    )
    parser.add_argument(
        "--pair",
        type=_parse_pair,
        action="append",
        default=None,
        help="Optional pair to process as '<id>,<id>'. Can be passed multiple times.",
    )
    parser.add_argument(
        "--min-co-visible-frames",
        type=int,
        default=1,
        help="Only auto-select pairs visible together in at least this many selected frames.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1/chat/completions",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--pair-responses-file", type=Path, default=None)
    parser.add_argument("--server-repo-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument(
        "--image-url-mode",
        choices=["auto", "file", "data"],
        default="auto",
        help="How to send images. 'auto' tries file:// first and falls back to data: URLs.",
    )
    args = parser.parse_args()

    if args.min_co_visible_frames <= 0:
        raise ValueError("min-co-visible-frames must be positive")

    selected_dir = args.selected_dir.resolve()
    if not selected_dir.is_dir():
        raise FileNotFoundError(f"Selected dir does not exist: {selected_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    image_url_mode = "file" if args.image_url_mode == "auto" else args.image_url_mode
    frames = _read_frames_json(selected_dir, args.metadata_format)
    shared_content = _build_user_content(
        selected_dir,
        frames,
        repo_root,
        args.server_repo_root,
        image_url_mode,
        args.image_source,
    )
    objects_by_frame = _objects_by_frame(frames)
    auto_pair_frames = _pair_frame_ids(objects_by_frame, args.min_co_visible_frames)

    if args.pair:
        pairs = sorted(set(args.pair))
        pair_frames = {
            pair: [
                frame_id
                for frame_id, object_map in objects_by_frame.items()
                if pair[0] in object_map and pair[1] in object_map
            ]
            for pair in pairs
        }
    else:
        pair_frames = auto_pair_frames
        pairs = sorted(pair_frames)

    if not pairs:
        raise ValueError("No object pairs found to process")

    relationships: list[Relationship] = []
    pair_response_rows: list[dict[str, Any]] = []

    progress = tqdm(pairs, desc="Pairwise Qwen extraction", unit="pair")
    for pair in progress:
        labels = _labels_for_pair(objects_by_frame, pair)
        pair_instruction = _build_pair_instruction(pair, labels, pair_frames.get(pair, []))
        payload = _build_pair_payload(
            args.model,
            args.max_tokens,
            args.metadata_format,
            shared_content,
            pair_instruction,
        )

        try:
            response_json = _post_json(args.endpoint, payload, args.api_key)
        except RuntimeError as exc:
            if args.image_url_mode != "auto" or "allowed-local-media-path" not in str(exc):
                raise
            shared_content = _build_user_content(
                selected_dir,
                frames,
                repo_root,
                args.server_repo_root,
                "data",
                args.image_source,
            )
            payload = _build_pair_payload(
                args.model,
                args.max_tokens,
                args.metadata_format,
                shared_content,
                pair_instruction,
            )
            response_json = _post_json(args.endpoint, payload, args.api_key)

        assistant_text = _extract_assistant_text(response_json)
        parsed_relationships = _parse_relationship_json(assistant_text)
        valid_relationships = [
            relationship
            for relationship in (
                _valid_pair_relationship(relationship, pair)
                for relationship in parsed_relationships
            )
            if relationship is not None
        ]
        relationships.extend(valid_relationships)
        pair_response_rows.append(
            {
                "pair": list(pair),
                "co_visible_frame_ids": pair_frames.get(pair, []),
                "assistant_text": assistant_text,
                "relationships": valid_relationships,
            }
        )

    output_file = args.output_file or (selected_dir / "qwen_relationships_pairwise_raw.json")
    output_file = output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_payload = {"relationships": _merge_relationships(relationships)}
    output_file.write_text(json.dumps(merged_payload, indent=2), encoding="utf-8")

    if args.pair_responses_file is not None:
        _write_jsonl(args.pair_responses_file.resolve(), pair_response_rows)

    print(f"Processed {len(pairs)} pairs")
    print(f"Saved merged pairwise relationships to {output_file}")


if __name__ == "__main__":
    main()
