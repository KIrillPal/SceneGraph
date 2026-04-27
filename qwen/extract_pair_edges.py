#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    from .extract_relationships import (
        _extract_assistant_text,
        _image_url,
        _post_json,
        format_relationship_payload,
    )
    from .prompts import RELATION_VOCABULARY
except ImportError:
    from extract_relationships import (
        _extract_assistant_text,
        _image_url,
        _post_json,
        format_relationship_payload,
    )
    from prompts import RELATION_VOCABULARY


Relationship = list[Any]


SYSTEM_PROMPT = """You are a precise spatial relationship classifier.

You receive a few images of the same two static objects and metadata with their ids, labels, and coordinates.
Predict one static spatial relationship from object A to object B.

Rules:
- Object A is always the first id in the requested pair.
- Object B is always the second id in the requested pair.
- The answer must describe the relation of object A to object B.
- Return exactly one word from the closed vocabulary, or return: none
- Do not output ids.
- Do not output JSON.
- Do not output explanation.
- If there is no clear relationship, return: none

Output format examples:
above
on
none

Closed vocabulary:
{vocabulary}
""".format(vocabulary="\n".join(f"- {relation}" for relation in RELATION_VOCABULARY))


def _default_output_filename(metadata_format: str, image_source: str) -> str:
    return f"qwen_pair_edges_{metadata_format}_{image_source}_raw.json"


def _read_pair_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    pairs = payload.get("pairs", [])
    if not isinstance(pairs, list):
        raise ValueError(f"Expected 'pairs' list in {path}")
    return pairs


def _frame_metadata_for_prompt(
    frame: dict[str, Any], metadata_format: str
) -> dict[str, Any]:
    objects = []
    for obj in frame.get("objects", []):
        item = {
            "id": int(obj["id"]),
            "label": str(obj.get("label", "object")),
        }
        if metadata_format == "bbox":
            item["bbox"] = obj["bbox"]
        elif metadata_format == "center":
            item["center"] = obj["center"]
        else:
            raise ValueError(f"Unsupported metadata format: {metadata_format}")
        objects.append(item)
    return {"frame_id": int(frame["frame_id"]), "objects": objects}


def _build_user_content(
    pair_metadata: dict[str, Any],
    metadata_path: Path,
    metadata_format: str,
    image_source: str,
    repo_root: Path,
    server_repo_root: Path,
    image_url_mode: str,
) -> list[dict[str, Any]]:
    pair = [int(value) for value in pair_metadata["pair"]]
    labels = pair_metadata.get("labels", {})
    source_note = ""
    if image_source == "marked_frames":
        source_note = (
            " The images contain numeric labels drawn on the two objects; "
            "these numbers correspond to object ids."
        )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "/no_think\n"
                f"Predict one static relationship from object {pair[0]} to object {pair[1]}. "
                f"Labels: {json.dumps(labels, ensure_ascii=True)}."
                f"{source_note} "
                "Use only the provided frames and coordinates. "
                "Return exactly one predicate word from the closed vocabulary, or none."
            ),
        }
    ]

    for frame in pair_metadata.get("selected_frames", []):
        frame_text = json.dumps(
            _frame_metadata_for_prompt(frame, metadata_format), ensure_ascii=True
        )
        image_rel = frame["images"][image_source]
        image_path = (metadata_path.parent / image_rel).resolve()
        content.append({"type": "text", "text": f"Frame metadata:\n{frame_text}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_url(
                        image_path,
                        repo_root,
                        server_repo_root,
                        image_url_mode,
                    )
                },
            }
        )
    return content


def _build_payload(
    model: str,
    max_tokens: int,
    user_content: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def _parse_relation_text(text: str, pair: tuple[int, int]) -> tuple[int, str, int] | None:
    text = text.strip()
    if not text or text.lower() == "none" or "none" == text.lower().strip(" ."):
        return None

    first_line = text.splitlines()[0].strip().strip("` .\"'")
    if first_line in RELATION_VOCABULARY:
        return pair[0], first_line, pair[1]

    try:
        payload = json.loads(text)
        relationships = payload.get("relationships", [])
        if relationships:
            rel = relationships[0]
            return _validate_relation(int(rel[0]), str(rel[1]), int(rel[2]), pair)
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        pass

    for predicate in sorted(RELATION_VOCABULARY, key=len, reverse=True):
        pattern = rf"^\s*(\d+)\s+{re.escape(predicate)}\s+(\d+)\s*[\.]?\s*$"
        match = re.match(pattern, first_line, flags=re.IGNORECASE)
        if not match:
            continue
        return _validate_relation(
            int(match.group(1)), predicate, int(match.group(2)), pair
        )
    return None


def _validate_relation(
    subject_id: int, predicate: str, object_id: int, pair: tuple[int, int]
) -> tuple[int, str, int] | None:
    if {subject_id, object_id} != set(pair):
        return None
    if predicate not in RELATION_VOCABULARY:
        return None
    return subject_id, predicate, object_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict one static relation per object pair using selected pair frames."
    )
    parser.add_argument("--pair-metadata", type=Path, required=True)
    parser.add_argument(
        "--metadata-format",
        choices=["bbox", "center"],
        default="bbox",
    )
    parser.add_argument(
        "--image-source",
        choices=["unmarked_frames", "marked_frames"],
        default="marked_frames",
    )
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--server-repo-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument(
        "--image-url-mode",
        choices=["auto", "file", "data"],
        default="auto",
    )
    parser.add_argument("--print-responses", action="store_true")
    args = parser.parse_args()

    metadata_path = args.pair_metadata.resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Pair metadata does not exist: {metadata_path}")

    repo_root = Path(__file__).resolve().parents[1]
    image_url_mode = "file" if args.image_url_mode == "auto" else args.image_url_mode
    pairs = _read_pair_metadata(metadata_path)
    relationships: list[Relationship] = []
    stats = {"raw": 0, "kept": 0, "none_or_invalid": 0}

    for pair_metadata in tqdm(pairs, desc="Pair static edge extraction", unit="pair"):
        pair = tuple(sorted(int(value) for value in pair_metadata["pair"]))
        user_content = _build_user_content(
            pair_metadata,
            metadata_path,
            args.metadata_format,
            args.image_source,
            repo_root,
            args.server_repo_root,
            image_url_mode,
        )
        payload = _build_payload(args.model, args.max_tokens, user_content)
        try:
            response_json = _post_json(args.endpoint, payload, args.api_key)
        except RuntimeError as exc:
            if args.image_url_mode != "auto" or "allowed-local-media-path" not in str(exc):
                raise
            image_url_mode = "data"
            user_content = _build_user_content(
                pair_metadata,
                metadata_path,
                args.metadata_format,
                args.image_source,
                repo_root,
                args.server_repo_root,
                "data",
            )
            payload = _build_payload(args.model, args.max_tokens, user_content)
            response_json = _post_json(args.endpoint, payload, args.api_key)

        answer = _extract_assistant_text(response_json).strip()
        stats["raw"] += 1
        if args.print_responses:
            print(f"Pair {pair[0]},{pair[1]}: {answer}")

        relation = _parse_relation_text(answer, pair)
        if relation is None:
            stats["none_or_invalid"] += 1
            continue
        subject_id, predicate, object_id = relation
        relationships.append(
            [subject_id, predicate, object_id, pair_metadata["co_visible_ranges"]]
        )
        stats["kept"] += 1

    output_file = args.output_file or (
        metadata_path.parent / _default_output_filename(args.metadata_format, args.image_source)
    )
    output_file = output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        format_relationship_payload({"relationships": relationships}), encoding="utf-8"
    )

    print(
        f"Processed {stats['raw']} pairs; kept={stats['kept']}; "
        f"none_or_invalid={stats['none_or_invalid']}"
    )
    print(f"Saved pair edge relationships to {output_file}")


if __name__ == "__main__":
    main()
