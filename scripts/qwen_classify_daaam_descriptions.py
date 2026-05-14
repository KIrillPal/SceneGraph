#!/usr/bin/env python3
"""Classify DAAAM object descriptions into short classes using Qwen."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
QWEN_DIR = REPO_ROOT / "qwen"
if str(QWEN_DIR) not in sys.path:
    sys.path.insert(0, str(QWEN_DIR))

from extract_relationships import _extract_assistant_text, _post_json  # noqa: E402


SYSTEM_PROMPT = """You convert object descriptions into concise object class names.

Rules:
- Return exactly one lowercase singular class name.
- Use 1-3 words max.
- Use underscores between words.
- Prefer common object/tracking/evaluation categories.
- Do not include colors, materials, sizes, attributes, ids, punctuation, explanations, or Markdown.
- If the description is unknown, empty, or too vague to name, return unknown.

Examples:
Description: A rectangular window with a dark frame and clear glass pane.
Class: window

Description: A large green metal dumpster with a flat top.
Class: dumpster

Description: A tall slender street light pole with a rectangular light fixture.
Class: street_light

Description: Smooth concrete sidewalk bordered by a curb.
Class: sidewalk

Description: A black pickup truck with a crew cab and short bed.
Class: pickup_truck

Description: A round manhole cover embedded in concrete.
Class: manhole

Description: unknown
Class: unknown
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Qwen to add a qwen_class field to DAAAM processed predictions."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/coda/predicted_detections_daaam/perframe_predicitons_processed.jsonl"),
        help="Input processed DAAAM JSONL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/coda/predicted_detections_daaam/perframe_predicitons_processed_qwen_classes.jsonl"),
        help="Output JSONL with qwen_class added.",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("data/coda/predicted_detections_daaam/qwen_description_class_cache.json"),
        help="JSON cache mapping descriptions to Qwen classes.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1/chat/completions",
        help="OpenAI-compatible Qwen chat completions endpoint.",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between uncached calls.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--field-name",
        default="qwen_class",
        help="Detection field to write. Use 'class' to overwrite the current class field.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Classify at most this many uncached descriptions, then write output with available classes.",
    )
    return parser.parse_args()


def load_records(input_path: Path) -> list[dict[str, Any]]:
    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def unique_descriptions(records: list[dict[str, Any]]) -> list[str]:
    seen = set()
    descriptions = []
    for record in records:
        for det in record.get("detections", []):
            description = str(det.get("description", "unknown")).strip()
            if description not in seen:
                seen.add(description)
                descriptions.append(description)
    return descriptions


def load_cache(cache_file: Path) -> dict[str, str]:
    if not cache_file.is_file():
        return {}
    with cache_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(key): str(value) for key, value in data.items()}


def save_cache(cache_file: Path, cache: dict[str, str]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_file.with_suffix(cache_file.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False, sort_keys=True)
    tmp_path.replace(cache_file)


def build_payload(description: str, model: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def normalize_class(text: str) -> str:
    text = text.strip().lower()
    text = text.splitlines()[0] if text else "unknown"
    text = re.sub(r"^class\s*:\s*", "", text)
    text = re.sub(r"[`'\".,;:()\[\]{}]", "", text)
    text = re.sub(r"[^a-z0-9_\-\s]", "", text)
    text = re.sub(r"[\s\-]+", "_", text).strip("_")
    if not text:
        return "unknown"
    parts = [part for part in text.split("_") if part]
    if len(parts) > 3:
        text = "_".join(parts[:3])
    return text


def classify_description(
    description: str,
    endpoint: str,
    model: str,
    api_key: str | None,
    max_tokens: int,
    max_retries: int,
) -> str:
    if not description or description.strip().lower() == "unknown":
        return "unknown"

    payload = build_payload(description, model, max_tokens)
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response_json = _post_json(endpoint, payload, api_key)
            return normalize_class(_extract_assistant_text(response_json))
        except Exception as exc:  # noqa: BLE001 - keep retries robust for API failures.
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(2**attempt)
    assert last_error is not None
    raise RuntimeError(f"Qwen classification failed after {max_retries} attempts") from last_error


def write_records(output_path: Path, records: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp_path.replace(output_path)


def apply_cache_to_records(records: list[dict[str, Any]], cache: dict[str, str], field_name: str) -> int:
    missing = 0
    for record in records:
        for det in record.get("detections", []):
            description = str(det.get("description", "unknown")).strip()
            class_name = cache.get(description)
            if class_name is None:
                missing += 1
                class_name = "unknown"
            det[field_name] = class_name
    return missing


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    descriptions = unique_descriptions(records)
    cache = load_cache(args.cache_file)

    uncached = [desc for desc in descriptions if desc not in cache]
    if args.limit is not None:
        uncached = uncached[: args.limit]

    print(f"Records: {len(records)}")
    print(f"Unique descriptions: {len(descriptions)}")
    print(f"Cached descriptions: {len(cache)}")
    print(f"Descriptions to classify now: {len(uncached)}")

    for idx, description in enumerate(uncached, start=1):
        class_name = classify_description(
            description=description,
            endpoint=args.endpoint,
            model=args.model,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
        )
        cache[description] = class_name
        save_cache(args.cache_file, cache)
        print(f"[{idx}/{len(uncached)}] {class_name}: {description[:100]}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    missing = apply_cache_to_records(records, cache, args.field_name)
    write_records(args.output, records)
    print(f"Wrote: {args.output}")
    print(f"Cache: {args.cache_file}")
    print(f"Detections without cached class: {missing}")


if __name__ == "__main__":
    main()
