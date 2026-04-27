#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .extract_relationships import (
        _build_user_content,
        _extract_assistant_text,
        _post_json,
        _read_frames_json,
    )
except ImportError:
    from extract_relationships import (
        _build_user_content,
        _extract_assistant_text,
        _post_json,
        _read_frames_json,
    )


SYSTEM_PROMPT = """You are a careful visual object color annotator.

You will receive an ordered sequence of frames with metadata for tracked objects.
Each object has a stable integer id across frames.

Task:
- Identify the dominant visible color or colors for each unique object id.
- Use the images as the main evidence and the JSON metadata to know object ids.
- If marked images are used, numeric labels drawn on objects correspond to object ids in the metadata.
- Return exactly one JSON object and no extra text.

Output format:
{
  "object_colors": [
    {"id": object_id, "label": "object_label", "color": "dominant color"}
  ]
}

Rules:
- Include every unique object id present in the metadata.
- Use concise common color names, e.g. "black", "white", "brown", "gray", "red".
- If an object has multiple clear colors, use a short phrase such as "white and blue".
- If color is unclear, use "unknown".
"""


def _build_payload(
    selected_dir: Path,
    model: str,
    max_tokens: int,
    repo_root: Path,
    server_repo_root: Path,
    image_url_mode: str,
    metadata_format: str,
    image_source: str,
) -> dict[str, object]:
    frames = _read_frames_json(selected_dir, metadata_format)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_content(
                    selected_dir,
                    frames,
                    repo_root,
                    server_repo_root,
                    image_url_mode,
                    image_source,
                ),
            },
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send selected frames to Qwen and ask for colors of tracked objects."
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
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1/chat/completions",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--server-repo-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument(
        "--image-url-mode",
        choices=["auto", "file", "data"],
        default="auto",
        help="How to send images. 'auto' tries file:// first and falls back to data: URLs.",
    )
    args = parser.parse_args()

    selected_dir = args.selected_dir.resolve()
    if not selected_dir.is_dir():
        raise FileNotFoundError(f"Selected dir does not exist: {selected_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    image_url_mode = "file" if args.image_url_mode == "auto" else args.image_url_mode
    payload = _build_payload(
        selected_dir,
        args.model,
        args.max_tokens,
        repo_root,
        args.server_repo_root,
        image_url_mode,
        args.metadata_format,
        args.image_source,
    )

    try:
        response_json = _post_json(args.endpoint, payload, args.api_key)
    except RuntimeError as exc:
        if args.image_url_mode != "auto" or "allowed-local-media-path" not in str(exc):
            raise
        payload = _build_payload(
            selected_dir,
            args.model,
            args.max_tokens,
            repo_root,
            args.server_repo_root,
            "data",
            args.metadata_format,
            args.image_source,
        )
        response_json = _post_json(args.endpoint, payload, args.api_key)

    print(_extract_assistant_text(response_json))


if __name__ == "__main__":
    main()
