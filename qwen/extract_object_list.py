#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from .extract_relationships import _data_image_url, _extract_assistant_text, _post_json, _server_image_url
    from .prompts import OBJECT_LIST_SYSTEM_PROMPT
except ImportError:
    from extract_relationships import _data_image_url, _extract_assistant_text, _post_json, _server_image_url
    from prompts import OBJECT_LIST_SYSTEM_PROMPT


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
EXCLUDED_OBJECT_NAMES = {"road", "floor", "ceiling", "wall", "grass"}

def natural_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=natural_key,
    )


def image_url(
    image_path: Path,
    repo_root: Path,
    server_repo_root: Path,
    image_url_mode: str,
) -> str:
    if image_url_mode == "data":
        return _data_image_url(image_path)
    return _server_image_url(image_path, repo_root, server_repo_root)


def build_user_content(
    image_paths: list[Path],
    repo_root: Path,
    server_repo_root: Path,
    image_url_mode: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "/no_think\n"
                "Review these selected video frames jointly. "
                "Return unique scene object descriptions, canonical classes, and static/dynamic labels as plain text lines."
            ),
        }
    ]

    for idx, image_path in enumerate(image_paths, start=1):
        content.append({"type": "text", "text": f"Selected frame {idx}: {image_path.name}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url(
                        image_path,
                        repo_root,
                        server_repo_root,
                        image_url_mode,
                    )
                },
            }
        )
    return content


def build_payload(
    image_paths: list[Path],
    model: str,
    max_tokens: int,
    repo_root: Path,
    server_repo_root: Path,
    image_url_mode: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": OBJECT_LIST_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_content(
                    image_paths,
                    repo_root,
                    server_repo_root,
                    image_url_mode,
                ),
            },
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def normalize_field(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def parse_objects(assistant_text: str) -> list[tuple[str, str, str]]:
    parsed = []
    seen = set()
    for raw_line in assistant_text.splitlines():
        line = raw_line.strip().strip("-*")
        if not line:
            continue
        if line.startswith("```"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        description = normalize_field(parts[0])
        class_name = normalize_field(parts[1]).replace(" ", "_")
        state = normalize_field(parts[2])
        if not description or not class_name or state not in {"static", "dynamic"}:
            continue
        if class_name in EXCLUDED_OBJECT_NAMES:
            continue
        key = (description, class_name)
        if key in seen:
            continue
        parsed.append((description, class_name, state))
        seen.add(key)

    if not parsed:
        raise ValueError("Qwen returned no valid objects")
    return parsed


def write_object_list(output_file: Path, objects: list[tuple[str, str, str]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{description}, {class_name}, {state}" for description, class_name, state in objects]
    output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract scene object descriptions, classes, and static/dynamic labels from selected frames using Qwen."
    )
    parser.add_argument("selected_image_dir", type=Path, help="Folder with selected keyframe images")
    parser.add_argument(
        "output_file",
        type=Path,
        nargs="?",
        default=None,
        help="Output .txt file. Defaults to <selected_image_dir>/objects_static_dynamic.txt",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1/chat/completions",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-27B")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--server-repo-root", type=Path, default=Path("/workspace"))
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument(
        "--image-url-mode",
        choices=["auto", "file", "data"],
        default="auto",
        help="How to send images. 'auto' tries file:// first and falls back to data: URLs.",
    )
    parser.add_argument(
        "--save-response-json",
        action="store_true",
        help="Also save the full API response JSON next to the output .txt.",
    )
    args = parser.parse_args()

    selected_image_dir = args.selected_image_dir.resolve()
    if not selected_image_dir.is_dir():
        raise FileNotFoundError(f"Selected image dir does not exist: {selected_image_dir}")

    image_paths = list_images(selected_image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No selected images found in: {selected_image_dir}")

    output_file = args.output_file
    if output_file is None:
        output_file = selected_image_dir / "objects_static_dynamic.txt"
    output_file = output_file.resolve()

    repo_root = Path(__file__).resolve().parents[1]
    image_url_mode = "file" if args.image_url_mode == "auto" else args.image_url_mode
    payload = build_payload(
        image_paths,
        args.model,
        args.max_tokens,
        repo_root,
        args.server_repo_root,
        image_url_mode,
    )

    try:
        response_json = _post_json(args.endpoint, payload, args.api_key)
    except RuntimeError as exc:
        if args.image_url_mode != "auto" or "allowed-local-media-path" not in str(exc):
            raise
        payload = build_payload(
            image_paths,
            args.model,
            args.max_tokens,
            repo_root,
            args.server_repo_root,
            "data",
        )
        response_json = _post_json(args.endpoint, payload, args.api_key)

    assistant_text = _extract_assistant_text(response_json)
    objects = parse_objects(assistant_text)
    write_object_list(output_file, objects)

    if args.save_response_json:
        response_file = output_file.with_suffix(".response.json")
        response_file.write_text(json.dumps(response_json, indent=2), encoding="utf-8")

    print(f"Wrote {len(objects)} objects to {output_file}")


if __name__ == "__main__":
    main()
