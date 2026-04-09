#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib import error, request


SYSTEM_PROMPT = """You are a detail-oriented Video Relationship Annotator.

Your job is to review an ordered sequence of sampled video frames and extract visually grounded spatial relationships between tracked objects.

Task context:
- Videos are sampled at 1 fps.
- Each frame contains detected objects with unique integer ids.
- The input sequence is ordered by time.

Rules:
- Use the full sequence jointly, not frame-by-frame in isolation.
- Extract only spatial, physical, or geometric relationships visible in the scene.
- Do not output actions, functions, intentions, attention, social relations, or object states.
- Do not output left of or right of.
- Use 3D scene reasoning from visual cues and common sense, not only 2D box coordinates.
- Only output relationships clearly supported by the provided frames and metadata.
- Newly appearing objects must also be considered from the first frame where they appear.
- Use only object ids that appear in the provided metadata.
- Do not explain your reasoning.
- Do not analyze step by step.
- Do not restate the input.
- Output exactly one JSON object and nothing else.

Predicate vocabulary:
["on", "under", "above", "below", "in front of", "behind", "inside", "around", "intersecting", "overlapping", "covering", "attached to", "leaning on", "against"]

Return exactly one valid JSON object and no extra text:
{
  "relationships": [
    [subject_id, predicate_verb, object_id, [[start_frame, end_frame], ...]]
  ]
}

Requirements:
- subject_id and object_id must be integers
- predicate_verb must be a string from the allowed vocabulary
- frame indices must be integers
- each interval must be [start_frame, end_frame] with start_frame <= end_frame
- if no valid relationships exist, return {"relationships": []}
"""


def _read_frames_json(selected_dir: Path) -> list[dict[str, Any]]:
    frames_path = selected_dir / "frames.json"
    with frames_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    frames = payload.get("frames", [])
    frames.sort(key=lambda item: int(item["frame_id"]))
    return frames


def _frame_image_path(selected_dir: Path, frame_id: int) -> Path:
    image_dir = selected_dir / "unmarked_frames"
    image_path = image_dir / f"frame_{frame_id:04d}.png"
    if image_path.is_file():
        return image_path
    image_path = image_dir / f"frame_{frame_id:04d}.jpg"
    if image_path.is_file():
        return image_path
    raise FileNotFoundError(f"Could not find image for frame {frame_id} in {image_dir}")


def _server_image_path(
    image_path: Path, repo_root: Path, server_repo_root: Path
) -> str:
    image_path = image_path.resolve()
    repo_root = repo_root.resolve()
    try:
        rel_path = image_path.relative_to(repo_root)
    except ValueError:
        return str(image_path)
    return str((server_repo_root / rel_path).as_posix())


def _build_user_content(
    selected_dir: Path,
    frames: list[dict[str, Any]],
    repo_root: Path,
    server_repo_root: Path,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "/no_think\n"
                "Analyze this ordered frame sequence and extract spatial relationships. "
                "Use the provided frame metadata together with the images. "
                "Return exactly one JSON object and no extra text."
            ),
        }
    ]

    for frame in frames:
        frame_id = int(frame["frame_id"])
        frame_text = json.dumps(frame, ensure_ascii=True)
        image_path = _frame_image_path(selected_dir, frame_id)
        content.append(
            {
                "type": "text",
                "text": f"Frame metadata:\n{frame_text}",
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _server_image_path(image_path, repo_root, server_repo_root)
                },
            }
        )
    return content


def _build_payload(
    selected_dir: Path,
    model: str,
    max_tokens: int,
    repo_root: Path,
    server_repo_root: Path,
) -> dict[str, Any]:
    frames = _read_frames_json(selected_dir)
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
                ),
            },
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def _post_json(
    url: str, payload: dict[str, Any], api_key: str | None
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from Qwen server:\n{body}") from exc


def _extract_assistant_text(response_json: dict[str, Any]) -> str:
    choices = response_json.get("choices", [])
    if not choices:
        raise ValueError(f"No choices in response: {response_json}")

    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return "\n".join(part for part in text_parts if part)
    raise ValueError(f"Unsupported assistant message content: {content}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run simple Qwen spatial-relationship extraction on selected frames."
    )
    parser.add_argument(
        "--selected-dir",
        type=Path,
        required=True,
        help="Directory with frames.json and unmarked_frames/.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1/chat/completions",
        help="OpenAI-compatible Qwen chat completions endpoint.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-27B",
        help="Model name exposed by the Qwen server.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum number of output tokens.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Where to save the raw assistant text. Defaults to <selected-dir>/qwen_relationships_raw.json.",
    )
    parser.add_argument(
        "--save-response-json",
        action="store_true",
        help="Also save the full API response JSON next to the raw text output.",
    )
    parser.add_argument(
        "--server-repo-root",
        type=Path,
        default=Path("/workspace"),
        help="Repo root as seen by the Qwen server. Defaults to /workspace.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional bearer token for the vLLM server.",
    )
    args = parser.parse_args()

    selected_dir = args.selected_dir.resolve()
    if not selected_dir.is_dir():
        raise FileNotFoundError(f"Selected dir does not exist: {selected_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    payload = _build_payload(
        selected_dir,
        args.model,
        args.max_tokens,
        repo_root,
        args.server_repo_root,
    )
    response_json = _post_json(args.endpoint, payload, args.api_key)
    assistant_text = _extract_assistant_text(response_json)

    output_file = args.output_file
    if output_file is None:
        output_file = selected_dir / "qwen_relationships_raw.json"
    output_file = output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(assistant_text, encoding="utf-8")

    if args.save_response_json:
        response_path = output_file.with_name(output_file.stem + "_response.json")
        response_path.write_text(json.dumps(response_json, indent=2), encoding="utf-8")

    print(f"Saved raw assistant output to {output_file}")


if __name__ == "__main__":
    main()
