#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any
from urllib import error, request

try:
    from .prompts import get_system_prompt
except ImportError:
    from prompts import get_system_prompt


def _metadata_filename(metadata_format: str) -> str:
    if metadata_format == "bbox":
        return "frames.json"
    if metadata_format == "center":
        return "frames_centers.json"
    raise ValueError(f"Unsupported metadata format: {metadata_format}")


def _read_frames_json(selected_dir: Path, metadata_format: str) -> list[dict[str, Any]]:
    frames_path = selected_dir / _metadata_filename(metadata_format)
    if not frames_path.is_file():
        raise FileNotFoundError(f"Could not find frame metadata: {frames_path}")
    with frames_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    frames = payload.get("frames", [])
    frames.sort(key=lambda item: int(item["frame_id"]))
    return frames


def _frame_image_path(selected_dir: Path, frame_id: int, image_source: str) -> Path:
    image_dir = selected_dir / image_source
    image_path = image_dir / f"frame_{frame_id:04d}.png"
    if image_path.is_file():
        return image_path
    image_path = image_dir / f"frame_{frame_id:04d}.jpg"
    if image_path.is_file():
        return image_path
    raise FileNotFoundError(f"Could not find image for frame {frame_id} in {image_dir}")


def _server_image_url(image_path: Path, repo_root: Path, server_repo_root: Path) -> str:
    image_path = image_path.resolve()
    repo_root = repo_root.resolve()
    try:
        rel_path = image_path.relative_to(repo_root)
    except ValueError:
        return image_path.as_uri()
    return (server_repo_root / rel_path).as_uri()


def _data_image_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    else:
        raise ValueError(f"Unsupported image extension: {image_path}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _image_url(
    image_path: Path,
    repo_root: Path,
    server_repo_root: Path,
    image_url_mode: str,
) -> str:
    if image_url_mode == "data":
        return _data_image_url(image_path)
    return _server_image_url(image_path, repo_root, server_repo_root)


def _build_user_content(
    selected_dir: Path,
    frames: list[dict[str, Any]],
    repo_root: Path,
    server_repo_root: Path,
    image_url_mode: str,
    image_source: str,
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
        image_path = _frame_image_path(selected_dir, frame_id, image_source)
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
    selected_dir: Path,
    model: str,
    max_tokens: int,
    repo_root: Path,
    server_repo_root: Path,
    image_url_mode: str,
    metadata_format: str,
    image_source: str,
) -> dict[str, Any]:
    frames = _read_frames_json(selected_dir, metadata_format)
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": get_system_prompt(metadata_format)},
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
        if "allowed-local-media-path" in body:
            body += (
                "\n\nHint: start vLLM with "
                "`--allowed-local-media-path /workspace` "
                "or another directory that contains the images."
            )
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
        help="Directory with frame metadata and selected frame images.",
    )
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
        help="Selected-frame image directory to send to the model.",
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
        default="Qwen/Qwen3.5-4B",
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
    parser.add_argument(
        "--image-url-mode",
        choices=["auto", "file", "data"],
        default="auto",
        help="How to send images to the server. 'auto' tries file:// first and falls back to data: URLs.",
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
