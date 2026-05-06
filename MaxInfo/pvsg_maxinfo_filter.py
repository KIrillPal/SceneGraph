#!/usr/bin/env python3
"""Select visually diverse keyframes from an image folder with CLIP features."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14-336"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MaxInfo-style keyframe selection for a folder of image frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Folder with image frames")
    parser.add_argument("--output_dir", type=Path, required=True, help="Folder for selected frames")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=20,
        help="Maximum number of selected frames",
    )
    parser.add_argument(
        "--include-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Always include first, middle, and last frame when possible",
    )
    parser.add_argument("--rank", type=int, default=8, help="SVD rank for MaxInfo selection")
    parser.add_argument(
        "--tol",
        type=float,
        default=0.23,
        help="rect_maxvol tolerance when maxvolpy is installed",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="CLIP batch size")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 CLIP on CUDA")
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL, help="HuggingFace CLIP model id")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output directory before writing selected frames",
    )
    return parser.parse_args()


def natural_key(path: Path) -> list[int | str]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def list_frames(input_dir: Path) -> list[Path]:
    return sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=natural_key,
    )


def extract_clip_features(
    frames: list[Path],
    model_name: str,
    batch_size: int | None,
    use_fp16: bool,
) -> np.ndarray:
    from transformers import AutoProcessor, CLIPModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = batch_size if batch_size is not None else (64 if device.type == "cuda" else 16)
    use_fp16 = use_fp16 and device.type == "cuda"

    if use_fp16:
        dtype_msg = "fp16"
    else:
        dtype_msg = "fp32"
    print(f"Loading CLIP {model_name} on {device} ({dtype_msg}, batch={batch_size})")

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    if use_fp16:
        model = model.half()
    processor = AutoProcessor.from_pretrained(model_name)

    features = []
    for start in tqdm(range(0, len(frames), batch_size), desc="Embedding frames", unit="batch"):
        batch_paths = frames[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            if use_fp16:
                inputs = {
                    key: value.half() if value.dtype == torch.float32 else value
                    for key, value in inputs.items()
                }
            batch_features = model.get_image_features(**inputs)
            batch_features = torch.nn.functional.normalize(batch_features.float(), dim=1)
        features.append(batch_features.cpu().numpy())

    return np.concatenate(features, axis=0).astype(np.float64)


def svd_reduce(features: np.ndarray, rank: int) -> np.ndarray:
    if len(features) <= 1:
        return features
    rank = max(1, min(rank, features.shape[1], len(features) - 1))
    u, _s, _vh = np.linalg.svd(features, full_matrices=False)
    return u[:, :rank]


def anchor_indices(num_frames: int) -> list[int]:
    if num_frames <= 0:
        return []
    anchors = {0, num_frames // 2, num_frames - 1}
    return sorted(anchors)


def unique_ordered(indices: list[int] | np.ndarray) -> list[int]:
    seen = set()
    ordered = []
    for idx in indices:
        idx = int(idx)
        if idx not in seen:
            ordered.append(idx)
            seen.add(idx)
    return ordered


def greedy_volume_order(matrix: np.ndarray) -> list[int]:
    remaining = list(range(len(matrix)))
    ordered: list[int] = []

    while remaining:
        if not ordered:
            scores = np.linalg.norm(matrix[remaining], axis=1)
        else:
            selected_matrix = matrix[ordered]
            gram = selected_matrix @ selected_matrix.T
            projection_matrix = (
                selected_matrix.T
                @ np.linalg.solve(gram + np.eye(gram.shape[0]) * 1e-8, selected_matrix)
            )
            residual = matrix[remaining] - matrix[remaining] @ projection_matrix
            scores = np.linalg.norm(residual, axis=1)

        best = remaining[int(np.argmax(scores))]
        ordered.append(best)
        remaining.remove(best)

    return ordered


def qr_pivot_order(matrix: np.ndarray) -> list[int]:
    try:
        from scipy.linalg import qr

        _q, _r, pivots = qr(matrix.T, pivoting=True, mode="economic")
        return unique_ordered(pivots)
    except Exception:
        return greedy_volume_order(matrix)


def maxvol_order(matrix: np.ndarray, tol: float) -> list[int]:
    try:
        try:
            from maxvolpy.maxvol import rect_maxvol
        except ImportError:
            from maxvolpy.maxvolpy.maxvol import rect_maxvol

        pivots, _ = rect_maxvol(matrix, tol=tol)
        candidates = unique_ordered(pivots)
    except Exception:
        candidates = []

    if candidates:
        candidate_order = qr_pivot_order(matrix[candidates])
        ordered = [candidates[idx] for idx in candidate_order]
    else:
        ordered = []

    for idx in qr_pivot_order(matrix):
        if idx not in ordered:
            ordered.append(idx)
    return ordered


def select_maxinfo_indices(
    features: np.ndarray,
    num_frames: int,
    include_anchors: bool,
    rank: int,
    tol: float,
) -> list[int]:
    total = len(features)
    if num_frames >= total:
        return list(range(total))

    reduced = svd_reduce(features, rank)
    selected = anchor_indices(total) if include_anchors else []
    selected = selected[:num_frames]

    for idx in maxvol_order(reduced, tol):
        if len(selected) >= num_frames:
            break
        if idx not in selected:
            selected.append(idx)

    return sorted(selected)


def write_outputs(
    output_dir: Path,
    frames: list[Path],
    selected_indices: list[int],
    overwrite: bool,
    rank: int,
    tol: float,
    include_anchors: bool,
) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_names = []
    for idx in selected_indices:
        src = frames[idx]
        dst = output_dir / src.name
        shutil.copyfile(src, dst)
        selected_names.append(src.name)

    report = {
        "input_dir": str(frames[0].parent) if frames else "",
        "output_dir": str(output_dir),
        "num_input_frames": len(frames),
        "num_selected_frames": len(selected_indices),
        "rank": int(rank),
        "tol": float(tol),
        "include_anchors": bool(include_anchors),
        "selection_method": "clip_svd_rect_maxvol_candidates_qr_pivot_budget",
        "selected_indices": [int(i) for i in selected_indices],
        "keyframes": selected_names,
    }
    # (output_dir / "keyframes.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # (output_dir / "keyframes.txt").write_text("\n".join(selected_names) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        print(f"Error: input_dir is not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)
    if args.num_frames < 1:
        print("Error: --num-frames must be >= 1", file=sys.stderr)
        sys.exit(1)

    frames = list_frames(input_dir)
    if not frames:
        print(f"Error: no image frames found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    features = extract_clip_features(frames, args.clip_model, args.batch_size, args.fp16)
    selected_indices = select_maxinfo_indices(
        features,
        num_frames=args.num_frames,
        include_anchors=args.include_anchors,
        rank=args.rank,
        tol=args.tol,
    )
    write_outputs(
        output_dir,
        frames,
        selected_indices,
        args.overwrite,
        args.rank,
        args.tol,
        args.include_anchors,
    )

    pct = 100.0 * len(selected_indices) / len(frames)
    print(f"Selected {len(selected_indices)} / {len(frames)} frames ({pct:.1f}%)")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
