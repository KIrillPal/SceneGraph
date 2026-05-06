#!/usr/bin/env python3
"""Extract PNG frames from a video at a target FPS."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from a video into a folder of PNG images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_video", type=Path, help="Path to input video")
    parser.add_argument("output_dir", type=Path, help="Folder where PNG frames will be saved")
    parser.add_argument("--fps", type=float, required=True, help="Target output FPS")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output_dir before writing frames",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Index used for the first output frame name",
    )
    return parser.parse_args()


def extract_frames(
    input_video: Path,
    output_dir: Path,
    target_fps: float,
    overwrite: bool,
    start_index: int,
) -> int:
    if target_fps <= 0:
        raise ValueError("--fps must be greater than 0")
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if not input_video.is_file():
        raise FileNotFoundError(input_video)

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if source_fps <= 0:
        cap.release()
        raise ValueError("Could not determine source video FPS")

    effective_fps = min(target_fps, source_fps)
    if target_fps > source_fps:
        print(
            f"Requested FPS ({target_fps:g}) is higher than source FPS ({source_fps:g}); "
            f"extracting at source FPS.",
            file=sys.stderr,
        )

    frame_idx = 0
    saved = 0
    next_time = 0.0
    step = 1.0 / effective_fps

    progress_total = total_frames if total_frames > 0 else None
    with tqdm(total=progress_total, desc="Extracting frames", unit="frame") as progress:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            current_time = frame_idx / source_fps
            if current_time + 1e-9 >= next_time:
                out_index = start_index + saved
                out_path = output_dir / f"frame_{out_index:06d}.png"
                if not cv2.imwrite(str(out_path), frame):
                    cap.release()
                    raise ValueError(f"Failed to write frame: {out_path}")
                saved += 1
                next_time += step

            frame_idx += 1
            progress.update(1)

    cap.release()
    return saved


def main() -> None:
    args = parse_args()
    try:
        saved = extract_frames(
            args.input_video.resolve(),
            args.output_dir.resolve(),
            args.fps,
            args.overwrite,
            args.start_index,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Saved {saved} PNG frames to {args.output_dir}")


if __name__ == "__main__":
    main()
