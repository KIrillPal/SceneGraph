from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from tqdm import tqdm

from .base import BaseSelector, FrameSample


@dataclass(frozen=True)
class _ObjectView:
    track_id: int
    class_name: str
    area: int
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    quality: float


@dataclass(frozen=True)
class _PairFrame:
    frame_id: int
    image: np.ndarray
    width: int
    height: int
    left: _ObjectView
    right: _ObjectView
    quality: float
    view_key: tuple[int, int, int]


class PairCoverageSelector(BaseSelector):
    """Select a few informative frames independently for each object pair."""

    def __init__(
        self,
        frames_per_pair: int = 3,
        min_area_ratio: float = 0.001,
        min_gap: int = 4,
        novelty_weight: float = 0.5,
    ) -> None:
        if frames_per_pair <= 0:
            raise ValueError("frames_per_pair must be positive")
        if min_gap < 0:
            raise ValueError("min_gap must be non-negative")
        self.frames_per_pair = frames_per_pair
        self.min_area_ratio = min_area_ratio
        self.min_gap = min_gap
        self.novelty_weight = novelty_weight

    def select_frames(self, frame_sample_iter: Iterator[FrameSample]) -> list[int]:
        raise NotImplementedError("Use select_pairs instead")

    def select_pairs(
        self,
        frame_sample_iter: Iterator[FrameSample],
        output_dir: str | Path,
    ) -> Path:
        raw_frames, max_areas = self._collect_frames(frame_sample_iter)
        pair_frames = self._build_pair_frames(raw_frames, max_areas)
        payload = self._build_payload(pair_frames, Path(output_dir))

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / "pair_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return metadata_path

    def _collect_frames(
        self, frame_sample_iter: Iterator[FrameSample]
    ) -> tuple[list[tuple[int, np.ndarray, int, int, list[_ObjectView]]], dict[int, int]]:
        pending_frames: list[tuple[int, np.ndarray, int, int, list[tuple[int, str, int, tuple[int, int, int, int], tuple[int, int]]]]] = []
        max_areas: dict[int, int] = {}

        for frame_id, (image, masks, _) in tqdm(
            enumerate(frame_sample_iter), desc="Scanning pair masks"
        ):
            height, width = image.shape[:2]
            min_area = int(round(height * width * self.min_area_ratio))
            objects = []

            for class_name in sorted(masks):
                for track_id in sorted(masks[class_name]):
                    resized_mask = self._resize_mask(
                        masks[class_name][track_id], (height, width)
                    )
                    area = int(resized_mask.sum())
                    if area < min_area:
                        continue
                    bbox = self._mask_to_bbox(resized_mask)
                    if bbox is None:
                        continue
                    center = self._get_mark_xy(resized_mask)
                    track_id = int(track_id)
                    max_areas[track_id] = max(max_areas.get(track_id, 0), area)
                    objects.append(
                        (
                            track_id,
                            str(class_name),
                            area,
                            (bbox[0], bbox[1], bbox[2], bbox[3]),
                            center,
                        )
                    )

            pending_frames.append((int(frame_id), image, int(width), int(height), objects))

        frames: list[tuple[int, np.ndarray, int, int, list[_ObjectView]]] = []
        for frame_id, image, width, height, objects in pending_frames:
            views = [
                _ObjectView(
                    track_id=track_id,
                    class_name=class_name,
                    area=area,
                    bbox=bbox,
                    center=center,
                    quality=self._object_quality(area, bbox, center, width, height, max_areas[track_id]),
                )
                for track_id, class_name, area, bbox, center in objects
            ]
            frames.append((frame_id, image, width, height, views))

        return frames, max_areas

    def _build_pair_frames(
        self,
        frames: list[tuple[int, np.ndarray, int, int, list[_ObjectView]]],
        max_areas: dict[int, int],
    ) -> dict[tuple[int, int], list[_PairFrame]]:
        pair_frames: dict[tuple[int, int], list[_PairFrame]] = {}
        for frame_id, image, width, height, objects in frames:
            objects_by_id = {obj.track_id: obj for obj in objects}
            for left_id, right_id in combinations(sorted(objects_by_id), 2):
                left = objects_by_id[left_id]
                right = objects_by_id[right_id]
                quality = min(left.quality, right.quality)
                pair_frames.setdefault((left_id, right_id), []).append(
                    _PairFrame(
                        frame_id=frame_id,
                        image=image,
                        width=width,
                        height=height,
                        left=left,
                        right=right,
                        quality=quality,
                        view_key=self._pair_view_key(left, right, width, height),
                    )
                )
        return pair_frames

    def _build_payload(
        self,
        pair_frames: dict[tuple[int, int], list[_PairFrame]],
        output_dir: Path,
    ) -> dict[str, Any]:
        pairs_payload: list[dict[str, Any]] = []
        for pair, frames in tqdm(sorted(pair_frames.items()), desc="Selecting pair frames"):
            selected = self._select_pair_frames(frames)
            pair_key = self._pair_key(pair)
            self._save_pair_images(output_dir, pair_key, selected)

            labels = {
                str(pair[0]): selected[0].left.class_name,
                str(pair[1]): selected[0].right.class_name,
            }
            pairs_payload.append(
                {
                    "pair": [pair[0], pair[1]],
                    "labels": labels,
                    "co_visible_ranges": self._frame_ids_to_ranges(
                        [frame.frame_id for frame in frames]
                    ),
                    "selected_frames": [
                        self._selected_frame_payload(output_dir, pair_key, frame)
                        for frame in selected
                    ],
                }
            )

        return {
            "frames_per_pair": self.frames_per_pair,
            "pairs": pairs_payload,
        }

    def _select_pair_frames(self, frames: list[_PairFrame]) -> list[_PairFrame]:
        selected: list[_PairFrame] = []
        covered_view_keys: set[tuple[int, int, int]] = set()
        remaining = set(range(len(frames)))

        while remaining and len(selected) < self.frames_per_pair:
            best_idx = None
            best_score = -1.0
            for idx in remaining:
                candidate = frames[idx]
                if selected and any(
                    abs(candidate.frame_id - frame.frame_id) < self.min_gap
                    for frame in selected
                ):
                    continue

                novelty = candidate.quality if candidate.view_key not in covered_view_keys else 0.0
                score = candidate.quality + self.novelty_weight * novelty
                if score > best_score:
                    best_idx = idx
                    best_score = score

            if best_idx is None:
                break
            candidate = frames[best_idx]
            selected.append(candidate)
            covered_view_keys.add(candidate.view_key)
            remaining.remove(best_idx)

        if not selected and frames:
            selected.append(max(frames, key=lambda frame: frame.quality))
        return sorted(selected, key=lambda frame: frame.frame_id)

    def _selected_frame_payload(
        self, output_dir: Path, pair_key: str, frame: _PairFrame
    ) -> dict[str, Any]:
        return {
            "frame_id": frame.frame_id,
            "objects": [
                self._object_payload(frame.left, frame.width, frame.height),
                self._object_payload(frame.right, frame.width, frame.height),
            ],
            "images": {
                "unmarked_frames": str(
                    Path("pair_images")
                    / "unmarked_frames"
                    / pair_key
                    / f"frame_{frame.frame_id:04d}.png"
                ),
                "marked_frames": str(
                    Path("pair_images")
                    / "marked_frames"
                    / pair_key
                    / f"frame_{frame.frame_id:04d}.png"
                ),
            },
        }

    def _object_payload(self, obj: _ObjectView, width: int, height: int) -> dict[str, Any]:
        return {
            "id": obj.track_id,
            "label": obj.class_name,
            "bbox": self._normalize_bbox(list(obj.bbox), width, height),
            "center": self._normalize_xy(obj.center, width, height),
        }

    def _save_pair_images(
        self, output_dir: Path, pair_key: str, selected: list[_PairFrame]
    ) -> None:
        for frame in selected:
            unmarked_path = (
                output_dir
                / "pair_images"
                / "unmarked_frames"
                / pair_key
                / f"frame_{frame.frame_id:04d}.png"
            )
            marked_path = (
                output_dir
                / "pair_images"
                / "marked_frames"
                / pair_key
                / f"frame_{frame.frame_id:04d}.png"
            )
            unmarked_path.parent.mkdir(parents=True, exist_ok=True)
            marked_path.parent.mkdir(parents=True, exist_ok=True)
            marked_image = frame.image.copy()
            for obj in (frame.left, frame.right):
                self._draw_text_with_outline(
                    marked_image,
                    str(obj.track_id),
                    obj.center,
                    self._track_color(obj.track_id),
                )
            if not cv2.imwrite(str(unmarked_path), frame.image):
                raise ValueError(f"Failed to save image: {unmarked_path}")
            if not cv2.imwrite(str(marked_path), marked_image):
                raise ValueError(f"Failed to save image: {marked_path}")

    def _object_quality(
        self,
        area: int,
        bbox: tuple[int, int, int, int],
        center: tuple[int, int],
        width: int,
        height: int,
        max_area: int,
    ) -> float:
        area_score = float(np.sqrt(min(area / max(float(max_area), 1.0), 1.0)))
        norm_dx = (center[0] - width * 0.5) / max(width * 0.5, 1.0)
        norm_dy = (center[1] - height * 0.5) / max(height * 0.5, 1.0)
        center_score = 1.0 - 0.35 * min(float(np.hypot(norm_dx, norm_dy)), 1.0)
        x1, y1, x2, y2 = bbox
        edge_hits = int(x1 <= 0) + int(y1 <= 0) + int(x2 >= width - 1) + int(y2 >= height - 1)
        edge_score = 1.0 - 0.15 * edge_hits
        return max(0.0, area_score * center_score * edge_score)

    def _pair_view_key(
        self, left: _ObjectView, right: _ObjectView, width: int, height: int
    ) -> tuple[int, int, int]:
        dx = (right.center[0] - left.center[0]) / max(float(width), 1.0)
        dy = (right.center[1] - left.center[1]) / max(float(height), 1.0)
        distance = float(np.hypot(dx, dy))
        return (
            self._signed_relation_bin(dx),
            self._signed_relation_bin(dy),
            int(np.clip(distance * 4.0, 0, 3)),
        )

    def _signed_relation_bin(self, value: float) -> int:
        if value < -0.05:
            return -1
        if value > 0.05:
            return 1
        return 0

    def _frame_ids_to_ranges(self, frame_ids: list[int]) -> list[list[int]]:
        ranges: list[list[int]] = []
        for frame_id in sorted(set(frame_ids)):
            if ranges and frame_id == ranges[-1][1] + 1:
                ranges[-1][1] = frame_id
            else:
                ranges.append([frame_id, frame_id])
        return ranges

    def _pair_key(self, pair: tuple[int, int]) -> str:
        return f"pair_{pair[0]:04d}_{pair[1]:04d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select informative frames independently for each co-visible object pair."
    )
    parser.add_argument("tracker_output_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--frames-per-pair", type=int, default=3)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--min-gap", type=int, default=4)
    args = parser.parse_args()

    selector = PairCoverageSelector(
        frames_per_pair=args.frames_per_pair,
        min_area_ratio=args.min_area_ratio,
        min_gap=args.min_gap,
    )
    metadata_path = selector.select_pairs(
        selector.read_data(args.tracker_output_dir), args.output_dir
    )
    print(f"Saved pair metadata to {metadata_path}")


if __name__ == "__main__":
    main()
