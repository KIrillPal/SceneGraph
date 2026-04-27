from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterator, TypeAlias

import numpy as np
from tqdm import tqdm

from .base import BaseSelector, FrameSample


ObjectKey: TypeAlias = tuple[int, int, int, int]
PairKey: TypeAlias = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class _RawObjectView:
    track_id: int
    class_name: str
    area: int
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]


@dataclass(frozen=True)
class _ObjectView:
    track_id: int
    class_name: str
    area: int
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    quality: float
    view_key: ObjectKey


@dataclass(frozen=True)
class _FrameCandidate:
    frame_id: int
    width: int
    height: int
    objects: dict[int, _ObjectView]
    pairs: dict[tuple[int, int], float]
    pair_view_keys: dict[PairKey, float]
    total_quality: float


class MaskCoverageSelector(BaseSelector):
    """Greedy frame selector driven by tracker masks.

    ``max_frames`` is an upper bound. The selector stops earlier when selected
    frames already cover the useful object, pair, and view-state evidence in the
    scene.
    """

    def __init__(
        self,
        max_frames: int = 32,
        min_gap: int = 4,
        views_per_object: int = 2,
        views_per_pair: int = 2,
        min_area_ratio: float = 0.001,
        min_gain: float = 0.05,
        position_bins: int = 3,
        object_weight: float = 2.0,
        pair_weight: float = 1.0,
        novelty_weight: float = 0.5,
    ) -> None:
        if max_frames < 2:
            raise ValueError("max_frames must be at least 2 to include first and last frames")
        if min_gap < 0:
            raise ValueError("min_gap must be non-negative")
        if views_per_object <= 0:
            raise ValueError("views_per_object must be positive")
        if views_per_pair <= 0:
            raise ValueError("views_per_pair must be positive")
        if position_bins <= 0:
            raise ValueError("position_bins must be positive")

        self.max_frames = max_frames
        self.min_gap = min_gap
        self.views_per_object = views_per_object
        self.views_per_pair = views_per_pair
        self.min_area_ratio = min_area_ratio
        self.min_gain = min_gain
        self.position_bins = position_bins
        self.object_weight = object_weight
        self.pair_weight = pair_weight
        self.novelty_weight = novelty_weight

    def select_frames(self, frame_sample_iter: Iterator[FrameSample]) -> list[int]:
        raw_frames, max_areas = self._collect_raw_frames(frame_sample_iter)
        candidates = self._build_candidates(raw_frames, max_areas)
        return self._select_greedy(candidates)

    def _collect_raw_frames(
        self,
        frame_sample_iter: Iterator[FrameSample],
    ) -> tuple[list[tuple[int, int, int, list[_RawObjectView]]], dict[int, int]]:
        raw_frames: list[tuple[int, int, int, list[_RawObjectView]]] = []
        max_areas: dict[int, int] = {}

        frame_bar = tqdm(enumerate(frame_sample_iter), desc="Scanning masks")
        for frame_id, (image, masks, _) in frame_bar:
            height, width = image.shape[:2]
            min_area = int(round(height * width * self.min_area_ratio))
            raw_objects: list[_RawObjectView] = []

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

                    x1, y1, x2, y2 = bbox
                    center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
                    raw_objects.append(
                        _RawObjectView(
                            track_id=int(track_id),
                            class_name=str(class_name),
                            area=area,
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            center=center,
                        )
                    )
                    max_areas[int(track_id)] = max(max_areas.get(int(track_id), 0), area)

            raw_frames.append((int(frame_id), int(width), int(height), raw_objects))

        return raw_frames, max_areas

    def _build_candidates(
        self,
        raw_frames: list[tuple[int, int, int, list[_RawObjectView]]],
        max_areas: dict[int, int],
    ) -> list[_FrameCandidate]:
        candidates: list[_FrameCandidate] = []

        for frame_id, width, height, raw_objects in raw_frames:
            objects: dict[int, _ObjectView] = {}
            for raw_obj in raw_objects:
                max_area = max(max_areas.get(raw_obj.track_id, raw_obj.area), 1)
                quality = self._object_quality(raw_obj, width, height, max_area)
                objects[raw_obj.track_id] = _ObjectView(
                    track_id=raw_obj.track_id,
                    class_name=raw_obj.class_name,
                    area=raw_obj.area,
                    bbox=raw_obj.bbox,
                    center=raw_obj.center,
                    quality=quality,
                    view_key=self._object_view_key(raw_obj, width, height, max_area),
                )

            pairs: dict[tuple[int, int], float] = {}
            pair_view_keys: dict[PairKey, float] = {}
            for left_id, right_id in combinations(sorted(objects), 2):
                left = objects[left_id]
                right = objects[right_id]
                quality = min(left.quality, right.quality)
                pair = (left_id, right_id)
                pairs[pair] = quality
                pair_view_keys[self._pair_view_key(left, right, width, height)] = quality

            total_quality = float(sum(obj.quality for obj in objects.values()))
            candidates.append(
                _FrameCandidate(
                    frame_id=frame_id,
                    width=width,
                    height=height,
                    objects=objects,
                    pairs=pairs,
                    pair_view_keys=pair_view_keys,
                    total_quality=total_quality,
                )
            )

        return candidates

    def _object_quality(
        self,
        obj: _RawObjectView,
        width: int,
        height: int,
        max_area: int,
    ) -> float:
        relative_area = min(float(obj.area) / float(max_area), 1.0)
        area_score = float(np.sqrt(relative_area))

        cx, cy = obj.center
        norm_dx = (cx - width * 0.5) / max(width * 0.5, 1.0)
        norm_dy = (cy - height * 0.5) / max(height * 0.5, 1.0)
        center_score = 1.0 - 0.35 * min(float(np.hypot(norm_dx, norm_dy)), 1.0)

        x1, y1, x2, y2 = obj.bbox
        edge_hits = int(x1 <= 0) + int(y1 <= 0) + int(x2 >= width - 1) + int(y2 >= height - 1)
        edge_score = 1.0 - 0.15 * edge_hits

        return max(0.0, area_score * center_score * edge_score)

    def _object_view_key(
        self,
        obj: _RawObjectView,
        width: int,
        height: int,
        max_area: int,
    ) -> ObjectKey:
        cx, cy = obj.center
        x_bin = self._position_bin(cx, width)
        y_bin = self._position_bin(cy, height)
        relative_area = min(float(obj.area) / float(max(max_area, 1)), 1.0)
        if relative_area < 0.25:
            scale_bin = 0
        elif relative_area < 0.6:
            scale_bin = 1
        else:
            scale_bin = 2
        return (obj.track_id, x_bin, y_bin, scale_bin)

    def _pair_view_key(
        self,
        left: _ObjectView,
        right: _ObjectView,
        width: int,
        height: int,
    ) -> PairKey:
        dx = (right.center[0] - left.center[0]) / max(float(width), 1.0)
        dy = (right.center[1] - left.center[1]) / max(float(height), 1.0)
        dx_bin = self._signed_relation_bin(dx)
        dy_bin = self._signed_relation_bin(dy)
        distance = float(np.hypot(dx, dy))
        distance_bin = int(np.clip(distance * 4.0, 0, 3))
        return (left.track_id, right.track_id, dx_bin, dy_bin, distance_bin)

    def _position_bin(self, value: float, extent: int) -> int:
        return int(np.clip(value / max(float(extent), 1.0) * self.position_bins, 0, self.position_bins - 1))

    def _signed_relation_bin(self, value: float) -> int:
        if value < -0.05:
            return -1
        if value > 0.05:
            return 1
        return 0

    def _select_greedy(self, candidates: list[_FrameCandidate]) -> list[int]:
        selected_indices: list[int] = []
        selected_frame_ids: list[int] = []
        object_counts: dict[int, int] = {}
        pair_counts: dict[tuple[int, int], int] = {}
        covered_object_views: set[ObjectKey] = set()
        covered_pair_views: set[PairKey] = set()
        remaining = set(range(len(candidates)))

        select_bar = tqdm(total=self.max_frames, desc="Selecting frames")
        for idx in dict.fromkeys([0, len(candidates) - 1]):
            if idx < 0 or idx not in remaining or len(selected_indices) >= self.max_frames:
                continue
            self._add_selected_candidate(
                candidates[idx],
                idx,
                selected_indices,
                selected_frame_ids,
                object_counts,
                pair_counts,
                covered_object_views,
                covered_pair_views,
                remaining,
            )
            select_bar.update(1)

        while remaining and len(selected_indices) < self.max_frames:
            best_idx: int | None = None
            best_score = self.min_gain

            for idx in remaining:
                score = self._candidate_score(
                    candidates[idx],
                    selected_frame_ids,
                    object_counts,
                    pair_counts,
                    covered_object_views,
                    covered_pair_views,
                )
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                break

            self._add_selected_candidate(
                candidates[best_idx],
                best_idx,
                selected_indices,
                selected_frame_ids,
                object_counts,
                pair_counts,
                covered_object_views,
                covered_pair_views,
                remaining,
            )
            select_bar.update(1)

        select_bar.close()
        return sorted(selected_frame_ids)

    def _add_selected_candidate(
        self,
        candidate: _FrameCandidate,
        candidate_idx: int,
        selected_indices: list[int],
        selected_frame_ids: list[int],
        object_counts: dict[int, int],
        pair_counts: dict[tuple[int, int], int],
        covered_object_views: set[ObjectKey],
        covered_pair_views: set[PairKey],
        remaining: set[int],
    ) -> None:
        selected_indices.append(candidate_idx)
        selected_frame_ids.append(candidate.frame_id)
        remaining.remove(candidate_idx)

        for track_id, obj in candidate.objects.items():
            object_counts[track_id] = object_counts.get(track_id, 0) + 1
            covered_object_views.add(obj.view_key)
        for pair in candidate.pairs:
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        covered_pair_views.update(candidate.pair_view_keys)

    def _candidate_score(
        self,
        candidate: _FrameCandidate,
        selected_frame_ids: list[int],
        object_counts: dict[int, int],
        pair_counts: dict[tuple[int, int], int],
        covered_object_views: set[ObjectKey],
        covered_pair_views: set[PairKey],
    ) -> float:
        if not candidate.objects:
            return 0.0
        if any(abs(candidate.frame_id - frame_id) < self.min_gap for frame_id in selected_frame_ids):
            return 0.0

        object_gain = 0.0
        object_novelty = 0.0
        for track_id, obj in candidate.objects.items():
            remaining_views = max(self.views_per_object - object_counts.get(track_id, 0), 0)
            if remaining_views:
                object_gain += obj.quality * remaining_views / self.views_per_object
            if obj.view_key not in covered_object_views:
                object_novelty += obj.quality

        pair_gain = 0.0
        pair_novelty = 0.0
        for pair, quality in candidate.pairs.items():
            remaining_views = max(self.views_per_pair - pair_counts.get(pair, 0), 0)
            if remaining_views:
                pair_gain += quality * remaining_views / self.views_per_pair

        for pair_key, quality in candidate.pair_view_keys.items():
            if pair_key not in covered_pair_views:
                pair_novelty += quality

        coverage_gain = self.object_weight * object_gain + self.pair_weight * pair_gain
        novelty_gain = self.novelty_weight * (object_novelty + pair_novelty)
        gain = coverage_gain + novelty_gain
        if gain <= 0.0:
            return 0.0

        return gain + 0.01 * candidate.total_quality


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select tracker frames using mask coverage and view novelty."
    )
    parser.add_argument("tracker_output_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--min-gap", type=int, default=4)
    parser.add_argument("--views-per-object", type=int, default=2)
    parser.add_argument("--views-per-pair", type=int, default=2)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--min-gain", type=float, default=0.05)
    args = parser.parse_args()

    selector = MaskCoverageSelector(
        max_frames=args.max_frames,
        min_gap=args.min_gap,
        views_per_object=args.views_per_object,
        views_per_pair=args.views_per_pair,
        min_area_ratio=args.min_area_ratio,
        min_gain=args.min_gain,
    )
    frame_ids = selector.select_frames(selector.read_data(args.tracker_output_dir))
    selector.save_frames(frame_ids, selector.read_data(args.tracker_output_dir), args.output_dir)
    print(f"Selected {len(frame_ids)} frames: {frame_ids}")


if __name__ == "__main__":
    main()
