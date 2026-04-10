from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
import re
from typing import Iterator, TypeAlias

import cv2
import numpy as np
from scipy.spatial import cKDTree

FrameObjectCenters: TypeAlias = dict[int, np.ndarray]
EdgeRelationship: TypeAlias = tuple[int, str, int]
FrameRelationships: TypeAlias = tuple[int, list[EdgeRelationship]]
FrameSample: TypeAlias = tuple[int, FrameObjectCenters, np.ndarray]


class BaseEdgeGenerator(ABC):
    """Base class for edge generation from per-frame tracker exports."""

    _INDEX_RE = re.compile(r"(\d+)(?!.*\d)")

    def _resolve_export_dir(self, path: Path) -> Path:
        if path.is_dir() and list(path.glob("frame_*.npz")):
            return path
        candidate = path / "tracker_outputs"
        if candidate.is_dir() and list(candidate.glob("frame_*.npz")):
            return candidate
        raise FileNotFoundError(f"Could not find frame_*.npz export in {path}")

    def _sort_key(self, path: Path) -> tuple[int, int | str]:
        match = self._INDEX_RE.search(path.stem)
        if match is None:
            return (1, path.stem)
        return (0, int(match.group(1)))

    def _get_frame_paths(self, export_dir: Path) -> list[Path]:
        frame_paths = sorted(export_dir.glob("frame_*.npz"), key=self._sort_key)
        if not frame_paths:
            raise FileNotFoundError(f"No frame_*.npz files found in {export_dir}")
        return frame_paths

    def _load_mask_dict(
        self,
        raw_obj: np.ndarray | dict | None,
    ) -> dict[str, dict[int, np.ndarray]]:
        if raw_obj is None:
            return {}
        if isinstance(raw_obj, np.ndarray) and raw_obj.dtype == object:
            raw_obj = raw_obj.item()
        if raw_obj is None:
            return {}

        out: dict[str, dict[int, np.ndarray]] = {}
        for class_name, track_map in dict(raw_obj).items():
            out[str(class_name)] = {
                int(track_id): np.asarray(value, dtype=np.bool_)
                for track_id, value in dict(track_map).items()
            }
        return out

    def _resize_mask(self, mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        height, width = image_size
        if mask.shape[:2] == (height, width):
            return np.asarray(mask, dtype=bool)
        resized = cv2.resize(
            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        )
        return resized > 0

    def _adaptive_erode_mask_area(
        self,
        mask: np.ndarray,
        target_area_ratio: float = 0.85,
    ) -> np.ndarray:
        if mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        original_area = cv2.countNonZero(mask)
        if original_area == 0:
            return mask

        kernel_size = 3
        while kernel_size <= 7:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            eroded = cv2.erode(mask, kernel, iterations=1)
            current_area = cv2.countNonZero(eroded)
            if current_area / original_area >= target_area_ratio:
                return eroded
            kernel_size += 2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.erode(mask, kernel, iterations=1)

    def _safe_erode(self, mask: np.ndarray, min_area_after: int = 32) -> np.ndarray:
        original_area = cv2.countNonZero(mask)
        if original_area < min_area_after * 2:
            return mask
        eroded = self._adaptive_erode_mask_area(mask)
        if cv2.countNonZero(eroded) < min_area_after:
            return mask
        return eroded

    def _statistical_outlier_removal(
        self,
        points: np.ndarray,
        k: int = 10,
        std_ratio: float = 1.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(points) < k + 1:
            return points, np.ones(len(points), dtype=bool)
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=k + 1)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        mask = mean_distances < (global_mean + std_ratio * global_std)
        return points[mask], mask

    def _extract_object_centers(
        self,
        masks: dict[str, dict[int, np.ndarray]],
        point_cloud: np.ndarray,
    ) -> FrameObjectCenters:
        centers: FrameObjectCenters = {}
        h_pc, w_pc = point_cloud.shape[:2]
        valid_points = np.isfinite(point_cloud).all(axis=2)

        for class_name in sorted(masks):
            for track_id in sorted(masks[class_name]):
                resized_mask = self._resize_mask(
                    masks[class_name][track_id], (h_pc, w_pc)
                )
                eroded_mask = self._safe_erode(resized_mask.astype(np.uint8))
                object_mask = valid_points & (eroded_mask > 0)
                points = np.asarray(point_cloud[object_mask], dtype=np.float32)
                if len(points) == 0:
                    continue
                points, _ = self._statistical_outlier_removal(points)
                if len(points) == 0:
                    continue
                centers[int(track_id)] = np.mean(points, axis=0).astype(np.float32)

        return centers

    def _load_frame_sample(self, frame_path: Path) -> FrameSample:
        with np.load(frame_path, allow_pickle=True) as data:
            frame_id = (
                int(data["frame_id"])
                if "frame_id" in data.files
                else int(self._sort_key(frame_path)[1])
            )
            masks = self._load_mask_dict(data["masks"])
            point_cloud = np.asarray(data["point_cloud"], dtype=np.float32)
            extrinsic = np.asarray(data["extrinsic"], dtype=np.float32)
        object_centers = self._extract_object_centers(masks, point_cloud)
        return frame_id, object_centers, extrinsic

    def read_data(self, tracker_output_dir: str | Path) -> Iterator[FrameSample]:
        """Yield `(frame_id, object_centers, extrinsic)` from tracker outputs."""
        export_dir = self._resolve_export_dir(Path(tracker_output_dir))
        for frame_path in self._get_frame_paths(export_dir):
            yield self._load_frame_sample(frame_path)

    def save_data(
        self,
        frame_relationships: list[FrameRelationships],
        output_file: str | Path,
    ) -> Path:
        """Save aggregated relationships in Qwen-like JSON format."""
        relationship_intervals: dict[EdgeRelationship, list[list[int]]] = {}

        for frame_id, edges in sorted(frame_relationships, key=lambda item: item[0]):
            unique_edges = sorted(
                set(edges), key=lambda item: (item[0], item[1], item[2])
            )
            for edge in unique_edges:
                intervals = relationship_intervals.setdefault(edge, [])
                if intervals and frame_id == intervals[-1][1] + 1:
                    intervals[-1][1] = int(frame_id)
                else:
                    intervals.append([int(frame_id), int(frame_id)])

        payload = {
            "relationships": [
                [subject_id, predicate, object_id, intervals]
                for (subject_id, predicate, object_id), intervals in sorted(
                    relationship_intervals.items(),
                    key=lambda item: (item[0][0], item[0][1], item[0][2]),
                )
            ]
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return output_path

    @abstractmethod
    def generate_edges(
        self,
        frame_sample_iter: Iterator[FrameSample],
    ) -> list[FrameRelationships]:
        """Generate per-frame relationships from tracker-export samples."""


BaseEdgeBuilder = BaseEdgeGenerator
