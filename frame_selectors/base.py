from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
import re
from typing import Any, Iterator, TypeAlias

import cv2
import numpy as np
from tqdm import tqdm

FrameMasks: TypeAlias = dict[str, dict[int, np.ndarray]]
FrameEmbeddings: TypeAlias = dict[str, dict[int, np.ndarray]]
FrameSample: TypeAlias = tuple[np.ndarray, FrameMasks, FrameEmbeddings]


class BaseSelector(ABC):
    """Base class for frame-selection algorithms built on per-frame tracker export."""

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

    def _load_object_dict(
        self,
        raw_obj: np.ndarray | dict | None,
        value_dtype: np.dtype,
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
                int(track_id): np.asarray(value, dtype=value_dtype)
                for track_id, value in dict(track_map).items()
            }
        return out

    def _load_frame_sample(self, frame_path: Path) -> FrameSample:
        with np.load(frame_path, allow_pickle=True) as data:
            image = np.asarray(data["image"])
            masks = self._load_object_dict(data["masks"], np.bool_)
            embeddings = self._load_object_dict(data["embeddings"], np.float32)
        return image, masks, embeddings

    def read_data(self, tracker_output_dir: str | Path) -> Iterator[FrameSample]:
        """Yield `(image, masks, embeddings)` from the per-frame tracker export."""
        export_dir = self._resolve_export_dir(Path(tracker_output_dir))
        for frame_path in self._get_frame_paths(export_dir):
            yield self._load_frame_sample(frame_path)

    def save_frames(
        self,
        frame_ids: list[int],
        frame_sample_iter: Iterator[FrameSample],
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Save selected frames, marked copies, and JSON metadata."""
        output_dir = Path(output_dir)
        marked_frames_dir = output_dir / "marked_frames"
        unmarked_frames_dir = output_dir / "unmarked_frames"
        marked_frames_dir.mkdir(parents=True, exist_ok=True)
        unmarked_frames_dir.mkdir(parents=True, exist_ok=True)

        target_frame_ids = {int(frame_id) for frame_id in frame_ids}
        if not target_frame_ids:
            metadata_path = output_dir / "frames.json"
            centers_metadata_path = output_dir / "frames_centers.json"
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump({"frames": []}, f, indent=2)
            with centers_metadata_path.open("w", encoding="utf-8") as f:
                json.dump({"frames": []}, f, indent=2)
            return {
                "marked_frames_dir": marked_frames_dir,
                "unmarked_frames_dir": unmarked_frames_dir,
                "metadata_path": metadata_path,
                "centers_metadata_path": centers_metadata_path,
            }

        metadata_frames: list[dict[str, Any]] = []
        centers_metadata_frames: list[dict[str, Any]] = []
        missing_frame_ids = set(target_frame_ids)

        save_bar = tqdm(enumerate(frame_sample_iter), desc="Saving frames")
        for frame_idx, (image, masks, _) in save_bar:
            if frame_idx not in target_frame_ids:
                continue

            height, width = image.shape[:2]
            marked_image = image.copy()
            detected_objects: list[dict[str, Any]] = []
            detected_object_centers: list[dict[str, Any]] = []

            for class_name in sorted(masks):
                for track_id in sorted(masks[class_name]):
                    resized_mask = self._resize_mask(
                        masks[class_name][track_id], (height, width)
                    )
                    bbox = self._mask_to_bbox(resized_mask)
                    if bbox is None:
                        continue

                    mark_xy = self._get_mark_xy(resized_mask)
                    color = self._track_color(track_id)
                    self._draw_text_with_outline(
                        marked_image, str(track_id), mark_xy, color
                    )
                    normalized_bbox = self._normalize_bbox(bbox, width, height)
                    normalized_center = self._normalize_xy(mark_xy, width, height)

                    detected_objects.append(
                        {
                            "id": int(track_id),
                            "label": class_name,
                            "bbox": normalized_bbox,
                        }
                    )
                    detected_object_centers.append(
                        {
                            "id": int(track_id),
                            "label": class_name,
                            "center": normalized_center,
                        }
                    )

            image_path = marked_frames_dir / f"frame_{frame_idx:04d}.png"
            if not cv2.imwrite(str(image_path), marked_image):
                raise ValueError(f"Failed to save marked frame: {image_path}")

            unmarked_image_path = unmarked_frames_dir / f"frame_{frame_idx:04d}.png"
            if not cv2.imwrite(str(unmarked_image_path), image):
                raise ValueError(
                    f"Failed to save unmarked frame: {unmarked_image_path}"
                )

            metadata_frames.append(
                {
                    "frame_id": int(frame_idx),
                    "frame_size": {"width": int(width), "height": int(height)},
                    "detected_objects": detected_objects,
                }
            )
            centers_metadata_frames.append(
                {
                    "frame_id": int(frame_idx),
                    "frame_size": {"width": int(width), "height": int(height)},
                    "detected_objects": detected_object_centers,
                }
            )
            missing_frame_ids.discard(frame_idx)

            if not missing_frame_ids:
                break

        if missing_frame_ids:
            missing_str = ", ".join(
                str(frame_id) for frame_id in sorted(missing_frame_ids)
            )
            raise ValueError(f"Requested frames were not found: {missing_str}")

        metadata_frames.sort(key=lambda item: item["frame_id"])
        centers_metadata_frames.sort(key=lambda item: item["frame_id"])
        metadata_path = output_dir / "frames.json"
        centers_metadata_path = output_dir / "frames_centers.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump({"frames": metadata_frames}, f, indent=2)
        with centers_metadata_path.open("w", encoding="utf-8") as f:
            json.dump({"frames": centers_metadata_frames}, f, indent=2)

        return {
            "marked_frames_dir": marked_frames_dir,
            "unmarked_frames_dir": unmarked_frames_dir,
            "metadata_path": metadata_path,
            "centers_metadata_path": centers_metadata_path,
        }

    def _resize_mask(self, mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        height, width = image_size
        if mask.shape[:2] == (height, width):
            return np.asarray(mask, dtype=bool)
        resized = cv2.resize(
            mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
        )
        return resized > 0

    def _mask_to_bbox(self, mask: np.ndarray) -> list[int] | None:
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    def _normalize_bbox(
        self, bbox: list[int], width: int, height: int
    ) -> list[int]:
        x1, y1, x2, y2 = bbox
        return [
            self._normalize_coord(x1, width),
            self._normalize_coord(y1, height),
            self._normalize_coord(x2, width),
            self._normalize_coord(y2, height),
        ]

    def _normalize_xy(
        self, xy: tuple[int, int], width: int, height: int
    ) -> list[int]:
        x, y = xy
        return [self._normalize_coord(x, width), self._normalize_coord(y, height)]

    def _normalize_coord(self, value: int | float, extent: int) -> int:
        if extent <= 1:
            return 0
        normalized = round(float(value) * 999.0 / float(extent - 1))
        return int(np.clip(normalized, 0, 999))

    def _get_mark_xy(self, mask: np.ndarray) -> tuple[int, int]:
        mask_uint8 = mask.astype(np.uint8)
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)
        if num_labels <= 1:
            return (mask.shape[1] // 2, mask.shape[0] // 2)

        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cx, cy = centroids[largest_idx]
        return (int(cx), int(cy))

    def _track_color(self, track_id: int) -> tuple[int, int, int]:
        rng = np.random.default_rng(int(track_id) + 1)
        color = rng.integers(64, 256, size=3, dtype=np.int32)
        return (int(color[0]), int(color[1]), int(color[2]))

    def _draw_text_with_outline(
        self,
        image: np.ndarray,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    @abstractmethod
    def select_frames(self, frame_sample_iter: Iterator[FrameSample]) -> list[int]:
        """Select frame indices from the frame iterator."""


BaseFrameSelector = BaseSelector


class UniformSelector(BaseSelector):
    def __init__(self, step: int):
        super().__init__()
        self.step = step

    def select_frames(self, frame_sample_iter: Iterator[FrameSample]) -> list[int]:
        select_bar = tqdm(enumerate(frame_sample_iter), desc="Selecting frames")
        return [i for i, _ in select_bar if i % self.step == 0]


if __name__ == "__main__":
    selector = UniformSelector(12)
    frame_iter = selector.read_data("data/7/tracker_ouputs")
    frame_ids = selector.select_frames(frame_iter)
    frame_iter = selector.read_data("data/7/tracker_ouputs")
    selector.save_frames(frame_ids, frame_iter, "data/7/selected_frames")
