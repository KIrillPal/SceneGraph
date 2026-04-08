from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
import json
from pathlib import Path
import pickle
import re
from typing import Any, Iterator, TypeAlias

import cv2
import numpy as np
from tqdm import tqdm

FrameMasks: TypeAlias = dict[str, dict[int, np.ndarray]]
FrameEmbeddings: TypeAlias = dict[str, dict[int, np.ndarray]]
FrameSample: TypeAlias = tuple[np.ndarray, FrameMasks, FrameEmbeddings]


class BaseSelector(ABC):
    """Base class for frame-selection algorithms built on tracker outputs."""

    _INDEX_RE = re.compile(r"(\d+)(?!.*\d)")
    _IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

    def _resolve_rerun_tracks_path(self, path: Path) -> Path:
        if path.is_file() and path.name == "tracks.pkl":
            return path
        if path.is_dir() and (path / "rerun_export" / "tracks.pkl").is_file():
            return path / "rerun_export" / "tracks.pkl"
        if (
            path.is_dir()
            and (path / "tracks.pkl").is_file()
            and path.name == "rerun_export"
        ):
            return path / "tracks.pkl"
        raise FileNotFoundError(
            f"Could not resolve rerun_export/tracks.pkl from {path}"
        )

    def _resolve_da3_dir(self, path: Path) -> Path:
        if not path.is_dir():
            raise FileNotFoundError(f"DA3 directory does not exist: {path}")
        return path

    def _get_sorted_paths(self, directory: Path, suffixes: set[str]) -> list[Path]:
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {directory}")

        paths = [
            p
            for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in suffixes
        ]
        paths.sort(key=self._sort_key)
        if not paths:
            raise FileNotFoundError(f"No matching files found in {directory}")
        return paths

    def _sort_key(self, path: Path) -> tuple[int, int | str]:
        match = self._INDEX_RE.search(path.stem)
        if match is None:
            return (1, path.stem)
        return (0, int(match.group(1)))

    def _load_tracks(self, tracks_pickle_path: Path) -> dict[int, dict[str, object]]:
        tracks: dict[int, dict[str, object]] = {}
        with tracks_pickle_path.open("rb") as f:
            raw_tracks: list[dict[str, Any]] = pickle.load(f)

        for track_data in raw_tracks:
            track_id = int(track_data["id"])
            track_record: dict[str, object] = {
                "cls": str(track_data.get("cls", "unknown")),
                "masks": {
                    int(frame_idx): np.asarray(mask, dtype=bool)
                    for frame_idx, mask in track_data["masks"].items()
                },
            }
            if isinstance(track_data.get("embeddings"), dict):
                track_record["embeddings"] = {
                    int(frame_idx): np.asarray(embedding, dtype=np.float32)
                    for frame_idx, embedding in track_data["embeddings"].items()
                }
            elif track_data.get("embedding") is not None:
                track_record["embedding"] = np.asarray(
                    track_data["embedding"], dtype=np.float32
                )
            tracks[track_id] = track_record
        return tracks

    def read_data(
        self,
        image_dir: str | Path,
        tracker_output_dir: str | Path,
        da3_output_dir: str | Path | None = None,
    ) -> Iterator[FrameSample]:
        """
        Yield per-frame inputs for downstream selectors.

        Notes:
            - `masks` and `embeddings` are grouped by class name first.
            - The nested key is the tracker object id because multiple objects can
              share the same class label in one frame.
            - Final tracker ids are loaded from `rerun_export/tracks.pkl`.
            - If the rerun export does not contain class labels, objects are grouped
              under the placeholder label `unknown`.
            - Embeddings are read only from the final tracker outputs if they were
              saved there. They are omitted otherwise.
            - `da3_output_dir` is accepted for future selectors that also need
              geometry, but the base loader does not consume it yet.
        """
        image_dir = Path(image_dir)
        tracks_pickle_path = self._resolve_rerun_tracks_path(Path(tracker_output_dir))

        if da3_output_dir is not None:
            self._resolve_da3_dir(Path(da3_output_dir))

        frame_paths = self._get_sorted_paths(image_dir, self._IMAGE_SUFFIXES)
        tracks = self._load_tracks(tracks_pickle_path)

        for frame_idx, image_path in enumerate(frame_paths):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")

            masks_by_class: defaultdict[str, dict[int, np.ndarray]] = defaultdict(dict)
            embeddings_by_class: defaultdict[str, dict[int, np.ndarray]] = defaultdict(
                dict
            )

            for track_id, track_data in tracks.items():
                mask = track_data["masks"].get(frame_idx)
                if mask is None:
                    continue

                class_name = str(track_data["cls"])
                mask_array = np.asarray(mask, dtype=bool)
                masks_by_class[class_name][track_id] = mask_array

                frame_embeddings = track_data.get("embeddings")
                if isinstance(frame_embeddings, dict) and frame_idx in frame_embeddings:
                    embeddings_by_class[class_name][track_id] = np.asarray(
                        frame_embeddings[frame_idx], dtype=np.float32
                    )
                    continue

                track_embedding = track_data.get("embedding")
                if track_embedding is not None:
                    embeddings_by_class[class_name][track_id] = np.asarray(
                        track_embedding, dtype=np.float32
                    )

            yield image, dict(masks_by_class), dict(embeddings_by_class)

    def save_frames(
        self,
        frame_ids: list[int],
        frame_sample_iter: Iterator[FrameSample],
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Save selected frames with track-id marks and export JSON metadata."""
        output_dir = Path(output_dir)
        marked_frames_dir = output_dir / "marked_frames"
        marked_frames_dir.mkdir(parents=True, exist_ok=True)

        target_frame_ids = {int(frame_id) for frame_id in frame_ids}
        if not target_frame_ids:
            metadata_path = output_dir / "frames.json"
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump({"frames": []}, f, indent=2)
            return {
                "marked_frames_dir": marked_frames_dir,
                "metadata_path": metadata_path,
            }

        metadata_frames: list[dict[str, Any]] = []
        missing_frame_ids = set(target_frame_ids)

        save_bar = tqdm(enumerate(frame_sample_iter))
        save_bar.set_description("Saving frames")
        for frame_idx, (image, masks, _) in save_bar:
            if frame_idx not in target_frame_ids:
                continue

            height, width = image.shape[:2]
            marked_image = image.copy()
            detected_objects: list[dict[str, Any]] = []

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

                    detected_objects.append(
                        {
                            "id": int(track_id),
                            "label": class_name,
                            "bbox": bbox,
                        }
                    )

            image_path = marked_frames_dir / f"frame_{frame_idx:04d}.png"
            if not cv2.imwrite(str(image_path), marked_image):
                raise ValueError(f"Failed to save marked frame: {image_path}")

            metadata_frames.append(
                {
                    "frame_id": int(frame_idx),
                    "frame_size": {"width": int(width), "height": int(height)},
                    "detected_objects": detected_objects,
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
        metadata_path = output_dir / "frames.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump({"frames": metadata_frames}, f, indent=2)

        return {
            "marked_frames_dir": marked_frames_dir,
            "metadata_path": metadata_path,
        }

    def _resize_mask(self, mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        height, width = image_size
        if mask.shape[:2] == (height, width):
            return np.asarray(mask, dtype=bool)
        resized = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        return resized > 0

    def _mask_to_bbox(self, mask: np.ndarray) -> list[int] | None:
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    def _get_mark_xy(self, mask: np.ndarray) -> tuple[int, int]:
        mask_uint8 = mask.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8
        )
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
        select_bar = tqdm(enumerate(frame_sample_iter))
        select_bar.set_description("Selecting frames")
        return [i for i, _ in select_bar if i % self.step == 0]


if __name__ == "__main__":
    selector = UniformSelector(8)
    frame_iter = selector.read_data("data/0/images", "data/0/tracker_outputs")
    frame_ids = selector.select_frames(frame_iter)
    frame_iter = selector.read_data("data/0/images", "data/0/tracker_outputs")
    selector.save_frames(frame_ids, frame_iter, "data/0/selected_frames")
