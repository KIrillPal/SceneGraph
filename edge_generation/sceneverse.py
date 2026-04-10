from __future__ import annotations

from itertools import combinations
from typing import Iterator, Sequence

import numpy as np
from tqdm import tqdm

from .base import BaseEdgeGenerator, EdgeRelationship, FrameRelationships, FrameSample

DEFAULT_EPS = 1e-6
DEFAULT_MARGIN_METERS = 0.1


class SceneVerseEdgeGenerator(BaseEdgeGenerator):
    def __init__(
        self,
        eps: float = DEFAULT_EPS,
        margin: float = DEFAULT_MARGIN_METERS,
        max_distance: float | None = None,
    ) -> None:
        self.eps = eps
        self.margin = margin
        self.max_distance = max_distance

    def _normalize(self, vector: Sequence[float]) -> np.ndarray | None:
        arr = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm <= self.eps:
            return None
        return arr / norm

    def _camera_pose(self, extrinsic: np.ndarray) -> np.ndarray:
        return np.asarray(extrinsic[:3, 3], dtype=np.float32)

    def _camera_up(self, extrinsic: np.ndarray) -> np.ndarray | None:
        # Exported camera poses use CV axes: x-right, y-down, z-forward.
        return self._normalize(-np.asarray(extrinsic[:3, 1], dtype=np.float32))

    def _anchor_axes(
        self,
        anchor_center: Sequence[float],
        extrinsic: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        camera_pose = self._camera_pose(extrinsic)
        forward = self._normalize(
            np.asarray(anchor_center, dtype=np.float32) - camera_pose
        )
        camera_up = self._camera_up(extrinsic)
        if forward is None or camera_up is None:
            return None

        right = self._normalize(np.cross(forward, camera_up))
        if right is not None:
            return right, forward

        fallback_right = np.asarray(extrinsic[:3, 0], dtype=np.float32)
        fallback_right = (
            fallback_right - float(np.dot(fallback_right, forward)) * forward
        )
        right = self._normalize(fallback_right)
        if right is None:
            return None
        return right, forward

    def get_semantic_edge(
        self,
        target_center: Sequence[float],
        anchor_center: Sequence[float],
        extrinsic: np.ndarray,
    ) -> str | None:
        axes = self._anchor_axes(anchor_center, extrinsic)
        if axes is None:
            return None
        right, forward = axes

        relative = np.asarray(target_center, dtype=np.float32) - np.asarray(
            anchor_center, dtype=np.float32
        )
        left_right_offset = float(np.dot(relative, right))
        depth_offset = float(np.dot(relative, forward))

        relation_parts: list[str] = []
        if left_right_offset < -self.margin:
            relation_parts.append("left")
        elif left_right_offset > self.margin:
            relation_parts.append("right")

        if depth_offset < -self.margin:
            relation_parts.append("front")
        elif depth_offset > self.margin:
            relation_parts.append("back")

        if not relation_parts:
            return None
        return "_".join(relation_parts)

    def _build_frame_edges(
        self,
        object_centers: dict[int, np.ndarray],
        extrinsic: np.ndarray,
    ) -> list[EdgeRelationship]:
        edges: list[EdgeRelationship] = []
        objects = sorted(object_centers.items(), key=lambda item: item[0])

        for (anchor_id, anchor_center), (target_id, target_center) in combinations(
            objects, 2
        ):
            anchor_center_arr = np.asarray(anchor_center, dtype=np.float32)
            target_center_arr = np.asarray(target_center, dtype=np.float32)
            if self.max_distance is not None:
                pair_distance = float(
                    np.linalg.norm(anchor_center_arr - target_center_arr)
                )
                if pair_distance > self.max_distance:
                    continue

            relation = self.get_semantic_edge(
                target_center=target_center_arr,
                anchor_center=anchor_center_arr,
                extrinsic=extrinsic,
            )
            if relation is None:
                continue
            edges.append((int(target_id), relation, int(anchor_id)))

        return edges

    def generate_edges(
        self,
        frame_sample_iter: Iterator[FrameSample],
    ) -> list[FrameRelationships]:
        frame_relationships: list[FrameRelationships] = []
        edge_bar = tqdm(frame_sample_iter, desc="Generating edges")
        for frame_id, object_centers, extrinsic in edge_bar:
            edges = self._build_frame_edges(object_centers, extrinsic)
            frame_relationships.append((int(frame_id), edges))
        return frame_relationships
