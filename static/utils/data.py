"""
Feature Extraction Utilities for SAM3 Tracker

Provides text embeddings, visual embeddings, and point cloud generation.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Tuple, List, Dict, Set

from sentence_transformers import SentenceTransformer, util
from utils.point import *
from utils.mask import *
import cv2
import numpy as np
import open3d as o3d
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def get_text_embeddings(tracks: Dict[int, Dict]) -> Dict[str, np.ndarray]:
    """
    Generate text embeddings for all unique track classes.

    Uses SentenceTransformer to encode class names into semantic embeddings.
    Each class is encoded once and reused across all tracks of that class.

    Args:
        tracks: Dict of track_id -> track_data with 'cls' field.

    Returns:
        text_embs: Dict of class_name -> embedding [1, D] where D is embedding dim.

    Example:
        >>> tracks = {0: {'cls': 'car'}, 1: {'cls': 'person'}}
        >>> embs = get_text_embeddings(tracks)
        >>> print(embs['car'].shape)  # (1, 384) for MiniLM-L6
    """
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    text_embs = {}

    # Get unique classes
    classes: Set[str] = set(tracks[j]["cls"] for j in tracks.keys())
    logger.info("Getting text embeddings for %d classes", len(classes))

    # Encode each class once
    for cls_ in tqdm(
        sorted(classes),
        desc="Text embeddings",
        unit="class",
        dynamic_ncols=True,
    ):
        text_embs[cls_] = text_model.encode(cls_).reshape([1, -1])

    return text_embs


def get_object_embedding(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract object-specific embedding from image features using mask.

    Computes weighted average of features within the object mask,
    then L2-normalizes the result.

    Args:
        features: Image features [H*W, D] where D is feature dimension.
        mask: Binary object mask [H, W].

    Returns:
        embedding: L2-normalized object embedding [D].

    Notes:
        - Mask is resized to 72x72 for consistent feature aggregation
        - Small epsilon (1e-8) prevents division by zero
    """
    # Resize mask to fixed resolution for consistent aggregation
    resized = cv2.resize(mask.astype(np.uint8), (72, 72)).reshape(-1, 1)

    # Weighted average of features within mask
    emb = np.sum(features * resized, axis=0) / (np.sum(resized) + 1e-8)

    # L2 normalize
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-8)


def get_obj_point_cloud(
    image: np.ndarray,
    point_cloud: np.ndarray,
    mask: np.ndarray,
) -> o3d.geometry.PointCloud:
    """
    Generate a colored 3D point cloud for one object from a dense frame point cloud.

    Resizes the 2D mask to the dense point-cloud resolution, keeps only valid
    3D points inside the mask, applies outlier removal, and colors them from
    the resized image.

    Args:
        image: BGR image [H, W, 3] for coloring points.
        point_cloud: Dense 3D points [h, w, 3] in world coordinates.
        mask: Object mask [H, W] to filter points.

    Returns:
        pcd: Open3D point cloud with points and colors.
            Returns empty cloud if no valid points remain.
    """
    h, w = point_cloud.shape[:2]
    image_r = cv2.resize(image, (w, h)).reshape((-1, 3))
    mask_r = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    eroded_mask = safe_erode(mask_r)
    valid_mask = np.isfinite(point_cloud).all(axis=2) & (eroded_mask > 0)
    pts_mask = point_cloud[valid_mask]
    pts_last, last_mask = statistical_outlier_removal(pts_mask)

    pcd = o3d.geometry.PointCloud()
    if len(pts_last) == 0:
        return pcd

    pcd.points = o3d.utility.Vector3dVector(pts_last)
    colors = image_r[valid_mask.reshape(-1)][last_mask]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def read_tracking_data(
    image_dir: str, sam3_tracks_dir: str, sam3_embeds_dir: str
) -> Tuple[List[np.ndarray], Dict[int, Dict], List[np.ndarray]]:
    frame_names = sorted(os.listdir(image_dir))
    images = []
    logger.info("Getting images from %s", image_dir)
    for name in tqdm(
        frame_names,
        desc="Loading images",
        unit="image",
        dynamic_ncols=True,
    ):
        images.append(cv2.imread(os.path.join(image_dir, name)))

    track_files = sorted(os.listdir(sam3_tracks_dir))
    raw_tracks = {}
    logger.info("Getting tracks from %s", sam3_tracks_dir)
    for i, track_file in enumerate(
        tqdm(track_files, desc="Loading tracks", unit="track", dynamic_ncols=True)
    ):
        track = np.load(os.path.join(sam3_tracks_dir, track_file), allow_pickle=True)[
            "arr_0"
        ]
        if track.tolist()["cls"] != 0:
            raw_tracks[i] = track.tolist()
    tracks = merge_masks(raw_tracks)

    frame_embeds = []
    emb_files = sorted(os.listdir(sam3_embeds_dir))
    logger.info("Getting frame embeddings from %s", sam3_embeds_dir)
    for i, emb_file in enumerate(
        tqdm(emb_files, desc="Loading embeddings", unit="file", dynamic_ncols=True)
    ):
        frame_embeds.append(
            np.load(os.path.join(sam3_embeds_dir, emb_file), allow_pickle=True)["arr_0"]
        )

    return images, tracks, frame_embeds


def _read_extrinsics(extrinsics_file: str | Path) -> np.ndarray:
    """Read camera poses from a text file into an array of shape [N, 4, 4]."""
    extrinsics = []
    with open(extrinsics_file, "r", encoding="utf-8") as f:
        for line in f:
            pose = line.strip()
            if not pose:
                continue
            extrinsics.append(np.array(pose.split(), dtype=float).reshape((4, 4)))
    return np.stack(extrinsics, axis=0)


def get_da3_frame_data(
    depth_dir: str | Path,
    extrinsics_file: str | Path,
    num_frames: int,
    conf_thresh_mul: float = 0.5,
) -> List[Dict[str, np.ndarray]]:
    """Load DA3 outputs as dense per-frame point clouds aligned to depth pixels."""
    depth_dir = Path(depth_dir)
    extrinsics = _read_extrinsics(extrinsics_file).astype(np.float32)
    frame_data: List[Dict[str, np.ndarray]] = []

    logger.info("Getting DA3 frame data from %s", depth_dir)
    for frame_idx in tqdm(
        range(num_frames),
        desc="Loading DA3 frames",
        unit="frame",
        dynamic_ncols=True,
    ):
        frame_file = depth_dir / f"frame_{frame_idx}.npz"
        data = np.load(frame_file)

        depth = np.asarray(data["depth"], dtype=np.float32)
        conf = np.asarray(data["conf"], dtype=np.float32)
        intrinsic = np.asarray(data["intrinsics"], dtype=np.float32)
        extrinsic = np.asarray(extrinsics[frame_idx], dtype=np.float32)

        w2c = np.linalg.inv(extrinsic)[:3, :]
        dense_points = depth_to_point_cloud_vectorized(
            depth[np.newaxis, :, :],
            intrinsic[np.newaxis, :, :],
            w2c[np.newaxis, :, :],
        )[0].astype(np.float32)

        valid_mask = conf > (float(conf.mean()) * conf_thresh_mul)
        dense_points[~valid_mask] = np.nan

        frame_data.append(
            {
                "point_cloud": dense_points,
                "intrinsic": intrinsic,
                "extrinsic": extrinsic,
            }
        )

    return frame_data


def _get_track_label(track: Any) -> str:
    classes = np.asarray(list(track.cls.values()), dtype=object)
    values, counts = np.unique(classes, return_counts=True)
    return str(values[np.argmax(counts)])


def save_tracker_outputs(
    save_path: str | Path,
    all_tracks: List[Any],
    images: List[np.ndarray],
    da3_frame_data: List[Dict[str, np.ndarray]],
) -> Path:
    """Save one compressed NPZ file per frame keyed by final tracker ids."""
    logger.info("Saving tracker outputs to %s", save_path)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    track_labels = {int(track.id): _get_track_label(track) for track in all_tracks}

    for frame_idx, image in enumerate(
        tqdm(images, desc="Saving frame exports", unit="frame", dynamic_ncols=True)
    ):
        masks_by_class: defaultdict[str, dict[int, np.ndarray]] = defaultdict(dict)
        embeddings_by_class: defaultdict[str, dict[int, np.ndarray]] = defaultdict(dict)

        for track in all_tracks:
            track_id = int(track.id)
            mask = track.masks.get(frame_idx)
            if mask is None:
                continue

            class_name = track_labels[track_id]
            masks_by_class[class_name][track_id] = np.asarray(mask, dtype=bool)

            embedding = track.embeddings.get(frame_idx)
            if embedding is not None:
                embeddings_by_class[class_name][track_id] = np.asarray(
                    embedding, dtype=np.float32
                ).reshape(-1)

        frame_file = save_path / f"frame_{frame_idx:06d}.npz"
        np.savez_compressed(
            frame_file,
            frame_id=np.int32(frame_idx),
            image=np.asarray(image),
            masks=dict(masks_by_class),
            embeddings=dict(embeddings_by_class),
            point_cloud=np.asarray(
                da3_frame_data[frame_idx]["point_cloud"], dtype=np.float32
            ),
            intrinsic=np.asarray(
                da3_frame_data[frame_idx]["intrinsic"], dtype=np.float32
            ),
            extrinsic=np.asarray(
                da3_frame_data[frame_idx]["extrinsic"], dtype=np.float32
            ),
        )

    return save_path
