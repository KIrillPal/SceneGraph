"""
Feature Extraction Utilities for SAM3 Tracker

Provides text embeddings, visual embeddings, and point cloud generation.
"""

import json
import logging
import os
import pickle
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
    points: np.ndarray,
    clean_mask: np.ndarray,
    mask: np.ndarray,
    h: int = 378,
    w: int = 504,
) -> o3d.geometry.PointCloud:
    """
    Generate colored 3D point cloud from image, depth points, and masks.

    Combines 2D mask with 3D points, applies outlier removal, and assigns
    RGB colors from the image to each point.

    Args:
        image: RGB image [H, W, 3] for coloring points.
        points: 3D points [N, 3] in world coordinates.
        clean_mask: Boolean mask [h*w] for valid pixels.
        mask: Object mask [H, W] to filter points.
        h: Target height for resizing (default: 378).
        w: Target width for resizing (default: 504).

    Returns:
        pcd: Open3D point cloud with points and colors.
            Returns empty cloud if no valid points remain.

    Notes:
        - Mask is resized to (w, h) for consistency
        - Safe erosion applied to remove boundary artifacts
        - Statistical outlier removal filters noisy 3D points
        - 2D coordinates (xs, ys) are computed but commented out
    """
    # Resize image and mask to target resolution
    image_r = cv2.resize(image, (w, h)).reshape((-1, 3))
    H, W = mask.shape
    mask_r = cv2.resize(mask.astype(np.uint8), (w, h))

    # Apply safe erosion to remove boundary artifacts
    eroded_mask = safe_erode(mask_r)

    # Apply clean_mask filter
    mask_r = eroded_mask.flatten()[clean_mask]

    # Filter 3D points by mask
    pts_mask = points[mask_r > 0]

    # Remove statistical outliers
    pts_last, last_mask = statistical_outlier_removal(pts_mask)

    # Compute 2D pixel coordinates (scaled to original mask size)
    indices = np.indices((h, w))
    ys = (indices[0] * H / h).astype(int)
    xs = (indices[1] * W / w).astype(int)

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()

    # Get 2D coordinates for valid points
    xs_masked = xs.flatten()[clean_mask][mask_r > 0]
    ys_masked = ys.flatten()[clean_mask][mask_r > 0]

    # Return empty cloud if no valid points
    if len(pts_last) == 0 or len(xs_masked) == 0:
        return pcd

    # Assign 3D points
    pcd.points = o3d.utility.Vector3dVector(pts_last)

    # Assign RGB colors (filtered by outlier removal mask)
    # Note: xs_last, ys_last computation commented out
    colors = image_r[clean_mask][mask_r > 0][last_mask]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def read_tracking_data(image_dir: str, 
                       sam3_tracks_dir: str, 
                       sam3_embeds_dir: str
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
        track = np.load(os.path.join(sam3_tracks_dir, track_file), allow_pickle=True)['arr_0']
        if track.tolist()['cls'] != 0:
            raw_tracks[i] = track.tolist()
    tracks = merge_masks(raw_tracks)

    frame_embeds = []
    emb_files = sorted(os.listdir(sam3_embeds_dir))
    logger.info("Getting frame embeddings from %s", sam3_embeds_dir)
    for i, emb_file in enumerate(
        tqdm(emb_files, desc="Loading embeddings", unit="file", dynamic_ncols=True)
    ):
        frame_embeds.append(np.load(os.path.join(sam3_embeds_dir, emb_file), allow_pickle=True)['arr_0'])

    return images, tracks, frame_embeds


def create_output_dirs(save_path: str | Path) -> Dict[str, Path]:
    """
    Create the tracker output directory layout from a single save path.

    Args:
        save_path: Base directory for one tracker run, e.g.
            'vis_data/run_2' or 'data/0/tracker_outputs'.

    Returns:
        Dict with the created output directories.
    """
    save_path = Path(save_path)

    out_dirs = {
        "base_dir": save_path,
        "out_dir": save_path / "outputs",
        "track_out_dir": save_path / "track_outputs",
        "out_meta_dir": save_path / "meta_outputs",
        "out_filtered_dir": save_path / "filtered_outputs",
        "out_point_dir": save_path / "point_outputs",
        "rerun_export_dir": save_path / "rerun_export",
    }

    for path in out_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return out_dirs


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


def save_rerun_export(
    save_path: str | Path,
    all_tracks: List[Any],
    track_voxels_history: Dict[int, Dict[int, np.ndarray]],
    extrinsics_file: str | Path,
    points_per_frame: List[np.ndarray],
    points_per_frame_masks: List[np.ndarray],
) -> Path:
    """
    Save the minimal rerun export required by the tracker pipeline.

    Saves only:
        - tracks.pkl
        - extrinsics.npy
        - points_per_frame.pkl
        - points_per_frame_masks.pkl
    """
    logger.info("Saving rerun export")
    out_dirs = create_output_dirs(save_path)
    rerun_export_dir = out_dirs["rerun_export_dir"]
    rerun_export_dir.mkdir(parents=True, exist_ok=True)

    tracks_serial = []
    for track in tqdm(
        all_tracks,
        desc="Preparing rerun export",
        unit="track",
        dynamic_ncols=True,
    ):
        track_id = int(track.id)
        voxels_by_frame = {
            int(frame_idx): np.asarray(points, dtype=np.float32)
            for frame_idx, points in track_voxels_history.get(track_id, {}).items()
        }
        tracks_serial.append(
            {
                "id": track_id,
                "masks": {int(k): v for k, v in track.masks.items()},
                "voxels_by_frame": voxels_by_frame,
            }
        )

    with open(rerun_export_dir / "tracks.pkl", "wb") as f:
        pickle.dump(tracks_serial, f, protocol=pickle.HIGHEST_PROTOCOL)

    np.save(rerun_export_dir / "extrinsics.npy", _read_extrinsics(extrinsics_file))

    with open(rerun_export_dir / "points_per_frame.pkl", "wb") as f:
        pickle.dump(points_per_frame, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(rerun_export_dir / "points_per_frame_masks.pkl", "wb") as f:
        pickle.dump(points_per_frame_masks, f, protocol=pickle.HIGHEST_PROTOCOL)

    return rerun_export_dir


def save_tracker_outputs(
    save_path: str | Path,
    masked_images: List[np.ndarray],
    tracks: Dict[int, Dict],
    points_per_frame: List[np.ndarray],
    points_per_frame_masks: List[np.ndarray],
    all_tracks: List[Any],
    track_voxels_history: Dict[int, Dict[int, np.ndarray]],
    extrinsics_file: str | Path,
) -> Dict[str, Path]:
    """
    Save tracker outputs into the standard output directory layout.

    Saves:
        - masked images to outputs/<run>
        - per-track npz files to track_outputs/<run>
        - track_names.json to meta_outputs/<run>
        - filtered masks to filtered_outputs/<run>
        - per-frame point clouds to point_outputs/<run>
        - minimal rerun export to rerun_export
    """
    logger.info("Saving tracker outputs to %s", save_path)
    out_dirs = create_output_dirs(save_path)

    for i, image in enumerate(
        tqdm(masked_images, desc="Saving masked images", unit="image", dynamic_ncols=True)
    ):
        cv2.imwrite(str(out_dirs["out_dir"] / f"{i}.png"), image)

    for track_id, track_data in tqdm(
        tracks.items(),
        desc="Saving track files",
        unit="track",
        dynamic_ncols=True,
        total=len(tracks),
    ):
        np.savez(out_dirs["track_out_dir"] / f"{track_id}.npz", track_data, pickle=True)

    track_names = {int(idx): {"cls": track["cls"]} for idx, track in tracks.items()}
    with open(out_dirs["out_meta_dir"] / "track_names.json", "w", encoding="utf-8") as f:
        json.dump(track_names, f, indent=4)

    for i, mask in enumerate(
        tqdm(
            points_per_frame_masks,
            desc="Saving filtered masks",
            unit="frame",
            dynamic_ncols=True,
        )
    ):
        np.save(out_dirs["out_filtered_dir"] / f"{i}_filtered.npy", mask)

    for i, points in enumerate(
        tqdm(points_per_frame, desc="Saving point clouds", unit="frame", dynamic_ncols=True)
    ):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(str(out_dirs["out_point_dir"] / f"{i}.ply"), pcd)

    save_rerun_export(
        save_path,
        all_tracks,
        track_voxels_history,
        extrinsics_file,
        points_per_frame,
        points_per_frame_masks,
    )

    return out_dirs
