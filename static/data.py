"""
Feature Extraction Utilities for SAM3 Tracker

Provides text embeddings, visual embeddings, and point cloud generation.
"""

from sentence_transformers import SentenceTransformer, util
from utils.point import *
from utils.mask import *
import cv2
import numpy as np
import open3d as o3d


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

    # Encode each class once
    for cls in classes:
        text_embs[cls] = text_model.encode(cls).reshape([1, -1])

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
