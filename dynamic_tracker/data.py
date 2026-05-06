"""
Feature Extraction Utilities for SAM3 Tracker

Provides text embeddings, visual embeddings, and point cloud generation.
"""

import cv2
import numpy as np
import open3d as o3d
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional
from mask_utils import safe_erode
from point_utils import statistical_outlier_removal


# ============================================================================
# CONSTANTS
# ============================================================================

class Config:
    """Feature extraction configuration."""
    # Text embeddings
    TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    # Object embeddings
    MASK_RESIZE = (72, 72)
    EPSILON = 1e-8
    
    # Point cloud generation
    PC_TARGET_H = 378
    PC_TARGET_W = 504
    
    # Keypoints
    MAX_KEYPOINTS = 100
    KEYPOINT_QUALITY = 0.01
    KEYPOINT_MIN_DIST = 2


# ============================================================================
# TEXT EMBEDDINGS
# ============================================================================

class TextEmbeddingCache:
    """Cache for text embeddings to avoid recomputation."""
    
    def __init__(self, model_name: str = Config.TEXT_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.cache: Dict[str, np.ndarray] = {}
    
    def get_embedding(self, class_name: str) -> np.ndarray:
        """
        Get cached text embedding for a class name.
        
        Args:
            class_name: Object class name (e.g., 'car', 'person').
        
        Returns:
            embedding: Text embedding [1, D].
        """
        class_name = class_name.lower().strip()
        
        if class_name not in self.cache:
            emb = self.model.encode(class_name).reshape(1, -1)
            self.cache[class_name] = emb
        
        return self.cache[class_name]
    
    def get_batch_embeddings(self, class_names: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple class names.
        
        Args:
            class_names: List of class names.
        
        Returns:
            embeddings: Array [N, D] of embeddings.
        """
        embeddings = [self.get_embedding(cls) for cls in class_names]
        return np.vstack(embeddings)


# Global cache instance (lazy initialization)
_text_embedding_cache: Optional[TextEmbeddingCache] = None


def get_text_embedding(
    class_name: str,
    cache: Optional[TextEmbeddingCache] = None
) -> np.ndarray:
    """
    Get text embedding for a single class name.
    
    Args:
        class_name: Object class name.
        cache: Optional embedding cache (creates new if None).
    
    Returns:
        embedding: Text embedding [1, D].
    """
    global _text_embedding_cache
    
    if cache is None:
        if _text_embedding_cache is None:
            _text_embedding_cache = TextEmbeddingCache()
        cache = _text_embedding_cache
    
    return cache.get_embedding(class_name)


def get_text_embeddings(
    tracks: Dict[int, Dict],
    cache: Optional[TextEmbeddingCache] = None
) -> Dict[str, np.ndarray]:
    """
    Generate text embeddings for all unique track classes.
    
    Uses SentenceTransformer to encode class names into semantic embeddings.
    Each class is encoded once and cached for reuse.
    
    Args:
        tracks: Dict of track_id -> track_data with 'cls' field.
        cache: Optional embedding cache (creates new if None).
    
    Returns:
        text_embs: Dict of class_name -> embedding [1, D].
    
    Example:
        >>> tracks = {0: {'cls': 'car'}, 1: {'cls': 'person'}}
        >>> embs = get_text_embeddings(tracks)
        >>> print(embs['car'].shape)  # (1, 384) for MiniLM-L6
    """
    # Get unique classes
    classes = set(track["cls"] for track in tracks.values())
    
    # Generate embeddings
    text_embs = {}
    for cls in classes:
        text_embs[cls] = get_text_embedding(cls, cache)
    
    return text_embs


# ============================================================================
# VISUAL EMBEDDINGS
# ============================================================================

def get_object_embedding(
    features: np.ndarray,
    mask: np.ndarray,
    resize: tuple = Config.MASK_RESIZE,
    epsilon: float = Config.EPSILON
) -> np.ndarray:
    """
    Extract object-specific embedding from image features using mask.
    
    Computes weighted average of features within the object mask,
    then L2-normalizes the result.
    
    Args:
        features: Image features [H*W, D] where D is feature dimension.
        mask: Binary object mask [H, W].
        resize: Target size for mask resizing (default: 72x72).
        epsilon: Small value to prevent division by zero.
    
    Returns:
        embedding: L2-normalized object embedding [D].
    
    Notes:
        - Mask is resized for consistent feature aggregation
        - L2 normalization ensures unit norm embedding
    """
    # Resize mask to fixed resolution
    mask_uint8 = mask.astype(np.uint8)
    resized = cv2.resize(mask_uint8, resize).reshape(-1, 1)
    
    # Weighted average of features within mask
    emb = np.sum(features * resized, axis=0) / (np.sum(resized) + epsilon)
    
    # L2 normalize
    norm = np.linalg.norm(emb)
    return emb / (norm + epsilon)


# ============================================================================
# POINT CLOUD GENERATION
# ============================================================================

def get_obj_point_cloud(
    image: np.ndarray,
    points: np.ndarray,
    clean_mask: np.ndarray,
    mask: np.ndarray,
    h: int = Config.PC_TARGET_H,
    w: int = Config.PC_TARGET_W,
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
    
    if len(pts_mask) == 0:
        return o3d.geometry.PointCloud()
    
    # Remove statistical outliers
    pts_last, last_mask = statistical_outlier_removal(pts_mask)
    
    if len(pts_last) == 0:
        return o3d.geometry.PointCloud()
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_last)
    
    # Assign RGB colors (filtered by outlier removal mask)
    colors = image_r[clean_mask][mask_r > 0][last_mask]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


# ============================================================================
# KEYPOINT DETECTION
# ============================================================================

def get_keypoints(
    image: np.ndarray,
    mask: np.ndarray,
    max_keypoints: int = Config.MAX_KEYPOINTS,
    quality: float = Config.KEYPOINT_QUALITY,
    min_distance: int = Config.KEYPOINT_MIN_DIST
) -> np.ndarray:
    """
    Detect Shi-Tomasi corners within object mask.
    
    Args:
        image: RGB image [H, W, 3].
        mask: Binary object mask [H, W].
        max_keypoints: Maximum number of keypoints to detect.
        quality: Quality level (0-1, higher = better corners).
        min_distance: Minimum distance between keypoints in pixels.
    
    Returns:
        corners: Keypoint coordinates [N, 2] or empty array if none found.
    
    Notes:
        - Uses Shi-Tomasi corner detection
        - Mask restricts detection to object region
        - Returns empty array (not None) if no corners found
    """
    # Apply mask to image
    masked_img = image.copy()
    masked_img[:, :, 0] *= mask
    masked_img[:, :, 1] *= mask
    masked_img[:, :, 2] *= mask
    
    # Convert to grayscale
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    
    # Detect corners
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_keypoints,
        qualityLevel=quality,
        minDistance=min_distance
    )
    
    # Convert to [N, 2] format
    if corners is not None:
        corners = np.array([corner[0] for corner in corners])
    else:
        corners = np.zeros((0, 2))
    
    return corners
