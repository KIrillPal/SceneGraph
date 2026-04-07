"""
Visualization Utilities for SAM3 Tracker

Provides track visualization, color generation, and class label extraction.
"""

import cv2
import logging
import numpy as np
from skimage.measure import label, regionprops
import distinctipy
from typing import List, Dict, Tuple, Any
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def get_current_tracks(tracker) -> List[Any]:
    """
    Get all active and lost tracks that meet minimum hit threshold.

    Filters tracks to include only those with sufficient confirmation
    (hits >= min_hits_to_activate), excluding unconfirmed tentative tracks.

    Args:
        tracker: Tracker object with 'tracks' and 'lost_tracks' attributes.

    Returns:
        List of confirmed tracks (active + lost).
    """
    tracks = [t for t in tracker.tracks if t.hits >= t.min_hits_to_activate]
    lost_tracks = [t for t in tracker.lost_tracks if t.hits >= t.min_hits_to_activate]
    current_tracks = tracks.copy()
    current_tracks.extend(lost_tracks)
    return current_tracks


def visualize_tracks(
    images: List[np.ndarray], all_tracks: List[Any], vis_type: str = "mask"
) -> List[np.ndarray]:
    """
    Visualize tracked objects on images with unique colors.

    Args:
        images: List of RGB frames [H, W, 3].
        all_tracks: List of track objects with 'masks' attribute.
        vis_type: Visualization mode:
            - 'mask': Overlay colored masks on images
            - 'mark': Draw track ID labels at object centers

    Returns:
        List of visualized frames with track overlays.

    Notes:
        - Uses distinctipy for unique color generation
        - Each track gets a consistent color across all frames
    """
    logger.info("Visualizing tracks in %s mode", vis_type)
    # Generate unique colors for each track
    colors = (np.array(distinctipy.get_colors(len(all_tracks))) * 255).astype(np.uint8)
    vis_images = []

    for i in tqdm(
        range(len(images)),
        desc=f"Visualizing {vis_type}",
        unit="frame",
        dynamic_ncols=True,
    ):
        if vis_type == "mask":
            # Overlay colored masks
            all_mask = np.zeros_like(images[i])
            for j, track in enumerate(all_tracks):
                masks = track.masks
                if i in masks.keys():
                    mask = masks[i]
                    all_mask[mask > 0] = colors[j]
            vis_image = cv2.addWeighted(images[i], 0.7, all_mask, 0.3, 0)
            vis_images.append(vis_image)

        elif vis_type == "mark":
            # Draw track ID labels at object centers
            img = images[i].copy()
            for j, track in enumerate(all_tracks):
                masks = track.masks
                if i in masks.keys():
                    mask = masks[i]
                    mark_xyc = get_mark_xy(mask)
                    mark_x, mark_y = mark_xyc[0], mark_xyc[1]
                    color_tuple = tuple(int(x) for x in colors[j])
                    cv2.putText(
                        img,
                        str(j),
                        (mark_x, mark_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color_tuple,
                        2,
                        cv2.LINE_AA,
                    )
            vis_images.append(img)

    return vis_images


def get_mark_xy(mask: np.ndarray) -> Tuple[int, int]:
    """
    Get centroid coordinates of the largest connected component in mask.

    Args:
        mask: Binary mask [H, W].

    Returns:
        Tuple of (center_x, center_y) pixel coordinates.

    Notes:
        - Uses skimage regionprops for centroid calculation
        - Returns centroid of largest object if multiple exist
    """
    labeled = label(mask)
    regions = regionprops(labeled)

    if regions:
        # Get largest connected component by area
        largest = max(regions, key=lambda r: r.area)
        cy, cx = largest.centroid
        mark = (int(cx), int(cy))
        return mark

    # Fallback: return center of image if no regions found
    return (mask.shape[1] // 2, mask.shape[0] // 2)


def get_cls(track, names: Dict[int, str]) -> str:
    """
    Get most frequent class label for a track.

    Aggregates class predictions across all frames and returns
    the majority vote class name.

    Args:
        track: Track object with 'cls' dict {frame_idx: class_id}.
        names: Dict mapping class_id -> class_name.

    Returns:
        Most frequent class name for this track.

    Example:
        >>> track.cls = {0: 'car', 1: 'car', 2: 'truck'}
        >>> names = {0: 'car', 1: 'truck'}
        >>> get_cls(track, names)
        'car'
    """
    classes = np.array(list(track.cls.values()))
    vals, counts = np.unique(classes, return_counts=True)
    cls_id = vals[np.argmax(counts)]
    return names[cls_id]
