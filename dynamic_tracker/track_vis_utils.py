"""
Visualization Utilities for SAM3 Tracker

Provides track visualization, color generation, and class label extraction.
"""

import cv2
import numpy as np
from skimage.measure import label, regionprops
import distinctipy
from typing import List, Dict, Tuple, Any


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
    # Generate unique colors for each track
    colors = (np.array(distinctipy.get_colors(len(all_tracks))) * 255).astype(np.uint8)
    vis_images = []

    for i in range(len(images)):
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

"""
Keypoint Visualization for Moving Objects

Visualizes Shi-Tomasi and CoTracker keypoints with track-colored trails.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.cm as cm

def visualize_all_tracks_keypoints(
    image: np.ndarray,
    tracks: List,
    current_frame: int,
    show_history: int = 10,
    show_tracks: bool = True,
    image_size: Optional[Tuple[int, int]] = None,
    debug: bool = False
) -> np.ndarray:
    """
    Visualize keypoints for all moving tracks with colored trails.
    Smaller keypoints, more visible trails.
    """
    vis_image = image.copy()
    if image_size is not None:
        vis_image = cv2.resize(vis_image, image_size)
        scale_x = image_size[0] / image.shape[1]
        scale_y = image_size[1] / image.shape[0]
    else:
        scale_x, scale_y = 1.0, 1.0
        image_size = (image.shape[1], image.shape[0])
    
    H, W = vis_image.shape[:2]
    
    # Generate unique colors for each track
    if len(tracks) > 0:
        cmap = cm.get_cmap('hsv', len(tracks))
        track_colors = {
            track.id: tuple(int(c * 255) for c in cmap(idx)[:3])
            for idx, track in enumerate(tracks)
        }
    else:
        track_colors = {}
    
    total_shi_tomasi = 0
    total_cotracker = 0
    
    for track_idx, track in enumerate(tracks):
        # Only visualize moving tracks
        if not track.is_dynamic or track.motion_state.status != 'MOVING':
            continue
        
        track_color = track_colors.get(track.id, (255, 255, 255))
        
        # === 1. Draw Motion Trails FIRST (so they're behind keypoints) ===
        if show_tracks:
            trail_points = []
            for frame_idx in range(max(0, current_frame - show_history), current_frame + 1):
                if frame_idx in track.moving_keypoints and track.moving_keypoints[frame_idx] is not None:
                    kps = track.moving_keypoints[frame_idx]
                    if len(kps) > 0:
                        kps_scaled = kps.copy()
                        kps_scaled[:, 0] *= scale_x
                        kps_scaled[:, 1] *= scale_y
                        
                        # Take first few keypoints
                        for kp in kps_scaled[:5]:
                            x, y = int(round(kp[0])), int(round(kp[1]))
                            if 0 <= x < W and 0 <= y < H:
                                trail_points.append((x, y))
            
            # Draw trail lines (THICKER for visibility)
            if len(trail_points) > 1:
                for i in range(len(trail_points) - 1):
                    alpha = (i + 1) / len(trail_points)
                    trail_color = tuple(int(c * alpha) for c in track_color)
                    cv2.line(
                        vis_image,
                        trail_points[i],
                        trail_points[i + 1],
                        trail_color,
                        thickness=3,  # ← THICKER trails
                        lineType=cv2.LINE_AA
                    )
        
        # === 2. Draw Shi-Tomasi Keypoints (SMALLER) ===
        if current_frame in track.keypoints and track.keypoints[current_frame] is not None:
            kps = track.keypoints[current_frame]
            if len(kps) > 0:
                total_shi_tomasi += len(kps)
                
                kps_scaled = kps.copy()
                kps_scaled[:, 0] *= scale_x
                kps_scaled[:, 1] *= scale_y
                
                for kp in kps_scaled:
                    x, y = int(round(kp[0])), int(round(kp[1]))
                    if 0 <= x < W and 0 <= y < H:
                        # SMALLER circles (radius 3 instead of 5-6)
                        cv2.circle(vis_image, (x, y), 4, (255, 255, 255), -1)  # White border
                        cv2.circle(vis_image, (x, y), 3, track_color, -1)  # Track color fill
        
        # === 3. Draw CoTracker Keypoints (SMALLER) ===
        if current_frame in track.moving_keypoints and track.moving_keypoints[current_frame] is not None:
            kps = track.moving_keypoints[current_frame]
            if len(kps) > 0:
                total_cotracker += len(kps)
                
                kps_scaled = kps.copy()
                kps_scaled[:, 0] *= scale_x
                kps_scaled[:, 1] *= scale_y
                
                for kp in kps_scaled:
                    x, y = int(round(kp[0])), int(round(kp[1]))
                    if 0 <= x < W and 0 <= y < H:
                        # SMALLER diamonds (size 4 instead of 6)
                        diamond_size = 4
                        pts = np.array([
                            [x, y - diamond_size],
                            [x + diamond_size, y],
                            [x, y + diamond_size],
                            [x - diamond_size, y]
                        ], np.int32)
                        cv2.fillPoly(vis_image, [pts], track_color)
                        cv2.polylines(vis_image, [pts], True, (255, 255, 255), 1)
        
        # === 4. Draw Track ID Label ===
        all_current_kps = []
        if current_frame in track.keypoints and track.keypoints[current_frame] is not None:
            kps = track.keypoints[current_frame].copy()
            kps[:, 0] *= scale_x
            kps[:, 1] *= scale_y
            all_current_kps.append(kps)
        
        if current_frame in track.moving_keypoints and track.moving_keypoints[current_frame] is not None:
            kps = track.moving_keypoints[current_frame].copy()
            kps[:, 0] *= scale_x
            kps[:, 1] *= scale_y
            all_current_kps.append(kps)
        
        if len(all_current_kps) > 0:
            all_kps = np.vstack(all_current_kps)
            if len(all_kps) > 0:
                x, y = int(np.mean(all_kps[:, 0])), int(np.mean(all_kps[:, 1]))
                
                label = f"#{track.id}"
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    vis_image,
                    (x - 5, y - text_h - 8),
                    (x + text_w + 5, y + 3),
                    (0, 0, 0),
                    -1
                )
                cv2.putText(
                    vis_image,
                    label,
                    (x, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    track_color,
                    1
                )
    
    # === 5. Draw Legend ===
    legend_h = 120
    cv2.rectangle(vis_image, (10, 10), (250, 10 + legend_h), (0, 0, 0), -1)
    cv2.putText(vis_image, "Keypoint Visualization", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.line(vis_image, (20, 45), (240, 45), (255, 255, 255), 1)
    
    cv2.circle(vis_image, (30, 65), 3, (200, 200, 200), -1)
    cv2.putText(vis_image, "Shi-Tomasi (detected)", (45, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    diamond_pts = np.array([[30, 90], [34, 94], [30, 98], [26, 94]], np.int32)
    cv2.fillPoly(vis_image, [diamond_pts], (200, 200, 200))
    cv2.polylines(vis_image, [diamond_pts], True, (255, 255, 255), 1)
    cv2.putText(vis_image, "CoTracker (predicted)", (45, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    cv2.line(vis_image, (20, 110), (240, 110), (255, 255, 255), 1)
    cv2.putText(vis_image, f"Shi-Tomasi: {total_shi_tomasi}", (20, 125), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.putText(vis_image, f"CoTracker: {total_cotracker}", (20, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
    
    if debug:
        print(f"Frame {current_frame}: Shi-Tomasi={total_shi_tomasi}, CoTracker={total_cotracker}")
    
    return vis_image


def create_keypoint_video(
    video_frames: List[np.ndarray],
    tracks: List,
    output_path: str,
    show_history: int = 10,
    fps: int = 30,
    debug: bool = False
) -> None:
    """
    Create video visualization of keypoints over all frames.
    """
    if len(video_frames) == 0:
        print("No frames to visualize!")
        return
    
    H, W = video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    print(f"Creating keypoint video: {len(video_frames)} frames...")
    
    for frame_idx, frame in enumerate(video_frames):
        vis_frame = visualize_all_tracks_keypoints(
            image=frame,
            tracks=tracks,
            current_frame=frame_idx,
            show_history=show_history,
            debug=debug
        )
        
        # Add frame counter
        cv2.putText(vis_frame, f"Frame {frame_idx}", (W - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(vis_frame)
        
        if debug and frame_idx % 10 == 0:
            print(f"Processed {frame_idx}/{len(video_frames)} frames")
    
    out.release()
    print(f"Saved keypoint video to {output_path}")