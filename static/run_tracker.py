
import argparse
import logging
import os

import numpy as np
from tqdm.auto import tqdm

from tracker import Simple3DTracker
from utils.data import (
    read_tracking_data, get_da3_pointclouds, get_text_embeddings,
    get_object_embedding, get_obj_point_cloud, save_tracker_outputs
)
from utils.track_vis import get_current_tracks, visualize_tracks

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Path to folder with images, sam3 tracks, sam3 embeddings and da3 outputs"
)
parser.add_argument("image_folder", type=str, help="Path to folder with images")
parser.add_argument("sam3_tracks", type=str, help="Path to folder with sam3 tracks")
parser.add_argument("sam3_embeds", type=str, help="Path to folder with sam3 embeddings")
parser.add_argument("da3_outputs", type=str, help="Path to folder with da3 outputs")
parser.add_argument(
    "save_path",
    type=str,
    help="Base output directory for one run, e.g. data/0/tracker_outputs",
)
args = parser.parse_args()      

logger.info("Getting images, tracks, and frame embeddings")
images, tracks, frame_embeds = read_tracking_data(args.image_folder, args.sam3_tracks, args.sam3_embeds)
depth_dir = os.path.join(args.da3_outputs, 'results_output')
extrinsics_file = os.path.join(args.da3_outputs, 'camera_poses.txt')

logger.info("Getting DA3 point clouds")
points_per_frame, points_per_frame_masks,h,w = get_da3_pointclouds(depth_dir, extrinsics_file, len(images))
logger.info("Getting text embeddings")
text_embs = get_text_embeddings(tracks)

tracker = Simple3DTracker()
track_voxels_history = {}

frame_progress = tqdm(
    range(len(images)),
    desc="Tracking frames",
    unit="frame",
    dynamic_ncols=True,
)
for i in frame_progress:
    detections = []
    for j in tracks.keys():
        if i in tracks[j]['masks'].keys():
            det = {}
            det['sam_id'] = j
            det['cls'] = tracks[j]['cls']
            det['mask'] = tracks[j]['masks'][i]
            pcd0 = get_obj_point_cloud(images[i], points_per_frame[i],
                                       points_per_frame_masks[i], det['mask'], h, w)
            det['points']=np.asarray(pcd0.points)
            if len(det['points']) < 5:
                continue
            det['embedding']=get_object_embedding(frame_embeds[i], det['mask']).reshape(1,-1)
            det['text_embedding']=text_embs[det['cls']]
            detections.append(det)
    tracker.update(detections, i)

    # Сохраняем voxelmap каждого существующего трека на каждом кадре
    existing_tracks = {int(t.id): t for t in tracker.tracks}
    for t in tracker.lost_tracks:
        existing_tracks[int(t.id)] = t

    for tid, t in existing_tracks.items():
        pcd_global = t.voxels.get_pcd()
        pcd_global = np.asarray(pcd_global.points)
        if len(pcd_global) == 0:
            continue
        pts_global = pcd_global.astype(np.float32)
        if tid not in track_voxels_history:
            track_voxels_history[tid] = {}
        track_voxels_history[tid][int(i)] = pts_global

    frame_progress.set_postfix(
        dets=len(detections),
        active=len(tracker.tracks),
        lost=len(tracker.lost_tracks),
    )

logger.info("Collecting final tracks")
all_tracks = get_current_tracks(tracker)
logger.info("Visualizing tracks")
masked_images = visualize_tracks(images, all_tracks, 'mask')
logger.info("Saving tracker outputs")
save_tracker_outputs(
    args.save_path,
    masked_images,
    tracks,
    points_per_frame,
    points_per_frame_masks,
    all_tracks,
    track_voxels_history,
    extrinsics_file,
)
logger.info("Done")
