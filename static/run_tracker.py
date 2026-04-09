import argparse
import logging
import os

import numpy as np
from tqdm.auto import tqdm

from tracker import Simple3DTracker
from utils.data import (
    get_da3_frame_data,
    get_object_embedding,
    get_obj_point_cloud,
    get_text_embeddings,
    read_tracking_data,
    save_tracker_outputs,
)
from utils.track_vis import get_current_tracks

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
images, tracks, frame_embeds = read_tracking_data(
    args.image_folder, args.sam3_tracks, args.sam3_embeds
)
depth_dir = os.path.join(args.da3_outputs, "results_output")
extrinsics_file = os.path.join(args.da3_outputs, "camera_poses.txt")

logger.info("Getting DA3 frame data")
da3_frame_data = get_da3_frame_data(depth_dir, extrinsics_file, len(images))
logger.info("Getting text embeddings")
text_embs = get_text_embeddings(tracks)

tracker = Simple3DTracker()

frame_progress = tqdm(
    range(len(images)),
    desc="Tracking frames",
    unit="frame",
    dynamic_ncols=True,
)
for i in frame_progress:
    detections = []
    point_cloud = da3_frame_data[i]["point_cloud"]
    for j in tracks.keys():
        if i in tracks[j]["masks"].keys():
            det = {}
            det["sam_id"] = j
            det["cls"] = tracks[j]["cls"]
            det["mask"] = tracks[j]["masks"][i]
            pcd0 = get_obj_point_cloud(images[i], point_cloud, det["mask"])
            det["points"] = np.asarray(pcd0.points)
            if len(det["points"]) < 5:
                continue
            det["embedding"] = get_object_embedding(
                frame_embeds[i], det["mask"]
            ).reshape(1, -1)
            det["text_embedding"] = text_embs[det["cls"]]
            detections.append(det)
    tracker.update(detections, i)

    frame_progress.set_postfix(
        dets=len(detections),
        active=len(tracker.tracks),
        lost=len(tracker.lost_tracks),
    )

logger.info("Collecting final tracks")
all_tracks = get_current_tracks(tracker)
logger.info("Saving tracker outputs")
save_tracker_outputs(
    args.save_path,
    all_tracks,
    images,
    da3_frame_data,
)
logger.info("Done")
