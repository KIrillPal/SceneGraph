import argparse
import os

import cv2
import numpy as np

from sam3.model_builder import build_sam3_video_predictor
from utils import (
    get_frame_embeddings, save_embeddings, split_by_tracks,
    segment_on_vigeo, merge_predicts, perform_nms
)

parser = argparse.ArgumentParser(
    description="Path to folder with images, .txt with unique objects in scene and save path."
)
parser.add_argument("image_folder", type=str, help="Path to folder with images")
parser.add_argument("txt_path", type=str, help="Path .txt with unique objects in scene")
parser.add_argument("save_path", type=str, help="Path to folder where to save predicted masks and embeddings")
args = parser.parse_args()      

os.makedirs(f'{args.save_path}/tracks', exist_ok=True)

predictor = build_sam3_video_predictor(checkpoint_path="/workspace/sam3/sam3.pt")
image_names = os.listdir(args.image_folder)
image = cv2.imread(f"{args.image_folder}/{image_names[0]}")
h, w, c = image.shape

# Save embeddings for further usage
frame_embeds = get_frame_embeddings(args.image_folder)
save_embeddings(frame_embeds, save_path=args.save_path)
# Read txt with unique objects into list
with open(args.txt_path, "r") as f:
    object_names = [line.strip() for line in f if line.strip()]
# Segment with text prompt
preds = {}
for name in object_names:
    text_prompt = {"text": name}
    outputs_per_frame = segment_on_vigeo(predictor, args.image_folder, text_prompt, prompt_idx=0)
    preds[name]=outputs_per_frame
# Merge predictions across classes, assigning unique global IDs.
all_preds, max_id = merge_predicts(object_names, preds)
# Non maximum supression
preds_filtered = perform_nms(all_preds, h, w)
# Split into 
tracks = split_by_tracks(preds_filtered,max_id)
for i in range(len(tracks)):
    np.savez(f'{args.save_path}/tracks/{i}.npz', tracks[i], pickle=True)
