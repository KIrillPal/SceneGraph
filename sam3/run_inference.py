import argparse
import os

# Keep SAM3 internals quiet; this script exposes one object-level progress bar.
os.environ["TQDM_DISABLE"] = "1"

import cv2
import numpy as np
from tqdm import tqdm

from sam3.model_builder import build_sam3_video_predictor
from utils import (
    get_frame_embeddings, save_embeddings, split_by_tracks,
    segment_on_vigeo, merge_predicts, perform_nms
)

parser = argparse.ArgumentParser(
    description="Path to folder with images, .txt with unique object prompts in scene and save path."
)
parser.add_argument("image_folder", type=str, help="Path to folder with images")
parser.add_argument(
    "txt_path",
    type=str,
    help="Path .txt with '<description>, <class>, <static|dynamic>' lines; first field is used as the SAM3 prompt and second as the saved class label",
)
parser.add_argument("save_path", type=str, help="Path to folder where to save predicted masks and embeddings")
parser.add_argument(
    "--score-threshold-detection",
    type=float,
    default=0.2,
    help="Minimum SAM3 detection confidence before tracking",
)
parser.add_argument(
    "--new-det-threshold",
    type=float,
    default=0.4,
    help="Minimum confidence for adding a detection as a new object track",
)
args = parser.parse_args()      


def normalize_class_name(value: str) -> str:
    return "_".join(value.strip().lower().split())


def read_object_prompts(txt_path: str) -> tuple[list[str], dict[str, str]]:
    object_names = []
    class_names = {}
    seen = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split(",")]
            name = parts[0]
            if not name or name in seen:
                continue
            class_name = normalize_class_name(parts[1]) if len(parts) >= 3 else name
            if not class_name:
                raise ValueError(f"Class field is empty in object prompt line: {line}")
            class_names[name] = class_name
            object_names.append(name)
            seen.add(name)
    return object_names, class_names

os.makedirs(f'{args.save_path}/tracks', exist_ok=True)

predictor = build_sam3_video_predictor(checkpoint_path="/workspace/sam3/sam3.pt")
predictor.model.score_threshold_detection = args.score_threshold_detection
predictor.model.new_det_thresh = args.new_det_threshold
image_names = os.listdir(args.image_folder)
image = cv2.imread(f"{args.image_folder}/{image_names[0]}")
h, w, c = image.shape

# Save embeddings for further usage
frame_embeds = get_frame_embeddings(args.image_folder)
save_embeddings(frame_embeds, save_path=args.save_path)
# Read txt with unique object prompts and canonical class labels.
object_names, object_class_names = read_object_prompts(args.txt_path)
# Segment with text prompt
preds = {}
object_progress = tqdm(
    object_names,
    desc="SAM3 objects",
    unit="object",
    dynamic_ncols=True,
    disable=False,
)
for name in object_progress:
    object_progress.set_postfix_str(f"current={name}")
    text_prompt = {"text": name}
    outputs_per_frame = segment_on_vigeo(predictor, args.image_folder, text_prompt, prompt_idx=0)
    preds[name]=outputs_per_frame
# Merge predictions across classes, assigning unique global IDs.
all_preds, max_id = merge_predicts(object_names, preds, object_class_names)
# Non maximum supression
preds_filtered = perform_nms(all_preds, h, w)
# Split into 
tracks = split_by_tracks(preds_filtered,max_id)
for i in range(len(tracks)):
    np.savez(f'{args.save_path}/tracks/{i}.npz', tracks[i], pickle=True)
