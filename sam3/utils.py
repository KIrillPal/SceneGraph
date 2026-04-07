import os
from typing import List, Dict, Tuple

import torch
from torchvision.ops import nms
from PIL import Image
import numpy as np
import cv2

from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

Sam3Output = Dict[str, np.ndarray]

def cv2pil(cv_image: np.ndarray) -> Image.Image:
    """
    Converts cv2 BGR image to PIL RGB image
    """
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def xywh2xyxy(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
    """Convert normalized [x,y,w,h] boxes to pixel [x_min,y_min,x_max,y_max]."""
    boxes_xyxy = boxes.copy()
    boxes_xyxy[:, 2] += boxes_xyxy[:, 0]
    boxes_xyxy[:, 3] += boxes_xyxy[:, 1]
    boxes_xyxy[:, 0] *= w
    boxes_xyxy[:, 2] *= w
    boxes_xyxy[:, 1] *= h
    boxes_xyxy[:, 3] *= h
    return boxes_xyxy

def get_frame_embeddings(image_path: str) -> List[torch.Tensor]:
    """
    Extract SAM3 visual embeddings for all frames in a folder.
    """
    model = build_sam3_image_model()
    processor = Sam3Processor(model, resolution=1008)
    frame_embeds = []
    frame_names = sorted(os.listdir(image_path))
    images = []
    for name in frame_names:
        images.append(cv2.imread("%s/%s" % (image_path, name)))
    for i in range(len(images)):
        image = cv2pil(images[i])
        inference_state = processor.set_image(image)
        frame_embed = inference_state["backbone_out"]["vision_features"]
        frame_embeds.append(frame_embed[0].reshape((256, -1)).T)
    return frame_embeds

def propagate_in_video(predictor: Sam3VideoPredictorMultiGPU, 
                       session_id: str, frame_idx: str, 
                       prop_direction: str = "both") -> Dict[int, Sam3Output]:
    """
    Propagates from frame 'frame_idx' to all video
    """
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            propagation_direction=prop_direction,
            start_frame_idx=frame_idx,
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame

def segment_on_vigeo(video_predictor: Sam3VideoPredictorMultiGPU, 
                     image_path: str, 
                     prompt: Dict[str, str], 
                     prompt_idx: int = 0) -> Dict[int, Sam3Output]:
    """
    Run Sam3 inference on image sequence or video
    Params:
        video_predictor - Sam3 video predictor
        image_path - path to a JPEG folder or an MP4 video file
        prompt - dict with text or bbox prompt
        prompt_idx - frame for which prompt will be used
    Returns: predicted output

    """
    video_path = image_path  # a JPEG folder or an MP4 video file
    # Start a session
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_path,
        )
    )
    session_id = response["session_id"]
    # print("id",session_id)
    # add text prompt
    if "text" in prompt.keys():
        response = video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=prompt_idx,  # Arbitrary frame index
                text=prompt["text"],
            )
        )
    # add bbox prompt
    if "bounding_boxes" in prompt.keys():
        response = video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=prompt_idx,  # Arbitrary frame index
                bounding_boxes=prompt["bounding_boxes"],
                bounding_box_labels=prompt["box_labels"],
            )
        )
    # propagate on video
    outputs_per_frame = propagate_in_video(video_predictor, session_id, prompt_idx)
    # close session
    response = video_predictor.handle_request(
        request=dict(
            type="close_session",
            session_id=session_id,
        )
    )
    return outputs_per_frame

def save_embeddings(frame_embeds: List[torch.Tensor], save_path: str) -> None:
    """
    Saves embedding from get_frame_embeddings function into .npz file for each frame
    """
    os.makedirs(f'{save_path}/embeds', exist_ok=True)
    for i in range(len(frame_embeds)):
        np.savez(f'{save_path}/embeds/{i}.npz', frame_embeds[i].cpu().detach().float().numpy(), i)

def merge_predicts(object_names: List[str], 
                   preds: Dict[str, Dict[int, Sam3Output]]) -> Tuple[List[Dict[str, List]], int]:
    """
    Merge predictions across classes, assigning unique global IDs.

    Returns: (objects_per_frame, total_objects_count)
    """
    n_frames = len(preds[object_names[0]])
    objects_per_frame = [
        {"ids": [], "masks": [], "boxes": [], "confs": [], "classes": []}
        for i in range(n_frames)
    ]
    id_shift = 0
    for name in object_names:
        # Find max local ID for this class
        max_id = 0
        obj_preds = preds[name]
        for i in range(len(obj_preds)):
            pred = obj_preds[i]
            ids = pred["out_obj_ids"]
            if len(ids) > 0:
                local_max = max(ids) + 1
                if local_max > max_id:
                    max_id = local_max
        # Add objects with global IDs
        for id_ in range(max_id):
            for i in range(len(obj_preds)):
                pred = obj_preds[i]
                ids = pred["out_obj_ids"].tolist()
                if id_ in ids:
                    mask_id = ids.index(id_)
                    objects_per_frame[i]["ids"].append(id_ + id_shift)
                    objects_per_frame[i]["masks"].append(
                        pred["out_binary_masks"][mask_id]
                    )
                    objects_per_frame[i]["boxes"].append(
                        pred["out_boxes_xywh"][mask_id]
                    )
                    objects_per_frame[i]["confs"].append(pred["out_probs"][mask_id])
                    objects_per_frame[i]["classes"].append(name)
        id_shift += max_id
        # print(name, max_id, id_shift)
    return objects_per_frame, id_shift

def perform_nms(all_preds: List[Dict[str, List]], h: int, w: int, 
                inner_iou: float = 0.7, outer_iou: float = 0.7) -> List[Dict[str, np.array]]:
    """
    Two-stage NMS: (1) class-aware, (2) class-agnostic.

    Returns filtered predictions with duplicates removed.
    """
    preds_filtered = []
    for pred in all_preds:
        ids = np.array(pred["ids"])
        boxes = xywh2xyxy(np.array(pred["boxes"]), h, w)
        confs = np.array(pred["confs"])
        classes = np.array(pred["classes"])
        masks = np.array(pred["masks"])
        filtered_ids = []
        filtered_boxes = []
        filtered_confs = []
        filtered_classes = []
        filtered_masks = []
        # Stage 1: NMS per class
        for cls in np.unique(classes):
            keep = nms(
                torch.tensor(boxes[classes == cls]),
                torch.tensor(confs[classes == cls]),
                iou_threshold=inner_iou,
            ).numpy()
            filtered_ids.extend(ids[classes == cls][keep])
            filtered_boxes.extend(boxes[classes == cls][keep])
            filtered_masks.extend(masks[classes == cls][keep])
            filtered_confs.extend(confs[classes == cls][keep])
            filtered_classes.extend(classes[classes == cls][keep])
        # Stage 1: NMS per class
        keep2 = nms(
            torch.tensor(filtered_boxes), torch.tensor(filtered_confs), outer_iou
        ).numpy()
        keep_ids = np.array(filtered_ids)[keep2]
        keep_boxes = np.array(filtered_boxes)[keep2]
        keep_masks = np.array(filtered_masks)[keep2]
        keep_classes = np.array(filtered_classes)[keep2]
        keep_confs = np.array(filtered_confs)[keep2]
        preds_filtered.append(
            {
                "ids": keep_ids,
                "boxes": keep_boxes,
                "masks": keep_masks,
                "classes": keep_classes,
                "confs": keep_confs,
            }
        )
    return preds_filtered


def split_by_tracks(preds: List[Dict[str, np.array]], max_id: int):
    """
    Group predictions by track ID instead of by frame.

    Returns: List of tracks, each with {cls, masks[frame_idx], confs[frame_idx]}
    """
    tracks = [{"cls": 0, "masks": {}, "confs": {}} for i in range(max_id)]
    for i, pred in enumerate(preds):
        ids = pred['ids']
        for j, id_ in enumerate(ids):
            tracks[id_]["cls"] = pred["classes"][j]
            tracks[id_]["masks"][i] = pred["masks"][j]
            tracks[id_]["confs"][i] = pred["confs"][j]
    return tracks