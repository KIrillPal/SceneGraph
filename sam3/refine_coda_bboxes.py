import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


Box = Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine projected CODA 2D boxes with SAM3 box prompts and save labels "
            "plus visualizations under <sequence>/2bbox_refined."
        )
    )
    parser.add_argument(
        "sequence_root",
        type=Path,
        help="Path to a CODA sequence folder, e.g. /workspace/DAAAM/data_coda/0",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("sam3.pt"),
        help="Path to SAM3 checkpoint inside the current environment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <sequence_root>/2bbox_refined.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help="Camera names to process. Defaults to all cameras found in labels.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1008,
        help="SAM3 square input resolution. Default matches SAM3 training resolution.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum SAM3 mask confidence. Low default avoids dropping prompted objects.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on processed label files for quick tests.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip label files whose refined output already exists.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run on CPU. This is much slower and mainly for debugging.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable CUDA bfloat16 autocast.",
    )
    return parser.parse_args()


def xyxy_to_normalized_cxcywh(box: Box, image_width: int, image_height: int) -> List[float]:
    x0, y0, x1, y1 = box
    x0 = min(max(x0, 0.0), float(image_width))
    x1 = min(max(x1, 0.0), float(image_width))
    y0 = min(max(y0, 0.0), float(image_height))
    y1 = min(max(y1, 0.0), float(image_height))
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    return [
        ((x0 + x1) * 0.5) / image_width,
        ((y0 + y1) * 0.5) / image_height,
        width / image_width,
        height / image_height,
    ]


def xyxy_to_xywh(box: Box) -> List[float]:
    x0, y0, x1, y1 = box
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def round_box(box: Iterable[float], ndigits: int = 3) -> List[float]:
    return [round(float(value), ndigits) for value in box]


def mask_to_xyxy(mask: np.ndarray) -> Optional[Box]:
    mask = np.asarray(mask).astype(bool)
    if mask.ndim == 3:
        mask = np.squeeze(mask)
    if mask.ndim != 2 or not mask.any():
        return None

    ys, xs = np.where(mask)
    # x1/y1 are exclusive so xywh width/height match the covered mask extent.
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def box_iou(box_a: Box, box_b: Box) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def amp_context(use_amp: bool):
    if not use_amp:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def get_best_refinement(
    state: Dict[str, Any], source_xyxy: Box
) -> Tuple[Optional[Box], Optional[float]]:
    scores = state.get("scores")
    masks = state.get("masks")
    boxes = state.get("boxes")
    if scores is None or len(scores) == 0:
        return None, None

    candidates = []
    for idx in range(len(scores)):
        score = float(scores[idx].detach().cpu().item())
        candidate_box = None
        if masks is not None and len(masks) > idx:
            candidate_box = mask_to_xyxy(masks[idx].detach().cpu().numpy())
        if candidate_box is None and boxes is not None and len(boxes) > idx:
            candidate_box = tuple(float(v) for v in boxes[idx].detach().cpu().tolist())
        if candidate_box is not None:
            candidates.append((box_iou(source_xyxy, candidate_box), score, candidate_box))

    if not candidates:
        best_idx = int(torch.argmax(scores).item())
        return None, float(scores[best_idx].detach().cpu().item())
    _, best_score, best_box = max(candidates, key=lambda item: (item[0], item[1]))
    return best_box, best_score


def find_cameras(labels_root: Path) -> List[str]:
    return sorted(path.name for path in labels_root.iterdir() if path.is_dir())


def label_files_for_camera(labels_root: Path, camera: str) -> List[Path]:
    camera_root = labels_root / camera
    return sorted(camera_root.glob("*/*.json"), key=lambda path: path.name)


def image_path_for_label(sequence_root: Path, label_data: Dict[str, Any]) -> Path:
    sequence = str(label_data["sequence"])
    camera = str(label_data["camera"])
    frame = int(label_data["frame"])
    return (
        sequence_root
        / "2d_rect"
        / camera
        / sequence
        / f"2d_rect_{camera}_{sequence}_{frame}.png"
    )


def output_label_path(output_root: Path, label_path: Path, labels_root: Path) -> Path:
    return output_root / "labels" / label_path.relative_to(labels_root)


def output_vis_path(output_root: Path, label_data: Dict[str, Any]) -> Path:
    sequence = str(label_data["sequence"])
    camera = str(label_data["camera"])
    frame = int(label_data["frame"])
    return (
        output_root
        / "visualizations_refined"
        / camera
        / sequence
        / f"refined_bboxes_{camera}_{sequence}_{frame}.png"
    )


def draw_box(
    image: np.ndarray,
    box: Iterable[float],
    color: Tuple[int, int, int],
    label: str,
    thickness: int,
) -> None:
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)
    if label:
        cv2.putText(
            image,
            label,
            (x0, max(12, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def refine_label_file(
    processor: Sam3Processor,
    label_path: Path,
    labels_root: Path,
    sequence_root: Path,
    output_root: Path,
    use_amp: bool,
) -> None:
    with label_path.open("r", encoding="utf-8") as handle:
        label_data = json.load(handle)

    image_path = image_path_for_label(sequence_root, label_data)
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image for {label_path}: {image_path}")

    image_pil = Image.open(image_path).convert("RGB")
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image_width = int(label_data.get("image_width", image_pil.width))
    image_height = int(label_data.get("image_height", image_pil.height))

    with amp_context(use_amp):
        state = processor.set_image(image_pil)

    refined_boxes = []
    for box_record in label_data.get("boxes", []):
        source_xyxy = tuple(float(v) for v in box_record["bbox_xyxy"])
        prompt_box = xyxy_to_normalized_cxcywh(source_xyxy, image_width, image_height)

        processor.reset_all_prompts(state)
        with amp_context(use_amp):
            refined_state = processor.add_geometric_prompt(prompt_box, True, state)
        refined_xyxy, score = get_best_refinement(refined_state, source_xyxy)

        status = "refined"
        if refined_xyxy is None:
            refined_xyxy = source_xyxy
            status = "fallback_no_sam3_mask"

        refined_record = dict(box_record)
        refined_record["source_bbox_xyxy"] = round_box(source_xyxy)
        refined_record["source_bbox_xywh"] = round_box(
            box_record.get("bbox_xywh", xyxy_to_xywh(source_xyxy))
        )
        refined_record["bbox_xyxy"] = round_box(refined_xyxy)
        refined_record["bbox_xywh"] = round_box(xyxy_to_xywh(refined_xyxy))
        refined_record["sam3_score"] = None if score is None else round(score, 6)
        refined_record["refinement_status"] = status
        refined_boxes.append(refined_record)

        if image_bgr is not None:
            draw_box(image_bgr, source_xyxy, (0, 0, 255), "src", 1)
            draw_box(image_bgr, refined_xyxy, (0, 255, 0), "sam3", 2)

    output_data = dict(label_data)
    output_data["format"] = "SAM3-refined 2D bbox labels; bbox_xyxy/bbox_xywh are refined"
    output_data["source_label"] = str(label_path)
    output_data["boxes"] = refined_boxes

    out_label = output_label_path(output_root, label_path, labels_root)
    out_label.parent.mkdir(parents=True, exist_ok=True)
    with out_label.open("w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=2)
        handle.write("\n")

    if image_bgr is not None:
        out_vis = output_vis_path(output_root, label_data)
        out_vis.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_vis), image_bgr)


def main() -> None:
    args = parse_args()
    sequence_root = args.sequence_root.resolve()
    labels_root = sequence_root / "projected_2d_bboxes" / "labels"
    if not labels_root.exists():
        raise FileNotFoundError(f"Projected bbox labels not found: {labels_root}")

    output_root = args.output_dir.resolve() if args.output_dir else sequence_root / "2bbox_refined"
    cameras = args.cameras if args.cameras else find_cameras(labels_root)
    label_paths: List[Path] = []
    for camera in cameras:
        label_paths.extend(label_files_for_camera(labels_root, camera))
    if args.max_frames is not None:
        label_paths = label_paths[: args.max_frames]

    if args.skip_existing:
        label_paths = [
            path
            for path in label_paths
            if not output_label_path(output_root, path, labels_root).exists()
        ]

    device = "cpu" if args.cpu else "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        device=device,
        eval_mode=True,
    )
    processor = Sam3Processor(
        model,
        resolution=args.resolution,
        device=device,
        confidence_threshold=args.confidence_threshold,
    )

    use_amp = device == "cuda" and not args.disable_amp
    for label_path in tqdm(label_paths, desc="refining bbox labels"):
        refine_label_file(
            processor=processor,
            label_path=label_path,
            labels_root=labels_root,
            sequence_root=sequence_root,
            output_root=output_root,
            use_amp=use_amp,
        )

    print(f"Wrote refined labels and visualizations to {output_root}")


if __name__ == "__main__":
    main()
