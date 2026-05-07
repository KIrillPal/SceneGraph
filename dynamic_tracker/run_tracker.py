import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm.auto import tqdm


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dynamic 3D tracker from images, SAM3 outputs, and DA3 outputs."
    )
    parser.add_argument("image_folder", type=Path, help="Path to folder with images")
    parser.add_argument(
        "sam3_outputs",
        type=Path,
        help="Path to SAM3 output folder containing tracks/ and embeds/",
    )
    parser.add_argument(
        "da3_outputs",
        type=Path,
        help="Path to DA3 output folder containing results_output/ and camera_poses.txt",
    )
    parser.add_argument(
        "save_path",
        nargs="?",
        default=Path("tracker_outputs"),
        type=Path,
        help="Output directory for tracker frame .npz files (default: tracker_outputs)",
    )
    parser.add_argument(
        "--dynamic-classes",
        required=True,
        type=Path,
        help="Text file with '<class>, <static|dynamic>' lines",
    )
    parser.add_argument(
        "--embedding-type",
        default="dino",
        choices=("dino", "sam3"),
        help="Appearance embedding source to use for association (default: dino)",
    )
    return parser.parse_args()


def _sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 10**12, path.name


def parse_class_statuses(class_status_file: Path) -> dict[str, str]:
    class_statuses = {}
    with class_status_file.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2 or not parts[0]:
                raise ValueError(
                    f"Bad class status line {line_no}: expected '<class>, <static|dynamic>'"
                )

            class_name, status = parts[0].lower(), parts[1].lower()
            if status not in {"static", "dynamic"}:
                raise ValueError(
                    f"Bad status on line {line_no}: expected 'static' or 'dynamic'"
                )
            class_statuses[class_name] = status

    if not class_statuses:
        raise ValueError("No classes found in dynamic classes file")

    return class_statuses


def parse_dynamic_classes(class_status_file: Path) -> list[str]:
    class_statuses = parse_class_statuses(class_status_file)
    return [name for name, status in class_statuses.items() if status == "dynamic"]


def read_images(image_dir: Path) -> list[np.ndarray]:
    frame_paths = sorted(
        [path for path in image_dir.iterdir() if path.is_file()], key=_sort_key
    )
    images = []
    logger.info("Getting images from %s", image_dir)
    for path in tqdm(frame_paths, desc="Loading images", unit="image", dynamic_ncols=True):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Failed to read image: {path}")
        images.append(image)
    return images


def read_sam3_tracks(tracks_dir: Path) -> dict[int, dict[str, Any]]:
    from mask_utils import merge_masks

    track_files = sorted(tracks_dir.glob("*.npz"), key=_sort_key)
    raw_tracks = {}

    logger.info("Getting tracks from %s", tracks_dir)
    for idx, track_file in enumerate(
        tqdm(track_files, desc="Loading tracks", unit="track", dynamic_ncols=True)
    ):
        track = np.load(track_file, allow_pickle=True)["arr_0"].tolist()
        if track["cls"] != 0:
            raw_tracks[idx] = track

    if not raw_tracks:
        return {}

    return merge_masks(raw_tracks)


def read_sam3_embeddings(embeds_dir: Path) -> list[np.ndarray]:
    emb_files = sorted(embeds_dir.glob("*.npz"), key=_sort_key)
    frame_embeds = []

    logger.info("Getting SAM3 frame embeddings from %s", embeds_dir)
    for emb_file in tqdm(
        emb_files, desc="Loading embeddings", unit="file", dynamic_ncols=True
    ):
        frame_embeds.append(np.load(emb_file, allow_pickle=True)["arr_0"])

    return frame_embeds


def build_video_tensor(images: list[np.ndarray]) -> torch.Tensor:
    video = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        video.append(transforms.ToTensor()(image_rgb).unsqueeze(0))
    return torch.cat(video)


def read_extrinsics(extrinsics_file: Path, num_frames: int) -> list[np.ndarray]:
    extrinsics = []
    with extrinsics_file.open("r", encoding="utf-8") as f:
        for line in f:
            pose = line.strip()
            if pose:
                extrinsics.append(np.array(pose.split(), dtype=float).reshape((4, 4)))

    if len(extrinsics) < num_frames:
        raise ValueError(
            f"Expected at least {num_frames} poses in {extrinsics_file}, got {len(extrinsics)}"
        )

    return extrinsics[:num_frames]


def get_dense_da3_frame_data(
    depth_dir: Path,
    extrinsics: list[np.ndarray],
    num_frames: int,
    conf_thresh_mul: float = 0.5,
) -> list[dict[str, np.ndarray]]:
    from point_utils import depth_to_point_cloud_vectorized

    frame_data = []

    logger.info("Getting dense DA3 frame data from %s", depth_dir)
    for frame_idx in tqdm(
        range(num_frames), desc="Loading DA3 frames", unit="frame", dynamic_ncols=True
    ):
        frame_file = depth_dir / f"frame_{frame_idx}.npz"
        data = np.load(frame_file)

        depth = np.asarray(data["depth"], dtype=np.float32)
        conf = np.asarray(data["conf"], dtype=np.float32)
        intrinsic = np.asarray(data["intrinsics"], dtype=np.float32)
        extrinsic = np.asarray(extrinsics[frame_idx], dtype=np.float32)

        w2c = np.linalg.inv(extrinsic)[:3, :]
        dense_points = depth_to_point_cloud_vectorized(
            depth[np.newaxis, :, :],
            intrinsic[np.newaxis, :, :],
            w2c[np.newaxis, :, :],
        )[0][0].astype(np.float32)

        valid_mask = conf > (float(conf.mean()) * conf_thresh_mul)
        dense_points[~valid_mask] = np.nan

        frame_data.append(
            {
                "point_cloud": dense_points,
                "intrinsic": intrinsic,
                "extrinsic": extrinsic,
            }
        )

    return frame_data


def _get_track_label(track: Any) -> str:
    classes = np.asarray(list(track.cls.values()), dtype=object)
    values, counts = np.unique(classes, return_counts=True)
    return str(values[np.argmax(counts)])


def save_tracker_outputs(
    save_path: Path,
    all_tracks: list[Any],
    images: list[np.ndarray],
    da3_frame_data: list[dict[str, np.ndarray]],
    object_states: dict[str, str],
) -> Path:
    logger.info("Saving tracker outputs to %s", save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    track_labels = {int(track.id): _get_track_label(track) for track in all_tracks}

    for frame_idx, image in enumerate(
        tqdm(images, desc="Saving frame exports", unit="frame", dynamic_ncols=True)
    ):
        masks_by_class: defaultdict[str, dict[int, np.ndarray]] = defaultdict(dict)
        embeddings_by_class: defaultdict[str, dict[int, np.ndarray]] = defaultdict(dict)

        for track in all_tracks:
            track_id = int(track.id)
            mask = track.masks.get(frame_idx)
            if mask is None:
                continue

            class_name = track_labels[track_id]
            masks_by_class[class_name][track_id] = np.asarray(mask, dtype=bool)

            embedding = track.embeddings.get(frame_idx)
            if embedding is not None:
                embeddings_by_class[class_name][track_id] = np.asarray(
                    embedding, dtype=np.float32
                ).reshape(-1)

        frame_file = save_path / f"frame_{frame_idx:06d}.npz"
        np.savez_compressed(
            frame_file,
            frame_id=np.int32(frame_idx),
            image=np.asarray(image),
            masks=dict(masks_by_class),
            embeddings=dict(embeddings_by_class),
            object_states=object_states,
            point_cloud=np.asarray(
                da3_frame_data[frame_idx]["point_cloud"], dtype=np.float32
            ),
            intrinsic=np.asarray(
                da3_frame_data[frame_idx]["intrinsic"], dtype=np.float32
            ),
            extrinsic=np.asarray(
                da3_frame_data[frame_idx]["extrinsic"], dtype=np.float32
            ),
        )

    return save_path


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    tracks_dir = args.sam3_outputs / "tracks"
    embeds_dir = args.sam3_outputs / "embeds"
    depth_dir = args.da3_outputs / "results_output"
    extrinsics_file = args.da3_outputs / "camera_poses.txt"

    required_paths = [
        args.image_folder,
        args.sam3_outputs,
        tracks_dir,
        args.da3_outputs,
        depth_dir,
        extrinsics_file,
        args.dynamic_classes,
    ]
    if args.embedding_type == "sam3":
        required_paths.append(embeds_dir)

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    return tracks_dir, embeds_dir, depth_dir


def main() -> None:
    args = parse_args()
    tracks_dir, embeds_dir, depth_dir = validate_inputs(args)
    extrinsics_file = args.da3_outputs / "camera_poses.txt"

    from data import (
        get_keypoints,
        get_obj_point_cloud,
        get_object_embedding,
        get_text_embeddings,
    )
    from point_utils import get_da3_pointclouds
    from tracker import Simple3DTracker
    from track_vis_utils import get_current_tracks

    object_states = parse_class_statuses(args.dynamic_classes)
    dynamic_classes = [
        class_name for class_name, status in object_states.items() if status == "dynamic"
    ]
    logger.info("Dynamic classes: %s", ", ".join(dynamic_classes))

    images = read_images(args.image_folder)
    tracks = read_sam3_tracks(tracks_dir)
    frame_embeds = (
        read_sam3_embeddings(embeds_dir) if args.embedding_type == "sam3" else None
    )

    if frame_embeds is not None and len(frame_embeds) < len(images):
        raise ValueError(
            f"Expected at least {len(images)} SAM3 embedding files, got {len(frame_embeds)}"
        )

    extrinsics = read_extrinsics(extrinsics_file, len(images))

    logger.info("Getting filtered DA3 point clouds")
    (
        points_per_frame,
        points_per_frame_masks,
        pixels_per_frame,
        h,
        w,
        tracker_extrinsics,
        intrinsics,
    ) = get_da3_pointclouds(str(depth_dir) + os.sep, str(extrinsics_file), len(images))
    da3_frame_data = get_dense_da3_frame_data(depth_dir, extrinsics, len(images))

    logger.info("Getting text embeddings")
    text_embs = get_text_embeddings(tracks)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Building video tensor on %s", device)
    video_tensor = build_video_tensor(images)

    dino_extractor = None
    if args.embedding_type == "dino":
        from dino_extractor import get_dino_extractor

        logger.info("Loading DINO extractor")
        dino_extractor = get_dino_extractor(device=device)

    tracker = Simple3DTracker(
        video_tensor,
        device=device,
        dynamic_classes_list=dynamic_classes,
        intrinsics_list=intrinsics,
        extrinsics_list=tracker_extrinsics,
    )

    frame_progress = tqdm(
        range(len(images)), desc="Tracking frames", unit="frame", dynamic_ncols=True
    )
    for frame_idx in frame_progress:
        frame_masks = []
        frame_track_ids = []
        for track_id, track in tracks.items():
            if frame_idx in track["masks"]:
                frame_masks.append(track["masks"][frame_idx])
                frame_track_ids.append(track_id)

        if args.embedding_type == "dino":
            visual_embeddings = dino_extractor.extract_from_frame(
                image=images[frame_idx], masks=frame_masks, batch_size=32
            )
        else:
            visual_embeddings = None

        detections = []
        for det_idx, track_id in enumerate(frame_track_ids):
            track = tracks[track_id]
            det = {
                "sam_id": track_id,
                "cls": track["cls"],
                "mask": track["masks"][frame_idx],
            }

            pcd = get_obj_point_cloud(
                images[frame_idx],
                points_per_frame[frame_idx],
                points_per_frame_masks[frame_idx],
                det["mask"],
                h,
                w,
            )
            det["points"] = np.asarray(pcd.points)
            if len(det["points"]) < 5:
                continue

            det["keypoints"] = get_keypoints(images[frame_idx], det["mask"])
            det["text_embedding"] = text_embs[det["cls"]]

            if args.embedding_type == "dino":
                det["embedding"] = visual_embeddings[det_idx]
            else:
                det["embedding"] = get_object_embedding(
                    frame_embeds[frame_idx], det["mask"]
                ).reshape(1, -1)

            detections.append(det)

        tracker.update(
            detections,
            frame_idx,
            pixels_per_frame[frame_idx],
            points_per_frame[frame_idx],
        )

        frame_progress.set_postfix(
            dets=len(detections), active=len(tracker.tracks), lost=len(tracker.lost_tracks)
        )

    logger.info("Collecting final tracks")
    all_tracks = get_current_tracks(tracker)
    save_tracker_outputs(args.save_path, all_tracks, images, da3_frame_data, object_states)
    logger.info("Done")


if __name__ == "__main__":
    main()
