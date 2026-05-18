#!/usr/bin/env python3
"""Sequential end-to-end pipeline runner for an input video or image folder."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
ACTIVE_LOG_FILE: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frame extraction, keyframe selection, Qwen, DA3, SAM3, and tracker sequentially.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_path", type=Path, nargs="?", help="Input .mp4 video or image folder")
    parser.add_argument("--video", type=Path, default=None, help="Input .mp4 video")
    parser.add_argument("--image-folder", type=Path, default=None, help="Input folder with extracted images")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory mounted as data/ in containers. Defaults to the output folder parent.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=argparse.SUPPRESS,
        help="Frame extraction FPS for video input. Defaults to 10.0 for video input; invalid with image folders.",
    )
    parser.add_argument("--num-keyframes", type=int, default=30)
    parser.add_argument(
        "--gpus",
        default="0",
        help="GPU policy: 'i' uses one GPU for all steps; 'i,j' uses i for pipeline and j for Qwen",
    )
    parser.add_argument("--embedding-type", choices=("dinov2", "dinov3", "sam"), default="dinov3")
    parser.add_argument(
        "--sam3-score-threshold",
        type=float,
        default=0.2,
        help="Minimum SAM3 detection confidence before tracking",
    )
    parser.add_argument(
        "--sam3-new-det-threshold",
        type=float,
        default=0.4,
        help="Minimum confidence for adding a SAM3 detection as a new object track",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing stage outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Pipeline log directory. Defaults to <output-folder>/logs",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help=(
            "Hugging Face cache directory on the server. Absolute paths are used as-is; "
            "relative paths are resolved from the repo root. Defaults to $HF_CACHE_DIR or .cache/huggingface."
        ),
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token. Defaults to $HF_TOKEN or $HUGGING_FACE_HUB_TOKEN.",
    )

    parser.add_argument("--skip-frame-extraction", action="store_true")
    parser.add_argument("--skip-keyframes", action="store_true")
    parser.add_argument("--skip-qwen", action="store_true")
    parser.add_argument("--skip-da3", action="store_true")
    parser.add_argument("--skip-sam3", action="store_true")
    parser.add_argument("--skip-tracker", action="store_true")
    parser.add_argument(
        "--objects-file",
        type=Path,
        default=None,
        help="Use an existing '<description>, <class>, <static|dynamic>' txt instead of calling Qwen",
    )

    parser.add_argument("--pipeline-image", default="dynamic-tracker")
    parser.add_argument("--da3-image", default="depth_anything_3_streaming:latest")
    parser.add_argument("--sam3-image", default="sam3:latest")
    parser.add_argument("--qwen-image", default="qwen-vllm")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--qwen-port", type=int, default=8000)
    parser.add_argument("--qwen-timeout", type=int, default=900, help="Seconds to wait for Qwen server")
    parser.add_argument(
        "--qwen-container-name",
        default=None,
        help="Defaults to qwen-pipeline-<output-folder-name>",
    )
    parser.add_argument(
        "--start-qwen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start/stop a Qwen Docker server for object extraction",
    )
    parser.add_argument(
        "--keep-qwen-running",
        action="store_true",
        help="Do not stop a Qwen server started by this script",
    )
    args = parser.parse_args()
    fps_was_set = hasattr(args, "fps")
    explicit_inputs = [value for value in (args.input_path, args.video, args.image_folder) if value is not None]
    if len(explicit_inputs) == 0:
        parser.error("provide an input video/image folder, --video, or --image-folder")
    if len(explicit_inputs) > 1:
        parser.error("provide only one input source")

    if args.image_folder is not None:
        args.input_mode = "images"
        args.input_source = args.image_folder
        args.video = None
    elif args.video is not None:
        args.input_mode = "video"
        args.input_source = args.video
        args.image_folder = None
    elif args.input_path.is_dir():
        args.input_mode = "images"
        args.input_source = args.input_path
        args.image_folder = args.input_path
        args.video = None
    else:
        args.input_mode = "video"
        args.input_source = args.input_path
        args.video = args.input_path
        args.image_folder = None

    if args.input_mode == "images" and fps_was_set:
        parser.error("--fps is only valid with video input; omit it when using --image-folder or an image folder input")
    if args.input_mode == "video" and not fps_was_set:
        args.fps = 10.0
    if args.input_mode == "images":
        args.fps = None
    return args


def parse_gpu_policy(value: str) -> tuple[str, str]:
    devices = [part.strip() for part in value.split(",") if part.strip()]
    if len(devices) == 1:
        return devices[0], devices[0]
    if len(devices) == 2:
        return devices[0], devices[1]
    raise ValueError("--gpus must be either 'i' or 'i,j'")


def default_scene_dir(args: argparse.Namespace) -> Path:
    source = args.input_source.resolve()
    if args.input_mode == "video":
        return source.parent
    return source.parent


def data_path(args: argparse.Namespace, path: Path) -> str:
    path = path.resolve()
    try:
        relative = path.relative_to(args.data_root)
    except ValueError as exc:
        raise ValueError(f"Path must be inside data root for container mapping: {path}") from exc
    return str(Path("data") / relative)


def list_images(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])


def list_npz(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(path.glob("*.npz"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_log(text: str) -> None:
    if ACTIVE_LOG_FILE is None:
        return
    ACTIVE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with ACTIVE_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(text)


def run_cmd(cmd: list[str], *, dry_run: bool = False, check: bool = True) -> subprocess.CompletedProcess[str] | None:
    command_line = "+ " + " ".join(cmd)
    print(command_line, flush=True)
    append_log(command_line + "\n")
    if dry_run:
        return None

    process = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with ACTIVE_LOG_FILE.open("a", encoding="utf-8") if ACTIVE_LOG_FILE else open(os.devnull, "w") as log_f:
        for line in process.stdout:
            print(line, end="")
            log_f.write(line)
            log_f.flush()

    returncode = process.wait()
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return subprocess.CompletedProcess(cmd, returncode)


def write_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def run_stage(summary: dict, log_dir: Path, name: str, func, *func_args) -> None:
    global ACTIVE_LOG_FILE

    log_file = log_dir / f"{len(summary['stages']) + 1:02d}_{name}.log"
    ACTIVE_LOG_FILE = log_file
    started = time.time()
    stage_summary = {
        "status": "running",
        "started_at": now_iso(),
        "log_file": str(log_file),
    }
    summary["stages"][name] = stage_summary
    write_summary(summary, log_dir / "pipeline_summary.json")

    print(f"\n=== {name} ===")
    append_log(f"=== {name} started at {stage_summary['started_at']} ===\n")
    try:
        func(*func_args)
    except Exception as exc:
        stage_summary.update(
            {
                "status": "failed",
                "finished_at": now_iso(),
                "seconds": round(time.time() - started, 3),
                "error": repr(exc),
            }
        )
        summary["status"] = "failed"
        summary["failed_stage"] = name
        summary["finished_at"] = now_iso()
        write_summary(summary, log_dir / "pipeline_summary.json")
        append_log(f"=== {name} failed: {exc!r} ===\n")
        ACTIVE_LOG_FILE = None
        raise

    stage_summary.update(
        {
            "status": "ok",
            "finished_at": now_iso(),
            "seconds": round(time.time() - started, 3),
        }
    )
    write_summary(summary, log_dir / "pipeline_summary.json")
    append_log(f"=== {name} finished at {stage_summary['finished_at']} ===\n")
    ACTIVE_LOG_FILE = None


def remove_if_overwrite(path: Path, overwrite: bool) -> None:
    if overwrite and path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def ensure_file(path: Path, message: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{message}: {path}")


def ensure_dir_has_files(path: Path, message: str) -> None:
    if not path.is_dir() or not any(path.iterdir()):
        raise FileNotFoundError(f"{message}: {path}")


def user_args() -> list[str]:
    return ["--user", f"{os.getuid()}:{os.getgid()}"]


def resolve_hf_cache_dir(path: Path | None) -> Path:
    if path is None:
        env_path = os.environ.get("HF_CACHE_DIR")
        path = Path(env_path) if env_path else REPO_ROOT / ".cache" / "huggingface"
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def validate_writable_dir(path: Path, label: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise PermissionError(f"{label} is not writable by current user: {path}") from exc


def configure_hf(args: argparse.Namespace) -> None:
    args.hf_cache_dir = resolve_hf_cache_dir(args.hf_cache_dir)
    validate_writable_dir(args.hf_cache_dir, "HF cache directory")

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    args.hf_token_present = bool(token)
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token


def hf_env_args(args: argparse.Namespace) -> list[str]:
    env_args = [
        "-e",
        "HF_CACHE_DIR=/tmp/hf_cache",
        "-e",
        "HF_HOME=/tmp/hf_cache",
        "-e",
        "HF_HUB_CACHE=/tmp/hf_cache/hub",
        "-e",
        "HUGGINGFACE_HUB_CACHE=/tmp/hf_cache/hub",
        "-e",
        "TRANSFORMERS_CACHE=/tmp/hf_cache/transformers",
        "-v",
        f"{args.hf_cache_dir}:/tmp/hf_cache",
    ]
    if args.hf_token_present:
        env_args.extend(["-e", "HF_TOKEN", "-e", "HUGGING_FACE_HUB_TOKEN"])
    return env_args


def cotracker_cache_args() -> list[str]:
    cache = Path(os.environ.get("COTRACKER_CACHE_DIR", Path.home() / ".cache" / "cotracker")).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    return [
        "-e",
        "COTRACKER_CHECKPOINT=/tmp/cotracker/scaled_online.pth",
        "-v",
        f"{cache}:/tmp/cotracker",
    ]


def docker_pipeline_base(image: str, worker_gpu: str, data_root: Path, args: argparse.Namespace) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        f"device={worker_gpu}",
        "--ipc=host",
        *user_args(),
        "-e",
        "HOME=/tmp",
        *hf_env_args(args),
        "-v",
        f"{REPO_ROOT}:/workspace",
        "-v",
        f"{data_root.resolve()}:/workspace/data",
        "-w",
        "/workspace",
        image,
    ]


def copy_video(video: Path, scene_dir: Path, overwrite: bool, dry_run: bool) -> Path:
    video = video.resolve()
    ensure_file(video, "Input video not found")
    scene_dir.mkdir(parents=True, exist_ok=True)
    dest = scene_dir / "input.mp4"
    if dest.exists() and not overwrite:
        return dest
    if video != dest.resolve():
        if dry_run:
            print(f"+ copy {video} {dest}")
        else:
            shutil.copy2(video, dest)
    return dest


def copy_image_folder(image_folder: Path, scene_dir: Path, overwrite: bool, dry_run: bool) -> Path:
    image_folder = image_folder.resolve()
    images = list_images(image_folder)
    if not images:
        raise FileNotFoundError(f"Input image folder has no supported images: {image_folder}")

    scene_dir.mkdir(parents=True, exist_ok=True)
    dest = scene_dir / "images"
    if dest.exists() and image_folder == dest.resolve():
        return dest
    if list_images(dest) and not overwrite:
        print(f"Skipping image copy; found images in {dest}")
        return dest
    remove_if_overwrite(dest, overwrite)

    if dry_run:
        print(f"+ copy_images {image_folder} {dest}")
        return dest

    dest.mkdir(parents=True, exist_ok=True)
    for image in images:
        shutil.copy2(image, dest / image.name)
    ensure_dir_has_files(dest, "Image copy produced no images")
    return dest


def prepare_image_folder(args: argparse.Namespace, scene_dir: Path) -> None:
    if args.skip_frame_extraction:
        ensure_dir_has_files(scene_dir / "images", "Image preparation skipped but images are missing")
        return
    copy_image_folder(args.image_folder, scene_dir, args.overwrite, args.dry_run)


def run_frame_extraction(args: argparse.Namespace, worker_gpu: str, scene_dir: Path, video_path: Path) -> None:
    images_dir = scene_dir / "images"
    if args.skip_frame_extraction:
        ensure_dir_has_files(images_dir, "Frame extraction skipped but images are missing")
        return
    if list_images(images_dir) and not args.overwrite:
        print(f"Skipping frame extraction; found images in {images_dir}")
        return

    cmd = docker_pipeline_base(args.pipeline_image, worker_gpu, args.data_root, args) + [
        "python",
        "utils/video_to_frames.py",
        data_path(args, video_path),
        data_path(args, images_dir),
        "--fps",
        str(args.fps),
        "--start-index",
        "0",
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    run_cmd(cmd, dry_run=args.dry_run)
    if not args.dry_run:
        ensure_dir_has_files(images_dir, "Frame extraction produced no images")


def run_keyframes(args: argparse.Namespace, worker_gpu: str, scene_dir: Path) -> None:
    selected_dir = scene_dir / "selected_keyframes"
    if args.skip_keyframes:
        ensure_dir_has_files(selected_dir, "Keyframe selection skipped but selected keyframes are missing")
        return
    if list_images(selected_dir) and not args.overwrite:
        print(f"Skipping keyframe selection; found selected frames in {selected_dir}")
        return

    cmd = docker_pipeline_base(args.pipeline_image, worker_gpu, args.data_root, args) + [
        "python",
        "MaxInfo/pvsg_maxinfo_filter.py",
        "--input_dir",
        data_path(args, scene_dir / "images"),
        "--output_dir",
        data_path(args, selected_dir),
        "--num-frames",
        str(args.num_keyframes),
        "--fp16",
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    run_cmd(cmd, dry_run=args.dry_run)
    if not args.dry_run:
        ensure_dir_has_files(selected_dir, "Keyframe selection produced no frames")


def qwen_health_url(port: int) -> str:
    return f"http://localhost:{port}/v1/models"


def qwen_endpoint(port: int) -> str:
    return f"http://localhost:{port}/v1/chat/completions"


def get_container_status(container_name: str) -> str | None:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", container_name],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def append_qwen_logs(container_name: str) -> None:
    if ACTIVE_LOG_FILE is None:
        return
    result = subprocess.run(
        ["docker", "logs", container_name],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    append_log("\n=== qwen docker logs snapshot ===\n")
    append_log(result.stdout)


def wait_for_qwen(port: int, timeout: int, container_name: str | None = None) -> None:
    deadline = time.time() + timeout
    url = qwen_health_url(port)
    while time.time() < deadline:
        if container_name is not None:
            status = get_container_status(container_name)
            if status is None:
                raise RuntimeError(f"Qwen container disappeared before becoming ready: {container_name}")
            if status not in {"created", "running"}:
                append_qwen_logs(container_name)
                raise RuntimeError(
                    f"Qwen container exited before becoming ready: {container_name} status={status}"
                )

        try:
            with request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    return
        except (OSError, error.URLError):
            pass
        time.sleep(5)
    if container_name is not None:
        append_qwen_logs(container_name)
    raise TimeoutError(f"Qwen server did not become ready within {timeout}s: {url}")


def start_qwen(args: argparse.Namespace, qwen_gpu: str, container_name: str) -> subprocess.Popen | None:
    run_cmd(["docker", "rm", "-f", container_name], dry_run=args.dry_run, check=False)
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--gpus",
        f"device={qwen_gpu}",
        "--ipc=host",
        "-p",
        f"{args.qwen_port}:8000",
        "-e",
        "HOME=/tmp",
        "-e",
        "FLASHINFER_DISABLE_VERSION_CHECK=1",
        *hf_env_args(args),
        "-v",
        f"{REPO_ROOT}:/workspace",
        "-w",
        "/workspace",
        args.qwen_image,
        "--model",
        args.qwen_model,
        "--allowed-local-media-path",
        "/workspace",
    ]
    run_cmd(cmd, dry_run=args.dry_run)
    logs_process = None
    if not args.dry_run and ACTIVE_LOG_FILE is not None:
        logs_file = ACTIVE_LOG_FILE.open("a", encoding="utf-8")
        logs_file.write("\n=== qwen docker logs ===\n")
        logs_file.flush()
        logs_process = subprocess.Popen(
            ["docker", "logs", "-f", container_name],
            cwd=REPO_ROOT,
            stdout=logs_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        logs_file.close()
    if not args.dry_run:
        try:
            wait_for_qwen(args.qwen_port, args.qwen_timeout, container_name)
        except Exception:
            stop_logs_process(logs_process)
            raise
    return logs_process


def stop_qwen(container_name: str, dry_run: bool) -> None:
    run_cmd(["docker", "rm", "-f", container_name], dry_run=dry_run, check=False)


def stop_logs_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def run_qwen_objects(args: argparse.Namespace, qwen_gpu: str, scene_dir: Path) -> None:
    objects_path = scene_dir / "objects.txt"
    if args.objects_file is not None:
        ensure_file(args.objects_file, "Objects file not found")
        remove_if_overwrite(objects_path, args.overwrite)
        if not objects_path.exists() or args.overwrite:
            if not args.dry_run:
                shutil.copy2(args.objects_file, objects_path)
            else:
                print(f"+ copy {args.objects_file} {objects_path}")
        return

    if args.skip_qwen:
        ensure_file(objects_path, "Qwen skipped but objects file is missing")
        return
    if objects_path.is_file() and not args.overwrite:
        print(f"Skipping Qwen object extraction; found {objects_path}")
        return

    container_name = args.qwen_container_name or f"qwen-pipeline-{scene_dir.name}"
    started = False
    qwen_logs_process = None
    try:
        if args.start_qwen:
            started = True
            qwen_logs_process = start_qwen(args, qwen_gpu, container_name)
        elif not args.dry_run:
            wait_for_qwen(args.qwen_port, 30)

        cmd = [
            sys.executable,
            "qwen/extract_object_list.py",
            str(scene_dir / "selected_keyframes"),
            str(objects_path),
            "--endpoint",
            qwen_endpoint(args.qwen_port),
            "--model",
            args.qwen_model,
            "--image-url-mode",
            "data",
        ]
        run_cmd(cmd, dry_run=args.dry_run)
    finally:
        if started and not args.keep_qwen_running:
            stop_qwen(container_name, args.dry_run)
        if started:
            stop_logs_process(qwen_logs_process)

    if not args.dry_run:
        ensure_file(objects_path, "Qwen did not produce objects file")


def run_da3(args: argparse.Namespace, worker_gpu: str, scene_dir: Path) -> None:
    output_dir = scene_dir / "da3_outputs"
    if args.skip_da3:
        ensure_file(output_dir / "camera_poses.txt", "DA3 skipped but camera poses are missing")
        ensure_dir_has_files(output_dir / "results_output", "DA3 skipped but results_output is missing")
        return
    if (output_dir / "camera_poses.txt").is_file() and (output_dir / "results_output").is_dir() and not args.overwrite:
        print(f"Skipping DA3; found {output_dir}")
        return
    remove_if_overwrite(output_dir, args.overwrite)

    da3_dir = REPO_ROOT / "Depth-Anything-3" / "da3_streaming"
    cache_dir = da3_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        f"device={worker_gpu}",
        "--ipc=host",
        *user_args(),
        "-e",
        "HOME=/tmp",
        *hf_env_args(args),
        "-v",
        f"{da3_dir}:/workspace/da3_streaming",
        "-v",
        f"{cache_dir}:/workspace/.cache",
        "-v",
        f"{args.data_root.resolve()}:/workspace/da3_streaming/data",
        "-w",
        "/workspace/da3_streaming",
        args.da3_image,
        "python",
        "da3_streaming.py",
        "--image_dir",
        data_path(args, scene_dir / "images"),
        "--output_dir",
        data_path(args, scene_dir / "da3_outputs"),
        "--chunk-param-mode",
        "dynamic",
    ]
    run_cmd(cmd, dry_run=args.dry_run)
    if not args.dry_run:
        ensure_file(output_dir / "camera_poses.txt", "DA3 did not produce camera poses")
        ensure_dir_has_files(output_dir / "results_output", "DA3 did not produce results_output")


def run_sam3(args: argparse.Namespace, worker_gpu: str, scene_dir: Path) -> None:
    output_dir = scene_dir / "sam3_outputs"
    if args.skip_sam3:
        ensure_dir_has_files(output_dir / "tracks", "SAM3 skipped but tracks are missing")
        return
    if (output_dir / "tracks").is_dir() and list_npz(output_dir / "tracks") and not args.overwrite:
        print(f"Skipping SAM3; found tracks in {output_dir}")
        return
    remove_if_overwrite(output_dir, args.overwrite)

    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        f"device={worker_gpu}",
        "--ipc=host",
        *user_args(),
        "-e",
        "HOME=/tmp",
        *hf_env_args(args),
        "-v",
        f"{REPO_ROOT / 'sam3'}:/workspace/sam3",
        "-v",
        f"{args.data_root.resolve()}:/workspace/sam3/data",
        "-w",
        "/workspace/sam3",
        args.sam3_image,
        "python",
        "run_inference.py",
        data_path(args, scene_dir / "images"),
        data_path(args, scene_dir / "objects.txt"),
        data_path(args, scene_dir / "sam3_outputs"),
        "--score-threshold-detection",
        str(args.sam3_score_threshold),
        "--new-det-threshold",
        str(args.sam3_new_det_threshold),
    ]
    run_cmd(cmd, dry_run=args.dry_run)
    if not args.dry_run:
        ensure_dir_has_files(output_dir / "tracks", "SAM3 did not produce tracks")


def run_tracker(args: argparse.Namespace, worker_gpu: str, scene_dir: Path) -> None:
    output_dir = scene_dir / "tracker_outputs"
    if args.skip_tracker:
        ensure_dir_has_files(output_dir, "Tracker skipped but tracker outputs are missing")
        return
    if list_npz(output_dir) and not args.overwrite:
        print(f"Skipping tracker; found frame outputs in {output_dir}")
        return
    remove_if_overwrite(output_dir, args.overwrite)

    base = docker_pipeline_base(args.pipeline_image, worker_gpu, args.data_root, args)
    cmd = base[:-1] + cotracker_cache_args() + [base[-1],
        "python",
        "dynamic_tracker/run_tracker.py",
        data_path(args, scene_dir / "images"),
        data_path(args, scene_dir / "sam3_outputs"),
        data_path(args, scene_dir / "da3_outputs"),
        data_path(args, scene_dir / "tracker_outputs"),
        "--dynamic-classes",
        data_path(args, scene_dir / "objects.txt"),
        "--embedding-type",
        args.embedding_type,
    ]
    run_cmd(cmd, dry_run=args.dry_run)
    if not args.dry_run:
        ensure_dir_has_files(output_dir, "Tracker did not produce outputs")


def main() -> None:
    args = parse_args()
    worker_gpu, qwen_gpu = parse_gpu_policy(args.gpus)

    scene_dir = default_scene_dir(args).resolve()
    args.data_root = (args.data_root.resolve() if args.data_root is not None else scene_dir.parent)
    try:
        scene_dir.relative_to(args.data_root)
    except ValueError as exc:
        raise ValueError(f"Output directory must be inside --data-root: {scene_dir}") from exc

    validate_writable_dir(scene_dir, "Output scene directory")

    configure_hf(args)

    scene_id = scene_dir.name
    video_path = None
    if args.input_mode == "video":
        video_path = copy_video(args.video, scene_dir, args.overwrite, args.dry_run)
    log_dir = (args.log_dir or (scene_dir / "logs")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "status": "running",
        "scene_id": scene_id,
        "scene_dir": str(scene_dir),
        "input_mode": args.input_mode,
        "input_source": str(args.input_source.resolve()),
        "input_video": str(args.video.resolve()) if args.video is not None else None,
        "input_image_folder": str(args.image_folder.resolve()) if args.image_folder is not None else None,
        "copied_video": str(video_path) if video_path is not None else None,
        "images_dir": str(scene_dir / "images"),
        "started_at": now_iso(),
        "worker_gpu": worker_gpu,
        "qwen_gpu": qwen_gpu,
        "dry_run": bool(args.dry_run),
        "settings": {
            "fps": args.fps,
            "num_keyframes": args.num_keyframes,
            "embedding_type": args.embedding_type,
            "sam3_score_threshold": args.sam3_score_threshold,
            "sam3_new_det_threshold": args.sam3_new_det_threshold,
            "pipeline_image": args.pipeline_image,
            "da3_image": args.da3_image,
            "sam3_image": args.sam3_image,
            "qwen_image": args.qwen_image,
            "qwen_model": args.qwen_model,
            "hf_cache_dir": str(args.hf_cache_dir),
            "hf_token_present": args.hf_token_present,
        },
        "stages": {},
    }
    write_summary(summary, log_dir / "pipeline_summary.json")

    print(f"Scene directory: {scene_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Input mode: {args.input_mode}")
    print(f"HF cache directory: {args.hf_cache_dir}")
    print(f"HF token present: {args.hf_token_present}")
    print(f"Worker GPU: {worker_gpu}; Qwen GPU: {qwen_gpu}")

    try:
        if args.input_mode == "video":
            run_stage(summary, log_dir, "extract_frames", run_frame_extraction, args, worker_gpu, scene_dir, video_path)
        else:
            run_stage(summary, log_dir, "prepare_images", prepare_image_folder, args, scene_dir)
        run_stage(summary, log_dir, "select_keyframes", run_keyframes, args, worker_gpu, scene_dir)
        run_stage(summary, log_dir, "qwen_objects", run_qwen_objects, args, qwen_gpu, scene_dir)
        run_stage(summary, log_dir, "da3", run_da3, args, worker_gpu, scene_dir)
        run_stage(summary, log_dir, "sam3", run_sam3, args, worker_gpu, scene_dir)
        run_stage(summary, log_dir, "tracker", run_tracker, args, worker_gpu, scene_dir)
    except Exception:
        print(f"Pipeline failed. Logs: {log_dir}", file=sys.stderr)
        raise

    summary["status"] = "ok"
    summary["finished_at"] = now_iso()
    write_summary(summary, log_dir / "pipeline_summary.json")

    print(f"Pipeline complete: {scene_dir}")
    print(f"Logs: {log_dir}")


if __name__ == "__main__":
    main()
