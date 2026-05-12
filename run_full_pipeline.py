#!/usr/bin/env python3
"""Sequential end-to-end pipeline runner for an input MP4 video."""

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
    parser.add_argument("input_video", type=Path, nargs="?", help="Input .mp4 video")
    parser.add_argument("--video", type=Path, default=None, help="Input .mp4 video")
    parser.add_argument(
        "--scene-id",
        default=None,
        help="Scene id under data/. Defaults to input video stem",
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--fps", type=float, default=10.0, help="Frame extraction FPS")
    parser.add_argument("--num-keyframes", type=int, default=30)
    parser.add_argument(
        "--gpus",
        default="0",
        help="GPU policy: 'i' uses one GPU for all steps; 'i,j' uses i for pipeline and j for Qwen",
    )
    parser.add_argument("--embedding-type", choices=("dinov2", "dinov3", "sam"), default="dinov3")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing stage outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Pipeline log directory. Defaults to data/<scene-id>/logs",
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
        help="Use an existing '<object>, <static|dynamic>' txt instead of calling Qwen",
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
        help="Defaults to qwen-pipeline-<scene-id>",
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
    args.video = args.video or args.input_video
    if args.video is None:
        parser.error("provide an input video as positional argument or with --video")
    return args


def parse_gpu_policy(value: str) -> tuple[str, str]:
    devices = [part.strip() for part in value.split(",") if part.strip()]
    if len(devices) == 1:
        return devices[0], devices[0]
    if len(devices) == 2:
        return devices[0], devices[1]
    raise ValueError("--gpus must be either 'i' or 'i,j'")


def rel_to_repo(path: Path) -> Path:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must be inside repo root for container mapping: {path}") from exc


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


def cache_args() -> list[str]:
    hf_cache = Path(os.environ.get("HF_CACHE_DIR", Path.home() / ".cache" / "huggingface")).expanduser()
    hf_cache.mkdir(parents=True, exist_ok=True)
    return ["-e", "HF_HOME=/tmp/hf_cache", "-v", f"{hf_cache}:/tmp/hf_cache"]


def cotracker_cache_args() -> list[str]:
    cache = Path(os.environ.get("COTRACKER_CACHE_DIR", Path.home() / ".cache" / "cotracker")).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    return [
        "-e",
        "COTRACKER_CHECKPOINT=/tmp/cotracker/scaled_online.pth",
        "-v",
        f"{cache}:/tmp/cotracker",
    ]


def docker_pipeline_base(image: str, worker_gpu: str, data_root: Path) -> list[str]:
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
        *cache_args(),
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


def run_frame_extraction(args: argparse.Namespace, worker_gpu: str, scene_dir: Path, video_path: Path) -> None:
    images_dir = scene_dir / "images"
    if args.skip_frame_extraction:
        ensure_dir_has_files(images_dir, "Frame extraction skipped but images are missing")
        return
    if list_images(images_dir) and not args.overwrite:
        print(f"Skipping frame extraction; found images in {images_dir}")
        return

    cmd = docker_pipeline_base(args.pipeline_image, worker_gpu, args.data_root) + [
        "python",
        "utils/video_to_frames.py",
        str(rel_to_repo(video_path)),
        str(rel_to_repo(images_dir)),
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

    cmd = docker_pipeline_base(args.pipeline_image, worker_gpu, args.data_root) + [
        "python",
        "MaxInfo/pvsg_maxinfo_filter.py",
        "--input_dir",
        str(rel_to_repo(scene_dir / "images")),
        "--output_dir",
        str(rel_to_repo(selected_dir)),
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


def wait_for_qwen(port: int, timeout: int) -> None:
    deadline = time.time() + timeout
    url = qwen_health_url(port)
    while time.time() < deadline:
        try:
            with request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    return
        except (OSError, error.URLError):
            pass
        time.sleep(5)
    raise TimeoutError(f"Qwen server did not become ready within {timeout}s: {url}")


def start_qwen(args: argparse.Namespace, qwen_gpu: str, container_name: str) -> subprocess.Popen | None:
    run_cmd(["docker", "rm", "-f", container_name], dry_run=args.dry_run, check=False)
    hf_cache = Path(os.environ.get("HF_CACHE_DIR", Path.home() / ".cache" / "huggingface")).expanduser()
    hf_cache.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        container_name,
        "--gpus",
        f"device={qwen_gpu}",
        "--ipc=host",
        "-p",
        f"{args.qwen_port}:8000",
        "-e",
        "FLASHINFER_DISABLE_VERSION_CHECK=1",
        "-e",
        "HF_HOME=/root/.cache/huggingface",
        "-v",
        f"{hf_cache}:/root/.cache/huggingface",
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
            wait_for_qwen(args.qwen_port, args.qwen_timeout)
        except Exception:
            if logs_process is not None:
                logs_process.terminate()
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
            qwen_logs_process = start_qwen(args, qwen_gpu, container_name)
            started = True
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
        f"data/{scene_dir.name}/images",
        "--output_dir",
        f"data/{scene_dir.name}/da3_outputs",
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
        *cache_args(),
        "-v",
        f"{REPO_ROOT / 'sam3'}:/workspace/sam3",
        "-v",
        f"{args.data_root.resolve()}:/workspace/sam3/data",
        "-w",
        "/workspace/sam3",
        args.sam3_image,
        "python",
        "run_inference.py",
        f"data/{scene_dir.name}/images",
        f"data/{scene_dir.name}/objects.txt",
        f"data/{scene_dir.name}/sam3_outputs",
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

    base = docker_pipeline_base(args.pipeline_image, worker_gpu, args.data_root)
    cmd = base[:-1] + cotracker_cache_args() + [base[-1],
        "python",
        "dynamic_tracker/run_tracker.py",
        f"data/{scene_dir.name}/images",
        f"data/{scene_dir.name}/sam3_outputs",
        f"data/{scene_dir.name}/da3_outputs",
        f"data/{scene_dir.name}/tracker_outputs",
        "--dynamic-classes",
        f"data/{scene_dir.name}/objects.txt",
        "--embedding-type",
        args.embedding_type,
    ]
    run_cmd(cmd, dry_run=args.dry_run)
    if not args.dry_run:
        ensure_dir_has_files(output_dir, "Tracker did not produce outputs")


def main() -> None:
    args = parse_args()
    worker_gpu, qwen_gpu = parse_gpu_policy(args.gpus)

    args.data_root = args.data_root.resolve()
    try:
        args.data_root.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("--data-root must be inside the repository for current container path mapping") from exc

    scene_id = args.scene_id or args.video.stem
    scene_dir = args.data_root / scene_id
    video_path = copy_video(args.video, scene_dir, args.overwrite, args.dry_run)
    log_dir = (args.log_dir or (scene_dir / "logs")).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "status": "running",
        "scene_id": scene_id,
        "scene_dir": str(scene_dir),
        "input_video": str(args.video.resolve()),
        "copied_video": str(video_path),
        "started_at": now_iso(),
        "worker_gpu": worker_gpu,
        "qwen_gpu": qwen_gpu,
        "dry_run": bool(args.dry_run),
        "settings": {
            "fps": args.fps,
            "num_keyframes": args.num_keyframes,
            "embedding_type": args.embedding_type,
            "pipeline_image": args.pipeline_image,
            "da3_image": args.da3_image,
            "sam3_image": args.sam3_image,
            "qwen_image": args.qwen_image,
            "qwen_model": args.qwen_model,
        },
        "stages": {},
    }
    write_summary(summary, log_dir / "pipeline_summary.json")

    print(f"Scene directory: {scene_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Worker GPU: {worker_gpu}; Qwen GPU: {qwen_gpu}")

    try:
        run_stage(summary, log_dir, "extract_frames", run_frame_extraction, args, worker_gpu, scene_dir, video_path)
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
