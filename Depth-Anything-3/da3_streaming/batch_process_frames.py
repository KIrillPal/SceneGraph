#!/usr/bin/env python3
import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path


def parse_devices(device_arg):
    devices = []
    for raw_device in device_arg.split(","):
        device = raw_device.strip()
        if not device:
            continue
        if device.startswith("cuda:"):
            device = device.split(":", 1)[1]
        if not device.isdigit():
            raise ValueError(
                f"Unsupported device '{raw_device}'. Use values like 'cuda:0,1' or '0,1'."
            )
        devices.append(device)

    if not devices:
        raise ValueError("At least one CUDA device must be provided.")

    return devices


def collect_jobs(frames_dir, output_dir):
    jobs = []
    skipped = []

    for image_dir in sorted(path for path in frames_dir.iterdir() if path.is_dir()):
        record_id = image_dir.name
        record_output_dir = output_dir / record_id

        if record_output_dir.exists():
            skipped.append((record_id, record_output_dir))
            continue

        jobs.append((record_id, image_dir, record_output_dir))

    return jobs, skipped


def worker(slot_name, gpu_id, jobs_queue, script_path, workdir, failures):
    while True:
        try:
            record_id, image_dir, output_dir = jobs_queue.get_nowait()
        except queue.Empty:
            return

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        command = [
            sys.executable,
            str(script_path),
            "--image_dir",
            str(image_dir),
            "--output_dir",
            str(output_dir),
        ]

        print(
            f"[{slot_name}] start {record_id} on cuda:{gpu_id} -> {output_dir}",
            flush=True,
        )
        try:
            subprocess.run(command, cwd=str(workdir), env=env, check=True)
            print(f"[{slot_name}] done {record_id}", flush=True)
        except subprocess.CalledProcessError as exc:
            failures.append((record_id, gpu_id, exc.returncode))
            print(
                f"[{slot_name}] failed {record_id} on cuda:{gpu_id} "
                f"with exit code {exc.returncode}",
                flush=True,
            )
        finally:
            jobs_queue.task_done()


def main():
    parser = argparse.ArgumentParser(
        description="Batch-process all ARKitScenes frame folders with DA3 streaming."
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("/workspace/data/arkitscenes_validation30_prepared/frames"),
        help="Directory containing one subdirectory per recording.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/data/arkitscenes_validation_prepared/da3"),
        help="Root output directory. Each recording is written to output-dir/<id>.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0,1",
        help="CUDA devices to use, for example 'cuda:0,1' or '0,1'.",
    )
    parser.add_argument(
        "--processes-per-gpu",
        type=int,
        default=1,
        help="Number of concurrent DA3 processes to run per GPU.",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=Path("/workspace/da3_streaming/da3_streaming.py"),
        help="Path to da3_streaming.py inside the container.",
    )
    args = parser.parse_args()

    if args.processes_per_gpu < 1:
        raise ValueError("--processes-per-gpu must be at least 1.")
    if not args.frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory does not exist: {args.frames_dir}")
    if not args.script.is_file():
        raise FileNotFoundError(f"DA3 streaming script does not exist: {args.script}")

    devices = parse_devices(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs, skipped = collect_jobs(args.frames_dir, args.output_dir)

    for record_id, output_dir in skipped:
        print(f"[skip] {record_id}: output directory already exists: {output_dir}")

    if not jobs:
        print("No recordings to process.")
        return 0

    jobs_queue = queue.Queue()
    for job in jobs:
        jobs_queue.put(job)

    failures = []
    threads = []
    failures_lock = threading.Lock()
    workdir = args.script.parent

    print(
        f"Processing {len(jobs)} recordings on devices {devices} "
        f"with {args.processes_per_gpu} process(es) per GPU.",
        flush=True,
    )

    for gpu_id in devices:
        for process_idx in range(args.processes_per_gpu):
            slot_name = f"gpu{gpu_id}-slot{process_idx}"
            thread = threading.Thread(
                target=worker,
                args=(
                    slot_name,
                    gpu_id,
                    jobs_queue,
                    args.script,
                    workdir,
                    ThreadSafeFailures(failures, failures_lock),
                ),
                daemon=True,
            )
            thread.start()
            threads.append(thread)

    for thread in threads:
        thread.join()

    if failures:
        print("Failed recordings:", flush=True)
        for record_id, gpu_id, returncode in failures:
            print(f"  {record_id} on cuda:{gpu_id}: exit code {returncode}", flush=True)
        return 1

    print("All recordings processed successfully.")
    return 0


class ThreadSafeFailures:
    def __init__(self, failures, lock):
        self.failures = failures
        self.lock = lock

    def append(self, item):
        with self.lock:
            self.failures.append(item)

    def __bool__(self):
        with self.lock:
            return bool(self.failures)

    def __iter__(self):
        with self.lock:
            return iter(list(self.failures))


if __name__ == "__main__":
    raise SystemExit(main())
