# Dynamic Tracker

Dynamic 3D tracker wrapper for SAM3 + DA3 outputs. It keeps the same tracker export format as the static tracker while adding CoTracker-based handling for dynamic object classes.

## Build Docker Image

From the repository root:

```bash
./dynamic_tracker/build.sh
```

The default image name is `dynamic-tracker`. Override it with:

```bash
IMAGE_NAME=my-dynamic-tracker ./dynamic_tracker/build.sh
```

## CoTracker Checkpoint

The dynamic tracker needs the CoTracker online checkpoint:

```text
scaled_online.pth
```

Download it on the host machine before running the tracker:

```bash
mkdir -p ~/.cache/cotracker
wget -O ~/.cache/cotracker/scaled_online.pth \
  https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
```

Verify:

```bash
ls -lh ~/.cache/cotracker/scaled_online.pth
```

`start.sh` mounts this host folder into the container:

```text
~/.cache/cotracker -> /root/.cache/cotracker
```

The default checkpoint path inside the container is:

```text
/root/.cache/cotracker/scaled_online.pth
```

If your checkpoint is elsewhere, override the cache directory:

```bash
COTRACKER_CACHE_DIR=/path/to/cotracker_cache ./dynamic_tracker/start.sh
```

or override the exact in-container checkpoint path:

```bash
COTRACKER_CHECKPOINT=/root/.cache/cotracker/scaled_online.pth ./dynamic_tracker/start.sh
```

## Start Container

Open a shell inside the container:

```bash
./dynamic_tracker/start.sh
```

Use a different GPU:

```bash
GPU_DEVICE=1 ./dynamic_tracker/start.sh
```

Mounted paths:

```text
repo root                  -> /workspace
~/.cache/huggingface       -> /root/.cache/huggingface
~/.cache/cotracker         -> /root/.cache/cotracker
```

## Inputs

The executable is:

```bash
python dynamic_tracker/run_tracker.py \
  <image_folder> \
  <sam3_outputs> \
  <da3_outputs> \
  [save_path] \
  --dynamic-classes <objects_static_dynamic.txt> \
  --embedding-type {dino,sam3}
```

Expected `sam3_outputs` layout:

```text
sam3_outputs/
  tracks/
  embeds/
```

`embeds/` is required when using:

```bash
--embedding-type sam3
```

Expected `da3_outputs` layout:

```text
da3_outputs/
  results_output/
  camera_poses.txt
```

The dynamic class file must use this format:

```text
chair, static
car, dynamic
person, dynamic
```

Only lines marked `dynamic` are treated as dynamic classes. Matching is case-insensitive.

If `save_path` is omitted, the output directory defaults to:

```text
tracker_outputs
```

## Example

Run directly through `start.sh`:

```bash
./dynamic_tracker/start.sh python dynamic_tracker/run_tracker.py \
  data/0/images \
  data/0/sam3_outputs \
  data/0/da3_outputs \
  data/0/tracker_outputs \
  --dynamic-classes data/0/objects_static_dynamic.txt \
  --embedding-type dino
```

Using SAM3 embeddings instead of DINO:

```bash
./dynamic_tracker/start.sh python dynamic_tracker/run_tracker.py \
  data/0/images \
  data/0/sam3_outputs \
  data/0/da3_outputs \
  data/0/tracker_outputs \
  --dynamic-classes data/0/objects_static_dynamic.txt \
  --embedding-type sam3
```

## Output

The dynamic tracker writes the same per-frame `.npz` format as the static tracker:

```text
frame_000000.npz
frame_000001.npz
...
```

Each file contains:

```text
frame_id
image
masks
embeddings
point_cloud
intrinsic
extrinsic
```

This output is compatible with the existing visualization and downstream frame-selection scripts.
