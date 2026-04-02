# SceneGraph
Knowledge-graph-based approach for spatial reasoning for static and dynamic scenes.

![demo](examples/images/demo.gif)

## Quick Start

The main pipeline is:

1. `SAM3` generates 2D masks/tracks and frame embeddings from the RGB frames.
2. `Depth-Anything-3` generates depth, intrinsics, and camera poses for the same frames.
3. The custom `3D tracker` fuses SAM3 and DA3 outputs into object tracks and voxel histories.
4. `Rerun` visualization builds a `.rrd` file for inspecting the scene, masks, and 3D tracks.

Expected high-level inputs:

- a folder with scene images
- a text file with one object name per line for SAM3 prompts

### 1. Run SAM3

See [sam3/README.md](sam3/README.md) for environment details.

```bash
python3 sam3/run_inference.py <image_folder> <objects_txt> <sam3_output_dir>
```

This produces:

- `<sam3_output_dir>/tracks`
- frame embeddings in `<sam3_output_dir>`

### 2. Run Depth-Anything-3

See [Depth-Anything-3/README.md](Depth-Anything-3/README.md) for container and weights setup.

```bash
cd Depth-Anything-3/da3_streaming
python3 da3_streaming.py --image_dir <image_folder> --output_dir <da3_output_dir>
```

This produces a DA3 output directory with:

- `results_output/`
- `camera_poses.txt`

### 3. Run the Custom 3D Tracker

Detailed tracker notes are in [static/README.md](static/README.md).

```bash
python static/run_tracker.py \
  <image_folder> \
  <sam3_output_dir>/tracks \
  <sam3_output_dir> \
  <da3_output_dir> \
  <tracker_output_dir>
```

Key tracker outputs:

- `<tracker_output_dir>/outputs`
- `<tracker_output_dir>/track_outputs`
- `<tracker_output_dir>/point_outputs`
- `<tracker_output_dir>/rerun_export`

### 4. Build the Visualization

The repo also includes a useful Rerun visualization in [visualization/tracker_layers_rerun.py](visualization/tracker_layers_rerun.py):

```bash
python visualization/tracker_layers_rerun.py \
  --export-dir <tracker_output_dir> \
  --save <tracker_output_dir>/tracker_layers.rrd
```

Open the resulting `.rrd` in Rerun to inspect:

- image masks
- per-frame scene point clouds
- tracked object point clouds
- merged voxel clouds and track ids
