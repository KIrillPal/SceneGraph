#!/usr/bin/env python3
"""
Adapter: replay notebook export through yolo_sgg RerunVisualizer.

Expected export directory content:
  - runpy_export_config.json
  - runpy_objects.pkl               # list[ list[object_dict] ] per frame
  - runpy_masks_clean.pkl           # list[ list[mask(H,W)] ] per frame
  - runpy_track_ids.pkl             # list[np.ndarray] per frame
  - runpy_class_names.pkl           # list[list[str] | None] per frame
  - runpy_T_w_c.npy                 # (N,4,4) c2w

Optional:
  - runpy_graph_edges.json          # list of {src, dst, label}
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Optional
from tqdm import tqdm

import networkx as nx
import numpy as np
import rerun as rr


def _import_yolo_sgg_modules() -> Any:
    here = Path(__file__).resolve()
    yolo_sgg_root = here.parents[2] / "yolo_sgg"
    if str(yolo_sgg_root) not in sys.path:
        sys.path.insert(0, str(yolo_sgg_root))
    from rerun_utils import RerunVisualizer  # pylint: disable=import-error

    return RerunVisualizer


class ExportObjectRegistryShim:
    """Small shim compatible with RerunVisualizer expectations."""

    def __init__(self) -> None:
        self._objects: dict[int, dict] = {}

    def set_frame_objects(self, objs: list[dict]) -> None:
        self._objects = {int(o["global_id"]): o for o in objs}

    def get_all_pcds_for_visualization(self) -> list[dict]:
        out: list[dict] = []
        for gid, obj in self._objects.items():
            out.append(
                {
                    "global_id": gid,
                    "points": np.asarray(obj.get("points", np.zeros((0, 3), np.float32)), dtype=np.float32),
                    "class_name": obj.get("class_name"),
                    "bbox_3d": obj.get("bbox_3d"),
                    "visible_current_frame": bool(obj.get("visible_current_frame", False)),
                    "observation_count": int(obj.get("observation_count", 0)),
                }
            )
        return out

    def get_all_objects(self) -> dict[int, dict]:
        return self._objects


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _load_graph(export_dir: Path, object_ids: set[int]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for oid in sorted(object_ids):
        graph.add_node(int(oid), data={})

    edges_path = export_dir / "runpy_graph_edges.json"
    if not edges_path.is_file():
        return graph

    with edges_path.open(encoding="utf-8") as f:
        edges = json.load(f)

    for e in edges:
        src = int(e["src"])
        dst = int(e["dst"])
        label = str(e.get("label", ""))
        graph.add_edge(src, dst, label=label, label_class=label)
    return graph


def run_from_export(
    export_dir: Path,
    spawn_viewer: bool = True,
    save_path: Optional[Path] = None,
    connect_uri: Optional[str] = None,
) -> None:
    export_dir = export_dir.resolve()

    cfg_path = export_dir / "runpy_export_config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing {cfg_path}")
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)

    objects_by_frame: list[list[dict]] = _load_pickle(export_dir / "runpy_objects.pkl")
    masks_clean_by_frame: list[list[np.ndarray]] = _load_pickle(export_dir / "runpy_masks_clean.pkl")
    track_ids_by_frame: list[np.ndarray] = _load_pickle(export_dir / "runpy_track_ids.pkl")
    class_names_by_frame: list[Optional[list[str]]] = _load_pickle(export_dir / "runpy_class_names.pkl")
    T_w_c_all = np.load(export_dir / "runpy_T_w_c.npy")

    rgb_paths = cfg["rgb_paths"]
    if not (len(rgb_paths) == len(objects_by_frame) == len(T_w_c_all)):
        raise ValueError("Inconsistent export lengths: rgb_paths/objects/T_w_c")

    RerunVisualizer = _import_yolo_sgg_modules()
    vis = RerunVisualizer(recording_id=str(cfg.get("recording_id", "tracker_runpy_adapter")))
    vis.init(
        int(cfg["img_w"]),
        int(cfg["img_h"]),
        float(cfg["fx"]),
        float(cfg["fy"]),
        float(cfg["cx"]),
        float(cfg["cy"]),
        spawn=spawn_viewer,
    )
    if save_path is not None:
        rr.save(str(save_path))
    elif connect_uri:
        rr.connect_grpc(connect_uri)

    all_ids = {int(o["global_id"]) for frame_objs in objects_by_frame for o in frame_objs}
    graph = _load_graph(export_dir, all_ids)
    registry = ExportObjectRegistryShim()

    for i in tqdm(list(range(len(rgb_paths)))):
        registry.set_frame_objects(objects_by_frame[i])
        vis.log_frame(
            frame_idx=i,
            object_registry=registry,
            persistent_graph=graph,
            T_w_c=np.asarray(T_w_c_all[i], dtype=np.float32),
            rgb_path=str(rgb_paths[i]),
            masks_clean=masks_clean_by_frame[i],
            track_ids=np.asarray(track_ids_by_frame[i]),
            class_names=class_names_by_frame[i],
            vis_edges=bool(cfg.get("vis_edges", False)),
        )

    print(f"Adapter replay completed. Frames: {len(rgb_paths)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Replay notebook export via yolo_sgg RerunVisualizer.")
    p.add_argument("--export-dir", type=Path, required=True)
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Write recording to .rrd file (recommended for headless/unstable gRPC).",
    )
    p.add_argument(
        "--no-spawn",
        action="store_true",
        help="Do not auto-spawn Rerun viewer.",
    )
    p.add_argument(
        "--connect",
        type=str,
        default=None,
        help="Explicit gRPC URI, e.g. rerun+http://127.0.0.1:9876/proxy",
    )
    args = p.parse_args()
    run_from_export(
        args.export_dir,
        spawn_viewer=not args.no_spawn,
        save_path=args.save,
        connect_uri=args.connect,
    )


if __name__ == "__main__":
    main()

