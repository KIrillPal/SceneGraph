"""YAML + argparse entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .evaluate import evaluate_scene


def load_config_dict(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    with p.open() as f:
        return yaml.safe_load(f) or {}


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dict(out[k], v)
        elif v is not None:
            out[k] = v
    return out


# Default packaged example (edit or pass --config).
_DEFAULT = Path(__file__).resolve().parent / "default_config.yaml"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ARKitScenes GT vs DA3 metrics: ATE RMSE, TAE, AbsRel, δ1, Chamfer, EMD.",
    )
    p.add_argument("--config", type=Path, default=None, help="YAML with metric parameters")
    p.add_argument("--scene-dir", type=Path, default=None, help="GT scene folder, or with --batch-da3-dir: raw/Validation (parent of <video_id>)")
    p.add_argument(
        "--da3-dir",
        type=Path,
        default=None,
        help="DA3 experiment root (single scene); ignored as root when --batch-da3-dir is set",
    )
    p.add_argument(
        "--batch-da3-dir",
        type=Path,
        default=None,
        help="Parent of per-scene DA3 folders (<video_id>/results_output/...); sequential run; requires --scene-dir = Validation root and --output-dir (or default under batch dir)",
    )
    p.add_argument(
        "--prediction-pt",
        type=Path,
        default=None,
        help="Optional: single da3.pt instead of/in addition to --da3-dir (legacy)",
    )
    p.add_argument(
        "--prediction-ply",
        type=Path,
        default=None,
        help="Predicted scene point cloud (default: <da3-dir>/pcd/combined_pcd.ply)",
    )
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root directory for outputs: metrics/, plots/, rrd/<scene_id>.rrd (also written to YAML config as output_dir)",
    )
    p.add_argument("--output-json", type=Path, default=None, help="Write numeric results as JSON")
    p.add_argument(
        "--print-yaml",
        action="store_true",
        help="Print results as YAML instead of JSON",
    )
    p.add_argument("--skip-cloud", action="store_true")
    p.add_argument("--skip-depth", action="store_true")
    p.add_argument("--skip-odometry", action="store_true")
    p.add_argument("--skip-tae", action="store_true")
    p.add_argument(
        "--quiet",
        action="store_true",
        help="No progress bars or stage messages (same as progress: false, verbose: false)",
    )
    p.add_argument(
        "--plot-rrd",
        action="store_true",
        help="Write Rerun .rrd (needs rerun-sdk); default <output-dir>/rrd/<scene_id>.rrd",
    )
    p.add_argument(
        "--plot-rrd-path",
        type=Path,
        default=None,
        help="Output .rrd file (overrides <output-dir>/rrd/<scene_id>.rrd)",
    )
    p.add_argument(
        "--plot-charts",
        action="store_true",
        help="Write charts under <output-dir>/plots or <da3-dir>/recon_metrics/charts (matplotlib for PNG)",
    )
    p.add_argument(
        "--plot-charts-dir",
        type=Path,
        default=None,
        help="Charts directory (overrides <output-dir>/plots)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override YAML: TAE device auto|cuda|cpu (default: from config, usually auto)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    file_cfg: dict[str, Any] = {}
    cfg_path = args.config
    if cfg_path is None and _DEFAULT.is_file():
        cfg_path = _DEFAULT
    if cfg_path is not None:
        file_cfg = load_config_dict(cfg_path)

    cli_override: dict[str, Any] = {}
    if args.scene_dir is not None:
        cli_override["scene_dir"] = args.scene_dir
    if args.da3_dir is not None:
        cli_override["da3_dir"] = args.da3_dir
    if args.prediction_pt is not None:
        cli_override["prediction_pt"] = args.prediction_pt
    if args.prediction_ply is not None:
        cli_override["prediction_ply"] = args.prediction_ply
    if args.max_frames is not None:
        cli_override["max_frames"] = args.max_frames
    if args.output_dir is not None:
        cli_override["output_dir"] = str(args.output_dir)
    if args.skip_cloud:
        cli_override["skip_cloud"] = True
    if args.skip_depth:
        cli_override["skip_depth"] = True
    if args.skip_odometry:
        cli_override["skip_odometry"] = True
    if args.skip_tae:
        cli_override["skip_tae"] = True
    if args.quiet:
        cli_override["progress"] = False
        cli_override["verbose"] = False
    if args.device is not None:
        cli_override["device"] = args.device
    if args.plot_rrd:
        cli_override["plot_rrd"] = True
    if args.plot_rrd_path is not None:
        cli_override["plot_rrd_path"] = str(args.plot_rrd_path)
    if args.plot_charts:
        cli_override["plot_charts"] = True
    if args.plot_charts_dir is not None:
        cli_override["plot_charts_dir"] = str(args.plot_charts_dir)
    if args.batch_da3_dir is not None:
        cli_override["batch_da3_dir"] = str(args.batch_da3_dir)

    merged = merge_dict(file_cfg, cli_override)

    batch_raw = merged.get("batch_da3_dir")
    if batch_raw:
        from .batch import run_batch

        scene = merged.get("scene_dir")
        if scene is None:
            raise SystemExit(
                "batch_da3_dir: scene_dir is required — path to raw/Validation "
                "(parent of <video_id> folders)."
            )
        out_raw = merged.get("output_dir")
        if not out_raw:
            br = Path(batch_raw)
            out_raw = str(br.parent / f"{br.name}_recon_metrics")
            if merged.get("verbose", True):
                print(f"[recon_metrics] batch: output_dir default → {out_raw}", flush=True)
        eval_cfg = {
            k: v
            for k, v in merged.items()
            if k
            not in {
                "scene_dir",
                "da3_dir",
                "prediction_pt",
                "prediction_ply",
                "batch_da3_dir",
                "output_dir",
            }
        }
        pred_ply_raw = merged.get("prediction_ply")
        overall = run_batch(
            validation_root=Path(scene),
            batch_da3_dir=Path(batch_raw),
            eval_cfg=eval_cfg,
            output_dir=Path(out_raw),
            prediction_ply=Path(pred_ply_raw) if pred_ply_raw else None,
            verbose=bool(merged.get("verbose", True)),
        )
        if args.print_yaml:
            print(yaml.safe_dump(overall, default_flow_style=False, sort_keys=False))
        else:
            compact = {
                "n_scenes_attempted": overall["n_scenes_attempted"],
                "n_scenes_ok": overall["n_scenes_ok"],
                "n_scenes_failed": overall["n_scenes_failed"],
                "metrics_mean": overall["metrics_mean"],
                "summary_yaml": overall.get("summary_yaml"),
                "summary_json": overall.get("summary_json"),
                "failures": [
                    {
                        "scene_id": f["scene_id"],
                        "error_first_line": (f["error"].split("\n")[0] if f.get("error") else "")[:400],
                    }
                    for f in overall.get("failures", [])
                ],
            }
            print(json.dumps(compact, indent=2, default=str))
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with args.output_json.open("w") as f:
                json.dump(overall, f, indent=2, default=str)
        return

    scene = merged.get("scene_dir")
    da3_dir = merged.get("da3_dir")
    pred_pt = merged.get("prediction_pt")
    pred_ply = merged.get("prediction_ply")

    if da3_dir is not None:
        prediction = Path(da3_dir)
    elif pred_pt is not None:
        prediction = Path(pred_pt)
    else:
        prediction = None

    if scene is None or prediction is None:
        raise SystemExit(
            "scene_dir and da3_dir (or legacy prediction_pt) are required. "
            "Example: --scene-dir .../Validation/47429971 --da3-dir .../da3/47429971"
        )

    eval_cfg = {
        k: v
        for k, v in merged.items()
        if k
        not in {
            "scene_dir",
            "da3_dir",
            "prediction_pt",
            "prediction_ply",
        }
    }
    results = evaluate_scene(scene, prediction, pred_ply, config=eval_cfg)

    if args.print_yaml:
        print(yaml.safe_dump(results, default_flow_style=False, sort_keys=False))
    else:
        print(json.dumps(results, indent=2))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
