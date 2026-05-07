"""Sequential batch evaluation over multiple DA3 scene folders."""

from __future__ import annotations

import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .evaluate import evaluate_scene
from .plot_charts import batch_mean_plot_options, plot_charts_enabled, save_batch_mean_curves

SKIP_AGG_KEYS = frozenset(
    {
        "output_dir",
        "metrics_dir",
        "scene_dir",
        "prediction",
        "pred_ply",
        "gt_mesh",
        "plot_rrd_path",
        "plot_charts_dir",
        "tae_device",
    }
)


def is_da3_experiment_dir(path: Path) -> bool:
    """Heuristics: DA3 streaming folder or single ``da3.pt``."""
    if not path.is_dir():
        return False
    if (path / "results_output").is_dir():
        return True
    if (path / "camera_poses.txt").is_file():
        return True
    if (path / "da3.pt").is_file():
        return True
    return False


def discover_batch_da3_dirs(batch_root: Path) -> list[Path]:
    """Sorted subdirectories that look like DA3 outputs (skip hidden names)."""
    batch_root = Path(batch_root)
    if not batch_root.is_dir():
        raise NotADirectoryError(batch_root)
    out: list[Path] = []
    for p in batch_root.iterdir():
        if not p.is_dir() or p.name.startswith("."):
            continue
        if is_da3_experiment_dir(p):
            out.append(p)
    return sorted(out, key=lambda x: x.name)


def mean_overall_metrics(per_scene: list[dict[str, Any]]) -> dict[str, float]:
    """Average numeric scalar fields across successful scene results."""
    buckets: dict[str, list[float]] = {}
    for r in per_scene:
        for k, v in r.items():
            if k in SKIP_AGG_KEYS:
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                fv = float(v)
                if not math.isfinite(fv):
                    continue
                buckets.setdefault(k, []).append(fv)
    return {k: float(np.mean(vals)) for k, vals in buckets.items() if vals}


def run_batch(
    *,
    validation_root: Path,
    batch_da3_dir: Path,
    eval_cfg: dict[str, Any],
    output_dir: Path,
    prediction_ply: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    For each DA3 subfolder under ``batch_da3_dir``, run ``evaluate_scene`` against
    ``validation_root / <video_id>``. Writes per-scene under ``output_dir/<video_id>/`` and
    summary under ``output_dir/overall/``.
    """
    validation_root = Path(validation_root)
    batch_da3_dir = Path(batch_da3_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    da3_list = discover_batch_da3_dirs(batch_da3_dir)
    if not da3_list:
        raise FileNotFoundError(
            f"No DA3 scene subfolders under {batch_da3_dir} "
            "(expected results_output/, camera_poses.txt, or da3.pt in each child)."
        )

    ok_results: list[tuple[str, dict[str, Any]]] = []
    failures: list[dict[str, str]] = []

    def say(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    for i, da3_path in enumerate(da3_list, start=1):
        vid = da3_path.name
        gt_path = validation_root / vid
        say(f"\n[recon_metrics] Batch [{i}/{len(da3_list)}] scene_id={vid}")
        if not gt_path.is_dir():
            err = f"GT scene directory missing: {gt_path}"
            say(f"  SKIP: {err}")
            failures.append({"scene_id": vid, "error": err})
            continue
        per_out = output_dir / vid
        cfg_i = dict(eval_cfg)
        cfg_i["output_dir"] = str(per_out.expanduser().resolve())
        ply_use: str | Path | None = prediction_ply
        try:
            res = evaluate_scene(gt_path, da3_path, ply_use, config=cfg_i)
            ok_results.append((vid, res))
            say(f"  OK → {per_out}")
        except Exception:
            tb = traceback.format_exc()
            failures.append({"scene_id": vid, "error": tb.strip()})
            say(f"  FAIL ({vid}): see batch_summary.yaml for traceback")

    scenes_payload = [{"scene_id": sid, **res} for sid, res in ok_results]
    mean_metrics = mean_overall_metrics([r for _, r in ok_results])
    overall: dict[str, Any] = {
        "n_scenes_attempted": len(da3_list),
        "n_scenes_ok": len(ok_results),
        "n_scenes_failed": len(failures),
        "metrics_mean": mean_metrics,
        "scenes_ok": [sid for sid, _ in ok_results],
        "scenes": scenes_payload,
        "failures": failures,
    }

    ov_dir = output_dir / "overall"
    ov_dir.mkdir(parents=True, exist_ok=True)
    json_path = ov_dir / "batch_summary.json"
    yaml_path = ov_dir / "batch_summary.yaml"
    metrics_json_path = ov_dir / "metrics.json"
    metrics_yaml_path = ov_dir / "metrics.yaml"
    overall["summary_json"] = str(json_path.resolve())
    overall["summary_yaml"] = str(yaml_path.resolve())
    overall["metrics_json"] = str(metrics_json_path.resolve())
    overall["metrics_yaml"] = str(metrics_yaml_path.resolve())

    bo = batch_mean_plot_options(eval_cfg)
    if bo["write_mean_plots"] and plot_charts_enabled(eval_cfg):
        mean_written = save_batch_mean_curves(
            output_dir,
            [sid for sid, _ in ok_results],
            min_videos=bo["min_videos_for_mean"],
            verbose=verbose,
        )
        if mean_written:
            overall["mean_plots_dir"] = str((ov_dir / "plots").resolve())
            overall["mean_plots"] = mean_written
            overall["min_videos_for_mean_curves"] = bo["min_videos_for_mean"]

    metrics_only = {
        "n_scenes_attempted": overall["n_scenes_attempted"],
        "n_scenes_ok": overall["n_scenes_ok"],
        "n_scenes_failed": overall["n_scenes_failed"],
        "metrics_mean": dict(mean_metrics),
    }
    if "min_videos_for_mean_curves" in overall:
        metrics_only["min_videos_for_mean_curves"] = overall["min_videos_for_mean_curves"]
    if overall.get("mean_plots_dir"):
        metrics_only["mean_plots_dir"] = overall["mean_plots_dir"]

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, default=str)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(overall, f, default_flow_style=False, sort_keys=False)
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_only, f, indent=2, default=str)
    with metrics_yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(metrics_only, f, default_flow_style=False, sort_keys=False)

    say("\n[recon_metrics] ========== batch summary ==========")
    say(f"  OK: {overall['n_scenes_ok']} / {overall['n_scenes_attempted']}")
    say(f"  Failed: {overall['n_scenes_failed']}")
    say(f"  Summary: {yaml_path}")
    if failures:
        say("  Failures (short):")
        for fb in failures:
            first_line = fb["error"].split("\n")[0][:200]
            say(f"    - {fb['scene_id']}: {first_line}")
    if mean_metrics:
        say("  metrics_mean (finite means over OK scenes):")
        for k in sorted(mean_metrics):
            say(f"    {k}: {mean_metrics[k]:.6g}")
    if overall.get("mean_plots_dir"):
        say(
            "  Mean curves (≥"
            f"{overall.get('min_videos_for_mean_curves', '?')} videos / index): "
            f"{overall['mean_plots_dir']}"
        )

    return overall
