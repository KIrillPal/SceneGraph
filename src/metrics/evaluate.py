"""End-to-end metric bundle for one scene."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from .arkit_frames import (
    enumerate_frames,
    find_mesh_path,
    gt_extrinsics_for_frames,
    load_gt_depth_meters,
)
from .cloud_metrics import chamfer_l2, emd_uniform_l1
from .depth_metrics import aggregate_depth_metrics
from .geometry import (
    absolute_rotation_error_per_frame_deg,
    align_w2c_extrinsics_batch,
    apply_sim3_points,
    ate_rmse_sim3,
    camera_center_from_w2c,
    relative_pose_errors,
)
from .ply_loader import load_ply_vertices
from .prediction_io import default_combined_ply, load_prediction
from .plot_charts import (
    plot_charts_enabled,
    plot_charts_options,
    save_metric_charts,
)
from .rrd_export import export_recon_metrics_rrd, plot_rrd_enabled, plot_rrd_options
from . import tae_metric


def _output_base(cfg: dict[str, Any]) -> Path | None:
    """Optional root directory: ``metrics/``, ``plots/``, ``rrd/`` below it."""
    raw = cfg.get("output_dir")
    if raw is None or raw == "":
        return None
    return Path(raw).expanduser()


def evaluate_scene(
    scene_dir: str | Path,
    prediction: str | Path,
    prediction_ply: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute odometry (ATE mean / RMSE on aligned camera centers), ARE/RPE, TAE, depth (AbsRel, δ1), cloud (Chamfer, EMD).

    ``prediction`` is either a DA3 experiment folder (``results_output/``, ``camera_poses.txt``)
    or a single ``da3.pt`` file. If it is a directory and ``prediction_ply`` is omitted,
    ``pcd/combined_pcd.ply`` is used when present.

    Predicted trajectory / depth frame order must match sorted RGB temporal order
    in ``scene_dir`` (same convention as ``da3_streaming`` + ``build_da3_pt.py``).

    Cloud alignment: the same Sim(3) estimated from camera-center correspondences
    maps predicted PLY (same frame as DA3 world coordinates) into the GT /
    ARKit world before Chamfer / EMD.
    """
    cfg = config or {}
    scene_dir = Path(scene_dir)
    pred_path = Path(prediction)
    ply_path = Path(prediction_ply) if prediction_ply else None
    output_base = _output_base(cfg)

    progress = bool(cfg.get("progress", True))
    verbose = bool(cfg.get("verbose", True))

    def say(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    prefer_pt = bool((cfg.get("prediction") or {}).get("prefer_da3_pt", True))

    if ply_path is None and pred_path.is_dir():
        dply = default_combined_ply(pred_path)
        if dply is not None:
            ply_path = dply

    max_frames = cfg.get("max_frames")
    skip_depth = bool(cfg.get("skip_depth", False))
    skip_cloud = bool(cfg.get("skip_cloud", False))
    skip_odom = bool(cfg.get("skip_odometry", False))
    skip_tae = bool(cfg.get("skip_tae", False))
    tcfg = cfg.get("tae") or {}

    align_cfg = cfg.get("alignment") or {}
    use_sim3 = bool(align_cfg.get("sim3", True))

    say("[recon_metrics] Enumerating GT frames (RGB / depth / intrinsics) …")
    frames = enumerate_frames(scene_dir)
    if max_frames is not None:
        frames = frames[: int(max_frames)]

    say(
        f"[recon_metrics] Loading predictions ({len(frames)} GT slots; source {pred_path}) …"
    )
    pred = load_prediction(
        pred_path,
        prefer_pt_in_dir=prefer_pt,
        max_frames=len(frames),
        progress=progress,
        verbose=verbose,
    )

    n_pred = pred["depth"].shape[0]
    n = min(len(frames), n_pred)
    if n < 2 and not skip_tae:
        skip_tae = True
    if n == 0:
        raise ValueError(
            f"No overlapping prediction and GT frames: {len(frames)} GT frame(s), "
            f"{n_pred} prediction frame(s). Check scene IDs and frame ordering."
        )

    frames = frames[:n]
    depth_pred = pred["depth"][:n]
    intr_pred = pred["intrinsic"][:n]
    ext_pred = pred["extrinsic"][:n]

    results: dict[str, Any] = {
        "frames_used": n,
        "scene_dir": str(scene_dir),
        "prediction": str(pred_path),
    }

    say("[recon_metrics] Interpolating GT camera poses …")
    gt_ext = gt_extrinsics_for_frames(scene_dir, frames)
    ref_c = camera_center_from_w2c(gt_ext)
    est_c = camera_center_from_w2c(ext_pred)

    r_a = np.eye(3)
    t_a = np.zeros(3)
    s_a = 1.0
    ate_val = float("nan")
    if not skip_odom:
        say("[recon_metrics] Aligning trajectories (Sim3) and ATE RMSE …")
    if n >= 3:
        ate_val, r_a, t_a, s_a = ate_rmse_sim3(est_c, ref_c, with_scale=use_sim3)
    elif n > 0:
        t_a = ref_c.mean(axis=0) - est_c.mean(axis=0)
        aligned = est_c + t_a
        ate_val = float(np.sqrt(np.mean(np.sum((aligned - ref_c) ** 2, axis=1))))

    if not skip_odom:
        results["ate_rmse"] = ate_val
        aligned_c = apply_sim3_points(est_c, r_a, t_a, float(s_a))
        results["ate_mean"] = float(np.mean(np.linalg.norm(aligned_c - ref_c, axis=1)))
        ext_pred_al = align_w2c_extrinsics_batch(ext_pred, r_a, t_a, float(s_a))
        are_deg = absolute_rotation_error_per_frame_deg(gt_ext, ext_pred_al)
        results["are_mean_deg"] = float(np.mean(are_deg))
        results["are_rmse_deg"] = float(np.sqrt(np.mean(are_deg**2)))
        pcfg = cfg.get("pose_error") or {}
        rpe_delta = max(1, int(pcfg.get("rpe_delta", 1)))
        fi, rpe_r, rpe_t = relative_pose_errors(gt_ext, ext_pred_al, delta=rpe_delta)
        results["rpe_delta"] = int(rpe_delta)
        if fi.size > 0:
            results["rpe_rot_mean_deg"] = float(np.mean(rpe_r))
            results["rpe_rot_rmse_deg"] = float(np.sqrt(np.mean(rpe_r**2)))
            results["rpe_trans_mean_m"] = float(np.mean(rpe_t))
            results["rpe_trans_rmse_m"] = float(np.sqrt(np.mean(rpe_t**2)))
        else:
            results["rpe_rot_mean_deg"] = float("nan")
            results["rpe_rot_rmse_deg"] = float("nan")
            results["rpe_trans_mean_m"] = float("nan")
            results["rpe_trans_rmse_m"] = float("nan")
    else:
        results["ate_rmse"] = float("nan")
        results["ate_mean"] = float("nan")
        results["are_mean_deg"] = float("nan")
        results["are_rmse_deg"] = float("nan")
        results["rpe_delta"] = int((cfg.get("pose_error") or {}).get("rpe_delta", 1))
        results["rpe_rot_mean_deg"] = float("nan")
        results["rpe_rot_rmse_deg"] = float("nan")
        results["rpe_trans_mean_m"] = float("nan")
        results["rpe_trans_rmse_m"] = float("nan")
    results["alignment_scale"] = float(s_a)

    if not skip_tae:
        say("[recon_metrics] TAE (temporal depth consistency) …")
        wss = tcfg.get("warp_source_samples")
        tae_dev = tcfg.get("device")
        if tae_dev is None:
            tae_dev = cfg.get("device", "auto")
        resolved_tae_dev = tae_metric.resolve_tae_device(str(tae_dev))
        results["tae_device"] = resolved_tae_dev
        if verbose and resolved_tae_dev == "cuda" and not progress:
            say("[recon_metrics] TAE using CUDA / torch.float32 warp")
        results["tae"] = float(
            tae_metric.tae(
                depth_pred.astype(np.float64),
                intr_pred.astype(np.float64),
                ext_pred.astype(np.float64),
                min_depth=float(tcfg.get("min_depth", 1e-4)),
                warp_source_samples=int(wss) if wss is not None else None,
                warp_sample_seed=int(tcfg.get("warp_sample_seed", 0)),
                device=str(tae_dev),
                progress=progress,
            )
        )
    else:
        results["tae"] = float("nan")
        results["tae_device"] = None

    if not skip_depth:
        dcfg = cfg.get("depth") or {}
        say("[recon_metrics] Depth metrics (AbsRel, δ1); loading GT depth maps …")
        depth_scale = float(dcfg.get("depth_mm_to_m", 1e-3))
        gt_depths: list[np.ndarray] = []
        for f in tqdm(
            frames,
            desc="GT depth",
            unit="frm",
            disable=not progress,
            ncols=100,
        ):
            gt_depths.append(load_gt_depth_meters(f.depth_path, depth_scale=depth_scale))
        dm = aggregate_depth_metrics(
            depth_pred,
            gt_depths,
            min_depth=float(dcfg.get("min_depth", 1e-3)),
            max_depth=float(dcfg.get("max_depth", 80.0)),
            scale_align=dcfg.get("scale_align", "median"),
            delta_thresh=float(dcfg.get("delta_thresh", 1.25)),
            eps=float(dcfg.get("eps", 1e-8)),
        )
        results.update(dm)
    else:
        results["absrel"] = float("nan")
        results["delta_1"] = float("nan")

    if not skip_cloud:
        if ply_path is None:
            raise FileNotFoundError(
                "Cloud metrics require a point cloud PLY. "
                f"No {pred_path / 'pcd/combined_pcd.ply'} found; pass prediction_ply or use --skip-cloud."
            )
        if not ply_path.is_file():
            raise FileNotFoundError(f"Prediction point cloud not found: {ply_path}")
        mesh = find_mesh_path(scene_dir)
        say(
            f"[recon_metrics] Point cloud: loading GT mesh {mesh.name} and prediction "
            f"{ply_path.name} …"
        )
        gt_pts, _ = load_ply_vertices(mesh)
        pr_pts, _ = load_ply_vertices(ply_path)
        pr_aligned = apply_sim3_points(pr_pts, r_a, t_a, s_a)
        ccfg = cfg.get("cloud") or {}
        rng_seed = int(ccfg.get("emd_seed", 0))
        say("[recon_metrics] Chamfer distance (subsampled) …")
        cm = chamfer_l2(
            gt_pts,
            pr_aligned,
            max_points_a=ccfg.get("chamfer_max_points_a"),
            max_points_b=ccfg.get("chamfer_max_points_b"),
            seed=rng_seed,
        )
        results.update(cm)
        say("[recon_metrics] EMD / assignment metric …")
        results["emd_l1"] = emd_uniform_l1(
            gt_pts,
            pr_aligned,
            max_points=int(ccfg.get("emd_max_points", 512)),
            seed=rng_seed,
        )
        results["pred_ply"] = str(ply_path)
        results["gt_mesh"] = str(mesh)
    else:
        results["chamfer_l2"] = float("nan")
        results["chamfer_l2_sqrt"] = float("nan")
        results["emd_l1"] = float("nan")

    if plot_rrd_enabled(cfg):
        dcfg = cfg.get("depth") or {}
        if skip_depth:
            say("[recon_metrics] Loading GT depths for Rerun (.rrd) …")
            depth_scale = float(dcfg.get("depth_mm_to_m", 1e-3))
            gt_depths_rrd: list[np.ndarray] = []
            for f in tqdm(
                frames,
                desc="GT depth (rrd)",
                unit="frm",
                disable=not progress,
                ncols=100,
            ):
                gt_depths_rrd.append(load_gt_depth_meters(f.depth_path, depth_scale=depth_scale))
        else:
            gt_depths_rrd = gt_depths

        ply_rrd = ply_path
        if ply_rrd is None and pred_path.is_dir():
            ply_rrd = default_combined_ply(pred_path)
        if ply_rrd is None or not ply_rrd.is_file():
            raise FileNotFoundError(
                "plot_rrd requires a prediction PLY (set prediction_ply or ensure "
                f"<da3-dir>/pcd/combined_pcd.ply exists). Got: {ply_rrd!r}"
            )
        mesh_rrd = find_mesh_path(scene_dir)
        say("[recon_metrics] Rerun export: point clouds, cameras, depth difference …")
        gt_pts_rrd, gt_cols_rrd = load_ply_vertices(mesh_rrd)
        pr_pts_rrd, pr_cols_rrd = load_ply_vertices(ply_rrd)
        pr_opts = plot_rrd_options(cfg)
        scene_id_r = scene_dir.name
        rrd_out = pr_opts["path"]
        if not rrd_out:
            if output_base is not None:
                rrd_out = output_base / "rrd" / f"{scene_id_r}.rrd"
            elif pred_path.is_dir():
                rrd_out = pred_path / "recon_metrics" / f"{scene_id_r}.rrd"
            else:
                rrd_out = pred_path.parent / f"{scene_id_r}_recon_metrics.rrd"
        rrd_out = Path(rrd_out)

        export_recon_metrics_rrd(
            rrd_path=rrd_out,
            frames=frames,
            depth_pred=depth_pred,
            intr_pred=intr_pred,
            ext_pred=ext_pred,
            gt_ext=gt_ext,
            gt_depths=gt_depths_rrd,
            gt_pts=gt_pts_rrd,
            gt_colors=gt_cols_rrd,
            pred_pts=pr_pts_rrd,
            pred_colors=pr_cols_rrd,
            r_align=r_a,
            t_align=t_a,
            s_align=s_a,
            ref_centers=ref_c,
            est_centers=est_c,
            depth_cfg=dict(dcfg),
            max_points_gt=int(pr_opts["max_points_gt"]),
            max_points_pred=int(pr_opts["max_points_pred"]),
            subsample_seed=int(pr_opts["seed"]),
            diff_vmax_m=float(pr_opts["diff_vmax_m"]),
            progress=progress,
        )
        results["plot_rrd_path"] = str(rrd_out.resolve())

    if plot_charts_enabled(cfg):
        scene_id = scene_dir.name
        ch_opts = plot_charts_options(cfg)
        charts_out = ch_opts["path"]
        if not charts_out:
            if output_base is not None:
                charts_out = output_base / "plots"
            elif pred_path.is_dir():
                charts_out = pred_path / "recon_metrics" / "charts" / scene_id
            else:
                charts_out = pred_path.parent / f"{scene_id}_recon_charts"
        charts_out = Path(charts_out)
        dcfg_ch = cfg.get("depth") or {}
        gt_for_charts: list[np.ndarray] | None = None
        if not skip_depth:
            gt_for_charts = gt_depths
        wss = tcfg.get("warp_source_samples")
        tae_dev_ch = tcfg.get("device")
        if tae_dev_ch is None:
            tae_dev_ch = cfg.get("device", "auto")
        tae_kw = {
            "min_depth": float(tcfg.get("min_depth", 1e-4)),
            "warp_source_samples": int(wss) if wss is not None else None,
            "warp_sample_seed": int(tcfg.get("warp_sample_seed", 0)),
            "device": str(tae_dev_ch),
        }
        say("[recon_metrics] Plot charts (matplotlib) …")
        pcfg_ch = cfg.get("pose_error") or {}
        written = save_metric_charts(
            charts_out,
            scene_id=scene_id,
            skip_tae=skip_tae,
            skip_depth=skip_depth,
            skip_odometry=skip_odom,
            depth_pred=depth_pred,
            intr_pred=intr_pred,
            ext_pred=ext_pred,
            gt_ext_w2c=gt_ext,
            gt_depths=gt_for_charts,
            dcfg=dict(dcfg_ch),
            tae_m=None if skip_tae else tae_metric,
            tae_window_size=int(ch_opts["tae_window_size"]),
            tae_kw=tae_kw,
            ref_centers=ref_c,
            est_centers=est_c,
            r_a=r_a,
            t_a=t_a,
            s_a=s_a,
            rpe_delta=int(pcfg_ch.get("rpe_delta", 1)),
            verbose=verbose,
        )
        if written:
            results["plot_charts_dir"] = str(charts_out.resolve())
        elif verbose:
            say(
                "[recon_metrics] plot_charts: no files written (try without "
                "--skip-tae / --skip-depth / --skip-odometry, or install matplotlib)"
            )

    if output_base is not None:
        output_base = output_base.resolve()
        metrics_dir = output_base / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        results["output_dir"] = str(output_base)
        results["metrics_dir"] = str(metrics_dir)
        with (metrics_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        with (metrics_dir / "metrics.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(results, f, default_flow_style=False, sort_keys=False)

    say("[recon_metrics] Done.")
    return results
