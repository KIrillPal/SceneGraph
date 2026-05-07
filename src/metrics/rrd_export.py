"""Rerun .rrd export: GT vs prediction point clouds, camera trajectories, depth difference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .arkit_frames import FrameSample, read_pincam
from .depth_metrics import resize_depth_to
from .geometry import apply_sim3_points, extrinsic_w2c_to_4x4, align_w2c_extrinsics_batch
from .ply_loader import load_ply_vertices


def _require_rerun():
    try:
        import rerun as rr  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "plot_rrd requires the `rerun-sdk` package. "
            "Install with: pip install rerun-sdk"
        ) from e
    return __import__("rerun")


def _subsample_points(
    pts: np.ndarray,
    cols: np.ndarray | None,
    max_n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    n = pts.shape[0]
    if n <= max_n:
        return pts.astype(np.float32, copy=False), cols
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_n, replace=False)
    pc = pts[idx].astype(np.float32, copy=False)
    cc = cols[idx] if cols is not None else None
    return pc, cc


def _log_w2c_camera(
    rr: Any,
    entity: str,
    ext_w2c: np.ndarray,
    intrinsic: np.ndarray,
    resolution_wh: tuple[int, int],
) -> None:
    """
    Place a Pinhole camera in the parent ``world`` frame.

    ``ext_w2c`` is OpenCV-style world-to-camera: p_cam = R @ p_world + t.
    Rerun's entity transform must be camera-to-world (camera origin and axes in
    world): using (R, t) from w2c would put the frustum at ``t`` instead of at
    the true camera center ``-R.T @ t`` and flip the orientation.
    """
    E = extrinsic_w2c_to_4x4(np.asarray(ext_w2c, dtype=np.float64))
    T_c2w = np.linalg.inv(E)
    R_c2w = T_c2w[:3, :3]
    t_c2w = T_c2w[:3, 3]
    rotation = Rotation.from_matrix(R_c2w)
    rr.log(
        entity,
        rr.Transform3D(
            translation=t_c2w.astype(np.float32),
            quaternion=rr.Quaternion(xyzw=rotation.as_quat()),
            relation=rr.TransformRelation.ParentFromChild,
        ),
    )
    rr.log(
        entity,
        rr.Pinhole(
            image_from_camera=np.asarray(intrinsic, dtype=np.float32),
            resolution=list(resolution_wh),
        ),
    )


def _depth_diff_rgb(diff_m: np.ndarray, valid: np.ndarray, vmax: float) -> np.ndarray:
    """Signed error (pred−GT) [m] → RGB (diverging): blue (−) / white (0) / red (+)."""
    x = diff_m / (vmax + 1e-12)
    x = np.clip(x, -1.0, 1.0)
    rgb = np.zeros((diff_m.shape[0], diff_m.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb[..., 2] = (np.clip(-x, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb[..., 1] = ((1.0 - np.abs(x)) * 255.0).astype(np.uint8)
    grey = np.uint8(32)
    rgb[~valid] = grey
    return rgb


def _build_blueprint(rrb: Any) -> Any:
    cam_gt = "world/camera_gt"
    cam_pr = "world/camera_pred"
    desc = """
# Reconstruction metrics (GT vs DA3)

**3D:** GT mesh vertices and predicted PLY (Sim(3) aligned to GT world) plus camera-center polylines (ARKit vs aligned prediction).

**2D:** GT RGB, GT / pred depth, signed depth error (pred−GT after median scale) on GT resolution.
""".strip()
    tabs = rrb.Tabs(
        rrb.Spatial2DView(name="GT RGB", origin=cam_gt, contents=[f"{cam_gt}/rgb"]),
        rrb.Spatial2DView(name="GT depth", origin=cam_gt, contents=[f"{cam_gt}/depth_gt"]),
        rrb.Spatial2DView(name="Pred depth", origin=cam_pr, contents=[f"{cam_pr}/depth_pred"]),
        rrb.Spatial2DView(
            name="Depth Δ",
            origin=cam_gt,
            contents=[f"{cam_gt}/depth_diff_rgb"],
        ),
        name="2D",
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D", origin="world"),
            rrb.Vertical(
                tabs,
                rrb.TextDocumentView(name="Info", origin="description"),
                row_shares=[2, 1],
            ),
        ),
    ), desc


def export_recon_metrics_rrd(
    *,
    rrd_path: Path,
    frames: list[FrameSample],
    depth_pred: np.ndarray,
    intr_pred: np.ndarray,
    ext_pred: np.ndarray,
    gt_ext: np.ndarray,
    gt_depths: list[np.ndarray],
    gt_pts: np.ndarray,
    gt_colors: np.ndarray | None,
    pred_pts: np.ndarray,
    pred_colors: np.ndarray | None,
    r_align: np.ndarray,
    t_align: np.ndarray,
    s_align: float,
    ref_centers: np.ndarray,
    est_centers: np.ndarray,
    depth_cfg: dict[str, Any],
    max_points_gt: int = 400_000,
    max_points_pred: int = 400_000,
    subsample_seed: int = 0,
    diff_vmax_m: float = 0.35,
    progress: bool = True,
) -> None:
    """Write one .rrd with static clouds + trajectories and time-varying cameras / depths."""
    rr = _require_rerun()
    import rerun.blueprint as rrb

    if not hasattr(rr, "script_add_args"):
        raise ImportError(
            "Installed `rerun` is too old or wrong package. "
            "Use: pip uninstall rerun -y && pip install rerun-sdk"
        )

    depth_pred = np.asarray(depth_pred, dtype=np.float64)
    intr_pred = np.asarray(intr_pred, dtype=np.float64)
    ext_pred = np.asarray(ext_pred, dtype=np.float64)
    gt_ext = np.asarray(gt_ext, dtype=np.float64)
    n = min(
        len(frames),
        depth_pred.shape[0],
        intr_pred.shape[0],
        ext_pred.shape[0],
        gt_ext.shape[0],
        len(gt_depths),
    )
    if n == 0:
        raise ValueError("export_recon_metrics_rrd: no frames to log")

    rrd_path = Path(rrd_path)
    rrd_path.parent.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    rr.script_add_args(parser)
    rr_args = parser.parse_args(["--save", str(rrd_path)])

    blueprint, md_desc = _build_blueprint(rrb)
    rr.script_setup(rr_args, "recon_metrics_viz", default_blueprint=blueprint)

    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    rr.log(
        "description",
        rr.TextDocument(md_desc, media_type=rr.MediaType.MARKDOWN),
        static=True,
    )

    gt_sub, gt_c_sub = _subsample_points(gt_pts, gt_colors, max_points_gt, subsample_seed)
    pr_aligned = apply_sim3_points(pred_pts, r_align, t_align, s_align)
    pr_sub, pr_c_sub = _subsample_points(pr_aligned, pred_colors, max_points_pred, subsample_seed + 1)

    rr.log(
        "world/gt_cloud",
        rr.Points3D(positions=gt_sub, colors=gt_c_sub, radii=0.01),
        static=True,
    )
    rr.log(
        "world/pred_cloud",
        rr.Points3D(positions=pr_sub, colors=pr_c_sub, radii=0.01),
        static=True,
    )

    est_aligned = apply_sim3_points(est_centers, r_align, t_align, s_align)
    rr.log(
        "world/traj_gt",
        rr.LineStrips3D([ref_centers.astype(np.float32)]),
        static=True,
    )
    rr.log(
        "world/traj_pred_aligned",
        rr.LineStrips3D([est_aligned.astype(np.float32)]),
        static=True,
    )

    ext_pred_al = align_w2c_extrinsics_batch(ext_pred[:n], r_align, t_align, s_align)

    min_depth = float(depth_cfg.get("min_depth", 1e-3))
    max_depth = float(depth_cfg.get("max_depth", 80.0))
    scale_align = depth_cfg.get("scale_align", "median")
    eps = float(depth_cfg.get("eps", 1e-8))

    from tqdm import tqdm

    for i in tqdm(
        range(n),
        desc="Rerun export",
        unit="frm",
        disable=not progress,
        ncols=100,
    ):
        rr.set_time("frame", sequence=i)

        frm = frames[i]
        gt_dep = np.asarray(gt_depths[i], dtype=np.float64)
        h_gt, w_gt = gt_dep.shape[:2]
        K_gt, _wh_pin = read_pincam(frm.pincam_path)
        wh_gt = (w_gt, h_gt)

        pr_raw = np.asarray(depth_pred[i], dtype=np.float64)
        if pr_raw.ndim == 3 and pr_raw.shape[-1] == 1:
            pr_raw = pr_raw[..., 0]
        ph, pw = pr_raw.shape[:2]
        pr = pr_raw.copy()
        if pr.shape != gt_dep.shape:
            pr = resize_depth_to(pr, (h_gt, w_gt))
        valid = (
            (gt_dep > min_depth)
            & (gt_dep < max_depth)
            & np.isfinite(gt_dep)
            & np.isfinite(pr)
            & (pr > min_depth)
        )
        pr_scaled = pr.copy()
        if scale_align == "median" and np.any(valid):
            s = float(np.median(gt_dep[valid] / (pr_scaled[valid] + eps)))
            if np.isfinite(s) and s > 1e-6:
                pr_scaled = pr_scaled * s
        elif scale_align is not None and str(scale_align).lower() not in ("none", ""):
            raise ValueError(f"export RRD: unknown scale_align {scale_align!r}")

        diff = pr_scaled - gt_dep
        diff_rgb = _depth_diff_rgb(diff, valid, diff_vmax_m)

        rgb = np.asarray(Image.open(frm.rgb_path).convert("RGB"))
        if rgb.shape[0] != h_gt or rgb.shape[1] != w_gt:
            rgb = np.asarray(Image.fromarray(rgb).resize((w_gt, h_gt), Image.Resampling.BILINEAR))

        _log_w2c_camera(rr, "world/camera_gt", gt_ext[i], K_gt, wh_gt)
        intr_i = intr_pred[i]
        _log_w2c_camera(rr, "world/camera_pred", ext_pred_al[i], intr_i, (pw, ph))

        rr.log("world/camera_gt/rgb", rr.Image(rgb).compress(jpeg_quality=88))
        rr.log("world/camera_gt/depth_gt", rr.DepthImage(gt_dep.astype(np.float32), meter=1.0))
        rr.log("world/camera_gt/depth_diff_rgb", rr.Image(diff_rgb))
        rr.log(
            "world/camera_pred/depth_pred",
            rr.DepthImage(pr_raw.astype(np.float32), meter=1.0),
        )

    rr.script_teardown(rr_args)


def plot_rrd_enabled(cfg: dict[str, Any]) -> bool:
    v = cfg.get("plot_rrd")
    if v is None:
        v = cfg.get("plot-rrd")
    if v is True:
        return True
    if isinstance(v, dict):
        return bool(v.get("enable", v.get("enabled", True)))
    return False


def plot_rrd_options(cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": None,
        "max_points_gt": 400_000,
        "max_points_pred": 400_000,
        "seed": 0,
        "diff_vmax_m": 0.35,
    }
    v = cfg.get("plot_rrd")
    if v is None:
        v = cfg.get("plot-rrd")
    if isinstance(v, dict):
        for k in out:
            if k in v:
                out[k] = v[k]
        if "rrd_path" in v:
            out["path"] = v["rrd_path"]
        if "path" in v:
            out["path"] = v["path"]
    p = cfg.get("plot_rrd_path") or cfg.get("plot-rrd-path")
    if p:
        out["path"] = p
    return out
