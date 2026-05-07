"""Save matplotlib charts (TAE sequence, depth per frame, alignment error)."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


def _safe_tight_layout() -> None:
    """``tight_layout`` can fail on some matplotlib + NumPy combinations."""
    try:
        import matplotlib.pyplot as plt

        plt.tight_layout()
    except Exception:
        pass


def _add_mean_rmse_hlines(
    ax,
    mean_v: float,
    rmse_v: float,
    *,
    mean_style: str = "--",
    rmse_style: str = ":",
    zorder: float = 2.5,
) -> None:
    """Reference lines: population mean and RMSE of the same series (constant vs time curve)."""
    if math.isfinite(mean_v):
        ax.axhline(
            mean_v,
            color="k",
            linestyle=mean_style,
            linewidth=1.2,
            alpha=0.85,
            label=f"Mean = {mean_v:.5g}",
            zorder=zorder,
        )
    if math.isfinite(rmse_v):
        ax.axhline(
            rmse_v,
            color="0.35",
            linestyle=rmse_style,
            linewidth=1.2,
            alpha=0.9,
            label=f"RMSE = {rmse_v:.5g}",
            zorder=zorder,
        )


def _batch_mean_scalar(
    batch_output_dir: Path,
    scene_ids: list[str],
    key: str,
) -> float | None:
    """Mean of ``key`` from ``<sid>/metrics/metrics.json`` over scenes that have it."""
    vals: list[float] = []
    root = Path(batch_output_dir)
    for sid in scene_ids:
        p = root / sid / "metrics" / "metrics.json"
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            v = data.get(key)
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                fv = float(v)
                if math.isfinite(fv):
                    vals.append(fv)
        except (json.JSONDecodeError, OSError):
            continue
    if not vals:
        return None
    return float(np.mean(vals))


def _batch_rms_scalar(
    batch_output_dir: Path,
    scene_ids: list[str],
    key: str,
) -> float | None:
    """RMS of ``key`` across episodes (sqrt of mean of per-episode squared values)."""
    vals: list[float] = []
    root = Path(batch_output_dir)
    for sid in scene_ids:
        p = root / sid / "metrics" / "metrics.json"
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            v = data.get(key)
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                fv = float(v)
                if math.isfinite(fv):
                    vals.append(fv)
        except (json.JSONDecodeError, OSError):
            continue
    if not vals:
        return None
    a = np.asarray(vals, dtype=np.float64)
    return float(np.sqrt(np.mean(a**2)))


def plot_charts_enabled(cfg: dict[str, Any]) -> bool:
    v = cfg.get("plot_charts")
    if v is None:
        v = cfg.get("plot-charts")
    if v is True:
        return True
    if isinstance(v, dict):
        return bool(v.get("enable", v.get("enabled", True)))
    return False


def plot_charts_options(cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": None,
        "tae_window_size": 2,
    }
    v = cfg.get("plot_charts")
    if v is None:
        v = cfg.get("plot-charts")
    chart_tae_k: int | None = None
    if isinstance(v, dict):
        if "path" in v:
            out["path"] = v["path"]
        if "charts_dir" in v:
            out["path"] = v["charts_dir"]
        if "tae_window_size" in v:
            chart_tae_k = int(v["tae_window_size"])
    p = cfg.get("plot_charts_dir") or cfg.get("plot-charts-dir")
    if p:
        out["path"] = p
    if chart_tae_k is not None:
        out["tae_window_size"] = chart_tae_k
    else:
        tae_ws = (cfg.get("tae") or {}).get("tae_window_size")
        if tae_ws is not None:
            out["tae_window_size"] = int(tae_ws)
    return out


def batch_mean_plot_options(cfg: dict[str, Any]) -> dict[str, Any]:
    """Options under ``batch_overall`` for episode-mean curves (batch + ``plot_charts``)."""
    raw = cfg.get("batch_overall") or {}
    mv = raw.get("min_videos_for_mean", raw.get("min_episodes_for_mean", 3))
    return {
        "min_videos_for_mean": max(1, int(mv)),
        "write_mean_plots": bool(raw.get("write_mean_plots", True)),
    }


def _mean_curve_xy(
    series: list[dict[int, float]],
    min_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices x with ≥ ``min_count`` values; y = mean; third return = counts."""
    keys: set[int] = set()
    for d in series:
        keys |= set(d.keys())
    xs: list[int] = []
    ys: list[float] = []
    ns: list[int] = []
    for k in sorted(keys):
        vals = [float(d[k]) for d in series if k in d and math.isfinite(d[k])]
        if len(vals) >= min_count:
            xs.append(k)
            ys.append(float(np.mean(vals)))
            ns.append(len(vals))
    return (
        np.asarray(xs, dtype=np.int64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(ns, dtype=np.int64),
    )


def _read_alignment_csv(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                k = int(float(row["frame_index"]))
                v = float(row["error_m"])
                if math.isfinite(v):
                    out[k] = v
            except (KeyError, ValueError):
                continue
    return out


def _read_depth_csv(path: Path) -> tuple[dict[int, float], dict[int, float]]:
    absrel_m: dict[int, float] = {}
    delta_m: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                k = int(float(row["frame_index"]))
                a = float(row["absrel"])
                d = float(row["delta_1"])
                if math.isfinite(a):
                    absrel_m[k] = a
                if math.isfinite(d):
                    delta_m[k] = d
            except (KeyError, ValueError):
                continue
    return absrel_m, delta_m


def _read_tae_csv(path: Path) -> tuple[dict[int, float], dict[int, float]]:
    """
    Per-window index (``frame_index`` column) maps to TAE and optionally
    ``window_center`` for metadata.
    """
    tae_m: dict[int, float] = {}
    xc_m: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                k = int(float(row["frame_index"]))
                raw_tae = row.get("tae", "").strip()
                if raw_tae == "":
                    continue
                v = float(raw_tae)
                if not math.isfinite(v):
                    continue
                tae_m[k] = v
                if "window_center" in row and row["window_center"].strip():
                    xc_m[k] = float(row["window_center"])
            except (KeyError, ValueError):
                continue
    return tae_m, xc_m


def _read_absrot_csv(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                k = int(float(row["frame_index"]))
                v = float(row["error_deg"])
                if math.isfinite(v):
                    out[k] = v
            except (KeyError, ValueError):
                continue
    return out


def _read_rpe_rot_csv(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                k = int(float(row["frame_i"]))
                v = float(row["rot_error_deg"])
                if math.isfinite(v):
                    out[k] = v
            except (KeyError, ValueError):
                continue
    return out


def _read_rpe_trans_csv(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                k = int(float(row["frame_i"]))
                v = float(row["trans_error_m"])
                if math.isfinite(v):
                    out[k] = v
            except (KeyError, ValueError):
                continue
    return out


def save_batch_mean_curves(
    batch_output_dir: Path,
    scene_ids_ok: list[str],
    *,
    min_videos: int,
    verbose: bool = True,
) -> list[str]:
    """
    Average per-frame (or per-window for TAE) series across episodes.

    Reads ``<batch_output_dir>/<scene_id>/plots/*.csv`` produced by
    ``save_metric_charts``. For each index ``k``, the mean is taken only if at
    least ``min_videos`` episodes provide a finite value at ``k``. Writes
    ``overall/plots/mean_*.csv`` and ``mean_*.png`` under ``batch_output_dir``.
    """
    out_dir = Path(batch_output_dir) / "overall" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def say(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    if not scene_ids_ok:
        return written

    # --- Alignment ---
    align_series: list[dict[int, float]] = []
    for sid in scene_ids_ok:
        p = Path(batch_output_dir) / sid / "plots" / "alignment_translation_error.csv"
        if p.is_file():
            align_series.append(_read_alignment_csv(p))
    if align_series:
        x_a, y_a, n_a = _mean_curve_xy(align_series, min_videos)
        if x_a.size > 0:
            csv_path = out_dir / "mean_alignment_translation_error.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["frame_index", "mean_error_m", "n_videos"])
                for i in range(len(x_a)):
                    w.writerow([int(x_a[i]), f"{y_a[i]:.8f}", int(n_a[i])])
            written.append(csv_path.name)
            say(f"[recon_metrics] Wrote {csv_path}")
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.plot(x_a, y_a, linewidth=1.5, color="C2", label="Mean error (curve)")
                bm = _batch_mean_scalar(batch_output_dir, scene_ids_ok, "ate_mean")
                br = _batch_mean_scalar(batch_output_dir, scene_ids_ok, "ate_rmse")
                if bm is not None and br is not None:
                    _add_mean_rmse_hlines(ax, bm, br)
                ax.set_xlabel("Frame index")
                ax.set_ylabel("Mean ‖t_align(est) − gt‖ (m)")
                ax.set_title(
                    f"Mean ATE translation error (≥{min_videos} videos); "
                    f"hlines = avg. episode ate_mean / ate_rmse"
                )
                ax.grid(True, alpha=0.3)
                h, lab = ax.get_legend_handles_labels()
                ax.legend(h, lab, loc="best", fontsize=8)
                _safe_tight_layout()
                png_path = out_dir / "mean_alignment_translation_error.png"
                plt.savefig(png_path, dpi=150)
                plt.close()
                written.append(png_path.name)
                say(f"[recon_metrics] Wrote {png_path}")
            except ImportError:
                say("[recon_metrics] matplotlib not installed; skipped mean_alignment_translation_error.png")
            except Exception as exc:
                say(
                    f"[recon_metrics] skipped mean_alignment_translation_error.png ({type(exc).__name__}: {exc})"
                )

    # --- Depth ---
    abs_series: list[dict[int, float]] = []
    del_series: list[dict[int, float]] = []
    for sid in scene_ids_ok:
        p = Path(batch_output_dir) / sid / "plots" / "depth_per_frame.csv"
        if p.is_file():
            ab, de = _read_depth_csv(p)
            abs_series.append(ab)
            del_series.append(de)
    if abs_series and del_series:
        xa, ya, na = _mean_curve_xy(abs_series, min_videos)
        xd, yd, nd = _mean_curve_xy(del_series, min_videos)
        common = np.intersect1d(xa, xd, assume_unique=True)
        if common.size > 0:
            idx_a = {int(v): i for i, v in enumerate(xa)}
            idx_d = {int(v): i for i, v in enumerate(xd)}
            y_ab = np.asarray([ya[idx_a[int(k)]] for k in common])
            y_de = np.asarray([yd[idx_d[int(k)]] for k in common])
            n_both = np.minimum(
                np.asarray([na[idx_a[int(k)]] for k in common]),
                np.asarray([nd[idx_d[int(k)]] for k in common]),
            )
            csv_path = out_dir / "mean_depth_per_frame.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["frame_index", "mean_absrel", "mean_delta_1", "n_videos"])
                for i, k in enumerate(common):
                    w.writerow(
                        [
                            int(k),
                            f"{y_ab[i]:.8f}",
                            f"{y_de[i]:.8f}",
                            int(n_both[i]),
                        ]
                    )
            written.append(csv_path.name)
            say(f"[recon_metrics] Wrote {csv_path}")
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax1 = plt.subplots(figsize=(10, 4))
                ax1.plot(common, y_ab, color="C0", linewidth=1.5, label="Mean AbsRel (curve)")
                b_abs_m = _batch_mean_scalar(batch_output_dir, scene_ids_ok, "absrel")
                b_abs_r = _batch_rms_scalar(batch_output_dir, scene_ids_ok, "absrel")
                if b_abs_m is not None and b_abs_r is not None:
                    _add_mean_rmse_hlines(ax1, b_abs_m, b_abs_r)
                ax1.set_xlabel("Frame index")
                ax1.set_ylabel("AbsRel", color="C0")
                ax1.tick_params(axis="y", labelcolor="C0")
                ax2 = ax1.twinx()
                ax2.plot(common, y_de, color="C1", linewidth=1.5, alpha=0.85, label="Mean δ1 (curve)")
                b_d_m = _batch_mean_scalar(batch_output_dir, scene_ids_ok, "delta_1")
                b_d_r = _batch_rms_scalar(batch_output_dir, scene_ids_ok, "delta_1")
                if b_d_m is not None and b_d_r is not None:
                    _add_mean_rmse_hlines(ax2, b_d_m, b_d_r)
                ax2.set_ylabel("δ1 accuracy", color="C1")
                ax2.set_ylim(0.0, 1.05)
                ax2.tick_params(axis="y", labelcolor="C1")
                ax1.set_title(
                    f"Mean depth metrics (≥{min_videos} videos); "
                    f"hlines = avg / RMS across episode scalars"
                )
                ax1.grid(True, alpha=0.3)
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=7)
                _safe_tight_layout()
                png_path = out_dir / "mean_depth_per_frame.png"
                plt.savefig(png_path, dpi=150)
                plt.close()
                written.append(png_path.name)
                say(f"[recon_metrics] Wrote {png_path}")
            except ImportError:
                say("[recon_metrics] matplotlib not installed; skipped mean_depth_per_frame.png")
            except Exception as exc:
                say(
                    f"[recon_metrics] skipped mean_depth_per_frame.png ({type(exc).__name__}: {exc})"
                )

    # --- TAE sliding window (index = window order / frame_index in CSV) ---
    tae_series: list[dict[int, float]] = []
    xc_series: list[dict[int, float]] = []
    for sid in scene_ids_ok:
        p = Path(batch_output_dir) / sid / "plots" / "tae_sequence.csv"
        if p.is_file():
            tae_d, xc_d = _read_tae_csv(p)
            tae_series.append(tae_d)
            xc_series.append(xc_d)
    if tae_series:
        x_t, y_t, n_t = _mean_curve_xy(tae_series, min_videos)
        if x_t.size > 0:
            x_plot_list: list[float] = []
            for k in x_t:
                centers: list[float] = []
                for ti, d in enumerate(tae_series):
                    if int(k) not in d:
                        continue
                    xcm = xc_series[ti]
                    if int(k) in xcm:
                        centers.append(float(xcm[int(k)]))
                if len(centers) >= min_videos:
                    x_plot_list.append(float(np.mean(centers)))
                else:
                    x_plot_list.append(float(k))
            x_plot = np.asarray(x_plot_list, dtype=np.float64)
            csv_path = out_dir / "mean_tae_sequence.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["window_index", "mean_window_center", "mean_tae", "n_videos"])
                for i in range(len(x_t)):
                    w.writerow(
                        [
                            int(x_t[i]),
                            f"{x_plot[i]:.4f}",
                            f"{y_t[i]:.8f}",
                            int(n_t[i]),
                        ]
                    )
            written.append(csv_path.name)
            say(f"[recon_metrics] Wrote {csv_path}")
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(x_plot, y_t, linewidth=1.5, label="Mean TAE (curve)")
                tm = _batch_mean_scalar(batch_output_dir, scene_ids_ok, "tae")
                tr = _batch_rms_scalar(batch_output_dir, scene_ids_ok, "tae")
                if tm is not None and tr is not None:
                    _add_mean_rmse_hlines(ax, tm, tr)
                ax.set_xlabel("Mean window center (frame units)")
                ax.set_ylabel("Mean TAE")
                ax.set_title(
                    f"Mean TAE (≥{min_videos} videos); hlines = avg / RMS of episode TAE"
                )
                ax.grid(True, alpha=0.3)
                h, lab = ax.get_legend_handles_labels()
                ax.legend(h, lab, loc="best", fontsize=8)
                _safe_tight_layout()
                png_path = out_dir / "mean_tae_sequence.png"
                plt.savefig(png_path, dpi=150)
                plt.close()
                written.append(png_path.name)
                say(f"[recon_metrics] Wrote {png_path}")
            except ImportError:
                say("[recon_metrics] matplotlib not installed; skipped mean_tae_sequence.png")
            except Exception as exc:
                say(
                    f"[recon_metrics] skipped mean_tae_sequence.png ({type(exc).__name__}: {exc})"
                )

    # --- Mean absolute rotation error ---
    ar_series: list[dict[int, float]] = []
    for sid in scene_ids_ok:
        p = Path(batch_output_dir) / sid / "plots" / "absolute_rotation_error_deg.csv"
        if p.is_file():
            ar_series.append(_read_absrot_csv(p))
    if ar_series:
        x_ar, y_ar, n_ar = _mean_curve_xy(ar_series, min_videos)
        if x_ar.size > 0:
            csv_path = out_dir / "mean_absolute_rotation_error_deg.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["frame_index", "mean_error_deg", "n_videos"])
                for i in range(len(x_ar)):
                    w.writerow([int(x_ar[i]), f"{y_ar[i]:.8f}", int(n_ar[i])])
            written.append(csv_path.name)
            say(f"[recon_metrics] Wrote {csv_path}")
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 3.5))
                ax.plot(x_ar, y_ar, linewidth=1.5, color="C3", label="Mean ARE (curve)")
                am = _batch_mean_scalar(batch_output_dir, scene_ids_ok, "are_mean_deg")
                ars = _batch_mean_scalar(batch_output_dir, scene_ids_ok, "are_rmse_deg")
                if am is not None and ars is not None:
                    _add_mean_rmse_hlines(ax, am, ars)
                ax.set_xlabel("Frame index")
                ax.set_ylabel("Mean absolute rotation error (deg)")
                ax.set_title(
                    f"Mean ARE (≥{min_videos} videos); "
                    f"hlines = avg episode mean / RMSE"
                )
                ax.grid(True, alpha=0.3)
                h, lab = ax.get_legend_handles_labels()
                ax.legend(h, lab, loc="best", fontsize=8)
                _safe_tight_layout()
                png_path = out_dir / "mean_absolute_rotation_error_deg.png"
                plt.savefig(png_path, dpi=150)
                plt.close()
                written.append(png_path.name)
                say(f"[recon_metrics] Wrote {png_path}")
            except ImportError:
                say(
                    "[recon_metrics] matplotlib not installed; skipped mean_absolute_rotation_error_deg.png"
                )
            except Exception as exc:
                say(
                    f"[recon_metrics] skipped mean_absolute_rotation_error_deg.png ({type(exc).__name__}: {exc})"
                )

    # --- Mean RPE (rotation + translation), keyed by pair start frame_i ---
    rpr_series: list[dict[int, float]] = []
    rpt_series: list[dict[int, float]] = []
    for sid in scene_ids_ok:
        pr = Path(batch_output_dir) / sid / "plots" / "rpe_rotation_error_deg.csv"
        pt = Path(batch_output_dir) / sid / "plots" / "rpe_translation_error_m.csv"
        if pr.is_file() and pt.is_file():
            rpr_series.append(_read_rpe_rot_csv(pr))
            rpt_series.append(_read_rpe_trans_csv(pt))
    if rpr_series and rpt_series:
        xr, yr, nr = _mean_curve_xy(rpr_series, min_videos)
        xt, yt, nt = _mean_curve_xy(rpt_series, min_videos)
        common = np.intersect1d(xr, xt, assume_unique=True)
        if common.size > 0:
            ir = {int(v): i for i, v in enumerate(xr)}
            it = {int(v): i for i, v in enumerate(xt)}
            y_r = np.asarray([yr[ir[int(k)]] for k in common])
            y_t = np.asarray([yt[it[int(k)]] for k in common])
            n_b = np.minimum(
                np.asarray([nr[ir[int(k)]] for k in common]),
                np.asarray([nt[it[int(k)]] for k in common]),
            )
            csv_path = out_dir / "mean_rpe_errors.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["frame_i", "mean_rot_error_deg", "mean_trans_error_m", "n_videos"])
                for i, k in enumerate(common):
                    w.writerow(
                        [
                            int(k),
                            f"{y_r[i]:.8f}",
                            f"{y_t[i]:.8f}",
                            int(n_b[i]),
                        ]
                    )
            written.append(csv_path.name)
            say(f"[recon_metrics] Wrote {csv_path}")
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax1 = plt.subplots(figsize=(10, 4))
                ax1.plot(common, y_r, color="C4", linewidth=1.5, label="Mean RPE rot (curve)")
                rm_ = _batch_mean_scalar(batch_output_dir, scene_ids_ok, "rpe_rot_mean_deg")
                rr_ = _batch_mean_scalar(batch_output_dir, scene_ids_ok, "rpe_rot_rmse_deg")
                if rm_ is not None and rr_ is not None:
                    _add_mean_rmse_hlines(ax1, rm_, rr_)
                ax1.set_xlabel("Frame i (pair start)")
                ax1.set_ylabel("Rotation error (deg)", color="C4")
                ax1.tick_params(axis="y", labelcolor="C4")
                ax2 = ax1.twinx()
                ax2.plot(common, y_t, color="C5", linewidth=1.5, alpha=0.85, label="Mean RPE trans (curve)")
                tm_ = _batch_mean_scalar(batch_output_dir, scene_ids_ok, "rpe_trans_mean_m")
                tr_ = _batch_mean_scalar(batch_output_dir, scene_ids_ok, "rpe_trans_rmse_m")
                if tm_ is not None and tr_ is not None:
                    _add_mean_rmse_hlines(ax2, tm_, tr_)
                ax2.set_ylabel("Translation error (m)", color="C5")
                ax2.tick_params(axis="y", labelcolor="C5")
                ax1.set_title(
                    f"Mean RPE (≥{min_videos} videos); hlines = avg episode mean / RMSE"
                )
                ax1.grid(True, alpha=0.3)
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=7)
                _safe_tight_layout()
                png_path = out_dir / "mean_rpe_errors.png"
                plt.savefig(png_path, dpi=150)
                plt.close()
                written.append(png_path.name)
                say(f"[recon_metrics] Wrote {png_path}")
            except ImportError:
                say("[recon_metrics] matplotlib not installed; skipped mean_rpe_errors.png")
            except Exception as exc:
                say(
                    f"[recon_metrics] skipped mean_rpe_errors.png ({type(exc).__name__}: {exc})"
                )

    return written


def save_metric_charts(
    out_dir: Path,
    *,
    scene_id: str,
    skip_tae: bool,
    skip_depth: bool,
    skip_odometry: bool,
    depth_pred: np.ndarray,
    intr_pred: np.ndarray,
    ext_pred: np.ndarray,
    gt_ext_w2c: np.ndarray,
    gt_depths: list[np.ndarray] | None,
    dcfg: dict[str, Any],
    tae_m: ModuleType | None,
    tae_window_size: int,
    tae_kw: dict[str, Any],
    ref_centers: np.ndarray,
    est_centers: np.ndarray,
    r_a: np.ndarray,
    t_a: np.ndarray,
    s_a: float,
    rpe_delta: int = 1,
    verbose: bool,
) -> list[str]:
    """
    Write CSV + PNG under ``out_dir``. Returns list of created artifact basenames.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def say(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    # --- TAE sliding window (same layout as metrics/tae_metric.py save-results) ---
    if not skip_tae and tae_m is not None:
        depths = np.asarray(depth_pred, dtype=np.float64)
        if depths.ndim == 4:
            depths = depths.squeeze(-1)
        K = max(2, int(tae_window_size))
        T = depths.shape[0]
        if T >= K:
            tae_seq = tae_m.tae_sequence(
                depths,
                intr_pred,
                ext_pred,
                window_size=K,
                progress=False,
                **tae_kw,
            )
            if tae_seq.size > 0:
                window_center = np.arange(len(tae_seq)) + (K - 1) / 2.0
                csv_path = out_dir / "tae_sequence.csv"
                with csv_path.open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["window_center", "frame_index", "tae"])
                    for i, (xc, val) in enumerate(zip(window_center, tae_seq)):
                        w.writerow([
                            f"{xc:.4f}",
                            i,
                            f"{val:.8f}" if np.isfinite(val) else "",
                        ])
                written.append(csv_path.name)
                say(f"[recon_metrics] Wrote {csv_path}")
                png_path = out_dir / "tae_sequence.png"
                try:
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(window_center, tae_seq, linewidth=1, label="TAE (window)")
                    finite = tae_seq[np.isfinite(tae_seq)]
                    if finite.size > 0:
                        _add_mean_rmse_hlines(
                            ax,
                            float(np.mean(finite)),
                            float(np.sqrt(np.mean(finite**2))),
                        )
                    ax.set_xlabel("Frame (window center)")
                    ax.set_ylabel("TAE")
                    ax.set_title(f"TAE over sliding windows (K={K}) — {scene_id}")
                    ax.grid(True, alpha=0.3)
                    h, lab = ax.get_legend_handles_labels()
                    ax.legend(h, lab, loc="best", fontsize=8)
                    _safe_tight_layout()
                    plt.savefig(png_path, dpi=150)
                    plt.close()
                    written.append(png_path.name)
                    say(f"[recon_metrics] Wrote {png_path}")
                except ImportError:
                    say("[recon_metrics] matplotlib not installed; skipped tae_sequence.png")

    # --- Depth AbsRel / δ1 per frame ---
    if not skip_depth and gt_depths is not None and len(gt_depths) > 0:
        from .depth_metrics import per_frame_depth_metrics

        fidx, absrels, deltas = per_frame_depth_metrics(
            depth_pred,
            gt_depths,
            min_depth=float(dcfg.get("min_depth", 1e-3)),
            max_depth=float(dcfg.get("max_depth", 80.0)),
            scale_align=dcfg.get("scale_align", "median"),
            delta_thresh=float(dcfg.get("delta_thresh", 1.25)),
            eps=float(dcfg.get("eps", 1e-8)),
        )
        if fidx.size > 0:
            csv_path = out_dir / "depth_per_frame.csv"
            with csv_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["frame_index", "absrel", "delta_1"])
                for i in range(len(fidx)):
                    w.writerow([int(fidx[i]), f"{absrels[i]:.8f}", f"{deltas[i]:.8f}"])
            written.append(csv_path.name)
            say(f"[recon_metrics] Wrote {csv_path}")
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax1 = plt.subplots(figsize=(10, 4))
                ax1.plot(fidx, absrels, color="C0", linewidth=1, label="AbsRel")
                _add_mean_rmse_hlines(
                    ax1,
                    float(np.mean(absrels)),
                    float(np.sqrt(np.mean(absrels**2))),
                )
                ax1.set_xlabel("Frame index")
                ax1.set_ylabel("AbsRel", color="C0")
                ax1.tick_params(axis="y", labelcolor="C0")
                ax2 = ax1.twinx()
                ax2.plot(fidx, deltas, color="C1", linewidth=1, alpha=0.85, label="δ1 accuracy")
                _add_mean_rmse_hlines(
                    ax2,
                    float(np.mean(deltas)),
                    float(np.sqrt(np.mean(deltas**2))),
                )
                ax2.set_ylabel("δ1 accuracy", color="C1")
                ax2.set_ylim(0.0, 1.05)
                ax2.tick_params(axis="y", labelcolor="C1")
                ax1.set_title(f"Per-frame depth vs GT — {scene_id}")
                ax1.grid(True, alpha=0.3)
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=7)
                _safe_tight_layout()
                png_path = out_dir / "depth_per_frame.png"
                plt.savefig(png_path, dpi=150)
                plt.close()
                written.append(png_path.name)
                say(f"[recon_metrics] Wrote {png_path}")
            except ImportError:
                say("[recon_metrics] matplotlib not installed; skipped depth_per_frame.png")

    # --- Camera translation error after Sim(3) ---
    if not skip_odometry and ref_centers.shape[0] == est_centers.shape[0] and ref_centers.shape[0] > 0:
        from .geometry import apply_sim3_points

        est_a = apply_sim3_points(est_centers, r_a, t_a, float(s_a))
        err = np.linalg.norm(est_a - ref_centers, axis=1)
        frames = np.arange(len(err), dtype=np.int32)
        csv_path = out_dir / "alignment_translation_error.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_index", "error_m"])
            for i in range(len(err)):
                w.writerow([int(frames[i]), f"{float(err[i]):.8f}"])
        written.append(csv_path.name)
        say(f"[recon_metrics] Wrote {csv_path}")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(frames, err, linewidth=1, color="C2", label="ATE transl. (per frame)")
            _add_mean_rmse_hlines(
                ax,
                float(np.mean(err)),
                float(np.sqrt(np.mean(err**2))),
            )
            ax.set_xlabel("Frame index")
            ax.set_ylabel("‖t_align(est) − gt‖ (m)")
            ax.set_title(f"ATE translation after Sim(3) — {scene_id}")
            ax.grid(True, alpha=0.3)
            h, lab = ax.get_legend_handles_labels()
            ax.legend(h, lab, loc="best", fontsize=8)
            _safe_tight_layout()
            png_path = out_dir / "alignment_translation_error.png"
            plt.savefig(png_path, dpi=150)
            plt.close()
            written.append(png_path.name)
            say(f"[recon_metrics] Wrote {png_path}")
        except ImportError:
            say("[recon_metrics] matplotlib not installed; skipped alignment_translation_error.png")

    # --- Absolute rotation error & RPE (Sim(3)-aligned pred vs GT w2c) ---
    if not skip_odometry and ext_pred.shape[0] > 0:
        from .geometry import (
            absolute_rotation_error_per_frame_deg,
            align_w2c_extrinsics_batch,
            relative_pose_errors,
        )

        g = np.asarray(gt_ext_w2c, dtype=np.float64)
        e = np.asarray(ext_pred, dtype=np.float64)
        n = min(g.shape[0], e.shape[0])
        g = g[:n]
        e = e[:n]
        ext_al = align_w2c_extrinsics_batch(e, r_a, t_a, float(s_a))
        are = absolute_rotation_error_per_frame_deg(g, ext_al)
        frames_ar = np.arange(len(are), dtype=np.int32)
        csv_ar = out_dir / "absolute_rotation_error_deg.csv"
        with csv_ar.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_index", "error_deg"])
            for i in range(len(are)):
                w.writerow([int(frames_ar[i]), f"{float(are[i]):.8f}"])
        written.append(csv_ar.name)
        say(f"[recon_metrics] Wrote {csv_ar}")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(frames_ar, are, linewidth=1, color="C3", label="ARE (per frame)")
            _add_mean_rmse_hlines(
                ax,
                float(np.mean(are)),
                float(np.sqrt(np.mean(are**2))),
            )
            ax.set_xlabel("Frame index")
            ax.set_ylabel("Absolute rotation error (deg)")
            ax.set_title(f"Geodesic ∠(R_gt, R_est) after Sim(3) — {scene_id}")
            ax.grid(True, alpha=0.3)
            h, lab = ax.get_legend_handles_labels()
            ax.legend(h, lab, loc="best", fontsize=8)
            _safe_tight_layout()
            png_path = out_dir / "absolute_rotation_error_deg.png"
            plt.savefig(png_path, dpi=150)
            plt.close()
            written.append(png_path.name)
            say(f"[recon_metrics] Wrote {png_path}")
        except ImportError:
            say("[recon_metrics] matplotlib not installed; skipped absolute_rotation_error_deg.png")
        except Exception as exc:
            say(
                f"[recon_metrics] skipped absolute_rotation_error_deg.png ({type(exc).__name__}: {exc})"
            )

        d = max(1, int(rpe_delta))
        fi, rpe_r, rpe_t = relative_pose_errors(g, ext_al, delta=d)
        if fi.size > 0:
            csv_rpe_r = out_dir / "rpe_rotation_error_deg.csv"
            with csv_rpe_r.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["frame_i", "rpe_delta", "rot_error_deg"])
                for i in range(len(fi)):
                    w.writerow([int(fi[i]), d, f"{float(rpe_r[i]):.8f}"])
            written.append(csv_rpe_r.name)
            say(f"[recon_metrics] Wrote {csv_rpe_r}")
            csv_rpe_t = out_dir / "rpe_translation_error_m.csv"
            with csv_rpe_t.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["frame_i", "rpe_delta", "trans_error_m"])
                for i in range(len(fi)):
                    w.writerow([int(fi[i]), d, f"{float(rpe_t[i]):.8f}"])
            written.append(csv_rpe_t.name)
            say(f"[recon_metrics] Wrote {csv_rpe_t}")
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax1 = plt.subplots(figsize=(10, 4))
                ax1.plot(fi, rpe_r, color="C4", linewidth=1, label="RPE rotation")
                _add_mean_rmse_hlines(
                    ax1,
                    float(np.mean(rpe_r)),
                    float(np.sqrt(np.mean(rpe_r**2))),
                )
                ax1.set_xlabel("Frame i (pair i→i+Δ)")
                ax1.set_ylabel("Rotation error (deg)", color="C4")
                ax1.tick_params(axis="y", labelcolor="C4")
                ax2 = ax1.twinx()
                ax2.plot(fi, rpe_t, color="C5", linewidth=1, alpha=0.85, label="RPE translation")
                _add_mean_rmse_hlines(
                    ax2,
                    float(np.mean(rpe_t)),
                    float(np.sqrt(np.mean(rpe_t**2))),
                )
                ax2.set_ylabel("Translation error (m)", color="C5")
                ax2.tick_params(axis="y", labelcolor="C5")
                ax1.set_title(
                    f"Relative pose error (Δ={d}) vs GT — {scene_id}"
                )
                ax1.grid(True, alpha=0.3)
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1 + h2, l1 + l2, loc="best", fontsize=7)
                _safe_tight_layout()
                png_path = out_dir / "rpe_errors.png"
                plt.savefig(png_path, dpi=150)
                plt.close()
                written.append(png_path.name)
                say(f"[recon_metrics] Wrote {png_path}")
            except ImportError:
                say("[recon_metrics] matplotlib not installed; skipped rpe_errors.png")
            except Exception as exc:
                say(f"[recon_metrics] skipped rpe_errors.png ({type(exc).__name__}: {exc})")

    return written
