"""Unit tests for reconstruction metric primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# src/ on path; package directory is metrics/
_SRC_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _SRC_ROOT.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from metrics.cloud_metrics import chamfer_l2, emd_uniform_l1  # noqa: E402
from metrics.depth_metrics import aggregate_depth_metrics, delta1_accuracy, absrel_map  # noqa: E402
from metrics.geometry import apply_sim3_points, ate_rmse_sim3, umeyama_sim3  # noqa: E402
from metrics.ply_loader import load_ply_vertices  # noqa: E402


def test_umeyama_recover_sim3():
    rng = np.random.default_rng(0)
    n = 50
    src = rng.normal(size=(n, 3))
    r_true, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(r_true) < 0:
        r_true[:, 2] *= -1
    t_true = rng.normal(size=3)
    s_true = 0.7
    dst = apply_sim3_points(src, r_true, t_true, s_true)
    r, t, s = umeyama_sim3(src, dst, with_scale=True)
    recon = apply_sim3_points(src, r, t, s)
    np.testing.assert_allclose(recon, dst, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(s, s_true, rtol=1e-3)
    np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-5)


def test_ate_rmse_zero_for_identical_trajectory():
    rng = np.random.default_rng(1)
    c = rng.normal(size=(20, 3))
    rmse, r, t, s = ate_rmse_sim3(c, c, with_scale=True)
    assert rmse < 1e-9
    np.testing.assert_allclose(s, 1.0, atol=1e-5)
    np.testing.assert_allclose(t, 0.0, atol=1e-5)


def test_chamfer_identical_clouds():
    rng = np.random.default_rng(2)
    pts = rng.uniform(-1, 1, size=(400, 3))
    out = chamfer_l2(pts, pts.copy(), max_points_a=1000, max_points_b=1000, seed=0)
    assert out["chamfer_l2"] < 1e-12


def test_absrel_and_delta1_perfect_maps():
    h, w = 4, 5
    gt = np.full((h, w), 2.0)
    pred = gt.copy()
    valid = np.ones((h, w), dtype=bool)
    assert absrel_map(pred, gt, valid) == pytest.approx(0.0)
    assert delta1_accuracy(pred, gt, valid, thresh=1.25) == pytest.approx(1.0)


def test_aggregate_depth_metrics_fixed_ratio():
    pred = np.ones((2, 8, 8), dtype=np.float64) * 1.0
    gt = [np.full((8, 8), 2.0) for _ in range(2)]
    outmedian = aggregate_depth_metrics(pred, gt, scale_align="median", min_depth=0.1)
    assert outmedian["absrel"] == pytest.approx(0.0, abs=1e-7)
    outnone = aggregate_depth_metrics(pred, gt, scale_align="none", min_depth=0.1)
    assert outnone["absrel"] == pytest.approx(0.5)


def test_emd_small_identical():
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    v = emd_uniform_l1(a, a.copy(), max_points=8, seed=0)
    assert v == pytest.approx(0.0, abs=1e-9)


def test_load_arkit_mesh_ply():
    mesh = _REPO_ROOT / "data/arkitscenes_validation30_raw/raw/Validation/41069043/41069043_3dod_mesh.ply"
    if not mesh.is_file():
        pytest.skip("dataset mesh not present")
    pts, colors = load_ply_vertices(mesh)
    assert pts.shape[1] == 3 and pts.shape[0] > 1000
    assert colors is not None


def test_tae_warp_source_samples():
    from metrics import tae_metric

    T, H, W = 4, 12, 16
    depths = np.ones((T, H, W), dtype=np.float64) * 2.0
    K = np.array([[10.0, 0, 8.0], [0, 10.0, 8.0], [0, 0, 1.0]])
    intr = np.stack([K] * T)
    ext = np.zeros((T, 3, 4), dtype=np.float64)
    ext[:, 0, 0] = 1
    ext[:, 1, 1] = 1
    ext[:, 2, 2] = 1
    full = tae_metric.tae(depths, intr, ext, device="cpu")
    sub = tae_metric.tae(
        depths, intr, ext, warp_source_samples=80, warp_sample_seed=0, device="cpu"
    )
    assert np.isfinite(full) and np.isfinite(sub)


def test_batch_da3_heuristic_and_means():
    from metrics.batch import discover_batch_da3_dirs, is_da3_experiment_dir, mean_overall_metrics

    assert is_da3_experiment_dir(Path("/nonexistent")) is False
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        a = root / "111"
        b = root / "222"
        a.mkdir()
        b.mkdir()
        (a / "camera_poses.txt").write_text("")
        (b / "da3.pt").write_bytes(b"x")
        dirs = discover_batch_da3_dirs(root)
        assert [p.name for p in dirs] == ["111", "222"]

    mm = mean_overall_metrics(
        [
            {"ate_rmse": 1.0, "tae": 2.0, "scene_dir": "/x"},
            {"ate_rmse": 3.0, "tae": float("nan")},
        ]
    )
    assert mm["ate_rmse"] == pytest.approx(2.0)
    assert "tae" in mm and mm["tae"] == pytest.approx(2.0)


def test_save_batch_mean_curves_min_videos():
    import tempfile

    from metrics.plot_charts import save_batch_mean_curves

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for sid, nfr in [("a", 5), ("b", 4), ("c", 3)]:
            pd = root / sid / "plots"
            pd.mkdir(parents=True)
            al = pd / "alignment_translation_error.csv"
            rows = ["frame_index,error_m"] + [f"{i},{0.1 * (i + 1):.6f}" for i in range(nfr)]
            al.write_text("\n".join(rows), encoding="utf-8")
        out = save_batch_mean_curves(root, ["a", "b", "c"], min_videos=3, verbose=False)
        assert "mean_alignment_translation_error.csv" in out
        data = (root / "overall" / "plots" / "mean_alignment_translation_error.csv").read_text()
        lines = data.strip().split("\n")
        assert lines[0] == "frame_index,mean_error_m,n_videos"
        body = lines[1:]
        for line in body:
            parts = line.split(",")
            assert int(parts[2]) >= 3
        last_idx = int(body[-1].split(",")[0])
        assert last_idx <= 2


def test_absolute_rotation_error_zero_identical():
    from metrics.geometry import absolute_rotation_error_per_frame_deg

    n = 8
    ext = np.zeros((n, 3, 4), dtype=np.float64)
    ext[:, 0, 0] = ext[:, 1, 1] = ext[:, 2, 2] = 1.0
    ext[:, 0, 3] = np.linspace(0, 0.5, n)
    err = absolute_rotation_error_per_frame_deg(ext, ext)
    np.testing.assert_allclose(err, 0.0, atol=1e-9)


def test_relative_pose_error_zero_identical():
    from metrics.geometry import relative_pose_errors

    n = 12
    ext = np.zeros((n, 3, 4), dtype=np.float64)
    ext[:, 0, 0] = ext[:, 1, 1] = ext[:, 2, 2] = 1.0
    ext[:, 1, 3] = np.linspace(0, 0.3, n)
    fi, r_rot, r_tr = relative_pose_errors(ext, ext, delta=2)
    assert fi.size == n - 2
    np.testing.assert_allclose(r_rot, 0.0, atol=1e-9)
    np.testing.assert_allclose(r_tr, 0.0, atol=1e-9)


def test_ate_mean_matches_mean_translation_error():
    from metrics.geometry import apply_sim3_points, ate_rmse_sim3

    rng = np.random.default_rng(42)
    ref = rng.normal(size=(30, 3))
    noise = 0.02 * rng.normal(size=ref.shape)
    est = ref + noise
    ate, r, t, s = ate_rmse_sim3(est, ref, with_scale=True)
    aligned = apply_sim3_points(est, r, t, s)
    err = np.linalg.norm(aligned - ref, axis=1)
    ate_mean = float(np.mean(err))
    ate_rmse_check = float(np.sqrt(np.mean(err**2)))
    assert ate == pytest.approx(ate_rmse_check)
    assert ate_mean <= ate_rmse_check + 1e-9


def test_save_batch_mean_curves_includes_pose_errors():
    import tempfile

    from metrics.plot_charts import save_batch_mean_curves

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for sid in ("a", "b", "c"):
            pd = root / sid / "plots"
            pd.mkdir(parents=True)
            rows_ar = ["frame_index,error_deg"] + [f"{i},{0.01 * i:.6f}" for i in range(4)]
            (pd / "absolute_rotation_error_deg.csv").write_text("\n".join(rows_ar), encoding="utf-8")
            rows_rr = ["frame_i,rpe_delta,rot_error_deg", "0,1,0.1", "1,1,0.2", "2,1,0.15"]
            rows_rt = ["frame_i,rpe_delta,trans_error_m", "0,1,0.01", "1,1,0.02", "2,1,0.012"]
            (pd / "rpe_rotation_error_deg.csv").write_text("\n".join(rows_rr), encoding="utf-8")
            (pd / "rpe_translation_error_m.csv").write_text("\n".join(rows_rt), encoding="utf-8")
        out = save_batch_mean_curves(root, ["a", "b", "c"], min_videos=3, verbose=False)
        assert "mean_absolute_rotation_error_deg.csv" in out
        assert "mean_rpe_errors.csv" in out


def test_mean_overall_includes_pose_metrics():
    from metrics.batch import mean_overall_metrics

    mm = mean_overall_metrics(
        [
            {"are_rmse_deg": 1.0, "rpe_trans_rmse_m": 0.1, "ate_mean": 0.05, "scene_dir": "/x"},
            {"are_rmse_deg": 3.0, "rpe_trans_rmse_m": 0.3, "ate_mean": 0.15},
        ]
    )
    assert mm["are_rmse_deg"] == pytest.approx(2.0)
    assert mm["rpe_trans_rmse_m"] == pytest.approx(0.2)
    assert mm["ate_mean"] == pytest.approx(0.1)
