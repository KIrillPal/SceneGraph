"""Sim(3) / pose helpers (Umeyama, world↔camera)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def extrinsic_w2c_to_4x4(ext: np.ndarray) -> np.ndarray:
    """(3, 4) or (4, 4) world-to-camera → (4, 4) homogeneous."""
    ext = np.asarray(ext, dtype=np.float64)
    if ext.shape == (4, 4):
        return ext
    if ext.shape != (3, 4):
        raise ValueError(f"Expected (3, 4) or (4, 4) extrinsic, got {ext.shape}")
    out = np.eye(4, dtype=np.float64)
    out[:3, :4] = ext
    return out


def extrinsics_w2c_to_c2w(ext_w2c: np.ndarray) -> np.ndarray:
    """Batch (N,3,4) or (N,4,4) w2c → (N,4,4) camera-to-world."""
    ext = np.asarray(ext_w2c, dtype=np.float64)
    n = ext.shape[0]
    out = np.zeros((n, 4, 4), dtype=np.float64)
    for i in range(n):
        T = extrinsic_w2c_to_4x4(ext[i])
        out[i] = np.linalg.inv(T)
    return out


def camera_center_from_w2c(ext_w2c: np.ndarray) -> np.ndarray:
    """Camera position in world frame: (N,3) from (N,3,4) w2c rows."""
    c2w = extrinsics_w2c_to_c2w(ext_w2c)
    return c2w[:, :3, 3].copy()


def load_arkit_traj_w2c(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse lowres_wide.traj: timestamp, rotvec(3), t_w2c(3) with R_w2c = Exp(rotvec).

    Returns:
        ts: (N,) timestamps
        ext_w2c: (N, 3, 4) rows [R|t] with p_cam = R @ p_world + t
    """
    rows_ts = []
    mats = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 7:
                continue
            ts = vals[0]
            w2c_R = Rotation.from_rotvec(vals[1:4]).as_matrix()
            w2c_t = np.asarray(vals[4:7], dtype=np.float64)
            E = np.zeros((3, 4), dtype=np.float64)
            E[:, :3] = w2c_R
            E[:, 3] = w2c_t
            rows_ts.append(ts)
            mats.append(E)
    if not mats:
        raise ValueError(f"No valid rows in {path}")
    return np.asarray(rows_ts, dtype=np.float64), np.stack(mats, axis=0)


def traj_interpolate_w2c(
    traj_ts: np.ndarray,
    traj_ext: np.ndarray,
    query_ts: np.ndarray,
) -> np.ndarray:
    """Piecewise linear (t) + SLERP (R). query_ts → (Nq, 3, 4) w2c."""
    from scipy.spatial.transform import Rotation, Slerp

    q = np.asarray(query_ts, dtype=np.float64)
    idx = np.searchsorted(traj_ts, q, side="left")
    idx = np.clip(idx, 1, len(traj_ts) - 1)
    a = idx - 1
    denom = traj_ts[idx] - traj_ts[a]
    w = np.where(denom > 1e-12, (q - traj_ts[a]) / denom, 0.0)
    w = np.clip(w, 0.0, 1.0)

    R_a = traj_ext[a, :, :3]
    R_b = traj_ext[idx, :, :3]
    t_a = traj_ext[a, :, 3]
    t_b = traj_ext[idx, :, 3]

    out = np.zeros((len(q), 3, 4), dtype=np.float64)
    for i in range(len(q)):
        slerp = Slerp([0.0, 1.0], Rotation.from_matrix([R_a[i], R_b[i]]))
        Ri = slerp(w[i]).as_matrix()
        ti = (1.0 - w[i]) * t_a[i] + w[i] * t_b[i]
        out[i, :, :3] = Ri
        out[i, :, 3] = ti

    below = q < traj_ts[0]
    above = q > traj_ts[-1]
    if np.any(below):
        out[below] = traj_ext[0]
    if np.any(above):
        out[above] = traj_ext[-1]
    return out


def umeyama_sim3(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    with_scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Least-squares Sim(3): dst ≈ scale * (R @ src.T).T + t  i.e. row-wise scale * p @ R.T + t.

    src, dst: (N, 3) matched points (same row = correspondence).
    Returns R (3,3), t (3,), scale (scalar).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[1] != 3:
        raise ValueError(f"src and dst must be (N, 3), got {src.shape}, {dst.shape}")
    n = src.shape[0]
    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    x = src - mean_src
    y = dst - mean_dst
    cov = (y.T @ x) / n
    u, d, vt = np.linalg.svd(cov)
    s_fix = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_fix[2, 2] = -1
    r = u @ s_fix @ vt
    var_src = float(np.sum(x * x) / n)
    if with_scale:
        scale = float(np.trace(np.diag(d) @ s_fix) / (var_src + 1e-12))
    else:
        scale = 1.0
    t = mean_dst - scale * (r @ mean_src)
    return r.astype(np.float64), t.astype(np.float64), scale


def apply_sim3_points(points: np.ndarray, r: np.ndarray, t: np.ndarray, scale: float) -> np.ndarray:
    """points (M,3) → scale * R @ p.T + t."""
    p = np.asarray(points, dtype=np.float64)
    return (scale * (r @ p.T).T + t).astype(np.float64)


def ate_rmse_sim3(
    est_centers: np.ndarray,
    ref_centers: np.ndarray,
    *,
    with_scale: bool = True,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """
    Align est → ref with Sim(3); return RMSE of translation errors and (R,t,s).

    est_centers / ref_centers: (N, 3) same frame order.
    """
    r, t, s = umeyama_sim3(est_centers, ref_centers, with_scale=with_scale)
    aligned = apply_sim3_points(est_centers, r, t, s)
    err = np.linalg.norm(aligned - ref_centers, axis=1)
    rmse = float(np.sqrt(np.mean(err**2)))
    return rmse, r, t, s


def align_w2c_extrinsics_batch(
    ext_w2c: np.ndarray,
    r: np.ndarray,
    t: np.ndarray,
    s: float,
) -> np.ndarray:
    """Apply Sim(3) from pred world into GT/ARKit world to each (3,4) w2c extrinsic."""
    ext_w2c = np.asarray(ext_w2c, dtype=np.float64)
    n = ext_w2c.shape[0]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = float(s) * np.asarray(r, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    out = np.zeros((n, 3, 4), dtype=np.float64)
    for i in range(n):
        E = extrinsic_w2c_to_4x4(ext_w2c[i])
        c2w = np.linalg.inv(E)
        c2w_new = T @ c2w
        E_new = np.linalg.inv(c2w_new)
        out[i, :, :] = E_new[:3, :]
    return out


def absolute_rotation_error_per_frame_deg(
    ext_gt_w2c: np.ndarray,
    ext_est_w2c: np.ndarray,
) -> np.ndarray:
    """
    Geodesic angle (degrees) between GT and estimated w2c rotation at each frame.

    Both are (N,3,4) with p_cam = R @ p_world + t; comparison is in world frame: R_gt @ R_est.T.
    """
    ext_gt_w2c = np.asarray(ext_gt_w2c, dtype=np.float64)
    ext_est_w2c = np.asarray(ext_est_w2c, dtype=np.float64)
    n = ext_gt_w2c.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        Rg = ext_gt_w2c[i, :3, :3]
        Re = ext_est_w2c[i, :3, :3]
        R_err = Rg @ Re.T
        out[i] = float(
            np.degrees(Rotation.from_matrix(R_err).magnitude()),
        )
    return out


def relative_pose_errors(
    ext_gt_w2c: np.ndarray,
    ext_est_w2c: np.ndarray,
    *,
    delta: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Relative pose error (RPE) using motion from frame i to i+delta (c2w convention).

    ΔT = inv(P_i) @ P_{i+delta} with P = inv(E) (camera-from-world).

    Returns:
        frame_i: (M,) start indices
        rpe_rot_deg: (M,) rotation part of inv(Δ_gt) @ Δ_est, degrees
        rpe_trans_m: (M,) translation part (Euclidean norm), meters
    """
    ext_gt_w2c = np.asarray(ext_gt_w2c, dtype=np.float64)
    ext_est_w2c = np.asarray(ext_est_w2c, dtype=np.float64)
    d = max(1, int(delta))
    n = ext_gt_w2c.shape[0]
    if n < d + 1:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    frame_i_list: list[int] = []
    rot_list: list[float] = []
    trans_list: list[float] = []
    for i in range(0, n - d):
        P_gt_i = np.linalg.inv(extrinsic_w2c_to_4x4(ext_gt_w2c[i]))
        P_gt_j = np.linalg.inv(extrinsic_w2c_to_4x4(ext_gt_w2c[i + d]))
        delta_gt = np.linalg.inv(P_gt_i) @ P_gt_j
        P_es_i = np.linalg.inv(extrinsic_w2c_to_4x4(ext_est_w2c[i]))
        P_es_j = np.linalg.inv(extrinsic_w2c_to_4x4(ext_est_w2c[i + d]))
        delta_est = np.linalg.inv(P_es_i) @ P_es_j
        F = np.linalg.inv(delta_gt) @ delta_est
        R = F[:3, :3]
        t = F[:3, 3]
        frame_i_list.append(i)
        rot_list.append(float(np.degrees(Rotation.from_matrix(R).magnitude())))
        trans_list.append(float(np.linalg.norm(t)))
    return (
        np.asarray(frame_i_list, dtype=np.int64),
        np.asarray(rot_list, dtype=np.float64),
        np.asarray(trans_list, dtype=np.float64),
    )
