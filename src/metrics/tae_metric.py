"""
Temporal Alignment Error (TAE) for video depth predictions.

TAE measures temporal consistency across consecutive frames by projecting
depth maps between views and comparing with the independently predicted depth.

Formula (Equation 7):

    TAE = (1 / 2(T-1)) * sum_{k=0}^{T-2} [
        AbsRel( f(depth_k,   p^k),         depth_{k+1} ) +
        AbsRel( f(depth_{k+1}, p_{-}^{k+1}), depth_k     )
    ]

where:
  - f(depth, transform): project depth map from source camera to target camera
  - p^k: transform from camera k to camera k+1
  - p_{-}^{k+1}: transform from camera k+1 to camera k (inverse of p^k)
  - AbsRel(a, b) = mean over valid pixels of |a - b| / b

Inputs: per-frame predicted depths, intrinsics, and extrinsics (camera-from-world).

TAE on ``device=auto`` or ``cuda``: PyTorch warp on GPU (float32) via ``scatter_reduce`` z-buffer.
CPU path matches previous NumPy float64 implementation.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

# -----------------------------------------------------------------------------
# AbsRel
# -----------------------------------------------------------------------------


def absrel(
    pred_depth: np.ndarray,
    ref_depth: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> float:
    """
    Absolute Relative Error: (1/N) * sum |pred - ref| / ref over valid pixels.

    Args:
        pred_depth: Predicted depth map (H, W).
        ref_depth: Reference depth map (H, W).
        valid_mask: Optional boolean (H, W). If None, valid = (ref > 0) & (pred > 0).
        eps: Small constant to avoid division by zero.

    Returns:
        Mean absolute relative error, or np.nan if no valid pixels.
    """
    pred = np.asarray(pred_depth, dtype=np.float64).ravel()
    ref = np.asarray(ref_depth, dtype=np.float64).ravel()
    if valid_mask is not None:
        valid = np.asarray(valid_mask, dtype=bool).ravel()
    else:
        valid = (ref > eps) & (pred > 0)
    if not np.any(valid):
        return np.nan
    rel = np.abs(pred[valid] - ref[valid]) / (ref[valid] + eps)
    return float(np.mean(rel))


# -----------------------------------------------------------------------------
# Camera transforms (extrinsic = world-to-camera [R|t])
# -----------------------------------------------------------------------------


def _extrinsic_to_4x4(extrinsic: np.ndarray) -> np.ndarray:
    """(3, 4) camera-from-world -> (4, 4) homogeneous."""
    E = np.asarray(extrinsic, dtype=np.float64)
    M = np.eye(4)
    M[:3, :4] = E
    return M


def transform_cam_i_to_cam_j(
    extrinsic_i: np.ndarray,
    extrinsic_j: np.ndarray,
) -> np.ndarray:
    """
    Transformation that maps 3D points from camera i to camera j.

    Both extrinsics are world-to-camera (p_cam = R @ p_world + t).
    So p_cam_j = M_j @ M_i^{-1} @ p_cam_i (in homogeneous coords).
    Returns (4, 4) matrix for homogeneous 3D points.
    """
    Mi = _extrinsic_to_4x4(extrinsic_i)
    Mj = _extrinsic_to_4x4(extrinsic_j)
    return Mj @ np.linalg.inv(Mi)


# -----------------------------------------------------------------------------
# PyTorch / CUDA helpers (TAE)
# -----------------------------------------------------------------------------


def resolve_tae_device(device: str | None) -> str:
    """``auto`` → CUDA if available, else CPU. ``cuda`` raises if unavailable."""
    import torch

    if device is None or str(device).lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    d = str(device).lower()
    if d == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("device='cuda' but torch.cuda.is_available() is False")
        return "cuda"
    if d == "cpu":
        return "cpu"
    raise ValueError(f"Unknown TAE device {device!r}; use 'auto', 'cuda', or 'cpu'")


def _absrel_torch(
    pred: "torch.Tensor",
    ref: "torch.Tensor",
    valid: "torch.Tensor",
    eps: float,
) -> float:
    import torch

    if not torch.any(valid):
        return float("nan")
    pr = pred[valid]
    rf = ref[valid]
    rel = (pr - rf).abs() / (rf + eps)
    return float(rel.mean().item())


def project_depth_to_target_torch(
    depth_src: "torch.Tensor",
    K_src: "torch.Tensor",
    K_tgt: "torch.Tensor",
    T_src_to_tgt: "torch.Tensor",
    H_tgt: int,
    W_tgt: int,
    min_depth: float,
    *,
    max_source_samples: Optional[int] = None,
    torch_gen: Optional["torch.Generator"] = None,
    dtype: Optional["torch.dtype"] = None,
) -> "torch.Tensor":
    """
    GPU warp: same semantics as ``project_depth_to_target`` (min-depth z-buffer).
    ``depth_src`` and intrinsics / transform must already live on the target device.
    """
    import torch

    device = depth_src.device
    if dtype is None:
        dtype = depth_src.dtype
    H_src, W_src = depth_src.shape

    fx_s, fy_s = K_src[0, 0], K_src[1, 1]
    cx_s, cy_s = K_src[0, 2], K_src[1, 2]
    fx_t, fy_t = K_tgt[0, 0], K_tgt[1, 1]
    cx_t, cy_t = K_tgt[0, 2], K_tgt[1, 2]

    v_grid, u_grid = torch.meshgrid(
        torch.arange(H_src, device=device, dtype=dtype),
        torch.arange(W_src, device=device, dtype=dtype),
        indexing="ij",
    )
    d = depth_src.reshape(-1)
    u_s = u_grid.reshape(-1)
    v_s = v_grid.reshape(-1)

    valid = d >= min_depth
    u_s = u_s[valid]
    v_s = v_s[valid]
    d = d[valid]

    nv = d.shape[0]
    if max_source_samples is not None:
        cap = int(max_source_samples)
        if cap < 1:
            raise ValueError("max_source_samples must be >= 1 when set")
        if nv > cap:
            if torch_gen is None:
                raise ValueError("torch_gen is required when max_source_samples is set for GPU warp")
            pick = torch.randperm(nv, generator=torch_gen, device=torch.device("cpu"))[:cap].to(
                device=device, dtype=torch.long
            )
            u_s = u_s[pick]
            v_s = v_s[pick]
            d = d[pick]

    x_s = (u_s - cx_s) * d / fx_s
    y_s = (v_s - cy_s) * d / fy_s
    z_s = d
    ones = torch.ones_like(z_s)
    p_src = torch.stack([x_s, y_s, z_s, ones], dim=0)
    p_tgt = (T_src_to_tgt @ p_src).T
    x_t, y_t, z_t = p_tgt[:, 0], p_tgt[:, 1], p_tgt[:, 2]

    front = z_t > min_depth
    x_t, y_t, z_t = x_t[front], y_t[front], z_t[front]

    u_t = fx_t * (x_t / z_t) + cx_t
    v_t = fy_t * (y_t / z_t) + cy_t

    ui = torch.round(u_t).to(torch.long)
    vi = torch.round(v_t).to(torch.long)
    in_fr = (ui >= 0) & (ui < W_tgt) & (vi >= 0) & (vi < H_tgt)
    ui = ui[in_fr]
    vi = vi[in_fr]
    z_t = z_t[in_fr]

    depth_out = torch.full((H_tgt, W_tgt), float("nan"), device=device, dtype=dtype)
    if ui.numel() == 0:
        return depth_out

    lin = vi * W_tgt + ui
    buf = torch.full((H_tgt * W_tgt,), float("inf"), device=device, dtype=dtype)
    buf.scatter_reduce_(0, lin, z_t, reduce="amin", include_self=True)
    buf = buf.view(H_tgt, W_tgt)
    nan_mask = torch.isinf(buf)
    depth_out = torch.where(nan_mask, depth_out, buf)
    return depth_out


def project_depth_to_target(
    depth_src: np.ndarray,
    K_src: np.ndarray,
    K_tgt: np.ndarray,
    T_src_to_tgt: np.ndarray,
    H_tgt: Optional[int] = None,
    W_tgt: Optional[int] = None,
    min_depth: float = 1e-4,
    occlusion: str = "min",
    *,
    max_source_samples: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Project a source depth map into the target camera view (warp).

    Steps:
      1. Unproject source pixels + depth to 3D in source camera.
      2. Transform 3D points to target camera with T_src_to_tgt.
      3. Project to target image plane; for each target pixel keep one depth
         (min depth by default to handle occlusions).

    Args:
        depth_src: (H_src, W_src) depth in source view.
        K_src: (3, 3) intrinsics for source camera.
        K_tgt: (3, 3) intrinsics for target camera.
        T_src_to_tgt: (4, 4) transform from source to target camera.
        H_tgt, W_tgt: Output size. Default: same as depth_src.
        min_depth: Ignore source pixels with depth < min_depth.
        occlusion: "min" = keep closest depth per target pixel.
        max_source_samples: If set, randomly sub-sample this many valid source
            pixels before warping (faster, approximate). ``None`` = use all pixels.
        rng: Random generator for sub-sampling; required if ``max_source_samples``
            is set and must be shared across calls for reproducible TAE.

    Returns:
        depth_tgt: (H_tgt, W_tgt). Unfilled pixels are np.nan (excluded from metrics).
    """
    depth_src = np.asarray(depth_src, dtype=np.float64)
    H_src, W_src = depth_src.shape
    if H_tgt is None:
        H_tgt = H_src
    if W_tgt is None:
        W_tgt = W_src

    fx_s, fy_s = K_src[0, 0], K_src[1, 1]
    cx_s, cy_s = K_src[0, 2], K_src[1, 2]
    fx_t, fy_t = K_tgt[0, 0], K_tgt[1, 1]
    cx_t, cy_t = K_tgt[0, 2], K_tgt[1, 2]

    u_s = np.arange(W_src, dtype=np.float64)
    v_s = np.arange(H_src, dtype=np.float64)
    u_s, v_s = np.meshgrid(u_s, v_s)
    d = depth_src.ravel()

    valid = d >= min_depth
    u_s = u_s.ravel()[valid]
    v_s = v_s.ravel()[valid]
    d = d[valid]

    nv = u_s.shape[0]
    if max_source_samples is not None:
        cap = int(max_source_samples)
        if cap < 1:
            raise ValueError("max_source_samples must be >= 1 when set")
        if nv > cap:
            if rng is None:
                raise ValueError("rng is required when max_source_samples is set")
            pick = rng.choice(nv, size=cap, replace=False)
            u_s = u_s[pick]
            v_s = v_s[pick]
            d = d[pick]
    x_s = (u_s - cx_s) * d / fx_s
    y_s = (v_s - cy_s) * d / fy_s
    z_s = d
    ones = np.ones_like(z_s)
    p_src = np.stack([x_s, y_s, z_s, ones], axis=0)  # (4, N)

    # 3D in target camera
    p_tgt = (T_src_to_tgt @ p_src).T  # (N, 4)
    x_t, y_t, z_t = p_tgt[:, 0], p_tgt[:, 1], p_tgt[:, 2]

    # Behind target camera
    valid = z_t > min_depth
    x_t, y_t, z_t = x_t[valid], y_t[valid], z_t[valid]

    u_t = fx_t * (x_t / z_t) + cx_t
    v_t = fy_t * (y_t / z_t) + cy_t

    # Rasterize: round to pixel, then occlusion handling
    ui = np.round(u_t).astype(np.int32)
    vi = np.round(v_t).astype(np.int32)
    in_frame = (ui >= 0) & (ui < W_tgt) & (vi >= 0) & (vi < H_tgt)
    ui = ui[in_frame]
    vi = vi[in_frame]
    z_t = z_t[in_frame]

    depth_tgt = np.full((H_tgt, W_tgt), np.nan, dtype=np.float64)
    if occlusion == "min":
        # Closest depth per target pixel: vectorized z-buffer merge.
        depth_tgt[...] = np.inf
        np.minimum.at(depth_tgt, (vi, ui), z_t)
        depth_tgt[~np.isfinite(depth_tgt)] = np.nan
    else:
        depth_tgt[vi, ui] = z_t

    return depth_tgt


# -----------------------------------------------------------------------------
# TAE
# -----------------------------------------------------------------------------


def _tae_pair_progress_iter(n_pairs: int, *, progress: bool, desc: str):
    """``range(n_pairs)`` or a tqdm iterator over frame pairs."""
    r = range(n_pairs)
    if not progress:
        return r
    from tqdm import tqdm

    return tqdm(r, desc=desc, unit="pair", ncols=100, leave=True)


def tae(
    depths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    *,
    min_depth: float = 1e-4,
    warp_source_samples: Optional[int] = None,
    warp_sample_seed: int = 0,
    device: str | None = "auto",
    progress: bool = False,
) -> float:
    """
    Temporal Alignment Error over a sequence of depth maps.

    With ``device='auto'`` uses CUDA when available (much faster for large maps).

    Args:
        device: ``auto`` (pick GPU if available), ``cuda``, or ``cpu``.
        progress: If True, show a tqdm bar over consecutive frame pairs.
    """
    depths = np.asarray(depths, dtype=np.float64)
    if depths.ndim == 4:
        depths = depths.squeeze(-1)
    T, H, W = depths.shape

    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.ndim == 2:
        intrinsics = np.broadcast_to(intrinsics[np.newaxis, ...], (T, 3, 3))
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    assert extrinsics.shape == (T, 3, 4), f"extrinsics shape {extrinsics.shape}"

    if T < 2:
        return np.nan

    dev = resolve_tae_device(device)
    bar_desc = "TAE (CUDA fp32)" if dev == "cuda" else "TAE (CPU)"
    if dev == "cuda":
        return _tae_cuda(
            depths,
            intrinsics,
            extrinsics,
            min_depth=min_depth,
            warp_source_samples=warp_source_samples,
            warp_sample_seed=warp_sample_seed,
            progress=progress,
            bar_desc=bar_desc,
        )

    proj_kw: dict = {"min_depth": min_depth}
    if warp_source_samples is not None:
        proj_kw["max_source_samples"] = int(warp_source_samples)
        proj_kw["rng"] = np.random.default_rng(int(warp_sample_seed))

    terms: list[float] = []
    for k in _tae_pair_progress_iter(T - 1, progress=progress, desc=bar_desc):
        T_k_to_kp1 = transform_cam_i_to_cam_j(extrinsics[k], extrinsics[k + 1])
        T_kp1_to_k = transform_cam_i_to_cam_j(extrinsics[k + 1], extrinsics[k])

        depth_k = depths[k]
        depth_kp1 = depths[k + 1]

        proj_k_to_kp1 = project_depth_to_target(
            depth_k,
            K_src=intrinsics[k],
            K_tgt=intrinsics[k + 1],
            T_src_to_tgt=T_k_to_kp1,
            H_tgt=H,
            W_tgt=W,
            **proj_kw,
        )
        valid = np.isfinite(proj_k_to_kp1) & (depth_kp1 > min_depth) & (proj_k_to_kp1 > 0)
        if np.any(valid):
            terms.append(absrel(proj_k_to_kp1, depth_kp1, valid_mask=valid, eps=min_depth))

        proj_kp1_to_k = project_depth_to_target(
            depth_kp1,
            K_src=intrinsics[k + 1],
            K_tgt=intrinsics[k],
            T_src_to_tgt=T_kp1_to_k,
            H_tgt=H,
            W_tgt=W,
            **proj_kw,
        )
        valid = np.isfinite(proj_kp1_to_k) & (depth_k > min_depth) & (proj_kp1_to_k > 0)
        if np.any(valid):
            terms.append(absrel(proj_kp1_to_k, depth_k, valid_mask=valid, eps=min_depth))

    if not terms:
        return np.nan
    return float(np.mean(terms))


def _tae_cuda(
    depths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    *,
    min_depth: float,
    warp_source_samples: Optional[int],
    warp_sample_seed: int,
    progress: bool,
    bar_desc: str,
) -> float:
    import torch

    tdev = torch.device("cuda")
    dtype = torch.float32
    T, H, W = depths.shape

    depths_t = torch.as_tensor(depths, device=tdev, dtype=dtype)
    intr_t = torch.as_tensor(intrinsics, device=tdev, dtype=dtype)

    t_gen = None
    if warp_source_samples is not None:
        t_gen = torch.Generator(device=torch.device("cpu"))
        t_gen.manual_seed(int(warp_sample_seed))

    terms: list[float] = []
    md = float(min_depth)

    for k in _tae_pair_progress_iter(T - 1, progress=progress, desc=bar_desc):
        T_k_to_kp1 = torch.as_tensor(
            transform_cam_i_to_cam_j(extrinsics[k], extrinsics[k + 1]),
            device=tdev,
            dtype=dtype,
        )
        T_kp1_to_k = torch.as_tensor(
            transform_cam_i_to_cam_j(extrinsics[k + 1], extrinsics[k]),
            device=tdev,
            dtype=dtype,
        )

        proj_k_to_kp1 = project_depth_to_target_torch(
            depths_t[k],
            intr_t[k],
            intr_t[k + 1],
            T_k_to_kp1,
            H,
            W,
            md,
            max_source_samples=warp_source_samples,
            torch_gen=t_gen,
            dtype=dtype,
        )
        dk1 = depths_t[k + 1]
        valid = torch.isfinite(proj_k_to_kp1) & (dk1 > md) & (proj_k_to_kp1 > 0)
        if torch.any(valid):
            terms.append(_absrel_torch(proj_k_to_kp1, dk1, valid, md))

        proj_kp1_to_k = project_depth_to_target_torch(
            dk1,
            intr_t[k + 1],
            intr_t[k],
            T_kp1_to_k,
            H,
            W,
            md,
            max_source_samples=warp_source_samples,
            torch_gen=t_gen,
            dtype=dtype,
        )
        dk = depths_t[k]
        valid = torch.isfinite(proj_kp1_to_k) & (dk > md) & (proj_kp1_to_k > 0)
        if torch.any(valid):
            terms.append(_absrel_torch(proj_kp1_to_k, dk, valid, md))

    if not terms:
        return np.nan
    return float(np.mean(terms))


def _tae_pair_terms_cuda(
    depths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    *,
    min_depth: float,
    warp_source_samples: Optional[int],
    warp_sample_seed: int,
    progress: bool,
    bar_desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    tdev = torch.device("cuda")
    dtype = torch.float32
    T, H, W = depths.shape

    depths_t = torch.as_tensor(depths, device=tdev, dtype=dtype)
    intr_t = torch.as_tensor(intrinsics, device=tdev, dtype=dtype)

    t_gen = None
    if warp_source_samples is not None:
        t_gen = torch.Generator(device=torch.device("cpu"))
        t_gen.manual_seed(int(warp_sample_seed))

    fwd = np.full(T - 1, np.nan, dtype=np.float64)
    bwd = np.full(T - 1, np.nan, dtype=np.float64)
    md = float(min_depth)

    for k in _tae_pair_progress_iter(T - 1, progress=progress, desc=bar_desc):
        T_k_to_kp1 = torch.as_tensor(
            transform_cam_i_to_cam_j(extrinsics[k], extrinsics[k + 1]),
            device=tdev,
            dtype=dtype,
        )
        T_kp1_to_k = torch.as_tensor(
            transform_cam_i_to_cam_j(extrinsics[k + 1], extrinsics[k]),
            device=tdev,
            dtype=dtype,
        )
        depth_k = depths_t[k]
        depth_kp1 = depths_t[k + 1]

        proj_k_to_kp1 = project_depth_to_target_torch(
            depth_k,
            intr_t[k],
            intr_t[k + 1],
            T_k_to_kp1,
            H,
            W,
            md,
            max_source_samples=warp_source_samples,
            torch_gen=t_gen,
            dtype=dtype,
        )
        valid = torch.isfinite(proj_k_to_kp1) & (depth_kp1 > md) & (proj_k_to_kp1 > 0)
        if torch.any(valid):
            fwd[k] = _absrel_torch(proj_k_to_kp1, depth_kp1, valid, md)

        proj_kp1_to_k = project_depth_to_target_torch(
            depth_kp1,
            intr_t[k + 1],
            intr_t[k],
            T_kp1_to_k,
            H,
            W,
            md,
            max_source_samples=warp_source_samples,
            torch_gen=t_gen,
            dtype=dtype,
        )
        valid2 = torch.isfinite(proj_kp1_to_k) & (depth_k > md) & (proj_kp1_to_k > 0)
        if torch.any(valid2):
            bwd[k] = _absrel_torch(proj_kp1_to_k, depth_k, valid2, md)

    return fwd, bwd


# -----------------------------------------------------------------------------
# TAE sequence (sliding window)
# -----------------------------------------------------------------------------


def _tae_pair_terms(
    depths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    *,
    min_depth: float = 1e-4,
    warp_source_samples: Optional[int] = None,
    warp_sample_seed: int = 0,
    device: str | None = "auto",
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute forward and backward AbsRel terms for each consecutive frame pair.

    For each k in 0..T-2:
      fwd[k] = AbsRel( f(depth_k, p^k), depth_{k+1} )
      bwd[k] = AbsRel( f(depth_{k+1}, p_{-}^{k+1}), depth_k )

    Returns:
        fwd: (T-1,) array; bwd: (T-1,) array. Invalid terms are np.nan.
    """
    depths = np.asarray(depths, dtype=np.float64)
    if depths.ndim == 4:
        depths = depths.squeeze(-1)
    T, H, W = depths.shape

    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.ndim == 2:
        intrinsics = np.broadcast_to(intrinsics[np.newaxis, ...], (T, 3, 3))
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    assert extrinsics.shape == (T, 3, 4), f"extrinsics shape {extrinsics.shape}"

    dev = resolve_tae_device(device)
    bar_desc = "TAE pairs (CUDA fp32)" if dev == "cuda" else "TAE pairs (CPU)"
    if dev == "cuda":
        return _tae_pair_terms_cuda(
            depths,
            intrinsics,
            extrinsics,
            min_depth=min_depth,
            warp_source_samples=warp_source_samples,
            warp_sample_seed=warp_sample_seed,
            progress=progress,
            bar_desc=bar_desc,
        )

    proj_kw: dict = {"min_depth": min_depth}
    if warp_source_samples is not None:
        proj_kw["max_source_samples"] = int(warp_source_samples)
        proj_kw["rng"] = np.random.default_rng(int(warp_sample_seed))

    fwd = np.full(T - 1, np.nan, dtype=np.float64)
    bwd = np.full(T - 1, np.nan, dtype=np.float64)

    for k in _tae_pair_progress_iter(T - 1, progress=progress, desc=bar_desc):
        T_k_to_kp1 = transform_cam_i_to_cam_j(extrinsics[k], extrinsics[k + 1])
        T_kp1_to_k = transform_cam_i_to_cam_j(extrinsics[k + 1], extrinsics[k])

        depth_k = depths[k]
        depth_kp1 = depths[k + 1]

        proj_k_to_kp1 = project_depth_to_target(
            depth_k,
            K_src=intrinsics[k],
            K_tgt=intrinsics[k + 1],
            T_src_to_tgt=T_k_to_kp1,
            H_tgt=H,
            W_tgt=W,
            **proj_kw,
        )
        valid = np.isfinite(proj_k_to_kp1) & (depth_kp1 > min_depth) & (proj_k_to_kp1 > 0)
        if np.any(valid):
            fwd[k] = absrel(proj_k_to_kp1, depth_kp1, valid_mask=valid, eps=min_depth)

        proj_kp1_to_k = project_depth_to_target(
            depth_kp1,
            K_src=intrinsics[k + 1],
            K_tgt=intrinsics[k],
            T_src_to_tgt=T_kp1_to_k,
            H_tgt=H,
            W_tgt=W,
            **proj_kw,
        )
        valid = np.isfinite(proj_kp1_to_k) & (depth_k > min_depth) & (proj_kp1_to_k > 0)
        if np.any(valid):
            bwd[k] = absrel(proj_kp1_to_k, depth_k, valid_mask=valid, eps=min_depth)

    return fwd, bwd


def tae_sequence(
    depths: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    window_size: int = 2,
    *,
    min_depth: float = 1e-4,
    warp_source_samples: Optional[int] = None,
    warp_sample_seed: int = 0,
    device: str | None = "auto",
    progress: bool = False,
) -> np.ndarray:
    """
    TAE over sliding windows of size window_size.

    For each window of consecutive frames [i, i+1, ..., i+window_size-1], compute
    the mean AbsRel over all consecutive pairs in that window (both directions).
    Returns one value per window; length is T - window_size + 1.

    Args:
        depths: (T, H, W) or (T, H, W, 1) predicted depth per frame.
        intrinsics: (T, 3, 3) or (3, 3) if same for all frames.
        extrinsics: (T, 3, 4) camera-from-world per frame.
        window_size: Number of consecutive frames per window (>= 2).
        min_depth: Ignore depths below this in projection and AbsRel.
        progress: If True, show tqdm while accumulating per-pair terms.

    Returns:
        (T - window_size + 1,) array of TAE values per window.
    """
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    fwd, bwd = _tae_pair_terms(
        depths,
        intrinsics,
        extrinsics,
        min_depth=min_depth,
        warp_source_samples=warp_source_samples,
        warp_sample_seed=warp_sample_seed,
        device=device,
        progress=progress,
    )
    T = len(fwd) + 1
    # pair_terms[k] = (fwd[k], bwd[k]) for pair (k, k+1)
    pair_terms = np.stack([fwd, bwd], axis=1)  # (T-1, 2)
    # Window [i .. i+window_size-1] uses pairs i .. i+window_size-2
    return np.array([
        np.nanmean(pair_terms[i : i + window_size - 1].ravel())
        for i in range(T - window_size + 1)
    ])


# -----------------------------------------------------------------------------
# CLI / example
# -----------------------------------------------------------------------------


def main() -> None:
    import argparse
    import csv
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Compute TAE from VGGT prediction file.")
    parser.add_argument("prediction", type=str, help="Path to .pth/.pt with depth, intrinsic, extrinsic")
    parser.add_argument("--max-frames", type=int, default=None, help="Use first N frames only")
    parser.add_argument(
        "--save-tae-sequence",
        action="store_true",
        help="Compute TAE over sliding windows and include in outputs (used with --save-results).",
    )
    parser.add_argument(
        "--tae-window-size",
        type=int,
        default=2,
        metavar="K",
        help="Window size for TAE sequence (sliding window of K consecutive frames). Default: 2.",
    )
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory to save TAE metric, TAE sequence (CSV + image), and results YAML.",
    )
    parser.add_argument(
        "--warp-source-samples",
        type=int,
        default=None,
        help="Approximate TAE: sample this many source pixels per warp (default: full image)",
    )
    parser.add_argument(
        "--warp-sample-seed",
        type=int,
        default=0,
        help="RNG seed when --warp-source-samples is set",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="TAE device: auto (GPU if available), cuda, or cpu",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm over frame pairs",
    )
    args = parser.parse_args()

    import torch

    raw = torch.load(args.prediction, map_location="cpu", weights_only=False)

    def to_np(x):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    depths = to_np(raw["depth"]).squeeze(-1)  # (T, H, W)
    intrinsic = to_np(raw["intrinsic"])
    extrinsic = to_np(raw["extrinsic"])

    if args.max_frames is not None:
        depths = depths[: args.max_frames]
        intrinsic = intrinsic[: args.max_frames]
        extrinsic = extrinsic[: args.max_frames]

    score = tae(
        depths,
        intrinsic,
        extrinsic,
        warp_source_samples=args.warp_source_samples,
        warp_sample_seed=args.warp_sample_seed,
        device=args.device,
        progress=not args.no_progress,
    )
    print(f"TAE = {score:.6f}  (device={resolve_tae_device(args.device)})")

    save_dir = Path(args.save_results) if args.save_results else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    tae_seq = None
    if args.save_tae_sequence:
        if args.tae_window_size < 2:
            raise SystemExit("--tae-window-size must be >= 2")
        tae_seq = tae_sequence(
            depths,
            intrinsic,
            extrinsic,
            args.tae_window_size,
            warp_source_samples=args.warp_source_samples,
            warp_sample_seed=args.warp_sample_seed,
            device=args.device,
            progress=not args.no_progress,
        )
        print(f"TAE sequence length (window_size={args.tae_window_size}): {len(tae_seq)}")

    if save_dir is not None:
        import yaml

        # Save metric and metadata as YAML
        results = {
            "tae": float(score),
            "num_frames": int(depths.shape[0]),
            "prediction_path": str(args.prediction),
        }
        if tae_seq is not None:
            results["tae_window_size"] = args.tae_window_size
            results["tae_sequence_length"] = len(tae_seq)
            results["tae_sequence"] = [float(x) if np.isfinite(x) else None for x in tae_seq]

        results_path = save_dir / "tae_results.yaml"
        with open(results_path, "w") as f:
            yaml.safe_dump(results, f, default_flow_style=False, sort_keys=False)
        print(f"Saved {results_path}")

        # Save TAE sequence as CSV and image
        if tae_seq is not None:
            csv_path = save_dir / "tae_sequence.csv"
            window_center = np.arange(len(tae_seq)) + (args.tae_window_size - 1) / 2
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["window_center", "frame_index", "tae"])
                for i, (xc, val) in enumerate(zip(window_center, tae_seq)):
                    w.writerow([f"{xc:.2f}", i, f"{val:.8f}" if np.isfinite(val) else ""])
            print(f"Saved {csv_path}")

            img_path = save_dir / "tae_sequence.png"
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(window_center, tae_seq, linewidth=1)
                ax.set_xlabel("Frame (window center)")
                ax.set_ylabel("TAE")
                ax.set_title(f"TAE over sliding windows (K={args.tae_window_size})")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(img_path, dpi=150)
                plt.close()
                print(f"Saved {img_path}")
            except ImportError:
                print("Skipping TAE sequence image (matplotlib not installed)")


if __name__ == "__main__":
    main()
