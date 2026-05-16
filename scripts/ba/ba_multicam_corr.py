"""3-DoF [ωx, ωy, Δfx] BA correction for PandaSet side cameras.

Use a front-cam-only converged model (zero-shot on side cams) to discover
left/right calibration bias. Pool per-pt residuals across all frames + scenes,
solve 1-step linearized GN with the model's per-pt 2x2 inv-cov as weight.

Output:
  - configs/ba/pandaset_3cam_corrections.json — per-cam (ωx, ωy, Δfx) + provenance
  - experiments/_cache_check/ps_3cam_ba_check/ — BEFORE/AFTER projection PNGs

Reproducible via config file. See configs/ba/pandaset_3cam_corr.json for schema.
"""
import argparse, json, hashlib, pickle, gzip, subprocess, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.model_depth import CalibNetDepth


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    cfg["_path"] = str(path)
    cfg["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return cfg


def build_model(model_cfg: dict, device: torch.device) -> CalibNetDepth:
    m = CalibNetDepth(
        img_size=model_cfg["img_size"],
        in_channels=model_cfg["in_channels"],
        n_layers=model_cfg["n_layers"],
        use_convnext=model_cfg["use_convnext"],
        use_frustum=model_cfg["use_frustum"],
        deform_mode=model_cfg["deform_mode"],
        use_frame_pose=model_cfg.get("use_frame_pose", False),
        frame_pose_dof=model_cfg.get("frame_pose_dof", 6),
    ).to(device)
    ckpt = Path(model_cfg["ckpt"])
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # BA only needs per-pt head; drop CLS frame_pose_head entirely so cov-shape
    # differences between ckpt (e.g. full Cholesky) and the model we build here
    # (default diagonal) don't fail load.
    state = {k: v for k, v in state.items() if not k.startswith("frame_pose_head")}
    miss, unexp = m.load_state_dict(state, strict=False)
    print(f"  model: miss={len(miss)} unexp={len(unexp)} ckpt={ckpt}")
    m.eval()
    return m


def ego_speed_at(root: Path, scene: str, cam: str, frame: int,
                  dt: float = 0.1) -> float:
    """Approx ego speed in m/s via central diff on the cam-in-world position
    (PS is ~10 Hz → dt=0.1). At sequence boundaries falls back to one-sided
    diff. Returns 0.0 if poses unavailable."""
    poses_p = root / scene / "camera" / cam / "poses.json"
    if not poses_p.exists():
        return 0.0
    poses = json.loads(poses_p.read_text())
    n = len(poses)
    if n < 2:
        return 0.0
    f_lo = max(0, frame - 1)
    f_hi = min(n - 1, frame + 1)
    if f_hi == f_lo:
        return 0.0
    p_lo = np.array([poses[f_lo]["position"]["x"],
                      poses[f_lo]["position"]["y"],
                      poses[f_lo]["position"]["z"]])
    p_hi = np.array([poses[f_hi]["position"]["x"],
                      poses[f_hi]["position"]["y"],
                      poses[f_hi]["position"]["z"]])
    return float(np.linalg.norm(p_hi - p_lo) / ((f_hi - f_lo) * dt))


def load_frame(root: Path, scene: str, cam: str, frame: int):
    sc = root / scene
    cd = sc / "camera" / cam
    if not (cd / "intrinsics.json").exists():
        return None
    intr = json.loads((cd / "intrinsics.json").read_text())
    poses = json.loads((cd / "poses.json").read_text())
    if frame >= len(poses):
        return None
    p = poses[frame]
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    q = p["heading"]
    R = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix().astype(np.float32)
    cpw = np.array([p["position"]["x"], p["position"]["y"], p["position"]["z"]], dtype=np.float32)
    T_wc = np.eye(4); T_wc[:3, :3] = R; T_wc[:3, 3] = cpw
    T_cw = np.linalg.inv(T_wc).astype(np.float32)
    ld = sc / "lidar" / f"{frame:02d}.pkl"
    if ld.exists():
        df = pickle.load(open(ld, "rb"))
    else:
        ld_gz = sc / "lidar" / f"{frame:02d}.pkl.gz"
        if not ld_gz.exists():
            return None
        df = pickle.load(gzip.open(ld_gz, "rb"))
    df = df[df["d"] == 0]
    pts_w = df[["x", "y", "z"]].values.astype(np.float32)
    homo = np.column_stack([pts_w, np.ones(len(pts_w), dtype=np.float32)])
    pts_c = (T_cw @ homo.T).T[:, :3]
    z = pts_c[:, 2]
    uv = (K @ pts_c.T).T[:, :2] / np.maximum(z[:, None], 1e-6)
    jpg = cd / f"{frame:02d}.jpg"
    if not jpg.exists():
        return None
    img = np.asarray(Image.open(jpg).convert("RGB"))
    return img, K, T_cw, pts_w, pts_c, uv, z


def build_bucket(uvd_local: np.ndarray, S: int, grid_n: int, K_per_cell: int,
                  rng: np.random.RandomState | None = None):
    """Replicate datasets/pandaset_full.py bucket layout.

    uvd_local is (n, C) with C ∈ {3, 4} — 3 for legacy uv+d caches, 4 when
    the dataset feeds intensity as the 4th channel (V3-i caches that the
    current ckpts were trained on).

    `rng` controls the per-cell shuffle that decides which K points each
    over-full cell keeps. Pass a seeded RandomState for run-to-run
    determinism (e.g. for tests or repeatable inference); pass None to
    use the global numpy RNG (legacy training behaviour)."""
    G = grid_n
    cell_S = float(S) / G
    C = uvd_local.shape[-1]
    cu = np.clip((uvd_local[:, 0] / cell_S).astype(np.int32), 0, G - 1)
    cv = np.clip((uvd_local[:, 1] / cell_S).astype(np.int32), 0, G - 1)
    cell_id = cv * G + cu
    n = uvd_local.shape[0]
    if n == 0:
        return (np.zeros((G * G, K_per_cell, C), dtype=np.float32),
                np.zeros((G * G, K_per_cell), dtype=bool))
    shuf = (rng.permutation(n) if rng is not None
            else np.random.permutation(n))
    sorted_idx = shuf[np.argsort(cell_id[shuf], kind="stable")]
    sorted_uvd = uvd_local[sorted_idx]
    sorted_cid = cell_id[sorted_idx]
    counts = np.bincount(sorted_cid, minlength=G * G)
    cell_starts = np.zeros(G * G + 1, dtype=np.int64)
    cell_starts[1:] = counts.cumsum()
    intra = np.arange(n, dtype=np.int64) - cell_starts[sorted_cid]
    keep = intra < K_per_cell
    slots = intra[keep]
    cells = sorted_cid[keep]
    bucket_uvd = np.zeros((G * G, K_per_cell, C), dtype=np.float32)
    bucket_valid = np.zeros((G * G, K_per_cell), dtype=bool)
    bucket_uvd[cells, slots] = sorted_uvd[keep]
    bucket_valid[cells, slots] = True
    return bucket_uvd, bucket_valid


@torch.no_grad()
def infer_tiles(model, img: np.ndarray, uv: np.ndarray, z: np.ndarray, K: np.ndarray,
                 ba_cfg: dict, device: torch.device, y_cam: np.ndarray | None = None,
                 intensity: np.ndarray | None = None,
                 bucket_rng: np.random.RandomState | None = None):
    # Inference must be deterministic: same image+pts in → same params out.
    # Training relies on the global numpy RNG to randomise per-cell point
    # selection (regularisation), but at inference we want a fixed-seed
    # shuffle so subsampling cells doesn't drift run-to-run.
    if bucket_rng is None:
        bucket_rng = np.random.RandomState(0)
    """Batched sliding-tile inference: one forward call per frame (B = n_tiles)
    instead of one per tile. Returns (uv_full[N,2], par[N,5], z_cam[N]) in
    FULL-image px units, identical contract to the per-tile version.

    ba_cfg knobs:
      - exclude_ground_y_cam : float | None
          When set, drop pts with cam-Y >= this value (in meters). PS cams sit
          ~1.5m above road → pts with y_cam ≳ +1.0 are road/ground, which the
          σ-head reports as low-σ (close + tight by pixel) and thus dominates
          BA weights despite carrying NO useful calibration signal. Excluding
          ground is required for BA to lock onto edges of buildings / vehicles
          / signs."""
    H, W = img.shape[:2]
    TILE = ba_cfg["tile_size"]
    S = ba_cfg["model_input_size"]
    max_pts = ba_cfg["max_pts_per_tile"]
    min_pts = ba_cfg["min_pts_per_tile"]
    stride = ba_cfg.get("tile_stride", 128)
    scale = TILE / S
    y_thresh = ba_cfg.get("exclude_ground_y_cam", None)
    us = list(range(0, max(W - TILE, 0) + 1, stride))
    vs = list(range(0, max(H - TILE, 0) + 1, stride))
    if us and us[-1] != W - TILE:
        us.append(W - TILE)
    if vs and vs[-1] != H - TILE:
        vs.append(H - TILE)

    # Pass 1: collect per-tile crop/uvd/idx (CPU). Skip tiles with < min_pts.
    crops_np: list = []
    uvd_np: list = []
    pad_mask_np: list = []
    buc_np: list = []
    buc_v_np: list = []
    idx_per_tile: list = []
    for v0 in vs:
        for u0 in us:
            crop = img[v0:v0 + TILE, u0:u0 + TILE]
            cH, cW = crop.shape[:2]
            if cH < TILE // 2 or cW < TILE // 2:
                continue
            if cH < TILE or cW < TILE:
                pad = np.zeros((TILE, TILE, 3), dtype=np.uint8)
                pad[:cH, :cW] = crop
                crop = pad
            in_tile = ((uv[:, 0] >= u0) & (uv[:, 0] < u0 + TILE) &
                       (uv[:, 1] >= v0) & (uv[:, 1] < v0 + TILE) & (z > 0.5))
            if y_thresh is not None and y_cam is not None:
                in_tile &= (y_cam < y_thresh)
            if int(in_tile.sum()) < min_pts:
                continue
            idx = np.where(in_tile)[0]
            if len(idx) > max_pts:
                # Deterministic top-K by closest depth (most reliable per-pt).
                # Was np.random.choice — pool was noisy across reruns.
                idx = idx[np.argsort(z[idx])[:max_pts]]
            u_local = (uv[idx, 0] - u0) * (S / TILE)
            v_local = (uv[idx, 1] - v0) * (S / TILE)
            z_local = z[idx]
            X = (uv[idx, 0] - K[0, 2]) * z_local / K[0, 0]
            Y = (uv[idx, 1] - K[1, 2]) * z_local / K[1, 1]
            d_eucl = np.sqrt(X * X + Y * Y + z_local * z_local)
            cols = [u_local, v_local, (d_eucl / 100.0).astype(np.float32)]
            use_intensity = bool(getattr(model, 'use_intensity', False))
            if use_intensity:
                if intensity is not None:
                    intens_pt = intensity[idx].astype(np.float32)
                else:
                    intens_pt = np.zeros_like(u_local, dtype=np.float32)
                cols.append(intens_pt)
            uvd_t = np.stack(cols, axis=1).astype(np.float32)
            C = uvd_t.shape[-1]
            # Pad to max_pts, build mask (True = pad / ignore).
            n = len(idx)
            uvd_pad = np.zeros((max_pts, C), dtype=np.float32)
            uvd_pad[:n] = uvd_t
            mask = np.ones(max_pts, dtype=bool)
            mask[:n] = False
            buc, buc_v = build_bucket(uvd_t, S, grid_n=16, K_per_cell=8,
                                       rng=bucket_rng)
            crop_r = np.asarray(Image.fromarray(crop).resize((S, S)))
            crops_np.append(crop_r)
            uvd_np.append(uvd_pad)
            pad_mask_np.append(mask)
            buc_np.append(buc)
            buc_v_np.append(buc_v)
            idx_per_tile.append((u0, v0, idx))

    if not crops_np:
        return None

    # Pass 2: one batched forward (B = n_tiles). vfp is per-tile but TILE is
    # constant across this frame so vfp is identical for every tile.
    B = len(crops_np)
    vfp = float(K[0, 0]) * S / TILE
    img_t = torch.from_numpy(np.stack(crops_np, axis=0)).permute(0, 3, 1, 2).float().div(255.0).to(device)
    dist_uvd = torch.from_numpy(np.stack(uvd_np, axis=0)).to(device)
    pad_mask = torch.from_numpy(np.stack(pad_mask_np, axis=0)).to(device)
    vfp_t = torch.full((B,), vfp, dtype=torch.float32, device=device)
    buc_t = torch.from_numpy(np.stack(buc_np, axis=0)).to(device)
    buc_v_t = torch.from_numpy(np.stack(buc_v_np, axis=0)).to(device)
    # fp16 to match training (accelerate launch --mixed_precision=fp16). bf16
    # would silently fall back to fp32 emulation on V100 (sm_70) and drift from
    # the trained distribution. Revisit when we run on Blackwell (sm_100+).
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(img_t, dist_uvd, key_padding_mask=pad_mask,
                    vfp=vfp_t, bucket_uvd=buc_t, bucket_valid=buc_v_t)
    params = out[0] if isinstance(out, tuple) else out
    params_np = params.float().cpu().numpy()  # (B, max_pts, 5)

    # Pass 3: per-tile slicing (real-only) → full-image px.
    out_uv, out_par, out_z = [], [], []
    for ti, (u0, v0, idx) in enumerate(idx_per_tile):
        n = len(idx)
        p = params_np[ti, :n]
        z_local = z[idx]
        out_uv.append(uv[idx])
        out_par.append(np.column_stack([
            p[:, 0] * scale,
            p[:, 1] * scale,
            np.exp(p[:, 2]) * scale,
            np.exp(p[:, 3]) * scale,
            np.tanh(p[:, 4]) * 0.99,
        ]))
        out_z.append(z_local)
    if not out_uv:
        return None
    return np.concatenate(out_uv), np.concatenate(out_par), np.concatenate(out_z)


# ── BA DoF library: name → (du(X,Y,Z,uv,K), dv(X,Y,Z,uv,K)) ───────────────
# All angles in DEGREES (d2r applied in solver), translations in METERS, fx
# in PIXELS. Sign convention: positive delta = correction to APPLY on top of
# the declared world→cam SE(3) / intrinsics so the projection lines up.
# Keep the legacy 3-DoF aliases ('df_common' under both 'dfx' and 'df_common'
# names) for back-compat with existing configs.
_D2R = np.pi / 180.0


def _jac_omega_x(X, Y, Z, uv, K):
    fx, fy = K[0, 0], K[1, 1]
    return -(fx * X * Y) / (Z * Z) * _D2R, (-fy - (fy * Y * Y) / (Z * Z)) * _D2R


def _jac_omega_y(X, Y, Z, uv, K):
    fx, fy = K[0, 0], K[1, 1]
    return (fx + (fx * X * X) / (Z * Z)) * _D2R, (fy * X * Y) / (Z * Z) * _D2R


def _jac_omega_z(X, Y, Z, uv, K):
    # Rotation about cam optical axis. Cross-couples with center offset but
    # PandaSet/Waymo lens roll is usually <0.01°.
    fx, fy = K[0, 0], K[1, 1]
    return -fx * Y / Z * _D2R, fy * X / Z * _D2R


def _jac_tx(X, Y, Z, uv, K):
    return K[0, 0] / Z, np.zeros_like(Z)


def _jac_ty(X, Y, Z, uv, K):
    return np.zeros_like(Z), K[1, 1] / Z


def _jac_tz(X, Y, Z, uv, K):
    # Mostly coupled with depth scale and dfx; include only when you really
    # mean a longitudinal mount-position bias (rare on automotive rigs).
    return -K[0, 0] * X / (Z * Z), -K[1, 1] * Y / (Z * Z)


def _jac_dfx(X, Y, Z, uv, K):
    fx, cx = K[0, 0], K[0, 2]
    return (uv[:, 0] - cx) / fx, np.zeros_like(Z)


def _jac_dfy(X, Y, Z, uv, K):
    fy, cy = K[1, 1], K[1, 2]
    return np.zeros_like(Z), (uv[:, 1] - cy) / fy


def _jac_df_common(X, Y, Z, uv, K):
    # Legacy 3-DoF "Δfx" — actually a common Δf in px applied symmetrically
    # to fx and fy (cameras with fixed aspect). Kept for back-compat with
    # the 3-DoF result tables in memory project_ps_calib_full_picture.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return (uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy


DOF_JAC = {
    "omega_x":   _jac_omega_x,
    "omega_y":   _jac_omega_y,
    "omega_z":   _jac_omega_z,
    "tx":        _jac_tx,
    "ty":        _jac_ty,
    "tz":        _jac_tz,
    "dfx":       _jac_dfx,
    "dfy":       _jac_dfy,
    "df_common": _jac_df_common,
}

# Default DoF lists for the legacy `param` string aliases.
_DOF_PRESETS = {
    "3dof": ["omega_x", "omega_y", "df_common"],
    "5dof": ["omega_x", "omega_y", "tx", "ty", "df_common"],
    "6dof": ["omega_x", "omega_y", "omega_z", "tx", "ty", "df_common"],
    "7dof": ["omega_x", "omega_y", "omega_z", "tx", "ty", "tz", "df_common"],
    # Pure extrinsic 6-DoF (no intrinsic): for the CaaaS sequence endpoint.
    "6dof_ext": ["omega_x", "omega_y", "omega_z", "tx", "ty", "tz"],
}


def resolve_dof_list(ba_cfg: dict) -> list:
    """ba.dof (explicit list) wins; else ba.param string preset; else 3dof."""
    if isinstance(ba_cfg.get("dof"), list) and ba_cfg["dof"]:
        return list(ba_cfg["dof"])
    return list(_DOF_PRESETS.get(ba_cfg.get("param", "3dof"),
                                  _DOF_PRESETS["3dof"]))


def solve_dofs(uv: np.ndarray, par: np.ndarray, z: np.ndarray, K: np.ndarray,
                dof_names: list, damping: float = 1e-3,
                huber_k: float | None = None, n_iter: int = 1):
    """Linearized GN over a config-supplied list of DoFs. Returns δ of
    len(dof_names) in declaration order.

    If `huber_k` is set, runs IRLS for `n_iter` iterations with a Huber
    M-estimator on the per-point Mahalanobis distance — same H = ΣJᵀWJ
    pipeline, just with W_i scaled by w(d_i) = min(1, k/d_i) where
    d_i² = r_iᵀ W_i r_i (post-correction residual). Plain 1-step (no
    outlier handling) when huber_k is None.
    """
    for name in dof_names:
        if name not in DOF_JAC:
            raise KeyError(f"unknown DoF '{name}' — valid: {sorted(DOF_JAC)}")
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (uv[:, 0] - cx) * z / fx
    Y = (uv[:, 1] - cy) * z / fy
    Z = z
    Jus, Jvs = [], []
    for name in dof_names:
        ju, jv = DOF_JAC[name](X, Y, Z, uv, K)
        Jus.append(np.broadcast_to(ju, Z.shape))
        Jvs.append(np.broadcast_to(jv, Z.shape))
    J_u = np.column_stack(Jus)
    J_v = np.column_stack(Jvs)
    r_u0, r_v0 = par[:, 0], par[:, 1]
    su, sv, rho = par[:, 2], par[:, 3], par[:, 4]
    det = su * su * sv * sv * (1 - rho * rho)
    Wuu0 = (sv * sv) / det
    Wvv0 = (su * su) / det
    Wuv0 = -(rho * su * sv) / det
    n = len(dof_names)

    delta = np.zeros(n)
    weights = np.ones_like(r_u0)
    iters = max(1, int(n_iter)) if huber_k is not None else 1
    for it in range(iters):
        Wuu = Wuu0 * weights
        Wvv = Wvv0 * weights
        Wuv = Wuv0 * weights
        H = np.zeros((n, n)); b = np.zeros(n)
        for i in range(n):
            for j in range(n):
                H[i, j] = ((J_u[:, i] * Wuu * J_u[:, j]).sum()
                           + (J_v[:, i] * Wvv * J_v[:, j]).sum()
                           + (J_u[:, i] * Wuv * J_v[:, j]).sum()
                           + (J_v[:, i] * Wuv * J_u[:, j]).sum())
            b[i] = ((J_u[:, i] * Wuu * r_u0).sum()
                    + (J_v[:, i] * Wvv * r_v0).sum()
                    + (J_u[:, i] * Wuv * r_v0).sum()
                    + (J_v[:, i] * Wuv * r_u0).sum())
        H += damping * np.eye(n)
        delta = np.linalg.solve(H, b)
        if huber_k is None:
            break
        # Post-correction residuals & Mahalanobis distance per point
        ru = r_u0 - J_u @ delta
        rv = r_v0 - J_v @ delta
        d2 = (ru * Wuu0 * ru) + (rv * Wvv0 * rv) + 2.0 * (ru * Wuv0 * rv)
        d = np.sqrt(np.maximum(d2, 1e-12))
        weights = np.where(d <= huber_k, 1.0, huber_k / d)
    # Cov(δ) = H^{-1}; useful for downstream (CaaaS, sequence-fuse).
    solve_dofs._last_cov = np.linalg.inv(H)
    solve_dofs._last_H = H
    solve_dofs._last_b = b
    solve_dofs._last_weights = weights
    return delta


# Back-compat thin wrappers — keep old call sites working.
def solve_3dof(uv, par, z, K, damping=1e-3):
    return solve_dofs(uv, par, z, K, _DOF_PRESETS["3dof"], damping)


def solve_5dof(uv, par, z, K, damping=1e-3):
    return solve_dofs(uv, par, z, K, _DOF_PRESETS["5dof"], damping)


def delta_to_dict(delta: np.ndarray, dof_names: list) -> dict:
    """Map solver output → human-readable dict keyed by DoF name with
    unit-aware values. Angles → degrees, translations → meters, fx → px."""
    out = {}
    for i, name in enumerate(dof_names):
        out[name] = float(delta[i])
    return out


def make_T_corr_from_dofs(dof_vals: dict) -> np.ndarray:
    """Build SE(3) T_corr from the DoF-value dict. Rotation = ZYX Euler in
    degrees (omega_z, omega_y, omega_x); translation in meters."""
    ox = dof_vals.get("omega_x", 0.0)
    oy = dof_vals.get("omega_y", 0.0)
    oz = dof_vals.get("omega_z", 0.0)
    # Compose in same order make_T_corr did for back-compat: xy Euler when
    # no z. With z present, use ZYX standard right-handed.
    if abs(oz) < 1e-12:
        R = Rotation.from_euler("xy", [ox, oy], degrees=True).as_matrix()
    else:
        R = Rotation.from_euler("zyx", [oz, oy, ox], degrees=True).as_matrix()
    T = np.eye(4); T[:3, :3] = R
    T[0, 3] = dof_vals.get("tx", 0.0)
    T[1, 3] = dof_vals.get("ty", 0.0)
    T[2, 3] = dof_vals.get("tz", 0.0)
    return T


def make_T_corr(ox_deg: float, oy_deg: float,
                tx_m: float = 0.0, ty_m: float = 0.0) -> np.ndarray:
    """Legacy positional API kept for callers outside this module."""
    return make_T_corr_from_dofs({
        "omega_x": ox_deg, "omega_y": oy_deg,
        "tx": tx_m, "ty": ty_m,
    })


def render_before_after(out_dir: Path, scene: str, frame: int, cam: str,
                         frame_data, dof_vals: dict):
    img, K, T_cw, pts_w, _, uv, z = frame_data
    # Apply BA corrections to BOTH the extrinsic (rotation + translation) and
    # the intrinsic (Δfx / Δfy / common Δf) so AFTER reflects everything we
    # solved for, not just rotation.
    K_new = K.copy()
    df_px = dof_vals.get("dfx", dof_vals.get("df_common", 0.0))
    if df_px:
        K_new[0, 0] = K[0, 0] + df_px
    df_common = dof_vals.get("df_common", 0.0)
    if df_common:
        K_new[1, 1] = K[1, 1] + df_common
    dfy_px = dof_vals.get("dfy", 0.0)
    if dfy_px:
        K_new[1, 1] = K[1, 1] + dfy_px
    T_corr = make_T_corr_from_dofs(dof_vals)
    T_cw_new = (T_corr @ T_cw).astype(np.float32)
    homo = np.column_stack([pts_w, np.ones(len(pts_w), dtype=np.float32)])
    pts_c2 = (T_cw_new @ homo.T).T[:, :3]
    z2 = pts_c2[:, 2]
    uv2 = (K_new @ pts_c2.T).T[:, :2] / np.maximum(z2[:, None], 1e-6)
    H_, W_ = img.shape[:2]
    v1 = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < W_) & (uv[:, 1] >= 0) & (uv[:, 1] < H_)
    v2 = (z2 > 0.5) & (uv2[:, 0] >= 0) & (uv2[:, 0] < W_) & (uv2[:, 1] >= 0) & (uv2[:, 1] < H_)
    # Keep native image resolution so 1-px shifts are visible. figsize × dpi
    # must equal (W_, 2*H_) exactly — don't shrink, don't crop with bbox_inches.
    fig, axes = plt.subplots(2, 1, figsize=(W_ / 100.0, H_ / 100.0 * 2), dpi=100)
    axes[0].imshow(img)
    axes[0].scatter(uv[v1, 0], uv[v1, 1], s=0.3, c=z[v1],
                     cmap="turbo_r", alpha=0.7, vmin=1, vmax=80, marker=".")
    axes[0].set_title(f"{cam} sc={scene} f={frame} — BEFORE")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].imshow(img)
    axes[1].scatter(uv2[v2, 0], uv2[v2, 1], s=0.3, c=z2[v2],
                     cmap="turbo_r", alpha=0.7, vmin=1, vmax=80, marker=".")
    title_bits = []
    if "omega_x" in dof_vals: title_bits.append(f"ωx={dof_vals['omega_x']:+.3f}°")
    if "omega_y" in dof_vals: title_bits.append(f"ωy={dof_vals['omega_y']:+.3f}°")
    if "omega_z" in dof_vals: title_bits.append(f"ωz={dof_vals['omega_z']:+.3f}°")
    if "tx"      in dof_vals: title_bits.append(f"tx={dof_vals['tx']*100:+.1f}cm")
    if "ty"      in dof_vals: title_bits.append(f"ty={dof_vals['ty']*100:+.1f}cm")
    if "tz"      in dof_vals: title_bits.append(f"tz={dof_vals['tz']*100:+.1f}cm")
    if df_px:    title_bits.append(f"Δfx={df_px:+.1f}px")
    if dfy_px:   title_bits.append(f"Δfy={dfy_px:+.1f}px")
    axes[1].set_title(f"{cam} — AFTER  " + " ".join(title_bits))
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.tight_layout()
    fp = out_dir / f"{scene}_f{frame:02d}_{cam}.png"
    plt.savefig(fp, dpi=100)
    plt.close(fig)
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"Config: {args.config} (sha256={cfg['_sha256']})")
    if 'description' in cfg:
        print(f"  description: {cfg['description']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"], device)

    root = Path(cfg["data"]["root"])
    scenes = cfg["data"]["scenes"]
    if scenes == "all":
        scenes = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit()])
    frames_cfg = cfg["data"]["frames"]
    cams = cfg["data"]["cams"]
    ba_cfg = cfg["ba"]

    vis_dir = Path(cfg["output"]["vis_dir"])
    vis_dir.mkdir(parents=True, exist_ok=True)
    for old in vis_dir.glob("*.png"):
        old.unlink()
    vis_scenes = set(cfg["output"]["vis_scenes"])
    vis_frames = set(cfg["output"]["vis_frames"])

    # Cache one frame_data per (scene, cam, frame) for vis (lazy)
    vis_cache = {}

    # Pool residuals across all (scene, frame) per cam
    corrections = {}
    t0 = time.time()
    for cam in cams:
        print(f"\n[BA] cam={cam}")
        pooled_uv, pooled_par, pooled_z = [], [], []
        K_ref = None
        per_scene = 0
        per_frame = 0
        for scene in scenes:
            cam_dir = root / scene / "camera" / cam
            if not cam_dir.exists():
                continue
            poses_p = cam_dir / "poses.json"
            if not poses_p.exists():
                continue
            n_frames = len(json.loads(poses_p.read_text()))
            if frames_cfg == "all":
                fr_iter = range(n_frames)
            else:
                fr_iter = [f for f in frames_cfg if f < n_frames]
            scene_used = False
            for frame in fr_iter:
                fd = load_frame(root, scene, cam, frame)
                if fd is None:
                    continue
                if K_ref is None:
                    K_ref = fd[1]
                if scene in vis_scenes and frame in vis_frames:
                    vis_cache[(scene, frame, cam)] = fd
                img, K, T_cw, pts_w, pts_c, uv, z = fd
                y_cam = pts_c[:, 1]  # cam-Y axis = downward; ground ≳ +1m
                res = infer_tiles(model, img, uv, z, K, ba_cfg, device, y_cam=y_cam)
                if res is None:
                    continue
                u, p, zc = res
                pooled_uv.append(u); pooled_par.append(p); pooled_z.append(zc)
                per_frame += 1
                scene_used = True
            if scene_used:
                per_scene += 1
            if per_scene % 10 == 0 and scene_used:
                el = time.time() - t0
                n_pt_so_far = sum(len(x) for x in pooled_uv)
                print(f"  {cam}: scenes={per_scene} frames={per_frame} pts={n_pt_so_far} t={el:.0f}s")
        if not pooled_uv:
            print(f"  {cam}: NO DATA")
            continue
        UV = np.concatenate(pooled_uv)
        PAR = np.concatenate(pooled_par)
        Z = np.concatenate(pooled_z)
        dof_names = resolve_dof_list(ba_cfg)
        delta = solve_dofs(UV, PAR, Z, K_ref, dof_names, damping=ba_cfg["damping"])
        dof_vals = delta_to_dict(delta, dof_names)
        fx_ref = float(K_ref[0, 0])
        rec = dict(dof_vals)
        rec["fx_ref"] = fx_ref
        # Δfx-style DoFs are reported as % of fx for quick reading.
        for k in ("dfx", "df_common"):
            if k in dof_vals:
                rec[f"{k}_pct"] = dof_vals[k] / fx_ref * 100.0
        if "dfy" in dof_vals:
            rec["dfy_pct"] = dof_vals["dfy"] / float(K_ref[1, 1]) * 100.0
        rec.update(n_scenes=int(per_scene), n_frames=int(per_frame),
                   n_pts=int(len(UV)))
        corrections[cam] = rec
        # Human-readable one-liner: only print the DoFs actually solved.
        bits = []
        if "omega_x"   in dof_vals: bits.append(f"ωx={dof_vals['omega_x']:+.4f}°")
        if "omega_y"   in dof_vals: bits.append(f"ωy={dof_vals['omega_y']:+.4f}°")
        if "omega_z"   in dof_vals: bits.append(f"ωz={dof_vals['omega_z']:+.4f}°")
        if "tx"        in dof_vals: bits.append(f"tx={dof_vals['tx']*100:+.1f}cm")
        if "ty"        in dof_vals: bits.append(f"ty={dof_vals['ty']*100:+.1f}cm")
        if "tz"        in dof_vals: bits.append(f"tz={dof_vals['tz']*100:+.1f}cm")
        for k in ("dfx", "df_common"):
            if k in dof_vals:
                bits.append(f"Δfx={dof_vals[k]:+.2f}px ({dof_vals[k]/fx_ref*100:+.3f}%)")
        if "dfy" in dof_vals:
            bits.append(f"Δfy={dof_vals['dfy']:+.2f}px "
                        f"({dof_vals['dfy']/float(K_ref[1,1])*100:+.3f}%)")
        print(f"  → " + "  ".join(bits) +
              f"  scenes={per_scene} frames={per_frame} pts={len(UV)}")

    # BEFORE/AFTER vis
    print("\n[VIS] rendering BEFORE/AFTER")
    for (scene, frame, cam), fd in sorted(vis_cache.items()):
        if cam not in corrections:
            continue
        # Extract the DoF-named keys (e.g. omega_x, tx, dfx) — skip
        # bookkeeping fields like n_pts / fx_ref / *_pct.
        rec = corrections[cam]
        dof_vals = {k: v for k, v in rec.items() if k in DOF_JAC}
        fp = render_before_after(vis_dir, scene, frame, cam, fd, dof_vals)
        print(f"  {fp}")

    # Provenance
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()[:10]
    except Exception:
        git_sha = "unknown"
    ckpt_p = Path(cfg["model"]["ckpt"])
    out_obj = {
        "provenance": {
            "config_path": cfg["_path"],
            "config_sha256_prefix": cfg["_sha256"],
            "git_sha": git_sha,
            "run_started": datetime.now().isoformat(timespec="seconds"),
            "model_ckpt": str(ckpt_p),
            "model_ckpt_mtime": datetime.fromtimestamp(ckpt_p.stat().st_mtime).isoformat(timespec="seconds"),
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "ba_param": "+".join(resolve_dof_list(ba_cfg)),
        "corrections": corrections,
    }
    out_p = Path(cfg["output"]["corrections_json"])
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out_obj, indent=2))
    print(f"\nwrote → {out_p}")
    print(f"vis  → {vis_dir}")


if __name__ == "__main__":
    main()
