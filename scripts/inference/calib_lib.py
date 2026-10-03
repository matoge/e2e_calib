"""Consolidated LiDAR–camera calibration inference library.

Two layers:

Layer 1 — CORE (pure, no I/O)
    calibrate_frame(model, image, pts_cam, uv_hat, z, intensity, K, hood_mask=None)
        → CalibResult(delta_cam, H_cam, diag)

    pool_frames([CalibResult, ...], mode='gate3', gate_c=3.0)
        → CalibResult(delta_joint_cam, H_joint_cam, k, diag)

    cam_delta_to_setting(delta_cam, setting_rot, setting_mp)
        → dict(rot_x=..., rot_y=..., rot_z=..., mp_x=..., mp_y=..., mp_z=...)
        # values are the SIGN-CORRECT additive delta for setting.json (rad + m)

Layer 2 — DATA LOADER (file-system aware, wraps build_woven_sequence_v3)
    load_frame_data(seq_dir, frame_idx, hood_mask_root=None)
        → FrameData
        # * time-discrepancy compensated (_pose_at_camera_time)
        # * KB4 projected into camera plane
        # * hood mask auto-resolved by car_id + applied

    build_model(exp_dir, device='cuda') → CalibNet2 in eval mode

All the tile-building / GN solving lives in one place. Both the CLI scripts
(apply_frame0.py, apply_sequence.py) and the FastAPI server import from here.
"""
from __future__ import annotations
import importlib.util, sys, json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts' / 'preprocessing'))

from build_woven_sequence_v3 import (
    _load_setting, _camera_calib_fcm, _load_metadata, _get_poses,
    _camera_delay_ms_for_frame, _pose_at_camera_time,
    _T_lidar_to_cam_at_camera_time, _load_pts_intensity,
)
from scripts.util.projection import project_lidar_into_image
from models.calibnet2 import CalibNet2
from scripts.ba.ba_torch import solve_pinhole_xyz
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Constants matching training config (S3 cam6, per-cache-crop 512 → resize 256)
# ---------------------------------------------------------------------------
CS = 512
S  = 256
GRID_N = 16
K_PER_CELL = 8
DOF = ('omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz')
R_TO_RDF = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class FrameData:
    """Everything you need to feed the model for one frame. Time-discrepancy
    already compensated by _pose_at_camera_time inside the loader."""
    fid: str
    frame_idx: int
    image_rgb: np.ndarray            # (IH, IW, 3) uint8
    pts_cam: np.ndarray              # (N, 3) metres, cam-frame at shutter time
    uv_hat: np.ndarray               # (N, 2) px, KB4 projection of pts_cam
    z: np.ndarray                    # (N,) depth
    intensity: np.ndarray            # (N,) 0..255
    K: np.ndarray                    # (3, 3) intrinsics
    dist: np.ndarray                 # (4,) KB4 k1..k4
    T_cl: np.ndarray                 # (4, 4) LiDAR→cam at camera time
    IW: int
    IH: int
    delay_ms: float
    R_cam_from_veh: np.ndarray       # (3, 3) — needed for setting.json ↔ cam frame
    setting_rot: np.ndarray          # (3,) rad [roll, pitch, yaw]  (current setting)
    setting_mp:  np.ndarray          # (3,) m   [x, y, z]           (current setting)
    n_hood_dropped: int = 0


@dataclass
class CalibResult:
    """Output of one-frame or joint calibration.
    All 6-DoF numbers are in CAMERA FRD frame: rot in degrees, trans in metres."""
    delta_cam: np.ndarray            # (6,) [ω_x, ω_y, ω_z, tx, ty, tz]
    H_cam: np.ndarray                # (6, 6) information matrix
    n_frames: int = 1
    n_tiles_fused: int = 0
    n_points: int = 0
    k: float = float('nan')          # χ²/6 measured across pooled frames (nan for single)
    per_frame: list = field(default_factory=list)   # list of dicts for post-hoc

    def cov(self, k_correct: bool = True) -> np.ndarray:
        c = np.linalg.inv(self.H_cam)
        if k_correct and np.isfinite(self.k) and self.k > 0:
            c = c * self.k
        return c

    def sd(self, k_correct: bool = True) -> np.ndarray:
        return np.sqrt(np.diag(self.cov(k_correct)))


# ---------------------------------------------------------------------------
# Layer 2 — data loader
# ---------------------------------------------------------------------------
def load_frame_data(seq_dir: Path, frame_idx: int,
                     hood_mask_root: Optional[Path] = None) -> FrameData:
    """Read a WovenSequence-format frame, KB4-project LiDAR with proper time
    compensation, optionally drop hood-mask points. Everything you need for
    model inference."""
    seq_dir = Path(seq_dir)
    setting = _load_setting(seq_dir)
    K, dist, R_cv, t_cv, IW, IH, delay_default = _camera_calib_fcm(setting)
    metadata = _load_metadata(seq_dir)
    frame_ids, poses = _get_poses(metadata)
    fid = frame_ids[frame_idx]

    delay_ms = _camera_delay_ms_for_frame(metadata, fid, delay_default)
    pose_curr = poses[fid]
    pose_cam = _pose_at_camera_time(poses, frame_ids, frame_idx, delay_ms)
    T_cl = _T_lidar_to_cam_at_camera_time(pose_curr, pose_cam, R_cv, t_cv)

    pts_flu, intensity = _load_pts_intensity(seq_dir, fid)
    pts_xyzi = np.column_stack([pts_flu.astype(np.float32),
                                 intensity.astype(np.float32)])
    _, pts_cam, uv_full, z_full, int_full = project_lidar_into_image(
        pts_xyzi, K, T_cl, IW, IH, is_fisheye=True, dist=dist, z_min=0.5)

    # ---- hood mask ----
    n_dropped = 0
    if hood_mask_root is not None:
        car_id = metadata.get('car_id')
        mp = hood_mask_root / f'car{car_id}.png' if car_id else None
        if mp and mp.exists():
            hood = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            uu = uv_full[:, 0].round().astype(int).clip(0, IW - 1)
            vv = uv_full[:, 1].round().astype(int).clip(0, IH - 1)
            keep = hood[vv, uu] == 0
            n_dropped = int((~keep).sum())
            pts_cam  = pts_cam[keep]
            uv_full  = uv_full[keep]
            z_full   = z_full[keep]
            int_full = int_full[keep]

    img_bgr = cv2.imread(str(seq_dir / 'tss4_fcm' / f'{fid}.jpg'))
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ---- extract setting rot / mp for downstream unit conversion ----
    setting_rot = np.array(setting['fcm']['rot'], dtype=np.float64)  # (3,) rad
    setting_mp  = np.array(setting['fcm']['mp'],  dtype=np.float64)  # (3,) m

    return FrameData(fid=fid, frame_idx=frame_idx,
                     image_rgb=image_rgb,
                     pts_cam=pts_cam, uv_hat=uv_full, z=z_full,
                     intensity=int_full, K=K.astype(np.float64), dist=dist,
                     T_cl=T_cl, IW=IW, IH=IH, delay_ms=delay_ms,
                     R_cam_from_veh=R_cv.astype(np.float64),
                     setting_rot=setting_rot, setting_mp=setting_mp,
                     n_hood_dropped=n_dropped)


def build_model(exp_dir: Path, device: str = 'cuda') -> CalibNet2:
    """Instantiate CalibNet2 with the exact hyperparameters saved in
    experiments/<name>/config.py and load best_model.pt."""
    exp_dir = Path(exp_dir)
    ck = exp_dir / 'best_model.pt'
    cfg_path = exp_dir / 'config.py'
    spec = importlib.util.spec_from_file_location('_cfg', cfg_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cfg = mod.CFG
    model = CalibNet2(img_size=cfg['img_size'],
                      frustum_grid_n=cfg.get('grid_n', 16),
                      n_iter=cfg['n_iter'],
                      use_info_head=cfg['use_info_head'],
                      point_mlp_fourier_n_freq=cfg.get('point_mlp_fourier_n_freq', 0),
                      in_channels=3).to(device).eval()
    sd = torch.load(ck, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model' in sd: sd = sd['model']
    miss, unexp = model.load_state_dict(sd, strict=False)
    assert not miss and not unexp, f'ckpt mismatch: {miss} {unexp}'
    return model


# ---------------------------------------------------------------------------
# Layer 1 — CORE inference (pure)
# ---------------------------------------------------------------------------
def _tile_grid(iw, ih, cs=CS):
    nx = max(1, int(np.ceil(iw / cs)))
    ny = max(1, int(np.ceil(ih / cs)))
    sx = max(0, iw - cs) / max(nx - 1, 1)
    sy = max(0, ih - cs) / max(ny - 1, 1)
    return [(int(round(sx * i)), int(round(sy * j)))
            for j in range(ny) for i in range(nx)]


def _stratified_grid(uv_loc_S, grid_n, max_per_cell=1):
    cell = np.clip((uv_loc_S / S * grid_n).astype(int), 0, grid_n - 1)
    cid = cell[:, 0] + cell[:, 1] * grid_n
    picks = []
    for c in np.unique(cid):
        idx = np.where(cid == c)[0]
        picks.extend(idx[:max_per_cell].tolist())
    return np.array(picks, dtype=int)


def _build_batch(img_full, uv_full, z_full, int_full, pts_cam_full, K, cells, dev):
    B = len(cells)
    Nq_max = GRID_N * GRID_N
    G2 = GRID_N * GRID_N
    imgs = np.zeros((B, 3, S, S), dtype=np.float32)
    dist_uvd = np.zeros((B, Nq_max, 4), dtype=np.float32)
    bucket_uvd = np.zeros((B, G2, K_PER_CELL, 4), dtype=np.float32)
    bucket_valid = np.zeros((B, G2, K_PER_CELL), dtype=bool)
    kpm = np.ones((B, Nq_max), dtype=bool)
    vfp = np.zeros(B, dtype=np.float32)
    per_tile = []
    scale = S / CS
    for b, (u0, v0) in enumerate(cells):
        m = ((uv_full[:, 0] >= u0) & (uv_full[:, 0] < u0 + CS)
             & (uv_full[:, 1] >= v0) & (uv_full[:, 1] < v0 + CS)
             & (z_full > 0.5))
        if int(m.sum()) < 8:
            per_tile.append(None); continue
        idx_in = np.where(m)[0]
        uv_loc_S = (uv_full[idx_in] - np.array([u0, v0], dtype=np.float32)) * scale
        d_norm = z_full[idx_in] / 100.0
        inten  = int_full[idx_in]

        cu = np.clip((uv_loc_S[:, 0] / (S / GRID_N)).astype(int), 0, GRID_N - 1)
        cv = np.clip((uv_loc_S[:, 1] / (S / GRID_N)).astype(int), 0, GRID_N - 1)
        cid = cv * GRID_N + cu
        uvd_raw = np.column_stack([uv_loc_S, d_norm, inten]).astype(np.float32)
        order = np.argsort(cid, kind='stable')
        cid_s = cid[order]; uvd_s = uvd_raw[order]
        counts = np.bincount(cid_s, minlength=G2)
        starts = np.zeros(G2 + 1, dtype=np.int64); starts[1:] = counts.cumsum()
        intra = np.arange(len(cid_s)) - starts[cid_s]
        keep = intra < K_PER_CELL
        bucket_uvd[b, cid_s[keep], intra[keep]] = uvd_s[keep]
        bucket_valid[b, cid_s[keep], intra[keep]] = True

        picks = _stratified_grid(uv_loc_S, GRID_N, 1)
        Nq = len(picks)
        dist_uvd[b, :Nq, 0] = uv_loc_S[picks, 0]
        dist_uvd[b, :Nq, 1] = uv_loc_S[picks, 1]
        dist_uvd[b, :Nq, 2] = d_norm[picks]
        dist_uvd[b, :Nq, 3] = inten[picks]
        kpm[b, :Nq] = False

        crop = img_full[v0:v0+CS, u0:u0+CS]
        crop_s = cv2.resize(crop, (S, S), interpolation=cv2.INTER_LINEAR)
        imgs[b] = crop_s.transpose(2, 0, 1).astype(np.float32) / 255.0
        vfp[b] = float(K[0, 0]) * S / CS
        per_tile.append(dict(u0=u0, v0=v0, idx=idx_in[picks],
                              pts_cam=pts_cam_full[idx_in[picks]],
                              uv_orig=uv_full[idx_in[picks]]))
    return (torch.from_numpy(imgs).to(dev),
            torch.from_numpy(dist_uvd).to(dev),
            torch.from_numpy(vfp).to(dev),
            torch.from_numpy(bucket_uvd).to(dev),
            torch.from_numpy(bucket_valid).to(dev),
            torch.from_numpy(kpm).to(dev),
            per_tile)


def calibrate_frame(model: CalibNet2, fd: FrameData,
                     device: str = 'cuda') -> CalibResult:
    """CORE: run the model on one frame (40 tiles), solve the outer GN, return
    per-frame δ_cam (deg / m) + information matrix.

    Zero I/O. The caller (Layer 2 loader or a bespoke driver) is responsible
    for time-discrepancy compensation before handing FrameData in — that
    happens inside load_frame_data via _pose_at_camera_time.
    """
    cells = _tile_grid(fd.IW, fd.IH, CS)
    imgs, q_in, vfp, buck, bvalid, kpm, per_tile = _build_batch(
        fd.image_rgb, fd.uv_hat, fd.z, fd.intensity, fd.pts_cam, fd.K,
        cells, device)

    with torch.no_grad():
        out = model(imgs, q_in, vfp=vfp, bucket_uvd=buck, bucket_valid=bvalid,
                    key_padding_mask=kpm, mode='calib')
    per_pt, W_head = (out if isinstance(out, tuple) else (out, None))

    all_pts, all_duv, all_W, all_valid = [], [], [], []
    for b, td in enumerate(per_tile):
        if td is None: continue
        Nq = len(td['idx'])
        mu_orig = per_pt[b, :Nq, :2].cpu().numpy() * (CS / S)
        all_pts.append(td['pts_cam']); all_duv.append(mu_orig)
        if W_head is not None:
            all_W.append(W_head[b, :Nq].cpu().numpy())
        else:
            sx = np.exp(per_pt[b, :Nq, 2].cpu().numpy()) * (CS/S)
            sy = np.exp(per_pt[b, :Nq, 3].cpu().numpy()) * (CS/S)
            W = np.zeros((Nq, 2, 2), dtype=np.float32)
            W[:, 0, 0] = 1.0 / (sx**2); W[:, 1, 1] = 1.0 / (sy**2)
            all_W.append(W)
        all_valid.append(np.ones(Nq, dtype=bool))
    if not all_pts:
        return CalibResult(delta_cam=np.zeros(6), H_cam=np.eye(6) * 1e-6,
                            n_frames=1, n_tiles_fused=0, n_points=0)

    pts_cam_all = np.concatenate(all_pts, 0)
    duv_all     = np.concatenate(all_duv, 0)
    W_all       = np.concatenate(all_W,   0)
    valid_all   = np.concatenate(all_valid, 0)

    pts_t = torch.from_numpy(pts_cam_all).double().unsqueeze(0).to(device)
    duv_t = torch.from_numpy(duv_all).double().unsqueeze(0).to(device)
    W_t   = torch.from_numpy(W_all).double().unsqueeze(0).to(device)
    K_t   = torch.from_numpy(fd.K.astype(np.float64)).unsqueeze(0).to(device)
    v_t   = torch.from_numpy(valid_all).unsqueeze(0).to(device)
    # robust='huber' at test time: InfoHead W already calibrates per-point
    # noise, but Huber bounds outlier influence (dynamic objects, OOD
    # scenes) that the learned W can't anticipate. Train path unchanged.
    delta, H = solve_pinhole_xyz(pts_t, duv_t, W_t, K_t, DOF,
                                  valid=v_t, n_iter=10, damping=1e-3,
                                  robust='huber', huber_k=1.5)
    return CalibResult(delta_cam=delta[0].cpu().numpy(),
                        H_cam=H[0].cpu().numpy(),
                        n_frames=1,
                        n_tiles_fused=sum(1 for t in per_tile if t is not None),
                        n_points=len(pts_cam_all))


def pool_frames(results: Sequence[CalibResult],
                mode: str = 'gate3', gate_c: float = 3.0,
                gate_iters: int = 2) -> CalibResult:
    """CORE: inverse-variance pool across per-frame CalibResults.

    Same math as scripts/eval/frame_fusion.py; here we work directly on
    (δ_cam, H_cam) rather than a saved .pt dump. The residual-space
    interpretation applies: r_f = δ_f is the "amount the current calibration
    is off" as reported by frame f."""
    if not results:
        return CalibResult(delta_cam=np.zeros(6), H_cam=np.eye(6)*1e-6)
    Hs = np.stack([r.H_cam for r in results], 0)             # (F, 6, 6)
    ds = np.stack([r.delta_cam for r in results], 0)         # (F, 6)
    keep = np.ones(len(Hs), dtype=bool)

    def _pool(H, d):
        Hsum = H.sum(0)
        bsum = np.einsum('fij,fj->i', H, d)
        return np.linalg.solve(Hsum, bsum), Hsum

    if mode == 'gate3':
        for _ in range(gate_iters):
            if keep.sum() < 2: break
            d_bar, _ = _pool(Hs[keep], ds[keep])
            e = ds - d_bar[None, :]
            c = np.einsum('fi,fij,fj->f', e, Hs, e) / 6.0
            new_keep = keep & (c <= gate_c)
            if new_keep.sum() == keep.sum(): break
            if new_keep.sum() < 2: break
            keep = new_keep
    elif mode == 'CI':
        Hs = Hs / len(Hs)  # uniform 1/F conservative fusion

    d_j, H_j = _pool(Hs[keep], ds[keep])
    # χ² over-dispersion measurement
    e = ds - d_j[None, :]
    chi2 = np.einsum('fi,fij,fj->f', e, Hs, e) / 6.0
    k = float(np.median(chi2[keep]))

    return CalibResult(delta_cam=d_j, H_cam=H_j,
                        n_frames=int(keep.sum()),
                        n_tiles_fused=sum(r.n_tiles_fused for i, r in enumerate(results) if keep[i]),
                        n_points=sum(r.n_points for i, r in enumerate(results) if keep[i]),
                        k=k,
                        per_frame=[dict(delta_cam=r.delta_cam.tolist(),
                                          H_cam=r.H_cam.tolist(),
                                          n_tiles=r.n_tiles_fused,
                                          n_points=r.n_points,
                                          kept=bool(keep[i]))
                                     for i, r in enumerate(results)])


# ---------------------------------------------------------------------------
# Coordinate conversion — CAM FRD (solver) → VEH FLU (setting.json)
# ---------------------------------------------------------------------------
def cam_delta_to_setting(delta_cam: np.ndarray,
                          R_cam_from_veh: np.ndarray) -> dict:
    """Convert solver output (cam FRD, deg + m) to LOOM-compatible ADDITIVE
    delta for setting.rot (rad, [roll, pitch, yaw]) and setting.mp (m).

    Sign: solver δ is the BIAS in the current calibration. The additive fix
    (LOOM: new = old + adj) is adj = −δ_vehicle.

    For small angles (< a few deg), the axis-angle vector transforms as
    a linear rotation: ω_veh = R_veh_from_cam @ ω_cam. Camera FRD → vehicle
    FLU axis components then map directly to (Δroll, Δpitch, Δyaw) since
    the vehicle-frame Euler is [roll (about X), pitch (about Y), yaw (about Z)]
    and X/Y/Z are the same veh basis axes the axis-angle is expressed in.
    """
    R_vc = np.linalg.inv(R_cam_from_veh)     # cam → veh
    omega_cam_rad = np.deg2rad(delta_cam[:3])
    omega_veh_rad = R_vc @ omega_cam_rad
    t_cam_m       = delta_cam[3:]
    t_veh_m       = R_vc @ t_cam_m
    return dict(
        rot_x = -float(omega_veh_rad[0]),
        rot_y = -float(omega_veh_rad[1]),
        rot_z = -float(omega_veh_rad[2]),
        mp_x  = -float(t_veh_m[0]),
        mp_y  = -float(t_veh_m[1]),
        mp_z  = -float(t_veh_m[2]),
    )


def cam_sd_to_setting(sd_cam: np.ndarray,
                       R_cam_from_veh: np.ndarray) -> dict:
    """Same transform for the σ, applied component-wise (rotation of the axis
    changes the axis labelling but preserves per-axis magnitudes when the
    covariance is treated as axis-aligned — this is the diagonal approximation
    the manual UI already uses)."""
    R_vc = np.linalg.inv(R_cam_from_veh)
    om = np.deg2rad(sd_cam[:3])
    om_veh = np.abs(R_vc) @ om
    t_veh  = np.abs(R_vc) @ sd_cam[3:]
    return dict(
        rot_x = float(om_veh[0]), rot_y = float(om_veh[1]), rot_z = float(om_veh[2]),
        mp_x  = float(t_veh[0]),  mp_y  = float(t_veh[1]),  mp_z  = float(t_veh[2]),
    )
