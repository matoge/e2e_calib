"""Solver only — pick the K points with smallest σ-ellipse (highest info)
   per sample, run BA with W=σ on those K points, and measure pose error
   + reproj error. Tests the hypothesis that σ is calibrated per-point but
   gets dragged down in solver by a few overconfident outliers.

   No training. Loads frozen baseline ckpt, no info_head needed."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.model_depth import CalibNetDepth
from scripts.ba.ba_torch import (
    solve_kb_xyz, make_info_from_sigma_rho, project_kb,
    _apply_extrinsic, _K_with_delta, _split_delta,
)

CACHE = Path.home() / 'cache' / 'kamikado_v3_tiled'
CKPT  = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'best_model.pt'
EXP_CFG_PATH = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'config.py'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IDX = 17
ROT = 0.30
T_M = 0.05
N_EVAL = 200
SEED = 7 + 1000
DOFS = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
KS = [10, 30, 60, 120, None]   # None = all valid


def _load_cfg():
    src = EXP_CFG_PATH.read_text(); ns = {}; exec(src, ns, ns); return ns['CFG']


def _build(cfg):
    return CalibNetDepth(
        img_size=cfg['img_size'], in_channels=cfg['in_channels'],
        n_layers=cfg['n_layers'],
        self_first=cfg.get('self_first', False),
        use_convnext=cfg.get('use_convnext', True),
        use_frustum=cfg.get('use_frustum', True),
        deform_mode=cfg.get('deform_mode', 'sl'),
        convnext_n_blocks=cfg.get('convnext_n_blocks', 2),
        convnext_fine_d=cfg.get('convnext_fine_d', None),
        convnext_stem_d=cfg.get('convnext_stem_d', None),
        use_info_head=True,
    )


def _major_axis(W2x2):
    """W → Σ → major axis (sqrt of largest eigenvalue of Σ).
       Returns (B,N). Smaller = higher info. Inputs in any units."""
    eye = torch.eye(2, device=W2x2.device,
                     dtype=W2x2.dtype).expand_as(W2x2)
    Sig = torch.linalg.inv(W2x2 + 1e-9 * eye)
    eigs = torch.linalg.eigvalsh(Sig)
    return eigs.clamp_min(0).max(dim=-1).values.sqrt()


def _select_topk_mask(major, valid, K):
    """For each sample pick K smallest-major-axis points among valid.
       major:(B,N), valid:(B,N) bool. Returns bool mask (B,N)."""
    B, N = major.shape
    if K is None or K >= N:
        return valid.clone()
    big = torch.full_like(major, float('inf'))
    score = torch.where(valid, major, big)
    idx = torch.topk(score, K, dim=-1, largest=False).indices  # (B,K)
    mask = torch.zeros_like(valid)
    mask.scatter_(1, idx, True)
    return mask & valid


def _project(P0, delta, K, dist, dofs):
    d = _split_delta(delta, dofs)
    omega = torch.stack([d['omega_x'], d['omega_y'], d['omega_z']], dim=-1)
    t_v   = torch.stack([d['tx'], d['ty'], d['tz']], dim=-1)
    P = _apply_extrinsic(P0, omega, t_v)
    Kn = _K_with_delta(K, d['dfx'], d['dfy'], d['dcx'], d['dcy'])
    return project_kb(P, Kn, dist)


def main():
    cfg = _load_cfg()
    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE, split='val', img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0, oversample=1,
        grid_n=cfg.get('grid_n', 16), center_band=0.0, preload=False,
    )
    inst = ds._load_inst(IDX)
    dist_one = inst['distortion'].clone().detach().to(torch.float32).reshape(1, 4)

    rng = np.random.RandomState(SEED)
    wins = []
    while len(wins) < N_EVAL:
        ox = float(rng.uniform(-ROT, ROT)); oy = float(rng.uniform(-ROT, ROT))
        oz = float(rng.uniform(-ROT, ROT))
        ypr = np.array([oz, oy, ox], dtype=np.float64)
        t = (rng.uniform(-1.0, 1.0, size=3) * T_M).astype(np.float64)
        win = ds.apply_perturbation_explicit(IDX, t, ypr)
        if win is None:
            continue
        wins.append(win)
    batch = collate_full(wins)
    moved = [t.to(DEVICE) if torch.is_tensor(t) else t for t in batch]
    (imgs, true_uvd, dist_uvd, pad_mask, vfp,
     bucket_uvd, bucket_valid, _,
     pts_cam_orig, duv_orig, K_orig, cs) = moved
    valid = ~pad_mask
    pad_full = ~valid
    B, N = pts_cam_orig.shape[:2]
    P0_orig = pts_cam_orig.clone()
    duv_oracle_orig = duv_orig.clone()
    if pad_full.any():
        duv_oracle_orig[pad_full] = 0.0
        P0_orig[pad_full] = torch.tensor([0., 0., 1.], dtype=P0_orig.dtype,
                                          device=P0_orig.device)
    dist = dist_one.to(DEVICE).expand(B, 4).contiguous()

    model = _build(cfg).to(DEVICE)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    img_norm = imgs.float().div(255.0)
    with torch.no_grad():
        per_pt, _ = model(img_norm, point_in, key_padding_mask=pad_mask, vfp=vfp,
                           bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    duv_pred_local = per_pt[..., :2]
    sx = per_pt[..., 2].exp(); sy = per_pt[..., 3].exp(); rho = per_pt[..., 4]
    W_sigma_local = make_info_from_sigma_rho(sx, sy, rho)
    if pad_full.any():
        duv_pred_local = duv_pred_local.clone()
        duv_pred_local[pad_full] = 0.0
    eye2 = torch.eye(2, device=DEVICE).expand(B, N, 2, 2)

    scale_l2o = (cs / float(cfg['img_size'])).reshape(-1, 1, 1)
    inv_l2o = (1.0 / scale_l2o).reshape(-1, 1, 1, 1)
    duv_pred_orig = duv_pred_local * scale_l2o
    W_eye_orig    = eye2 * inv_l2o.pow(2)
    W_sigma_orig  = W_sigma_local * inv_l2o.pow(2)

    # σ ellipse major-axis in local px (used as a per-point info ranking)
    major_local = _major_axis(W_sigma_local)   # (B,N)

    # oracle target uv (orig px)
    uv0_orig = project_kb(P0_orig, K_orig, dist)
    uv_target_orig = uv0_orig + duv_oracle_orig

    # 6-DoF oracle pose (Δuv = GT, W = I, all valid points) — gold standard
    delta_oracle, _ = solve_kb_xyz(P0_orig, duv_oracle_orig,
                                    eye2, K_orig, dist, DOFS,
                                    valid=valid, n_iter=6, damping=1e-3)

    print(f"σ-topK sweep on idx={IDX}, N_eval={N_EVAL}, "
          f"perturb rot=±{ROT}°, t=±{T_M}m")
    print(f"{'K':>6}  {'%pts':>6}  {'δ-MSE':>11}  "
          f"{'reproj mean':>11}  {'reproj med':>11}  {'reproj p95':>11}  "
          f"{'rot°RMS':>8}  {'t·m·RMS':>9}")

    rows = []
    for K in KS:
        sel = _select_topk_mask(major_local, valid, K)
        n_avg = sel.sum(dim=-1).float().mean().item()
        # solve with W=σ on this subset
        with torch.no_grad():
            delta, _ = solve_kb_xyz(
                P0_orig, duv_pred_orig, W_sigma_orig,
                K_orig, dist, DOFS,
                valid=sel, n_iter=6, damping=1e-3,
            )
            mse = (delta - delta_oracle).pow(2).mean(dim=-1).mean().item()
            uv_proj = _project(P0_orig, delta, K_orig, dist, DOFS)
            err = torch.linalg.vector_norm(
                uv_proj - uv_target_orig, dim=-1)
            err = err.masked_fill(~valid, float('nan'))
            flat = err[~torch.isnan(err)].cpu().numpy()
            rot_rms = (delta[:, :3] - delta_oracle[:, :3]) \
                       .pow(2).mean(dim=-1).sqrt().mean().item()
            t_rms = (delta[:, 3:] - delta_oracle[:, 3:]) \
                       .pow(2).mean(dim=-1).sqrt().mean().item()
        Kstr = 'all' if K is None else str(K)
        print(f"{Kstr:>6}  {n_avg:6.1f}  "
              f"{mse:11.4e}  "
              f"{flat.mean():11.3f}  "
              f"{np.median(flat):11.3f}  "
              f"{np.percentile(flat, 95):11.3f}  "
              f"{rot_rms:8.4f}  {t_rms:9.4f}")
        rows.append((K, mse, flat.mean(), np.median(flat),
                      np.percentile(flat, 95), rot_rms, t_rms))


if __name__ == '__main__':
    main()
