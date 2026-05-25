"""Cheat oracle: pick the K points whose duv_pred is closest to duv_oracle
   (smallest per-point residual), solve 6-DoF on those K with W=I, and
   measure reproj on the SAME K points. If this isn't sub-px, the solver
   itself is broken."""
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
    solve_kb_xyz, project_kb,
    _apply_extrinsic, _K_with_delta, _split_delta,
)

CACHE = Path.home() / 'cache' / 'kamikado_v3_tiled'
CKPT  = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'best_model.pt'
EXP_CFG_PATH = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'config.py'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IDX = 17; ROT = 0.30; T_M = 0.05; N_EVAL = 200; SEED = 7 + 1000
DOFS = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
KS = [20, 50, 100, None]


def _load_cfg():
    src = EXP_CFG_PATH.read_text(); ns = {}; exec(src, ns, ns); return ns['CFG']


def _build(cfg):
    return CalibNetDepth(
        img_size=cfg['img_size'], in_channels=cfg['in_channels'],
        n_layers=cfg['n_layers'], self_first=cfg.get('self_first', False),
        use_convnext=cfg.get('use_convnext', True),
        use_frustum=cfg.get('use_frustum', True),
        deform_mode=cfg.get('deform_mode', 'sl'),
        convnext_n_blocks=cfg.get('convnext_n_blocks', 2),
        convnext_fine_d=cfg.get('convnext_fine_d', None),
        convnext_stem_d=cfg.get('convnext_stem_d', None),
        use_info_head=True,
    )


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
    P0 = pts_cam_orig.clone()
    duv_oracle = duv_orig.clone()
    if pad_full.any():
        duv_oracle[pad_full] = 0.0
        P0[pad_full] = torch.tensor([0., 0., 1.], dtype=P0.dtype, device=P0.device)
    dist = dist_one.to(DEVICE).expand(B, 4).contiguous()

    model = _build(cfg).to(DEVICE)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    model.load_state_dict(sd, strict=False)
    model.eval()
    point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    with torch.no_grad():
        per_pt, _ = model(imgs.float().div(255.0), point_in,
                           key_padding_mask=pad_mask, vfp=vfp,
                           bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    duv_pred_local = per_pt[..., :2]
    if pad_full.any():
        duv_pred_local = duv_pred_local.clone()
        duv_pred_local[pad_full] = 0.0
    scale_l2o = (cs / float(cfg['img_size'])).reshape(-1, 1, 1)
    duv_pred_orig = duv_pred_local * scale_l2o
    eye2 = torch.eye(2, device=DEVICE).expand(B, N, 2, 2)

    # CHEAT score: |duv_pred − duv_oracle| (orig px) — small = "I picked the truth"
    score = torch.linalg.vector_norm(
        duv_pred_orig - duv_oracle, dim=-1)             # (B,N)
    big = torch.full_like(score, float('inf'))
    score = torch.where(valid, score, big)

    uv0 = project_kb(P0, K_orig, dist)
    uv_target = uv0 + duv_oracle

    oracle_mag = torch.linalg.vector_norm(duv_oracle, dim=-1)  # (B,N) orig px

    print(f"CHEAT topK (smallest |duv_pred-duv_oracle|) on idx={IDX}, N={N_EVAL}")
    print(f"{'K':>6}  {'%pts':>6}  "
          f"{'orig|Δuv| mean':>14}  {'orig|Δuv| med':>13}  "
          f"{'orig|Δuv| p95':>13}  {'orig|Δuv| max':>13}  | "
          f"{'sel_resid mean':>14}  {'sel_resid med':>13}  "
          f"{'sel_resid p95':>13}  {'sel_resid max':>13}  | "
          f"{'reproj_K mean':>14}  {'reproj_K med':>13}  "
          f"{'reproj_K p95':>13}  {'reproj_all mean':>15}  "
          f"{'rot°RMS':>8}  {'t·m·RMS':>9}")

    # 6-DoF "oracle" pose: solve with Δuv=GT, W=I, ALL valid points
    with torch.no_grad():
        delta_oracle, _ = solve_kb_xyz(P0, duv_oracle, eye2, K_orig, dist, DOFS,
                                        valid=valid, n_iter=6, damping=1e-3)

    for K in KS:
        if K is None or K >= N:
            sel = valid.clone()
        else:
            idx = torch.topk(score, K, dim=-1, largest=False).indices
            sel = torch.zeros_like(valid); sel.scatter_(1, idx, True); sel &= valid

        # selected per-point residuals (BEFORE solve)
        sel_resid_t = score.masked_fill(~sel, float('nan'))
        flat_sel = sel_resid_t[~torch.isnan(sel_resid_t)].cpu().numpy()
        # original |Δuv_oracle| on selected pts (= how perturbed they were)
        sel_orig_mag_t = oracle_mag.masked_fill(~sel, float('nan'))
        flat_orig_mag = sel_orig_mag_t[~torch.isnan(sel_orig_mag_t)].cpu().numpy()

        with torch.no_grad():
            delta, _ = solve_kb_xyz(
                P0, duv_pred_orig, eye2,           # W = I
                K_orig, dist, DOFS,
                valid=sel, n_iter=6, damping=1e-3,
            )
            uv_proj = _project(P0, delta, K_orig, dist, DOFS)
            err = torch.linalg.vector_norm(uv_proj - uv_target, dim=-1)
            err_K  = err.masked_fill(~sel,   float('nan'))
            err_all = err.masked_fill(~valid, float('nan'))
            flat_K   = err_K[~torch.isnan(err_K)].cpu().numpy()
            flat_all = err_all[~torch.isnan(err_all)].cpu().numpy()
            rot_rms = (delta[:, :3] - delta_oracle[:, :3]).pow(2).mean(dim=-1).sqrt().mean().item()
            t_rms   = (delta[:, 3:] - delta_oracle[:, 3:]).pow(2).mean(dim=-1).sqrt().mean().item()

        Kstr = 'all' if K is None else str(K)
        n_avg = sel.sum(dim=-1).float().mean().item()
        print(f"{Kstr:>6}  {n_avg:6.1f}  "
              f"{flat_orig_mag.mean():14.3f}  {np.median(flat_orig_mag):13.3f}  "
              f"{np.percentile(flat_orig_mag, 95):13.3f}  "
              f"{flat_orig_mag.max():13.3f}  | "
              f"{flat_sel.mean():14.3f}  {np.median(flat_sel):13.3f}  "
              f"{np.percentile(flat_sel, 95):13.3f}  "
              f"{flat_sel.max():13.3f}  | "
              f"{flat_K.mean():14.3f}  {np.median(flat_K):13.3f}  "
              f"{np.percentile(flat_K, 95):13.3f}  "
              f"{flat_all.mean():15.3f}  "
              f"{rot_rms:8.4f}  {t_rms:9.4f}")


if __name__ == '__main__':
    main()
