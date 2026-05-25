"""Solver sanity: feed Δuv_oracle as the "prediction", solve all 6-DoF
   with W=I on all valid points, and measure reproj on all valid points.
   If solver + scale conversions are correct, reproj should be ~0 px."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from scripts.ba.ba_torch import (
    solve_kb_xyz, project_kb,
    _apply_extrinsic, _K_with_delta, _split_delta,
)

CACHE = Path.home() / 'cache' / 'kamikado_v3_tiled'
EXP_CFG_PATH = (Path(__file__).resolve().parents[2]
                / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'config.py')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IDX = 17; ROT = 0.30; T_M = 0.05; N_EVAL = 50; SEED = 7 + 1000
DOFS = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']


def _load_cfg():
    src = EXP_CFG_PATH.read_text(); ns = {}; exec(src, ns, ns); return ns['CFG']


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
    eye2 = torch.eye(2, device=DEVICE).expand(B, N, 2, 2)

    uv0 = project_kb(P0, K_orig, dist)
    uv_target = uv0 + duv_oracle

    print(f'[sanity] N={N_EVAL}  rot=±{ROT}°  t=±{T_M}m')
    print(f'[sanity] do-nothing reproj = ||Δuv_oracle||')
    do_nothing = torch.linalg.vector_norm(duv_oracle, dim=-1)
    do_nothing = do_nothing.masked_fill(~valid, float('nan'))
    flat = do_nothing[~torch.isnan(do_nothing)].cpu().numpy()
    print(f'  mean {flat.mean():.4f}  med {np.median(flat):.4f}  '
          f'p95 {np.percentile(flat, 95):.4f}  max {flat.max():.4f}')

    # ---- Test 1: feed Δuv_oracle as prediction, W=I, all valid ----
    with torch.no_grad():
        delta, _ = solve_kb_xyz(
            P0, duv_oracle, eye2, K_orig, dist, DOFS,
            valid=valid, n_iter=12, damping=1e-3,
        )
        uv_proj = _project(P0, delta, K_orig, dist, DOFS)
        err = torch.linalg.vector_norm(uv_proj - uv_target, dim=-1)
        err = err.masked_fill(~valid, float('nan'))
        flat = err[~torch.isnan(err)].cpu().numpy()
    print(f'[sanity] Test 1: Δuv=ORACLE, W=I, all pts → reproj should be ~0:')
    print(f'  mean {flat.mean():.4f}  med {np.median(flat):.4f}  '
          f'p95 {np.percentile(flat, 95):.4f}  max {flat.max():.4f}')
    print(f'  delta sample 0: {delta[0].cpu().numpy()}')

    # ---- Test 2: feed Δuv=0 as prediction, W=I, all valid ----
    # Solver should return δ ≈ 0 → reproj ≈ ||Δuv_oracle|| (= do-nothing)
    duv_zero = torch.zeros_like(duv_oracle)
    with torch.no_grad():
        delta, _ = solve_kb_xyz(
            P0, duv_zero, eye2, K_orig, dist, DOFS,
            valid=valid, n_iter=12, damping=1e-3,
        )
        uv_proj = _project(P0, delta, K_orig, dist, DOFS)
        err = torch.linalg.vector_norm(uv_proj - uv_target, dim=-1)
        err = err.masked_fill(~valid, float('nan'))
        flat = err[~torch.isnan(err)].cpu().numpy()
    print(f'[sanity] Test 2: Δuv=0, W=I, all pts → reproj should be ≈ do-nothing:')
    print(f'  mean {flat.mean():.4f}  med {np.median(flat):.4f}  '
          f'p95 {np.percentile(flat, 95):.4f}  max {flat.max():.4f}')
    print(f'  delta sample 0 (should be ~0): {delta[0].cpu().numpy()}')


if __name__ == '__main__':
    main()
