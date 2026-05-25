"""PandaSet-pinhole counterpart of eval_shared_256x800.py.

Same shared-δ 800-tile BA aggregation pattern, but uses
`solve_pinhole_xyz_shared` (no KB distortion). Reuses `_build_subwin`,
`_draw_pert`, `compute_overlay_geom` from eval_shared_256x800.py.

Usage:
  CUDA_VISIBLE_DEVICES=12 python scripts/eval/eval_ps_shared_256x800.py \
      --exp experiments/ps_full_n4_img256_grid32_dgx3_100ep \
      --cache /mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full \
      --target-idx 17 --n-inst 200 --n-per-inst 4 --cs 256 \
      --levels '0.5,0.05;1.0,0.10;1.5,0.20' --n-seeds 4
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.model_depth import CalibNetDepth
from scripts.ba.ba_torch import (
    solve_pinhole_xyz_shared, make_info_from_sigma_rho,
)
from scripts.eval import eval_shared_256x800 as _ks

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DOFS = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
PRIOR_DIAG = torch.tensor(
    [1.0/9.0, 1.0/9.0, 1.0/9.0, 25.0, 25.0, 25.0], dtype=torch.float32)
BA_N_ITER = 2
DAMPING = 1e-3


def _load_cfg(exp_dir):
    src = (exp_dir / 'config.py').read_text()
    ns = {}; exec(src, ns, ns); return ns['CFG']


def _build_model(cfg):
    return CalibNetDepth(
        img_size=cfg['img_size'], in_channels=cfg['in_channels'],
        n_layers=cfg['n_layers'],
        self_first=cfg.get('self_first', False),
        use_convnext=cfg.get('use_convnext', True),
        use_frustum=cfg.get('use_frustum', True),
        frustum_grid_n=cfg.get('grid_n', 16),
        frustum_dense=cfg.get('frustum_dense', False),
        use_pose_emb=cfg.get('use_pose_emb', False),
        deform_mode=cfg.get('deform_mode', 'none'),
        convnext_n_blocks=cfg.get('convnext_n_blocks', 2),
        convnext_fine_d=cfg.get('convnext_fine_d', None),
        convnext_stem_d=cfg.get('convnext_stem_d', None),
    )


def _solve_one_pinhole(model, ds_imgs, *, target_idx, n_inst, cs, n_per_inst,
                        rng, ypr_target, t_target, cfg, label):
    if cs == 512:
        u0v0_list = [(0, 0)]
        assert n_per_inst == 1
    elif cs == 256:
        all4 = [(0, 0), (256, 0), (0, 256), (256, 256)]
        assert 1 <= n_per_inst <= 4
        u0v0_list = all4[:n_per_inst]
    else:
        raise ValueError(f'unsupported cs={cs}')

    target_inst = ds_imgs._load_inst(int(target_idx))
    wins = []
    for (u0, v0) in u0v0_list:
        w = _ks._build_subwin(ds_imgs, target_inst, t_target, ypr_target,
                              u0=u0, v0=v0, cs=cs)
        if w is not None:
            wins.append(w)
    assert len(wins) >= 1, f'target idx={target_idx} returned no sub-crops'

    target_b = n_inst * n_per_inst
    tries = 0
    while len(wins) < target_b and tries < 32 * target_b:
        ridx = int(rng.randint(0, len(ds_imgs.fnames)))
        inst_r = ds_imgs._load_inst(ridx)
        for (u0, v0) in u0v0_list:
            if len(wins) >= target_b:
                break
            w = _ks._build_subwin(ds_imgs, inst_r, t_target, ypr_target,
                                   u0=u0, v0=v0, cs=cs)
            tries += 1
            if w is not None:
                wins.append(w)
    assert len(wins) == target_b, \
        f'[{label}] could not build batch ({len(wins)}/{target_b})'

    moved = [t.to(DEVICE) if torch.is_tensor(t) else t for t in collate_full(wins)]
    (imgs, true_uvd, dist_uvd, pad_mask, vfp,
     bucket_uvd, bucket_valid, _,
     pts_cam_orig, duv_orig, K_orig, cs_b, delta1_se3) = moved
    valid = ~pad_mask
    pad_full = ~valid
    B, N = pts_cam_orig.shape[:2]
    P0_orig = pts_cam_orig.detach().clone()
    if pad_full.any():
        P0_orig[pad_full] = torch.tensor([0.0, 0.0, 1.0],
                                          dtype=P0_orig.dtype,
                                          device=P0_orig.device)

    use_intensity = getattr(model, 'use_intensity', True)
    if use_intensity:
        point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
    else:
        point_in = dist_uvd[..., :3]
    img_norm = imgs.float().div(255.0)
    with torch.no_grad():
        out = model(img_norm, point_in, key_padding_mask=pad_mask, vfp=vfp,
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
    per_pt = out[0] if isinstance(out, tuple) else out
    W_direct = None
    if isinstance(out, tuple):
        last = out[-1]
        if torch.is_tensor(last) and last.dim() == 4 and last.shape[-2:] == (2, 2):
            W_direct = last
    duv_pred_local = per_pt[..., :2].detach()
    if pad_full.any():
        duv_pred_local = duv_pred_local.clone()
        duv_pred_local[pad_full] = 0.0
    if W_direct is not None:
        W_sigma_local = W_direct.detach()
    else:
        sx = per_pt[..., 2].exp()
        sy = per_pt[..., 3].exp()
        rho = per_pt[..., 4]
        W_sigma_local = make_info_from_sigma_rho(sx, sy, rho).detach()

    scale_l2o = (cs_b / float(cfg['img_size'])).reshape(-1, 1, 1)
    inv_l2o = (1.0 / scale_l2o).reshape(-1, 1, 1, 1)
    duv_pred_orig = duv_pred_local * scale_l2o
    W_sigma_orig = W_sigma_local * inv_l2o.pow(2)

    prior = PRIOR_DIAG.to(DEVICE)
    with torch.no_grad():
        delta_shared, H_last = solve_pinhole_xyz_shared(
            P0_orig, duv_pred_orig, W_sigma_orig, K_orig, DOFS,
            valid=valid, n_iter=BA_N_ITER, damping=DAMPING, prior_diag=prior,
        )
    return delta_shared, B, H_last


def _parse_levels(s):
    out = []
    for part in s.split(';'):
        rd, tm = part.split(',')
        out.append((float(rd), float(tm)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, type=Path)
    ap.add_argument('--cache', required=True, type=Path)
    ap.add_argument('--target-idx', type=int, default=17)
    ap.add_argument('--n-inst', type=int, default=200)
    ap.add_argument('--n-per-inst', type=int, default=4)
    ap.add_argument('--cs', type=int, default=256)
    ap.add_argument('--levels', type=str, default='0.5,0.05;1.0,0.10;1.5,0.20')
    ap.add_argument('--n-seeds', type=int, default=4)
    args = ap.parse_args()

    cfg = _load_cfg(args.exp)
    print(f'cfg: img_size={cfg["img_size"]} grid_n={cfg.get("grid_n",16)} '
          f'deform={cfg.get("deform_mode","none")} pe={cfg.get("use_pose_emb",False)} '
          f'fdense={cfg.get("frustum_dense",False)}')
    model = _build_model(cfg).to(DEVICE)
    sd = torch.load(args.exp / 'best_model.pt', map_location=DEVICE, weights_only=False)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    model.load_state_dict(sd)
    model.eval()
    print(f'model: {args.exp.name}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    ds = PandaSetCalibDatasetFull(
        cache_dir=args.cache, split='val',
        img_size=cfg['img_size'],
        min_crop_px=cfg.get('min_crop_px', 128),
        max_crop_px=cfg.get('max_crop_px', 512),
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=cfg.get('grid_n', 16),
        center_band=0.0, preload=False,
    )
    print(f'val frames: {len(ds.fnames)}  target_idx={args.target_idx}')

    inst0 = ds._load_inst(args.target_idx)
    fx = float(inst0['K_full'].numpy()[0, 0])
    print(f'fx={fx:.1f}px (orig camera)')

    levels = _parse_levels(args.levels)
    summary = []
    for li, (rot_deg, t_m) in enumerate(levels):
        omegas, ts = [], []
        for k in range(args.n_seeds):
            rng = np.random.RandomState(1000 + 100 * li + k)
            ypr_t, t_t = _ks._draw_pert(rng, rot_deg=rot_deg, t_m=t_m)
            rng2 = np.random.RandomState(2000 + 100 * li + k)
            d, B, _ = _solve_one_pinhole(
                model, ds,
                target_idx=args.target_idx, n_inst=args.n_inst,
                cs=args.cs, n_per_inst=args.n_per_inst, rng=rng2,
                ypr_target=ypr_t, t_target=t_t, cfg=cfg,
                label=f'L{li}-s{k}')
            tgt = np.array([ypr_t[2], ypr_t[1], ypr_t[0]], dtype=np.float64)
            d_np = d.detach().cpu().numpy()
            omega_err = float(np.linalg.norm(d_np[:3] - tgt))
            t_err = float(np.linalg.norm(d_np[3:] - t_t))
            omegas.append(omega_err); ts.append(t_err)
            print(f'  L{li} s{k}  ±{rot_deg}°/±{t_m}m  ω={omega_err:.4f}°  t={t_err:.4f}m  (B={B})',
                  flush=True)
        omega_mean = float(np.mean(omegas))
        t_mean = float(np.mean(ts))
        omega_px = float(fx * np.tan(np.deg2rad(omega_mean)))
        summary.append(dict(
            level=li, rot_deg=rot_deg, t_m=t_m,
            omega_deg=omega_mean, omega_px_at_fx=omega_px, t_m_residual=t_mean,
            omega_per_seed=omegas, t_per_seed=ts,
        ))
        print(f'  L{li} MEAN  ω={omega_mean:.4f}° ({omega_px:.3f}px@fx)  t={t_mean:.4f}m')

    out_dir = args.exp / 'ba'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f'shared_{args.cs}x{args.n_inst*args.n_per_inst}_idx{args.target_idx}.json'
    out_path.write_text(json.dumps(dict(
        exp=args.exp.name, cache=str(args.cache), target_idx=args.target_idx,
        n_inst=args.n_inst, n_per_inst=args.n_per_inst, cs=args.cs,
        n_seeds=args.n_seeds, fx=fx, summary=summary,
    ), indent=2))
    print(f'\nsaved → {out_path}')


if __name__ == '__main__':
    main()
