"""Visualize the 256×800 result on idx=17's parent tile (IH=IW=512).

3-stage overlay (yellow=GT / red=perturbed / green=BA-corrected) on the FULL
LiDAR cloud, so we can eyeball whether the 0.45 px residual is real or a bug.

Reuses _build_subwin / _solve_one from eval_shared_256x800.py to make sure the
pose comes from the SAME pipeline that produced the residual table.
"""
from __future__ import annotations
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull
from models.model_depth import CalibNetDepth
from scripts.util.projection import project_kannala
from scripts.eval.eval_shared_256x800 import (
    _load_cfg, _build_model, _draw_pert, _solve_one, CACHE, CKPT, DEVICE,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--idx', type=int, default=17)
    ap.add_argument('--n-shared-512', type=int, default=200)
    ap.add_argument('--n-shared-256', type=int, default=200,
                    help='# instances; total tiles = N × 4 sub-crops')
    ap.add_argument('--rot-deg', type=float, default=0.30)
    ap.add_argument('--t-m', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=7 + 1000)
    ap.add_argument('--out', type=Path,
                    default=REPO / 'scripts' / '_debug' / '_outputs'
                            / 'vis_256x800_3stage_idx17.png')
    args = ap.parse_args()

    cfg = _load_cfg()
    print(f'[viz] idx={args.idx}  rot=±{args.rot_deg}°  t=±{args.t_m}m  seed={args.seed}')

    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE, split='val',
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=cfg.get('grid_n', 16),
        center_band=0.0, preload=False,
    )
    inst = ds._load_inst(int(args.idx))
    assert inst.get('is_fisheye', False)
    dist_one = inst['distortion'].clone().detach().to(torch.float32).reshape(1, 4)
    parent = np.array(Image.open(io.BytesIO(inst['jpg_bytes'])).convert('RGB'))
    pH, pW = parent.shape[:2]
    print(f'[viz] parent tile {pW}×{pH} (tile_u0={int(inst.get("tile_u0",0))}, '
          f'tile_v0={int(inst.get("tile_v0",0))})')

    K_full = inst['K_full'].numpy().astype(np.float64)
    dist_kb = inst['distortion'].numpy().astype(np.float64)
    pts_full_cam = inst['pts'].numpy().astype(np.float64)        # (N,3) cam-frame
    z_full = pts_full_cam[:, 2]
    tile_u0 = float(inst.get('tile_u0', 0))
    tile_v0 = float(inst.get('tile_v0', 0))
    print(f'[viz] full LiDAR pts (z>0): {int((z_full > 0.5).sum())} / {len(z_full)}')

    # ---- model + ckpt
    model = _build_model(cfg).to(DEVICE)
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and ('state_dict' in sd or 'model' in sd):
        sd = sd.get('state_dict') or sd.get('model')
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- same δ_target as the residual table
    rng = np.random.RandomState(args.seed)
    ypr_target, t_target = _draw_pert(rng, rot_deg=args.rot_deg, t_m=args.t_m)
    target_xyz = np.array([ypr_target[2], ypr_target[1], ypr_target[0]],
                            dtype=np.float64)
    print(f'[viz] δ_target ω=[{target_xyz[0]:+.4f},{target_xyz[1]:+.4f},'
          f'{target_xyz[2]:+.4f}]° t=[{t_target[0]:+.4f},{t_target[1]:+.4f},'
          f'{t_target[2]:+.4f}]m')

    # ---- solve under both regimes
    rng_a = np.random.RandomState(args.seed + 1)
    delta_512, B_512, _ = _solve_one(model, ds, target_idx=args.idx,
                                       n_inst=args.n_shared_512, cs=512,
                                       n_per_inst=1, rng=rng_a,
                                       ypr_target=ypr_target, t_target=t_target,
                                       dist_one=dist_one, cfg=cfg, label='512')
    rng_b = np.random.RandomState(args.seed + 1)
    delta_256, B_256, _ = _solve_one(model, ds, target_idx=args.idx,
                                       n_inst=args.n_shared_256, cs=256,
                                       n_per_inst=4, rng=rng_b,
                                       ypr_target=ypr_target, t_target=t_target,
                                       dist_one=dist_one, cfg=cfg, label='256')

    print(f'[viz] B_512={B_512}  δ̂_512={delta_512.cpu().numpy()}')
    print(f'[viz] B_256={B_256}  δ̂_256={delta_256.cpu().numpy()}')

    # ---- apply δ_target / δ̂_solved to FULL LiDAR cloud, project to parent
    # Sign convention (matches build_window):
    #   R_off = R_gt @ Rotation.from_euler('zyx', ypr).as_matrix()
    #   pts_cam_off = R_off.T @ (pts_world - cp_off) for one frame; for tile
    #   inst R_gt = I and cp = 0, so pts_cam_pert = R_pert.T @ (pts_full_cam - t_target).
    R_pert = Rotation.from_euler('zyx', ypr_target, degrees=True).as_matrix()
    pts_pert = (pts_full_cam - t_target) @ R_pert      # = (R_pert.T @ (P - t).T).T

    def _ba_correct(delta):
        """Empirically δ̂ ≈ +target_xyz (same sign), so δ̂ represents
        R_pert / t_target directly. Build_window applies
            P_off = R_pert.T · (P - t_target)
        ⇒ P     = R_pert · P_off + t_target
        ⇒ P_corr = R(δ̂) · P_pert + t(δ̂)  (row-vec: P_pert @ R_d.T + t_v)."""
        d = delta.cpu().numpy().astype(np.float64)
        omega = d[:3]                # (ω_x, ω_y, ω_z) deg
        t_v = d[3:]                  # tx, ty, tz m
        R_d = Rotation.from_rotvec(np.deg2rad(omega)).as_matrix()
        return pts_pert @ R_d.T + t_v

    pts_corr_512 = _ba_correct(delta_512)
    pts_corr_256 = _ba_correct(delta_256)

    # KB project all four sets in parent coords, then shift by tile origin
    def _proj(pts):
        uv = project_kannala(pts.astype(np.float64), K_full, dist_kb)
        # parent → tile-local
        uv = uv - np.array([tile_u0, tile_v0], dtype=np.float64)
        return uv.astype(np.float64)

    uv_gt = _proj(pts_full_cam)
    uv_pert = _proj(pts_pert)
    uv_512 = _proj(pts_corr_512)
    uv_256 = _proj(pts_corr_256)

    def _in(uv, z):
        return ((z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < pW)
                & (uv[:, 1] >= 0) & (uv[:, 1] < pH))

    # residual stats — print BEFORE plotting so the numbers anchor the figure
    def _stats(uv_a, uv_b, z):
        m = _in(uv_a, z) & _in(uv_b, z)
        d = np.linalg.norm(uv_a[m] - uv_b[m], axis=1)
        return d.mean(), np.median(d), np.percentile(d, 90), len(d)

    s_p = _stats(uv_pert, uv_gt, pts_full_cam[:, 2])
    s_5 = _stats(uv_512,  uv_gt, pts_corr_512[:, 2])
    s_2 = _stats(uv_256,  uv_gt, pts_corr_256[:, 2])
    print(f'[viz] full-cloud reproj-px vs GT  '
          f'(N_in_image after each transform):')
    print(f'  perturbed   mean={s_p[0]:.3f}  med={s_p[1]:.3f}  p90={s_p[2]:.3f}  N={s_p[3]}')
    print(f'  BA 512×200  mean={s_5[0]:.3f}  med={s_5[1]:.3f}  p90={s_5[2]:.3f}  N={s_5[3]}')
    print(f'  BA 256×{B_256:<3} mean={s_2[0]:.3f}  med={s_2[1]:.3f}  p90={s_2[2]:.3f}  N={s_2[3]}')

    # ---- 4-panel figure: GT / perturbed / BA-512 / BA-256
    panels = [
        ('GT ({} pts)'.format(int(_in(uv_gt, pts_full_cam[:, 2]).sum())),
            uv_gt, pts_full_cam[:, 2], 'yellow'),
        ('Perturbed (target δ applied)  '
         'mean={:.2f}px med={:.2f}px'.format(s_p[0], s_p[1]),
            uv_pert, pts_pert[:, 2], 'red'),
        ('BA 200×512 corrected  '
         'mean={:.2f}px med={:.2f}px'.format(s_5[0], s_5[1]),
            uv_512, pts_corr_512[:, 2], 'orange'),
        ('BA {}×256 corrected  '
         'mean={:.2f}px med={:.2f}px'.format(B_256, s_2[0], s_2[1]),
            uv_256, pts_corr_256[:, 2], 'lime'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(2*pW/100, 2*pH/100), dpi=140)
    for ax, (title, uv, z, colr) in zip(axes.flat, panels):
        m = _in(uv, z)
        ax.imshow(parent)
        ax.scatter(uv[m, 0], uv[m, 1], s=8, c=colr, marker='.',
                    linewidths=0, alpha=0.9)
        ax.set_xlim(0, pW); ax.set_ylim(pH, 0); ax.axis('off')
        ax.set_title(title, fontsize=11)
    fig.suptitle(
        f'idx={args.idx} ({ds.fnames[args.idx]})  '
        f'δ_target ω=[{target_xyz[0]:+.3f},{target_xyz[1]:+.3f},{target_xyz[2]:+.3f}]° '
        f't=[{t_target[0]:+.3f},{t_target[1]:+.3f},{t_target[2]:+.3f}]m   '
        f'(seed={args.seed})',
        y=1.0, fontsize=11,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'[viz] wrote → {args.out}')


if __name__ == '__main__':
    main()
