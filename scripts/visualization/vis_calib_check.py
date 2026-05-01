"""Calib-mode viz — single-image overlay.

calib_mode = same frame (fi_A == fi_B). patch_A and patch_B are the same
image cropped at slightly different centers. The standard cross-frame viz
(`vis_pred_check.py`) draws two side-by-side panels with cross-panel
ConnectionPatch dotted lines, which becomes visually chaotic for calib_mode
because the two panels show essentially the same scene and the crossing
lines fan everywhere.

This script renders a single-panel overlay per pair:
  • the (resized) anchor image
  • anchor sensor points (uv_A)
  • hyp positions (extrinsic-perturbed projection)  → 'x' marker
  • predicted positions (hyp + Δuv from model)      → '+' marker
  • GT positions (true projection)                  → 'o' ring
  • 1σ + 2σ ellipses around prediction
  • short arrows hyp→pred (model correction) and pred→gt (residual)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.cross_frame_unified import CalibNetUnifiedFrame
from datasets.pandaset_pair import PandaSetCrossFrameDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 64


def load_model(ckpt_dir: Path, ckpt_name='last_model.pt', n_cross=4, n_intra=2,
               in_ch=3, out_dim=5, uv_only_query=True):
    sd = torch.load(ckpt_dir / ckpt_name, map_location=DEVICE, weights_only=False)
    m = CalibNetUnifiedFrame(in_channels=in_ch, img_size=IMG_SIZE,
                              n_intra_layers=n_intra, n_cross_layers=n_cross,
                              out_dim=out_dim, uv_only_query=uv_only_query).to(DEVICE)
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


@torch.no_grad()
def run_one(model, sample):
    batch = {k: (v.unsqueeze(0).to(DEVICE) if torch.is_tensor(v) else v)
             for k, v in sample.items()}
    raw_AB, _ = model(
        patch_A=batch['patch_A'], uvd_A=batch['uvd_A'],
        patch_B=batch['patch_B'], uvd_B=batch['uvd_B'],
        pose_AB_6dof=batch['pose_AB_6dof'], pose_BA_6dof=batch['pose_BA_6dof'],
        uv_B_hat_of_A=batch['uv_B_hat_of_A'], uv_A_hat_of_B=batch['uv_A_hat_of_B'],
        pad_A=batch['pad_A'], pad_B=batch['pad_B'],
        uvd_A_full=batch['uvd_A_full'], uvd_B_full=batch['uvd_B_full'],
        pad_A_full=batch['pad_A_full'], pad_B_full=batch['pad_B_full'],
    )
    raw = raw_AB[0].cpu().numpy()
    delta = raw[:, :2]
    log_sx, log_sy = raw[:, 2], raw[:, 3]
    rho = np.tanh(raw[:, 4]) * 0.99
    sx, sy = np.exp(log_sx), np.exp(log_sy)
    uv_hat = sample['uv_B_hat_of_A'].numpy()
    uv_gt = sample['uv_B_gt_of_A'].numpy()
    uv_pred = uv_hat + delta
    pad = sample['pad_A'].numpy()
    return dict(
        patch_B=sample['patch_B'].numpy(),
        uv_A=sample['uvd_A'].numpy()[:, :2],
        uv_hat=uv_hat, uv_pred=uv_pred, uv_gt=uv_gt,
        sx=sx, sy=sy, rho=rho, pad=pad,
    )


def draw_calib(ax, r, k_show=16):
    img = np.transpose(r['patch_B'], (1, 2, 0)).clip(0, 1)
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    valid = ~r['pad']
    in_patch = ((r['uv_hat'][:, 0] >= 0) & (r['uv_hat'][:, 0] < IMG_SIZE) &
                (r['uv_hat'][:, 1] >= 0) & (r['uv_hat'][:, 1] < IMG_SIZE) &
                (r['uv_gt'][:, 0] >= 0) & (r['uv_gt'][:, 0] < IMG_SIZE) &
                (r['uv_gt'][:, 1] >= 0) & (r['uv_gt'][:, 1] < IMG_SIZE))
    keep = np.where(valid & in_patch)[0]
    if len(keep) == 0:
        ax.set_title('no valid pts', fontsize=10, loc='left')
        return
    err_hyp = np.linalg.norm(r['uv_hat'][keep] - r['uv_gt'][keep], axis=-1).mean()
    err_pred = np.linalg.norm(r['uv_pred'][keep] - r['uv_gt'][keep], axis=-1).mean()
    sig_mean = 0.5 * (r['sx'][keep] + r['sy'][keep]).mean()
    sx_k, sy_k, rho_k = r['sx'][keep], r['sy'][keep], r['rho'][keep]
    d_k = r['uv_gt'][keep] - r['uv_pred'][keep]
    inv_det = 1.0 / np.clip(sx_k**2 * sy_k**2 * (1 - rho_k**2), 1e-12, None)
    mahal2 = inv_det * (sy_k**2 * d_k[:, 0]**2 - 2 * rho_k * sx_k * sy_k * d_k[:, 0] * d_k[:, 1]
                        + sx_k**2 * d_k[:, 1]**2)
    cover_2s = float((mahal2 < 4.0).mean())

    if len(keep) > k_show:
        step = max(1, len(keep) // k_show)
        keep_show = keep[::step][:k_show]
    else:
        keep_show = keep
    cmap = plt.get_cmap('tab10')
    for k, i in enumerate(keep_show):
        c = cmap(k % 10)
        xh, yh = r['uv_hat'][i]; xp, yp = r['uv_pred'][i]; xg, yg = r['uv_gt'][i]
        sx, sy, rho = r['sx'][i], r['sy'][i], r['rho'][i]
        cov = np.array([[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]])
        # 2σ ellipse only (1σ skipped for clarity)
        try:
            vals, vecs = np.linalg.eigh(cov)
            vals = np.clip(vals, 1e-6, None)
            angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
            w, h = 2.0 * 2.0 * np.sqrt(vals[::-1])
            ax.add_patch(Ellipse((xp, yp), width=w, height=h, angle=angle,
                                  facecolor='none', edgecolor=c, lw=1.0,
                                  alpha=0.6, zorder=4))
        except Exception:
            pass
        # GT→HAT (target Δuv, dotted) — what the model SHOULD predict
        ax.add_patch(FancyArrowPatch((xh, yh), (xg, yg),
                                      arrowstyle='-|>', mutation_scale=10,
                                      color=c, lw=1.4, ls=(0, (3, 2)),
                                      alpha=0.55, zorder=5))
        # HAT→PRED (model's actual correction, solid)
        ax.add_patch(FancyArrowPatch((xh, yh), (xp, yp),
                                      arrowstyle='-|>', mutation_scale=10,
                                      color=c, lw=1.8, alpha=0.95, zorder=6))
        # markers
        ax.plot(xh, yh, 'x', color=c, markersize=8, mew=1.8, zorder=7)
        ax.plot(xg, yg, marker='o', markersize=10, markeredgecolor=c,
                 markerfacecolor='none', mew=1.6, alpha=0.95, zorder=8)
        ax.plot(xp, yp, '+', color=c, markersize=10, mew=1.8, zorder=9)
    ax.set_title(f'hyp {err_hyp:.2f} → pred {err_pred:.2f} px   '
                 f'σ̄={sig_mean:.2f}   2σ-cov={100*cover_2s:.0f}%   '
                 f'({len(keep)} pts)',
                 fontsize=10, loc='left')


def main(args):
    ckpt = Path(args.ckpt)
    model = load_model(ckpt, ckpt_name=args.ckpt_name,
                       n_cross=args.n_cross_layers, n_intra=args.n_intra_layers,
                       out_dim=args.out_dim, uv_only_query=True)
    print(f'loaded {ckpt.name}/{args.ckpt_name}')

    ds = PandaSetCrossFrameDataset(
        scenes_root=args.scenes_root, split=args.split,
        cameras=args.cameras,
        img_size=IMG_SIZE, max_points=args.max_points,
        baseline_range=(args.baseline_min, args.baseline_max),
        sigma_ypr=args.sigma_ypr, sigma_t=args.sigma_t,
        virtual_epoch_len=args.n_pairs * 8, seed=args.seed,
        lidar_subdir=args.lidar_subdir, calib_mode=True,
    )
    n_pairs = min(args.n_pairs, len(ds))
    cols = 4
    rows = (n_pairs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 4.0),
                              facecolor='#f6f4ed')
    axes = np.array(axes).reshape(rows, cols)
    plotted = 0
    for i in range(len(ds) * 4):
        if plotted >= n_pairs:
            break
        s = ds[i % len(ds)]
        if s is None:
            continue
        r = run_one(model, s)
        ax = axes[plotted // cols, plotted % cols]
        draw_calib(ax, r)
        plotted += 1
    for k in range(plotted, rows * cols):
        axes[k // cols, k % cols].axis('off')
    fig.suptitle(f'{ckpt.name} — calib viz (same-frame, sensor→cam Δuv)  '
                  f'σ_ypr={args.sigma_ypr}° σ_t={args.sigma_t}m  split={args.split}',
                  fontsize=10)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=120, bbox_inches='tight', facecolor='#f6f4ed')
    print(f'saved → {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--ckpt-name', default='last_model.pt')
    ap.add_argument('--scenes-root', required=True)
    ap.add_argument('--cameras', default='front_camera')
    ap.add_argument('--split', default='val')
    ap.add_argument('--train-frac', type=float, default=0.8)
    ap.add_argument('--max-points', type=int, default=256)
    ap.add_argument('--baseline-min', type=int, default=1)
    ap.add_argument('--baseline-max', type=int, default=5)
    ap.add_argument('--sigma-ypr', type=float, default=0.5)
    ap.add_argument('--sigma-t', type=float, default=0.05)
    ap.add_argument('--lidar-subdir', default='lidar')
    ap.add_argument('--n-cross-layers', type=int, default=4)
    ap.add_argument('--n-intra-layers', type=int, default=2)
    ap.add_argument('--out-dim', type=int, default=5)
    ap.add_argument('--n-pairs', type=int, default=24)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='calib_viz.png')
    args = ap.parse_args()
    main(args)
