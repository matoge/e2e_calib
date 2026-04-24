"""Visual sanity check for CalibNetCrossFrame.

For a handful of (fi_A, fi_B) pairs, show:
  - patch_A with A's LiDAR points overlaid
  - patch_B with three layers:
      • uv_hat (hypothesis projection of A's points)  — red ×
      • uv_pred = uv_hat + Δ_model                    — green ●
      • uv_gt   (ground-truth projection)             — blue ○
  - per-point Σ as ellipses around uv_pred (uncertainty)

If the model is doing what we want, green ● should land on top of blue ○ ,
much closer than the red × did.

Output: experiments/{ckpt_dir}/vis/sample_NN.png (4-grid overview)
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from datasets.pandaset_pair import PandaSetCrossFrameDataset
from models.cross_frame import CalibNetCrossFrame

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def overlay_points(ax, uv, color, marker='o', size=20, label=None, ec='white', lw=0.4, alpha=1.0):
    ax.scatter(uv[:, 0], uv[:, 1], c=color, marker=marker, s=size,
               edgecolors=ec, linewidths=lw, alpha=alpha, label=label, zorder=5)


def draw_uncertainty(ax, uv, sigma_uv, rho, color='lime', max_n=40):
    """Draw 1-σ ellipse for each point (sub-sampled to max_n)."""
    if len(uv) > max_n:
        idx = np.linspace(0, len(uv) - 1, max_n).astype(int)
    else:
        idx = np.arange(len(uv))
    for i in idx:
        sx, sy = sigma_uv[i]
        # decompose 2x2 cov [[sx² , ρ·sx·sy], [ρ·sx·sy, sy²]]
        cov = np.array([[sx*sx, rho[i]*sx*sy], [rho[i]*sx*sy, sy*sy]])
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        w, h = 2 * np.sqrt(np.abs(eigvals))
        e = Ellipse((uv[i, 0], uv[i, 1]), w, h, angle=angle,
                    facecolor='none', edgecolor=color, lw=0.5, alpha=0.4)
        ax.add_patch(e)


def make_grid(samples, model, out_path, n_show=6):
    """Run inference and produce a 6-row grid (one row per sample).

    Each row: [patch_A, patch_B w/ overlays, scatter zoom of B around mean].
    """
    fig, axes = plt.subplots(n_show, 3, figsize=(13, 4.0 * n_show), dpi=110)
    fig.patch.set_facecolor('#f6f4ed')

    keys = ['patch_A', 'uvd_A', 'patch_B', 'uvd_B',
            'pose_AB_6dof', 'pose_BA_6dof',
            'uv_B_hat_of_A', 'uv_A_hat_of_B', 'pad_A', 'pad_B']

    for ri, s in enumerate(samples[:n_show]):
        batch = {k: s[k].unsqueeze(0).to(DEVICE) for k in keys}
        with torch.no_grad():
            raw_AB, _ = model(**batch)
        raw = raw_AB[0].cpu().numpy()                      # (N, 5)
        N = (~s['pad_A']).sum().item()
        mu = raw[:N, :2]                                   # (N, 2) Δ patch-px
        log_sx, log_sy = raw[:N, 2], raw[:N, 3]
        rho = np.tanh(raw[:N, 4]) * 0.99
        sx = np.exp(log_sx); sy = np.exp(log_sy)

        uv_hat = s['uv_B_hat_of_A'][:N].cpu().numpy()
        uv_gt  = s['uv_B_gt_of_A'][:N].cpu().numpy()
        uv_pred = uv_hat + mu

        patch_A_np = (s['patch_A'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        patch_B_np = (s['patch_B'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # A's points on A
        uv_A_patch = s['uvd_A'][:N, :2].cpu().numpy()

        # ── col 0: patch_A with A points
        ax = axes[ri, 0] if n_show > 1 else axes[0]
        ax.imshow(patch_A_np)
        overlay_points(ax, uv_A_patch, '#c13c14', marker='o', size=18, label='A pts')
        ax.set_title(f'frame A — scene {s["scene"]}, fi={s["fi_A"]} → fi={s["fi_B"]}',
                      fontsize=10, loc='left')
        ax.set_xticks([]); ax.set_yticks([])

        # ── col 1: patch_B with hat / pred / gt overlay
        ax = axes[ri, 1] if n_show > 1 else axes[1]
        ax.imshow(patch_B_np)
        overlay_points(ax, uv_hat,  '#c13c14', marker='x', size=22, label='hyp', ec='none', lw=1.2)
        overlay_points(ax, uv_pred, '#0fa550', marker='o', size=20, label='pred', ec='black', lw=0.5)
        overlay_points(ax, uv_gt,   '#1e6fff', marker='o', size=10, label='gt', ec='black', lw=0.4, alpha=0.85)
        # uncertainty ellipses around pred
        draw_uncertainty(ax, uv_pred, np.stack([sx, sy], axis=1), rho)
        err_hat  = np.linalg.norm(uv_hat  - uv_gt, axis=1).mean()
        err_pred = np.linalg.norm(uv_pred - uv_gt, axis=1).mean()
        ax.set_title(f'frame B — hyp_err {err_hat:.2f}px → pred_err {err_pred:.2f}px',
                      fontsize=10, loc='left')
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(frameon=False, fontsize=8, loc='lower right')

        # ── col 2: zoom-in scatter only (no image)
        ax = axes[ri, 2] if n_show > 1 else axes[2]
        ax.set_facecolor('#fff')
        # arrows from hyp → pred
        for i in range(N):
            ax.annotate('', xy=uv_pred[i], xytext=uv_hat[i],
                         arrowprops=dict(arrowstyle='-|>', color='#888', lw=0.5))
        overlay_points(ax, uv_hat,  '#c13c14', marker='x', size=22, label='hyp')
        overlay_points(ax, uv_pred, '#0fa550', marker='o', size=22, label='pred')
        overlay_points(ax, uv_gt,   '#1e6fff', marker='o', size=12, label='gt')
        ax.set_xlim(0, 64); ax.set_ylim(64, 0)
        ax.set_aspect('equal')
        ax.set_title(f'arrows: hyp→pred (model prediction)', fontsize=10, loc='left')
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight', facecolor='#f6f4ed')
    plt.close(fig)
    print(f'saved → {out_path}')


def main(args):
    out_dir = Path(args.ckpt_dir) / 'vis'
    out_dir.mkdir(parents=True, exist_ok=True)

    # load model
    sd = torch.load(Path(args.ckpt_dir) / 'best_model.pt',
                    map_location=DEVICE, weights_only=True)
    # detect deform_mode from state dict
    deform = 'sl' if any('deform_img' in k for k in sd.keys()) else 'none'
    n_cross = sum(1 for k in sd.keys() if k.startswith('cross_blocks.') and k.endswith('.proj.weight'))
    print(f'detected deform_mode={deform}, n_cross_layers={n_cross}')

    model = CalibNetCrossFrame(img_size=args.img_size,
                                n_cross_layers=n_cross, deform_mode=deform).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()

    # build val dataset
    ds = PandaSetCrossFrameDataset(
        scenes_root=args.scenes_root, split='val', train_frac=0.80,
        img_size=args.img_size, max_points=args.max_points,
        baseline_range=(args.baseline_min, args.baseline_max),
        sigma_ypr=args.sigma_ypr, sigma_t=args.sigma_t,
        seed=args.seed,
    )
    samples = [ds[i] for i in range(args.n_pairs)]
    print(f'sampled {len(samples)} pairs from val')

    make_grid(samples, model, out_dir / 'sample_grid.png', n_show=args.n_pairs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='experiments/cross_frame_v08_deform_sl')
    ap.add_argument('--scenes-root', default='/mnt/mininas/datasets/pandaset')
    ap.add_argument('--img-size', type=int, default=64)
    ap.add_argument('--max-points', type=int, default=256)
    ap.add_argument('--baseline-min', type=int, default=1)
    ap.add_argument('--baseline-max', type=int, default=20)
    ap.add_argument('--sigma-ypr', type=float, default=1.0)
    ap.add_argument('--sigma-t',   type=float, default=0.20)
    ap.add_argument('--n-pairs', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    main(args)
