"""Dataset-side visualization for calib_mode.

Shows the raw input that frame_token will be built from:
  • the camera image (patch_B)
  • HAT-projected LiDAR points: where the model SEES them (input)  → 'x' marker
  • GT-projected LiDAR points: where they SHOULD be (target)        → 'o' ring
  • Δuv = GT - HAT (target offset the model must learn)              → arrow

This is purely a dataset-side check: does the perturbation actually
produce a non-trivial offset that the model can learn from?
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from datasets.pandaset_pair import PandaSetCrossFrameDataset

IMG_SIZE = 64


def draw_one(ax, sample, k_show=20):
    patch = sample['patch_A'].numpy()  # show A patch (camera content)
    img = np.transpose(patch, (1, 2, 0)).clip(0, 1)
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    # uvd_A: model input position = HAT-projected (calib_mode) or GT-projected (cross-frame)
    # uv_B_gt_of_A: GT target (where LiDAR really lands)
    uvd_A = sample['uvd_A'].numpy()
    uv_input = uvd_A[:, :2]
    # In calib_mode, also fetch GT-projected target (uv_B_gt_of_A in patch_B coords;
    # but we want it in patch_A coords for overlay on patch_A. Approximate: in
    # calib mode patch_A and patch_B differ only by crop center, so the *delta*
    # (Δuv) carries over. We compute Δuv from uv_B_hat - uv_B_gt and apply to
    # uv_input.
    uv_B_gt = sample['uv_B_gt_of_A'].numpy()
    uv_B_hat = sample['uv_B_hat_of_A'].numpy()
    # Δuv in patch-local coords (same magnitude in patch_A since translations
    # are absorbed by the local-coord transform)
    delta = uv_B_gt - uv_B_hat
    uv_target = uv_input + delta  # GT position in patch_A coords (approximate)
    pad = sample['pad_A'].numpy()
    valid = ~pad
    in_patch = ((uv_input[:, 0] >= 0) & (uv_input[:, 0] < IMG_SIZE) &
                (uv_input[:, 1] >= 0) & (uv_input[:, 1] < IMG_SIZE))
    keep = np.where(valid & in_patch)[0]
    if len(keep) == 0:
        ax.set_title('no valid pts', fontsize=8, loc='left')
        return
    delta_norm = np.linalg.norm(delta[keep], axis=-1)
    mean_shift = delta_norm.mean()
    max_shift = delta_norm.max()
    if len(keep) > k_show:
        step = max(1, len(keep) // k_show)
        sel = keep[::step][:k_show]
    else:
        sel = keep
    cmap = plt.get_cmap('tab10')
    for k, i in enumerate(sel):
        c = cmap(k % 10)
        xh, yh = uv_input[i]; xg, yg = uv_target[i]
        # × at HAT (model input — visually misaligned LiDAR)
        ax.plot(xh, yh, 'x', color=c, markersize=7, mew=1.6, alpha=0.95, zorder=6)
        # ○ at GT (target — where LiDAR should land per camera evidence)
        ax.plot(xg, yg, 'o', color=c, markersize=9, markerfacecolor='none',
                 markeredgecolor=c, mew=1.6, alpha=0.95, zorder=7)
        # arrow HAT→GT (Δuv the model must learn)
        ax.add_patch(FancyArrowPatch((xh, yh), (xg, yg), arrowstyle='-|>',
                                      mutation_scale=10, color=c, lw=1.6,
                                      alpha=0.85, zorder=5))
    ax.set_title(f'Δuv: μ={mean_shift:.2f} max={max_shift:.2f} px  ({len(keep)} pts)',
                 fontsize=8, loc='left')


def main(args):
    ds = PandaSetCrossFrameDataset(
        scenes_root=args.scenes_root, split=args.split,
        cameras=args.cameras, img_size=IMG_SIZE,
        max_points=args.max_points,
        baseline_range=(args.baseline_min, args.baseline_max),
        sigma_ypr=args.sigma_ypr, sigma_t=args.sigma_t,
        virtual_epoch_len=args.n_pairs * 8, seed=args.seed,
        lidar_subdir=args.lidar_subdir,
        calib_mode=args.calib_mode,
    )
    cols = 4
    rows = (args.n_pairs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 4.0),
                              facecolor='#f6f4ed')
    axes = np.array(axes).reshape(rows, cols)
    plotted = 0
    for i in range(len(ds) * 4):
        if plotted >= args.n_pairs:
            break
        s = ds[i % len(ds)]
        if s is None:
            continue
        ax = axes[plotted // cols, plotted % cols]
        draw_one(ax, s)
        plotted += 1
    for k in range(plotted, rows * cols):
        axes[k // cols, k % cols].axis('off')
    mode = 'calib' if args.calib_mode else 'cross-frame'
    fig.suptitle(f'Dataset-side check ({mode})  σ_ypr={args.sigma_ypr}° σ_t={args.sigma_t}m  '
                  f'split={args.split}  scenes={Path(args.scenes_root).name}\n'
                  f'  ×=HAT (model input)   ○=GT (target)   arrow=Δuv',
                  fontsize=10)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out, dpi=120, bbox_inches='tight', facecolor='#f6f4ed')
    print(f'saved → {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes-root', required=True)
    ap.add_argument('--cameras', default='front_camera')
    ap.add_argument('--split', default='val')
    ap.add_argument('--max-points', type=int, default=256)
    ap.add_argument('--baseline-min', type=int, default=1)
    ap.add_argument('--baseline-max', type=int, default=5)
    ap.add_argument('--sigma-ypr', type=float, default=0.5)
    ap.add_argument('--sigma-t', type=float, default=0.05)
    ap.add_argument('--lidar-subdir', default='lidar')
    ap.add_argument('--calib-mode', action='store_true')
    ap.add_argument('--n-pairs', type=int, default=24)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='dataset_calib_check.png')
    args = ap.parse_args()
    main(args)
