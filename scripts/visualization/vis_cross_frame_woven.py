"""Zero-shot visual sanity check for CalibNetCrossFrame on WovenSequence.

Loads the v13_mix (or any cross-frame) checkpoint and runs it against
WovenSequenceCrossFrameDataset (the target sequence is
  .../llinking_27/tf_long2/sequence=ip654_...
with VLS128 LiDAR + rectified pinhole tss4_fcm at 3840x2160).

Emits one N-row PNG grid identical in layout to vis_cross_frame.py:
    col 0  patch_A with A LiDAR points
    col 1  patch_B with uv_hat (×), uv_pred (●), uv_gt (○) + σ-ellipses
    col 2  pure scatter zoom with arrows hat→pred

If the model is doing what we want, green ● lands on blue ○ and
per-pair pred_err << hyp_err.
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

from datasets.woven_sequence_pair import WovenSequenceCrossFrameDataset
from models.cross_frame import CalibNetCrossFrame

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_SEQ = ('/home/hfunaya/git/loom/backend/assets/woven_sequence/llinking_27/'
               'tf_long2/sequence=ip654_1337941440921107425_'
               '16943630305775105398_1749030654176-1749030664176')


def overlay_points(ax, uv, color, marker='o', size=20, label=None,
                   ec='white', lw=0.4, alpha=1.0):
    ax.scatter(uv[:, 0], uv[:, 1], c=color, marker=marker, s=size,
               edgecolors=ec, linewidths=lw, alpha=alpha, label=label, zorder=5)


def draw_uncertainty(ax, uv, sigma_uv, rho, color='lime', max_n=40):
    if len(uv) > max_n:
        idx = np.linspace(0, len(uv) - 1, max_n).astype(int)
    else:
        idx = np.arange(len(uv))
    for i in idx:
        sx, sy = sigma_uv[i]
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
    fig, axes = plt.subplots(n_show, 3, figsize=(13, 4.0 * n_show), dpi=110)
    fig.patch.set_facecolor('#f6f4ed')

    keys = ['patch_A', 'uvd_A', 'patch_B', 'uvd_B',
            'pose_AB_6dof', 'pose_BA_6dof',
            'uv_B_hat_of_A', 'uv_A_hat_of_B', 'pad_A', 'pad_B']

    stats = []
    for ri, s in enumerate(samples[:n_show]):
        batch = {k: s[k].unsqueeze(0).to(DEVICE) for k in keys}
        with torch.no_grad():
            raw_AB, _ = model(**batch)
        raw = raw_AB[0].cpu().numpy()
        N = (~s['pad_A']).sum().item()
        mu = raw[:N, :2]
        log_sx, log_sy = raw[:N, 2], raw[:N, 3]
        rho = np.tanh(raw[:N, 4]) * 0.99
        sx = np.exp(log_sx); sy = np.exp(log_sy)

        uv_hat = s['uv_B_hat_of_A'][:N].cpu().numpy()
        uv_gt  = s['uv_B_gt_of_A'][:N].cpu().numpy()
        uv_pred = uv_hat + mu

        patch_A_np = (s['patch_A'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        patch_B_np = (s['patch_B'].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        uv_A_patch = s['uvd_A'][:N, :2].cpu().numpy()

        # col 0
        ax = axes[ri, 0] if n_show > 1 else axes[0]
        ax.imshow(patch_A_np)
        overlay_points(ax, uv_A_patch, '#c13c14', marker='o', size=18, label='A pts')
        ax.set_title(f'frame A — fi={s["fi_A"]} → fi={s["fi_B"]}  (Δ={s["fi_B"] - s["fi_A"]:+d})',
                      fontsize=10, loc='left')
        ax.set_xticks([]); ax.set_yticks([])

        # col 1
        ax = axes[ri, 1] if n_show > 1 else axes[1]
        ax.imshow(patch_B_np)
        overlay_points(ax, uv_hat,  '#c13c14', marker='x', size=22, label='hyp', ec='none', lw=1.2)
        overlay_points(ax, uv_pred, '#0fa550', marker='o', size=20, label='pred', ec='black', lw=0.5)
        overlay_points(ax, uv_gt,   '#1e6fff', marker='o', size=10, label='gt', ec='black', lw=0.4, alpha=0.85)
        draw_uncertainty(ax, uv_pred, np.stack([sx, sy], axis=1), rho)
        err_hat  = float(np.linalg.norm(uv_hat  - uv_gt, axis=1).mean())
        err_pred = float(np.linalg.norm(uv_pred - uv_gt, axis=1).mean())
        ax.set_title(f'frame B — hyp_err {err_hat:.2f}px → pred_err {err_pred:.2f}px',
                      fontsize=10, loc='left')
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(frameon=False, fontsize=8, loc='lower right')

        # col 2
        ax = axes[ri, 2] if n_show > 1 else axes[2]
        ax.set_facecolor('#fff')
        for i in range(N):
            ax.annotate('', xy=uv_pred[i], xytext=uv_hat[i],
                         arrowprops=dict(arrowstyle='-|>', color='#888', lw=0.5))
        overlay_points(ax, uv_hat,  '#c13c14', marker='x', size=22, label='hyp')
        overlay_points(ax, uv_pred, '#0fa550', marker='o', size=22, label='pred')
        overlay_points(ax, uv_gt,   '#1e6fff', marker='o', size=12, label='gt')
        ax.set_xlim(0, 64); ax.set_ylim(64, 0)
        ax.set_aspect('equal')
        ax.set_title(f'arrows: hyp→pred', fontsize=10, loc='left')
        ax.grid(alpha=0.2)

        stats.append((err_hat, err_pred))
        print(f'  pair {ri:2d}: fi_A={s["fi_A"]:3d} fi_B={s["fi_B"]:3d} '
              f'N={N:3d}  hyp_err={err_hat:6.2f}px  pred_err={err_pred:6.2f}px  '
              f'gap={err_hat - err_pred:+6.2f}px')

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight', facecolor='#f6f4ed')
    plt.close(fig)
    print(f'saved → {out_path}')

    eh = np.mean([x[0] for x in stats])
    ep = np.mean([x[1] for x in stats])
    print(f'\nmean over {len(stats)} pairs: hyp={eh:.2f}px  pred={ep:.2f}px  '
          f'reduction={100*(1 - ep/eh):.1f}%')


def main(args):
    out_dir = Path(args.ckpt_dir) / 'vis_woven'
    out_dir.mkdir(parents=True, exist_ok=True)

    sd = torch.load(Path(args.ckpt_dir) / 'best_model.pt',
                    map_location=DEVICE, weights_only=True)
    deform = 'sl' if any('deform_img' in k for k in sd.keys()) else 'none'
    n_cross = sum(1 for k in sd.keys()
                  if k.startswith('cross_blocks.') and k.endswith('.proj.weight'))
    print(f'detected deform_mode={deform}, n_cross_layers={n_cross}')

    model = CalibNetCrossFrame(img_size=args.img_size,
                                n_cross_layers=n_cross,
                                deform_mode=deform).to(DEVICE)
    model.load_state_dict(sd)
    model.eval()

    ds = WovenSequenceCrossFrameDataset(
        seq_root=args.seq_root,
        img_size=args.img_size, max_points=args.max_points,
        baseline_range=(args.baseline_min, args.baseline_max),
        sigma_ypr=args.sigma_ypr, sigma_t=args.sigma_t,
        crop_range=(args.crop_min, args.crop_max),
        img_scale_div=args.img_scale_div,
        virtual_epoch_len=args.n_pairs,
        seed=args.seed,
    )
    samples = [ds[i] for i in range(args.n_pairs)]
    print(f'sampled {len(samples)} pairs from {Path(args.seq_root).name}')

    make_grid(samples, model, out_dir / 'sample_grid.png', n_show=args.n_pairs)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='experiments/cross_frame_v13_mix')
    ap.add_argument('--seq-root', default=DEFAULT_SEQ)
    ap.add_argument('--img-size', type=int, default=64)
    ap.add_argument('--max-points', type=int, default=256)
    ap.add_argument('--baseline-min', type=int, default=1)
    ap.add_argument('--baseline-max', type=int, default=20)
    ap.add_argument('--sigma-ypr', type=float, default=1.0)
    ap.add_argument('--sigma-t',   type=float, default=0.20)
    ap.add_argument('--crop-min',  type=int, default=256)
    ap.add_argument('--crop-max',  type=int, default=512)
    ap.add_argument('--img-scale-div', type=int, default=2)
    ap.add_argument('--n-pairs', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    main(args)
