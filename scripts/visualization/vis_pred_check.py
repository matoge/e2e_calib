"""Train/val prediction check — uses PandaSetCrossFrameDataset directly so
viz path is byte-identical to training. Plots, per pair:

    patch_A | patch_B
              × = uv_B_hat_of_A   (input hypothesis)
              ○ = uv_B_gt_of_A    (target)
              + = uv_B_hat_of_A + Δμ   (raw model prediction = train metric)

err shown in title is `mean(|pred - gt|)` over IN-PATCH points = same quantity
as `err_AB` reported in train.log. No BA. No make_pair shortcut.
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
import matplotlib.cm as cm
from matplotlib.patches import ConnectionPatch, Ellipse, FancyArrowPatch

from datasets.pandaset_pair import PandaSetCrossFrameDataset
from models.cross_frame_multi import CalibNetMultiFrame
from models.cross_frame_unified import CalibNetUnifiedFrame

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 64


def load_model(ckpt_dir, ckpt_name='best_model.pt'):
    """Auto-detect unified vs legacy multi from state_dict layout.

    Unified has `encoder.cnn.*` / `encoder.fuse.*`; legacy has `cnn.*` at the
    top level. `cross_blocks.*.proj.weight` exists in both with same shape
    so we can read out_dim either way.
    """
    sd = torch.load(ckpt_dir / ckpt_name, map_location=DEVICE, weights_only=True)
    is_unified = any(k.startswith('encoder.cnn.') for k in sd)
    n_cross = sum(1 for k in sd if k.startswith('cross_blocks.') and k.endswith('.proj.weight'))
    proj_w = [sd[k] for k in sd if k.startswith('cross_blocks.') and k.endswith('.proj.weight')]
    out_dim = proj_w[0].shape[0] if proj_w else 5
    if is_unified:
        n_intra = max(1, sum(1 for k in sd if k.startswith('encoder.intra.') and k.endswith('.norm_sa.weight')))
        # max_kv_frames inferred from sampling_offsets shape:
        # n_heads*n_levels*n_points*2 = sampling_offsets.weight.shape[0]
        # we know n_heads=4, n_points=4 by construction; solve for n_levels.
        so_w = sd.get('cross_blocks.0.deform.sampling_offsets.weight')
        n_levels = max(1, so_w.shape[0] // (4 * 4 * 2)) if so_w is not None else 2
        m = CalibNetUnifiedFrame(img_size=IMG_SIZE,
                                  n_cross_layers=n_cross, n_intra_layers=n_intra,
                                  max_kv_frames=n_levels, out_dim=out_dim).to(DEVICE)
    else:
        deform = 'sl' if any('deform_img' in k for k in sd) else 'none'
        n_intra = max(1, sum(1 for k in sd if k.startswith('intra_blocks.') and k.endswith('.norm_sa.weight')))
        m = CalibNetMultiFrame(img_size=IMG_SIZE, deform_mode=deform,
                                n_cross_layers=n_cross, n_intra_layers=n_intra,
                                out_dim=out_dim).to(DEVICE)
    m.load_state_dict(sd); m.eval()
    return m, out_dim


@torch.no_grad()
def run_one(model, sample, multi_frame=False):
    """Run model on a single dataset sample, return uv_hat, uv_gt, uv_pred (local)."""
    batch = {k: (v.unsqueeze(0).to(DEVICE) if torch.is_tensor(v) else v)
             for k, v in sample.items()}
    kw = dict(
        patch_A=batch['patch_A'], uvd_A=batch['uvd_A'],
        patch_B=batch['patch_B'], uvd_B=batch['uvd_B'],
        pose_AB_6dof=batch['pose_AB_6dof'], pose_BA_6dof=batch['pose_BA_6dof'],
        uv_B_hat_of_A=batch['uv_B_hat_of_A'], uv_A_hat_of_B=batch['uv_A_hat_of_B'],
        pad_A=batch['pad_A'], pad_B=batch['pad_B'],
        uvd_A_full=batch['uvd_A_full'], uvd_B_full=batch['uvd_B_full'],
        pad_A_full=batch['pad_A_full'], pad_B_full=batch['pad_B_full'],
    )
    if multi_frame:
        for k in ['patch_M', 'uvd_M', 'pad_M', 'uvd_M_full', 'pad_M_full',
                  'pose_AM_6dof', 'uv_M_hat_of_A', 'uv_M_hat_of_B']:
            kw[k] = batch[k]
    raw_AB, _ = model(**kw)
    raw = raw_AB[0].cpu().numpy()           # (max_pts, 5)
    uv_hat = sample['uv_B_hat_of_A'].numpy()
    uv_gt  = sample['uv_B_gt_of_A'].numpy()
    pad    = sample['pad_A'].numpy()
    delta  = raw[:, :2]
    log_sx, log_sy = raw[:, 2], raw[:, 3]
    rho_raw = raw[:, 4]
    sx = np.exp(log_sx); sy = np.exp(log_sy)
    rho = np.tanh(rho_raw) * 0.99
    sigma  = 0.5 * (sx + sy)
    uv_pred = uv_hat + delta
    return dict(
        patch_A=sample['patch_A'].numpy(),
        patch_B=sample['patch_B'].numpy(),
        uv_A=sample['uvd_A'].numpy()[:, :2],
        uv_hat=uv_hat, uv_gt=uv_gt, uv_pred=uv_pred,
        sigma=sigma, sx=sx, sy=sy, rho=rho, pad=pad,
    )


def draw_pair(fig, ax_A, ax_B, r, k_show=12):
    pA = np.transpose(r['patch_A'], (1, 2, 0)).clip(0, 1)
    pB = np.transpose(r['patch_B'], (1, 2, 0)).clip(0, 1)
    ax_A.imshow(pA); ax_A.set_xticks([]); ax_A.set_yticks([])
    ax_B.imshow(pB); ax_B.set_xticks([]); ax_B.set_yticks([])

    valid = ~r['pad']
    in_patch = ((r['uv_hat'][:, 0] >= 0) & (r['uv_hat'][:, 0] < IMG_SIZE) &
                (r['uv_hat'][:, 1] >= 0) & (r['uv_hat'][:, 1] < IMG_SIZE) &
                (r['uv_gt' ][:, 0] >= 0) & (r['uv_gt' ][:, 0] < IMG_SIZE) &
                (r['uv_gt' ][:, 1] >= 0) & (r['uv_gt' ][:, 1] < IMG_SIZE))
    keep = np.where(valid & in_patch)[0]
    if len(keep) == 0:
        ax_B.set_title('no valid pts', fontsize=9, loc='left')
        return
    err_hyp = np.linalg.norm(r['uv_hat'][keep] - r['uv_gt'][keep], axis=-1).mean()
    err_pred = np.linalg.norm(r['uv_pred'][keep] - r['uv_gt'][keep], axis=-1).mean()
    sigma_mean = r['sigma'][keep].mean()
    # 2σ-coverage on the kept points
    sx_k, sy_k, rho_k = r['sx'][keep], r['sy'][keep], r['rho'][keep]
    d_k = r['uv_gt'][keep] - r['uv_pred'][keep]
    inv_det = 1.0 / np.clip(sx_k**2 * sy_k**2 * (1.0 - rho_k**2), 1e-12, None)
    mahal2 = inv_det * (sy_k**2 * d_k[:, 0]**2
                        - 2 * rho_k * sx_k * sy_k * d_k[:, 0] * d_k[:, 1]
                        + sx_k**2 * d_k[:, 1]**2)
    cover_2s = float((mahal2 < 4.0).mean())                     # 2σ → mahal² < 4

    # subsample for display
    if len(keep) > k_show:
        step = max(1, len(keep) // k_show)
        keep_show = keep[::step][:k_show]
    else:
        keep_show = keep

    cmap = plt.get_cmap('tab10')
    for k, i in enumerate(keep_show):
        c = cmap(k % 10)
        xa, ya = r['uv_A'][i]
        xh, yh = r['uv_hat'][i]
        xp, yp = r['uv_pred'][i]
        xg, yg = r['uv_gt'][i]
        ax_A.plot(xa, ya, 'o', color=c, markersize=6, markeredgecolor='white',
                   mew=0.8, zorder=5)
        ax_A.annotate(str(k), (xa, ya), ha='center', va='center',
                       fontsize=7, color='black', weight='bold', zorder=6)
        # cross-panel dotted line: A → B(GT)
        fig.add_artist(ConnectionPatch(
            xyA=(xa, ya), xyB=(xg, yg), coordsA='data', coordsB='data',
            axesA=ax_A, axesB=ax_B, color=c, lw=1.4, ls=(0,(2,2)), alpha=0.85, zorder=3))
        # covariance ellipses around prediction (1σ thin, 2σ thick) +
        # Mahalanobis distance from GT → calibration check
        sx, sy, rho = r['sx'][i], r['sy'][i], r['rho'][i]
        cov = np.array([[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]])
        try:
            inv = np.linalg.inv(cov)
            d = np.array([xg - xp, yg - yp])
            mahal = float(np.sqrt(d @ inv @ d))                  # σ-distance from GT
        except Exception:
            mahal = float('nan')
        try:
            vals, vecs = np.linalg.eigh(cov)
            vals = np.clip(vals, 1e-6, None)
            angle = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
            for n_sig, lw, alpha in ((1.0, 0.7, 0.45), (2.0, 1.4, 0.85)):
                w, h = 2.0 * n_sig * np.sqrt(vals[::-1])
                ax_B.add_patch(Ellipse((xp, yp), width=w, height=h, angle=angle,
                                        facecolor='none', edgecolor=c, lw=lw,
                                        alpha=alpha, zorder=4))
        except Exception:
            pass
        # GT marker: green ring if within 2σ, red if outside (calibration check)
        ring_c = '#0a8a2a' if mahal < 2.0 else '#c41a1a'
        ax_B.plot(xg, yg, marker='o', markersize=9, markeredgecolor=ring_c,
                   markerfacecolor='none', mew=2.0, alpha=0.9, zorder=8)
        # arrow: hyp → pred (correction the model applies)
        ax_B.add_patch(FancyArrowPatch((xh, yh), (xp, yp),
                                        arrowstyle='-|>', mutation_scale=10,
                                        color=c, lw=1.6, alpha=0.85, zorder=5))
        # arrow: pred → gt (residual error after correction)
        ax_B.add_patch(FancyArrowPatch((xp, yp), (xg, yg),
                                        arrowstyle='-|>', mutation_scale=8,
                                        color=c, lw=1.0, ls=':', alpha=0.55, zorder=5))
        # markers
        ax_B.plot(xh, yh, 'x', color=c, markersize=6, mew=1.4, alpha=0.8, zorder=6)
        ax_B.plot(xg, yg, 'o', color=c, markersize=5, markeredgecolor=c,
                   markerfacecolor='white', mew=1.2, alpha=0.95, zorder=7)
        ax_B.plot(xp, yp, '+', color=c, markersize=10, mew=1.8, zorder=8)

    ax_A.set_title(f'A  ({len(keep)} pts)', fontsize=9, loc='left')
    ax_B.set_title(f'B  hyp {err_hyp:.2f} → pred {err_pred:.2f} px '
                   f'  σ̄={sigma_mean:.2f}  2σ-cover={100*cover_2s:.0f}%',
                   fontsize=9, loc='left')


def main(args):
    ckpt = Path(args.ckpt)
    model, out_dim = load_model(ckpt, ckpt_name=args.ckpt_name)
    print(f'loaded {ckpt.name}/{args.ckpt_name}  out_dim={out_dim}')

    # mirror the training-time dataset construction.
    ds = PandaSetCrossFrameDataset(
        scenes_root=args.scenes_root, train_frac=args.train_frac,
        split=args.split, img_size=IMG_SIZE, max_points=256,
        baseline_range=(args.baseline_min, args.baseline_max),
        sigma_ypr=args.sigma_ypr, sigma_t=args.sigma_t,
        crop_range=(128, 256), cameras=args.cameras,
        triplet=args.multi_frame, virtual_epoch_len=200, seed=args.seed,
    )

    # Tile layout: `n_cols` pairs per row, top→bottom sorted by Δfi (easy → hard)
    n_cols  = args.n_cols
    n_rows  = (args.n_pairs + n_cols - 1) // n_cols
    n_pairs = n_rows * n_cols
    fig, axes = plt.subplots(n_rows, 2 * n_cols,
                              figsize=(7 * n_cols, 6 * n_rows), dpi=160)
    fig.patch.set_facecolor('#f6f4ed')
    if n_rows == 1:
        axes = axes[None, :]

    # sample many candidates, sort by baseline, pick n_pairs evenly distributed
    # so cells go short → long Δfi (top-left = easiest, bottom-right = hardest).
    n_pool = max(n_pairs * 8, 32)
    pool = []
    for k in range(n_pool):
        s = ds[k]
        bl = abs(s['fi_B'] - s['fi_A']) if 'fi_A' in s else 0
        pool.append((bl, s))
    pool.sort(key=lambda x: x[0])
    step = len(pool) / n_pairs
    picks = [pool[int(round(i * step))] for i in range(n_pairs)]
    seen = set(); uniq = []
    for bl, s in picks:
        key = (s['fi_A'], s['fi_B'], s['scene'])
        if key not in seen:
            seen.add(key); uniq.append((bl, s))
    while len(uniq) < n_pairs and pool:
        bl, s = pool.pop()
        key = (s['fi_A'], s['fi_B'], s['scene'])
        if key not in seen:
            seen.add(key); uniq.append((bl, s))
    uniq.sort(key=lambda x: x[0])

    # Fill grid in row-major order: easy → hard left-to-right, top-to-bottom
    for idx, (bl, s) in enumerate(uniq[:n_pairs]):
        row = idx // n_cols
        col = idx % n_cols
        ax_A = axes[row, 2 * col]
        ax_B = axes[row, 2 * col + 1]
        r = run_one(model, s, multi_frame=args.multi_frame)
        draw_pair(fig, ax_A, ax_B, r)
        ax_A.set_ylabel(f'Δfi={bl}', fontsize=10)

    fig.suptitle(f'{ckpt.name} / {args.ckpt_name} — split={args.split}  '
                 f'(× hyp, ○ GT, + model pred)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'saved → {args.out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--ckpt-name', default='best_model.pt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-pairs', type=int, default=6,
                    help='total pairs to display (rounded up to fit n-cols × rows).')
    ap.add_argument('--n-cols', type=int, default=3,
                    help='pairs per row in tile layout (each pair spans 2 cols).')
    ap.add_argument('--scenes-root', default='/mnt/nvme6t/pandaset_39')
    ap.add_argument('--train-frac', type=float, default=0.8)
    ap.add_argument('--split', default='train', choices=['train', 'val'])
    ap.add_argument('--cameras', default='front_camera')
    ap.add_argument('--baseline-min', type=int, default=1)
    ap.add_argument('--baseline-max', type=int, default=20)
    ap.add_argument('--sigma-ypr', type=float, default=1.0)
    ap.add_argument('--sigma-t', type=float, default=0.2)
    ap.add_argument('--multi-frame', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    main(args)
