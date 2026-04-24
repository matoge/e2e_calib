"""Inspect predicted σ distribution across val pairs.

For v15_uvd_mix on PandaSet val, aggregate (σu, σv, √(σu·σv), ρ) over
all points in N pairs and draw:
  - histogram of √(σu·σv) (geometric mean σ, a reasonable scalar σ)
  - scatter σu vs σv
  - histogram of |ρ|
  - per-point σ vs actual residual |uv_pred - uv_gt| (calibration check)
  - CDF of σ to pick thresholds

Run: python scripts/visualization/vis_sigma_distribution.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import argparse
import random as _r
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.pandaset_pair import _SceneData
from models.cross_frame import CalibNetCrossFrame
from scripts.eval.cross_frame_lln import make_pair, infer_batch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 64


def load_model(ckpt_dir):
    sd = torch.load(ckpt_dir / 'best_model.pt', map_location=DEVICE, weights_only=True)
    deform = 'sl' if any('deform_img' in k for k in sd) else 'none'
    n_cross = sum(1 for k in sd if k.startswith('cross_blocks.') and k.endswith('.proj.weight'))
    n_intra = max(1, sum(1 for k in sd if k.startswith('intra_blocks.') and k.endswith('.norm_sa.weight')))
    proj_w = [sd[k] for k in sd if k.startswith('cross_blocks.') and k.endswith('.proj.weight')]
    out_dim = proj_w[0].shape[0] if proj_w else 5
    m = CalibNetCrossFrame(img_size=IMG_SIZE, deform_mode=deform,
                            n_cross_layers=n_cross, n_intra_layers=n_intra,
                            out_dim=out_dim).to(DEVICE)
    m.load_state_dict(sd); m.eval()
    return m, out_dim


def full_to_local_B(uv_full, box_B):
    u0, v0, cw, ch = box_B
    return np.stack([(uv_full[:, 0] - u0) * IMG_SIZE / cw,
                     (uv_full[:, 1] - v0) * IMG_SIZE / ch], axis=1)


def main(args):
    ckpt = Path(args.ckpt)
    model, out_dim = load_model(ckpt)
    print(f'loaded {ckpt.name}  out_dim={out_dim}')

    root = Path('/mnt/mininas/datasets/pandaset')
    names = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit()])
    shuffled = sorted([str(root / n) for n in names])
    _r.Random(42).shuffle(shuffled)
    val_roots = shuffled[int(len(shuffled) * 0.8):]
    scenes = []
    for sr in val_roots:
        scn = _SceneData(Path(sr)); scn.precompute_all(preload_images=False); scenes.append(scn)

    rng = np.random.default_rng(args.seed)
    all_sx, all_sy, all_rho, all_err = [], [], [], []
    n_ok = 0; attempts = 0
    while n_ok < args.n_pairs and attempts < args.n_pairs * 5:
        attempts += 1
        scn = scenes[int(rng.integers(len(scenes)))]
        bl = int(rng.integers(args.baseline_min, args.baseline_max + 1)) * int(rng.choice([-1, 1]))
        fi_A = int(rng.integers(5, scn.n_frames - abs(bl) - 5))
        fi_B = fi_A + bl
        s = make_pair(scn, rng, fi_A, fi_B, IMG_SIZE, 256, args.sigma_ypr, args.sigma_t, (128, 256))
        if s is None: continue
        raw = infer_batch(model, [s])[0]
        N = s['N_valid']
        if N < 8: continue

        mu = raw[:N, :2]
        if out_dim == 7:
            log_sx, log_sy = raw[:N, 3], raw[:N, 4]; rho = raw[:N, 6]
        else:
            log_sx, log_sy = raw[:N, 2], raw[:N, 3]; rho = np.tanh(raw[:N, 4]) * 0.99
        sx, sy = np.exp(log_sx), np.exp(log_sy)
        all_sx.append(sx); all_sy.append(sy); all_rho.append(rho)

        # per-point residual err (pred vs gt) in local B coords
        uv_hat_l = s['uv_B_hat_of_A'][:N].numpy()
        uv_gt_l  = full_to_local_B(s['uv_gt_full'][:N], s['box_B'])
        err = np.linalg.norm((uv_hat_l + mu) - uv_gt_l, axis=-1)
        all_err.append(err)
        n_ok += 1

    sx = np.concatenate(all_sx); sy = np.concatenate(all_sy)
    rho = np.concatenate(all_rho); err = np.concatenate(all_err)
    sig = np.sqrt(sx * sy)
    print(f'total points: {len(sig)}  (σ in px; err in px @ 64-patch scale)')
    print(f'σ stats: mean={sig.mean():.3f}  median={np.median(sig):.3f}  '
          f'q90={np.quantile(sig, 0.9):.3f}  q95={np.quantile(sig, 0.95):.3f}  '
          f'q99={np.quantile(sig, 0.99):.3f}  max={sig.max():.3f}')
    print(f'err stats: mean={err.mean():.3f}  median={np.median(err):.3f}  '
          f'q90={np.quantile(err, 0.9):.3f}  q95={np.quantile(err, 0.95):.3f}  '
          f'q99={np.quantile(err, 0.99):.3f}  max={err.max():.3f}')

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=120)
    fig.patch.set_facecolor('#f6f4ed')

    # 1. σ histogram (log x)
    ax = axes[0, 0]
    bins = np.logspace(np.log10(0.1), np.log10(max(50, sig.max())), 60)
    ax.hist(sig, bins=bins, color='#174734', alpha=0.85)
    for q, lbl in [(np.median(sig), 'med'), (np.quantile(sig, 0.9), 'q90'),
                   (np.quantile(sig, 0.99), 'q99')]:
        ax.axvline(q, color='#c13c14', ls='--', lw=1); ax.text(q, ax.get_ylim()[1]*0.9, f' {lbl} {q:.2f}', fontsize=8, color='#c13c14')
    ax.set_xscale('log'); ax.set_xlabel('√(σu·σv) [px]'); ax.set_ylabel('# points')
    ax.set_title('σ magnitude (all val points)')

    # 2. σu vs σv scatter
    ax = axes[0, 1]
    ax.scatter(sx, sy, s=1, alpha=0.15, color='#174734')
    ax.plot([0, 30], [0, 30], 'k--', lw=0.5, alpha=0.4)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.1, max(30, sx.max())); ax.set_ylim(0.1, max(30, sy.max()))
    ax.set_xlabel('σu [px]'); ax.set_ylabel('σv [px]')
    ax.set_title('σu vs σv (log-log, y=x dashed)')

    # 3. |ρ| histogram
    ax = axes[0, 2]
    ax.hist(np.abs(rho), bins=50, color='#174734', alpha=0.85)
    ax.set_xlabel('|ρ_uv|'); ax.set_ylabel('# points')
    ax.set_title('|correlation coefficient|')

    # 4. σ vs actual err (calibration check)
    ax = axes[1, 0]
    ax.scatter(sig, err, s=1, alpha=0.15, color='#174734')
    ax.plot([0.1, 30], [0.1, 30], color='#c13c14', lw=1, ls='--', label='err=σ')
    ax.plot([0.1, 30], [0.2, 60], color='#c13c14', lw=0.8, ls=':', label='err=2σ')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.1, max(30, sig.max())); ax.set_ylim(0.05, max(err.max(), 60))
    ax.set_xlabel('σ = √(σu·σv) [px]'); ax.set_ylabel('actual err |pred - gt| [px]')
    ax.set_title('calibration: σ vs realised err (want err ≈ σ)')
    ax.legend(fontsize=8)

    # 5. CDF of σ
    ax = axes[1, 1]
    sorted_sig = np.sort(sig)
    cdf = np.arange(1, len(sorted_sig) + 1) / len(sorted_sig)
    ax.plot(sorted_sig, cdf, color='#174734', lw=1.5)
    for frac, color in [(0.5, '#888'), (0.8, '#888'), (0.95, '#888')]:
        idx = int(frac * len(sorted_sig)); thr = sorted_sig[idx]
        ax.axhline(frac, color=color, ls=':', lw=0.8)
        ax.axvline(thr, color=color, ls=':', lw=0.8)
        ax.text(thr * 1.05, frac - 0.02, f'{int(frac*100)}%: σ≤{thr:.2f}', fontsize=8, color='#555')
    ax.set_xscale('log'); ax.set_xlabel('σ [px]'); ax.set_ylabel('CDF')
    ax.set_title('σ CDF (pick threshold here)')
    ax.grid(alpha=0.3)

    # 6. z = err / σ (if well-calibrated, this should have std ≈ 1)
    ax = axes[1, 2]
    z = err / np.clip(sig, 0.05, None)
    ax.hist(np.clip(z, 0, 10), bins=80, color='#174734', alpha=0.85)
    ax.axvline(1.0, color='#c13c14', ls='--', lw=1); ax.text(1.0, ax.get_ylim()[1]*0.9, ' z=1 (calibrated)', fontsize=8, color='#c13c14')
    ax.set_xlabel('z = err / σ  (std should be ≈ 1)'); ax.set_ylabel('# points')
    ax.set_title(f'z = err/σ  (std={z.std():.2f}, mean={z.mean():.2f})')

    plt.suptitle(f'{ckpt.name} — σ distribution & calibration ({n_ok} val pairs, {len(sig)} pts)',
                  fontsize=13, y=0.999)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches='tight', facecolor='#f6f4ed')
    print(f'saved → {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='experiments/cross_frame_v15_uvd_mix')
    ap.add_argument('--n-pairs', type=int, default=100)
    ap.add_argument('--baseline-min', type=int, default=1)
    ap.add_argument('--baseline-max', type=int, default=20)
    ap.add_argument('--sigma-ypr', type=float, default=1.0)
    ap.add_argument('--sigma-t',   type=float, default=0.20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='experiments/sigma_distribution_v15.png')
    args = ap.parse_args()
    main(args)
