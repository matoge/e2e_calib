"""LLN + BA — beginner-friendly version.

Story per pair (big, readable): each row has two patches side-by-side.
  A (left):  K query points, colored 0..K-1 with numbers, big markers.
  B (right):  for each point k (SAME color):
      ×  at hypothesis location (where the perturbed pose sent it — WRONG)
      ● at model's corrected prediction (= hyp + Δ)
      ★ at ground-truth location
      + arrow from × → ● (the model's correction)
  A ●_k ——— B ★_k: dotted line across panels (GT correspondence, same color).

At a glance:
  - If a same-color ● lands on a ★ on the right, the model got it right.
  - The ×s show how far the perturbed hypothesis was off.
  - The dotted line says "this A point and this B star are the same 3D point".

No pivot star — every highlighted point is a query. Fewer points (4-6) so the
markers aren't crowded.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import argparse, random
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import ConnectionPatch

from datasets.pandaset_pair import _SceneData, _ypr_t_to_mat, _invert_mat
from models.cross_frame import CalibNetCrossFrame
from scripts.eval.cross_frame_lln import make_pair, ba_recover, pose_errors, infer_batch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def project_batch(P_cam, K):
    z = np.clip(P_cam[:, 2], 1e-6, None)
    uv = (K @ P_cam.T)[:2] / z
    return uv.T


def main(args):
    ckpt_dir = Path(args.ckpt_dir)
    out_dir  = ckpt_dir / 'lln'
    out_dir.mkdir(parents=True, exist_ok=True)

    sd = torch.load(ckpt_dir / 'best_model.pt', map_location=DEVICE, weights_only=True)
    deform = 'sl' if any('deform_img' in k for k in sd.keys()) else 'none'
    n_cross = sum(1 for k in sd if k.startswith('cross_blocks.') and k.endswith('.proj.weight'))
    n_intra = max(1, sum(1 for k in sd if k.startswith('intra_blocks.') and k.endswith('.norm_sa.weight')))
    model = CalibNetCrossFrame(img_size=args.img_size, deform_mode=deform,
                                n_cross_layers=n_cross, n_intra_layers=n_intra).to(DEVICE)
    model.load_state_dict(sd); model.eval()

    root = Path(args.scenes_root)
    names = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit()])
    shuffled = sorted([str(root / n) for n in names])
    random.Random(42).shuffle(shuffled)
    cutoff = int(len(shuffled) * args.train_frac)
    scenes = []
    for sr in shuffled[cutoff:]:
        scn = _SceneData(Path(sr)); scn.precompute_all(); scenes.append(scn)

    rng = np.random.default_rng(args.seed)
    bl = args.baseline
    samples = []
    while len(samples) < args.n_pairs:
        scn = scenes[int(rng.integers(len(scenes)))]
        fi_A = int(rng.integers(scn.n_frames))
        fi_B = fi_A + bl * int(rng.choice([-1, 1]))
        if fi_B < 0 or fi_B >= scn.n_frames: continue
        s = make_pair(scn, rng, fi_A, fi_B, args.img_size, args.max_points,
                       args.sigma_ypr, args.sigma_t, (args.crop_min, args.crop_max))
        if s is None: continue
        samples.append(s)

    raw_list = []
    for i in range(0, len(samples), args.batch_size):
        raw_list.append(infer_batch(model, samples[i:i + args.batch_size]))
    raw_all = np.concatenate(raw_list, axis=0)

    K_SHOW = args.k_show
    n_rows = args.n_show
    # each pair is a row with 2 big panels (A | B), big enough to see markers.
    # figsize gives each panel ~6 × 6 inches.
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 6.5 * n_rows), dpi=110)
    fig.patch.set_facecolor('#f6f4ed')
    if n_rows == 1: axes = axes[None, :]

    for ri in range(n_rows):
        s = samples[ri]; raw = raw_all[ri]
        N = s['N_valid']
        if N < K_SHOW: continue

        mu = raw[:N, :2]
        log_sx, log_sy = raw[:N, 2], raw[:N, 3]
        rho = np.tanh(raw[:N, 4]) * 0.99
        sx, sy = np.exp(log_sx), np.exp(log_sy)

        K_int = s['K']
        P_A = s['P_A_cam'][:N]
        homo = np.column_stack([P_A, np.ones(N)])
        uv_hat = project_batch((homo @ s['T_AB_hat'].T)[:, :3], K_int)
        uv_gt  = project_batch((homo @ s['T_AB_gt'].T)[:, :3],  K_int)
        u0b, v0b, cwb, chb = s['box_B']
        scale_u = cwb / args.img_size; scale_v = chb / args.img_size
        uv_pred = uv_hat + np.stack([mu[:, 0] * scale_u, mu[:, 1] * scale_v], axis=1)

        to_local_B = lambda uv: np.stack([(uv[:, 0] - u0b) * (args.img_size / cwb),
                                           (uv[:, 1] - v0b) * (args.img_size / chb)], axis=1)
        uv_hat_l  = to_local_B(uv_hat)
        uv_pred_l = to_local_B(uv_pred)
        uv_gt_l   = to_local_B(uv_gt)

        uvd_A = s['uvd_A'][:N].numpy()
        uv_A_l = uvd_A[:, :2]

        # pick K_SHOW spatially-spread points whose hyp AND gt land inside patch_B
        in_patch = ((uv_hat_l[:, 0] >= 0) & (uv_hat_l[:, 0] < args.img_size) &
                    (uv_hat_l[:, 1] >= 0) & (uv_hat_l[:, 1] < args.img_size) &
                    (uv_gt_l[:, 0]  >= 0) & (uv_gt_l[:, 0]  < args.img_size) &
                    (uv_gt_l[:, 1]  >= 0) & (uv_gt_l[:, 1]  < args.img_size) &
                    (uv_A_l[:, 0]   >= 0) & (uv_A_l[:, 0]   < args.img_size) &
                    (uv_A_l[:, 1]   >= 0) & (uv_A_l[:, 1]   < args.img_size))
        valid_idx = np.where(in_patch)[0]
        if len(valid_idx) == 0: continue
        if len(valid_idx) > K_SHOW:
            step = max(1, len(valid_idx) // K_SHOW)
            sel = valid_idx[::step][:K_SHOW]
        else:
            sel = valid_idx

        cmap = cm.get_cmap('tab10')
        colors = [cmap(k % 10) for k in range(len(sel))]

        ax_A = axes[ri, 0]; ax_B = axes[ri, 1]
        pA = (s['patch_A'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pB = (s['patch_B'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        ax_A.imshow(pA); ax_A.set_xticks([]); ax_A.set_yticks([])
        ax_B.imshow(pB); ax_B.set_xticks([]); ax_B.set_yticks([])

        err_h_sel = np.linalg.norm(uv_hat[sel] - uv_gt[sel], axis=1).mean()
        err_p_sel = np.linalg.norm(uv_pred[sel] - uv_gt[sel], axis=1).mean()

        ax_A.set_title(f'A  scene {s["scene"]}/fi={s["fi_A"]}\n'
                        f'red ★ = PIVOT (patch center)   ● = other query points',
                        fontsize=12, loc='left')
        ax_B.set_title(f'B  fi={s["fi_B"]} (Δ{s["fi_B"]-s["fi_A"]:+d})\n'
                        f'red ★ = hyp (=patch center, WRONG)    green ★ = model pred    blue ★ = GT\n'
                        f'green arrow = model correction for pivot    hyp err {err_h_sel:.1f}→pred {err_p_sel:.1f}px',
                        fontsize=12, loc='left')

        # ---- PIVOT: special highlight ----
        # In A: patch center.
        # In B: hypothesis = patch center (by construction).
        #        prediction  = center + Δ_center (model's correction, we use mu of
        #                       the point closest to center as the pivot's Δ proxy).
        #        gt          = projection under T_AB_gt of the 3D pivot.
        piv_A_xy = (args.img_size / 2, args.img_size / 2)
        piv_B_hat = (args.img_size / 2, args.img_size / 2)
        # find the query point closest to pivot in A (uv_A ≈ center)
        dist_to_A_center = np.linalg.norm(uv_A_l - np.array([args.img_size/2, args.img_size/2]), axis=1)
        piv_idx = int(np.argmin(dist_to_A_center))
        piv_B_pred = (uv_hat_l[piv_idx, 0] + mu[piv_idx, 0],
                      uv_hat_l[piv_idx, 1] + mu[piv_idx, 1])
        piv_B_gt  = (uv_gt_l[piv_idx, 0], uv_gt_l[piv_idx, 1])

        # draw pivot star on A (big red)
        ax_A.plot(*piv_A_xy, marker='*', color='#c13c14', markersize=32,
                   markeredgecolor='white', mew=2.0, zorder=8)
        # cross-panel line A-pivot → B-hypothesis (same physical thing, both
        # are "where the pivot lands" — hyp coincides with A's center by crop)
        cp_pivot = ConnectionPatch(
            xyA=piv_A_xy, xyB=piv_B_hat,
            coordsA='data', coordsB='data',
            axesA=ax_A, axesB=ax_B,
            color='#c13c14', lw=1.5, ls='--', alpha=0.85, zorder=4)
        fig.add_artist(cp_pivot)
        # on B: hyp (red ★, center), pred (green ★), gt (blue ★)
        ax_B.plot(*piv_B_hat, marker='*', color='#c13c14', markersize=32,
                   markeredgecolor='white', mew=2.0, zorder=8)
        ax_B.plot(*piv_B_pred, marker='*', color='#0fa550', markersize=28,
                   markeredgecolor='white', mew=2.0, zorder=9)
        ax_B.plot(*piv_B_gt, marker='*', color='#1e6fff', markersize=28,
                   markeredgecolor='white', mew=2.0, zorder=10)
        # arrow on B: hyp ★ → pred ★ (pivot's predicted correction)
        ax_B.annotate('', xy=piv_B_pred, xytext=piv_B_hat,
                       arrowprops=dict(arrowstyle='-|>', color='#0fa550',
                                        lw=2.8, alpha=0.95), zorder=7)
        # short line pred → gt (how much remaining error for pivot)
        ax_B.plot([piv_B_pred[0], piv_B_gt[0]], [piv_B_pred[1], piv_B_gt[1]],
                   color='#1e6fff', lw=1.2, ls=':', alpha=0.8, zorder=6)

        # draw points + cross-panel lines (A ● → B ★)
        for k, i in enumerate(sel):
            c = colors[k]
            # A: big filled circle with number
            ax_A.plot(uv_A_l[i, 0], uv_A_l[i, 1], 'o', color=c, markersize=20,
                       markeredgecolor='white', mew=2.0, zorder=5)
            ax_A.annotate(str(k), (uv_A_l[i, 0], uv_A_l[i, 1]),
                           ha='center', va='center', fontsize=11, color='black',
                           weight='bold', zorder=6)

            # B: hyp × (large, same color)
            ax_B.plot(uv_hat_l[i, 0], uv_hat_l[i, 1], 'x', color=c, markersize=18,
                       mew=3.0, zorder=5)
            # B: arrow hyp → pred
            ax_B.annotate('', xy=uv_pred_l[i], xytext=uv_hat_l[i],
                           arrowprops=dict(arrowstyle='-|>', color=c, lw=2.0, alpha=0.9),
                           zorder=4)
            # B: pred ●
            ax_B.plot(uv_pred_l[i, 0], uv_pred_l[i, 1], 'o', color=c, markersize=14,
                       markeredgecolor='black', mew=0.8, zorder=6)
            # B: GT ★
            ax_B.plot(uv_gt_l[i, 0], uv_gt_l[i, 1], '*', color=c, markersize=22,
                       markeredgecolor='white', mew=1.5, zorder=7)
            ax_B.annotate(str(k), (uv_hat_l[i, 0] + 2, uv_hat_l[i, 1] - 2),
                           fontsize=10, color=c, weight='bold', zorder=8)

            # cross-panel dotted line: A ● → B ★ (same 3D point under GT)
            cp = ConnectionPatch(
                xyA=(uv_A_l[i, 0], uv_A_l[i, 1]),    xyB=(uv_gt_l[i, 0], uv_gt_l[i, 1]),
                coordsA='data', coordsB='data',
                axesA=ax_A, axesB=ax_B,
                color=c, lw=0.8, ls=':', alpha=0.7, zorder=3)
            fig.add_artist(cp)

    plt.suptitle(f'baseline ±{bl} frames   perturb σ_ypr={args.sigma_ypr}° / σ_t={args.sigma_t}m\n'
                  f'[A: ● points] — dotted line → [B: × hyp, ● model pred, ★ GT]   '
                  f'same color = same 3D point',
                  fontsize=13, y=0.998)
    plt.tight_layout()
    out = out_dir / f'vis_lln_bl{bl}.png'
    plt.savefig(out, dpi=110, bbox_inches='tight', facecolor='#f6f4ed')
    print(f'saved → {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default='experiments/cross_frame_v10_padclean_deform')
    ap.add_argument('--scenes-root', default='/mnt/mininas/datasets/pandaset')
    ap.add_argument('--train-frac', type=float, default=0.80)
    ap.add_argument('--img-size', type=int, default=64)
    ap.add_argument('--max-points', type=int, default=256)
    ap.add_argument('--baseline', type=int, default=10)
    ap.add_argument('--n-pairs', type=int, default=4)
    ap.add_argument('--n-show', type=int, default=4)
    ap.add_argument('--k-show', type=int, default=5, help='points highlighted per pair')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--sigma-ypr', type=float, default=1.0)
    ap.add_argument('--sigma-t',   type=float, default=0.20)
    ap.add_argument('--crop-min', type=int, default=128)
    ap.add_argument('--crop-max', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    main(args)
