"""Visual check: does BA-recovered pose actually put the same 3D point on the
same image feature? Subpixel-scale markers — no stars, no 20-px ellipses.

For each pair, panel B shows three projections of the same world point:
  - faint × = hyp (T_AB_hat)
  - small red + = BA-recovered (T_AB_rec)
  - small blue ○ = GT (T_AB_gt)  [only for reference; noisy on PandaSet]

If red + lands on the same image feature as the blue ○, BA is actually
resolving geometry regardless of what the rot_improvement% number says.

Two variants side-by-side per pair:
  Left:  perturbed  (σ_ypr=1.0°, σ_t=0.20m)  → usual eval setup
  Right: no-perturb (σ=0)  → hyp already equals GT. Model should predict
         Δ≈0; BA should not move the pose. Any systematic drift here is
         the model's learned dataset-bias, NOT a real residual.

Run: python scripts/visualization/vis_ba_reproject.py [--ckpt <dir>]
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

from datasets.pandaset_pair import _SceneData, _project, _ypr_t_to_mat
from models.cross_frame import CalibNetCrossFrame
from scripts.eval.cross_frame_lln import make_pair, infer_batch, ba_recover

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(ckpt_dir):
    sd = torch.load(ckpt_dir / 'best_model.pt', map_location=DEVICE, weights_only=True)
    deform = 'sl' if any('deform_img' in k for k in sd) else 'none'
    n_cross = sum(1 for k in sd if k.startswith('cross_blocks.') and k.endswith('.proj.weight'))
    n_intra = max(1, sum(1 for k in sd if k.startswith('intra_blocks.') and k.endswith('.norm_sa.weight')))
    proj_w = [sd[k] for k in sd if k.startswith('cross_blocks.') and k.endswith('.proj.weight')]
    out_dim = proj_w[0].shape[0] if proj_w else 5
    m = CalibNetCrossFrame(img_size=64, deform_mode=deform,
                            n_cross_layers=n_cross, n_intra_layers=n_intra,
                            out_dim=out_dim).to(DEVICE)
    m.load_state_dict(sd); m.eval()
    return m, out_dim


def one_panel(ax, image_B_full, pts_3D_A, scn, T_AB_hat, T_AB_rec, T_AB_gt,
              box_B, img_size, K, title, k_show=12):
    """Draw 3 projections of N 3D points into image B on a cropped axis."""
    u0, v0, cw, ch = box_B
    # show full-res patch_B at its native resolution so we can see subpixel
    # (cropping to box_B from the full frame B image)
    IH, IW = image_B_full.shape[:2]
    u0c, v0c = max(int(u0), 0), max(int(v0), 0)
    u1c, v1c = min(int(u0 + cw), IW), min(int(v0 + ch), IH)
    ax.imshow(image_B_full[v0c:v1c, u0c:u1c])
    ax.set_xticks([]); ax.set_yticks([])

    # select k_show spatially-spread points (by 2D position in B under hyp)
    homo = np.concatenate([pts_3D_A, np.ones((len(pts_3D_A), 1))], axis=1)
    uv_hat_full, z_hat = _project(pts_3D_A, np.linalg.inv(T_AB_hat_inv_like_w2c(T_AB_hat, scn)), K)
    # Actually we want: point in A-cam → B-cam via T_AB → then project via K.
    # But we have pts_3D_A already in A-cam, and T_AB_hat (A-cam → B-cam).
    P_Bh = (homo @ T_AB_hat.T)[:, :3]
    P_Brec = (homo @ T_AB_rec.T)[:, :3]
    P_Bgt = (homo @ T_AB_gt.T)[:, :3]
    uv_hat = (K @ P_Bh.T)[:2] / np.clip(P_Bh[:, 2], 1e-6, None);  uv_hat = uv_hat.T
    uv_rec = (K @ P_Brec.T)[:2] / np.clip(P_Brec[:, 2], 1e-6, None); uv_rec = uv_rec.T
    uv_gt  = (K @ P_Bgt.T)[:2] / np.clip(P_Bgt[:, 2], 1e-6, None);  uv_gt  = uv_gt.T

    # keep only points that land inside the cropped patch for all three
    def in_patch(uv):
        return ((uv[:, 0] >= u0c) & (uv[:, 0] < u1c) &
                (uv[:, 1] >= v0c) & (uv[:, 1] < v1c))
    m = in_patch(uv_hat) & in_patch(uv_rec) & in_patch(uv_gt)
    idx = np.where(m)[0]
    if len(idx) == 0:
        ax.set_title(f'{title}\n(no co-visible pts)', fontsize=9, loc='left'); return
    # subsample evenly across the patch
    if len(idx) > k_show:
        step = max(1, len(idx) // k_show)
        idx = idx[::step][:k_show]

    cmap = cm.get_cmap('tab10')
    for k, i in enumerate(idx):
        c = cmap(k % 10)
        # convert full-image coord → patch-local coord by subtracting crop origin
        h = (uv_hat[i, 0] - u0c, uv_hat[i, 1] - v0c)
        r = (uv_rec[i, 0] - u0c, uv_rec[i, 1] - v0c)
        g = (uv_gt[i,  0] - u0c, uv_gt[i,  1] - v0c)
        # fine line segment hyp→rec to show direction of BA correction
        ax.plot([h[0], r[0]], [h[1], r[1]], color=c, lw=0.8, alpha=0.5, zorder=3)
        # fine line segment rec→gt to show residual error
        ax.plot([r[0], g[0]], [r[1], g[1]], color=c, lw=0.6, alpha=0.35,
                 linestyle=':', zorder=3)
        ax.plot(*h, marker='x', color=c, markersize=4, mew=0.9,
                 alpha=0.55, zorder=5)
        ax.plot(*g, marker='o', color=c, markersize=3, mew=0.9,
                 markerfacecolor='none', markeredgecolor=c, alpha=0.85, zorder=6)
        ax.plot(*r, marker='+', color=c, markersize=5, mew=1.1,
                 zorder=7)

    # summary err
    err_hyp = np.linalg.norm(uv_hat[idx] - uv_gt[idx], axis=-1).mean()
    err_rec = np.linalg.norm(uv_rec[idx] - uv_gt[idx], axis=-1).mean()
    ax.set_title(f'{title}  hyp {err_hyp:.2f} → rec {err_rec:.2f} px',
                 fontsize=9, loc='left')


def T_AB_hat_inv_like_w2c(T, scn):
    # dummy helper; not used, left for docstring
    return T


def run_pair(model, scn, fi_A, fi_B, rng, out_dim, sigma_ypr, sigma_t, img_size=64):
    s = make_pair(scn, rng, fi_A, fi_B, img_size, 256, sigma_ypr, sigma_t, (128, 256))
    if s is None: return None
    raw = infer_batch(model, [s])[0]
    N = s['N_valid']
    if N < 8: return None

    mu = raw[:N, :2]
    if out_dim == 7:
        log_sx, log_sy = raw[:N, 3], raw[:N, 4]; rho = raw[:N, 6]
        delta_d = raw[:N, 2]; sigma_d = np.exp(raw[:N, 5])
    else:
        log_sx, log_sy = raw[:N, 2], raw[:N, 3]; rho = np.tanh(raw[:N, 4]) * 0.99
        delta_d = sigma_d = None
    sx, sy = np.exp(log_sx), np.exp(log_sy)
    Sigma = np.zeros((N, 2, 2), np.float32)
    Sigma[:, 0, 0] = sx * sx; Sigma[:, 1, 1] = sy * sy
    Sigma[:, 0, 1] = Sigma[:, 1, 0] = rho * sx * sy

    res = ba_recover(s, mu, Sigma,
                     delta_d_pred=delta_d, sigma_d_pred=sigma_d)
    if res is None: return None
    θ, _ = res
    δT = _ypr_t_to_mat(θ[:3], θ[3:6])
    T_rec = δT @ s['T_AB_hat']

    return dict(sample=s, T_rec=T_rec)


def main(args):
    ckpt = Path(args.ckpt)
    model, out_dim = load_model(ckpt)
    print(f'loaded {ckpt.name}  out_dim={out_dim}')

    # use PandaSet val scenes, same split as training (seed=42)
    import random as _r
    root = Path('/mnt/mininas/datasets/pandaset')
    names = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit()])
    shuffled = sorted([str(root / n) for n in names])
    _r.Random(42).shuffle(shuffled)
    cutoff = int(len(shuffled) * 0.8)
    val_roots = shuffled[cutoff:]
    scenes = []
    for sr in val_roots[:6]:
        scn = _SceneData(Path(sr)); scn.precompute_all(preload_images=True); scenes.append(scn)

    rng = np.random.default_rng(args.seed)
    n_rows = args.n_pairs

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows), dpi=130)
    fig.patch.set_facecolor('#f6f4ed')
    if n_rows == 1: axes = axes[None, :]

    row = 0; tries = 0
    while row < n_rows and tries < n_rows * 5:
        tries += 1
        scn = scenes[int(rng.integers(len(scenes)))]
        bl = int(rng.integers(args.baseline_min, args.baseline_max + 1)) * int(rng.choice([-1, 1]))
        fi_A = int(rng.integers(5, scn.n_frames - abs(bl) - 5))
        fi_B = fi_A + bl

        # LEFT: perturbed
        rng_p = np.random.default_rng(rng.integers(2**31))
        left = run_pair(model, scn, fi_A, fi_B, rng_p, out_dim,
                         sigma_ypr=1.0, sigma_t=0.20)
        # RIGHT: no perturbation
        rng_np = np.random.default_rng(rng.integers(2**31))
        right = run_pair(model, scn, fi_A, fi_B, rng_np, out_dim,
                          sigma_ypr=0.0, sigma_t=0.0)
        if left is None or right is None: continue

        s_l, s_r = left['sample'], right['sample']
        image_B = scn.load_image(fi_B)
        K = s_l['K']
        pts_A_l = s_l['P_A_cam'][:s_l['N_valid']]
        pts_A_r = s_r['P_A_cam'][:s_r['N_valid']]

        one_panel(axes[row, 0], image_B, pts_A_l, scn,
                   s_l['T_AB_hat'], left['T_rec'], s_l['T_AB_gt'],
                   s_l['box_B'], 64, K,
                   title=f'scene {scn.root.name}  fi {fi_A}→{fi_B}  perturbed (σ 1°/0.2m)')
        one_panel(axes[row, 1], image_B, pts_A_r, scn,
                   s_r['T_AB_hat'], right['T_rec'], s_r['T_AB_gt'],
                   s_r['box_B'], 64, K,
                   title=f'scene {scn.root.name}  fi {fi_A}→{fi_B}  no-perturb (σ=0)')
        row += 1

    plt.suptitle(
        f'{ckpt.name} — BA reprojection check   '
        f'× hyp   ● = BA-recovered (+)   ○ = GT\n'
        f'LEFT: perturbed.  RIGHT: hyp=GT identity check (any offset = learned dataset bias).',
        fontsize=12, y=0.998)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor='#f6f4ed')
    print(f'saved → {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='experiments/cross_frame_v15_uvd_mix')
    ap.add_argument('--n-pairs', type=int, default=4)
    ap.add_argument('--baseline-min', type=int, default=5)
    ap.add_argument('--baseline-max', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='experiments/ba_reproject_v15.png')
    args = ap.parse_args()
    main(args)
