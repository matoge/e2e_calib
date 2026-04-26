"""BA reprojection check — patch_A | patch_B pair view.

Same layout as vis_lln_ba.py (which the report uses):
  - patch_A on the left: query points (numbered dots) at their A-local uv
  - patch_B on the right: for each point, three projections
      × = hyp (T_AB_hat)
      ○ = GT  (T_AB_gt)
      + = BA-recovered (T_AB_rec)
  - thin cross-panel dotted lines A●→B_gt for the same 3D point
  - small in-panel segments:  ×──+ (BA correction), +·· ○ (residual to GT)

Markers are deliberately small (4–6 px) so subpixel differences are
visible when zoomed in. No hero stars.

Two side-by-side variants per row:
  LEFT  σ_ypr=1°, σ_t=0.2m  (normal eval)
  RIGHT σ=0                  (hyp == GT identity check; any + drift is
                              the learned dataset bias)
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
from matplotlib.patches import ConnectionPatch

from datasets.pandaset_pair import _SceneData, _ypr_t_to_mat
from models.cross_frame_multi import CalibNetMultiFrame
from scripts.eval.cross_frame_lln import make_pair, infer_batch, ba_recover

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 64


def load_model(ckpt_dir, ckpt_name='best_model.pt'):
    sd = torch.load(ckpt_dir / ckpt_name, map_location=DEVICE, weights_only=True)
    deform = 'sl' if any('deform_img' in k for k in sd) else 'none'
    n_cross = sum(1 for k in sd if k.startswith('cross_blocks.') and k.endswith('.proj.weight'))
    n_intra = max(1, sum(1 for k in sd if k.startswith('intra_blocks.') and k.endswith('.norm_sa.weight')))
    proj_w = [sd[k] for k in sd if k.startswith('cross_blocks.') and k.endswith('.proj.weight')]
    out_dim = proj_w[0].shape[0] if proj_w else 5
    m = CalibNetMultiFrame(img_size=IMG_SIZE, deform_mode=deform,
                            n_cross_layers=n_cross, n_intra_layers=n_intra,
                            out_dim=out_dim).to(DEVICE)
    m.load_state_dict(sd); m.eval()
    return m, out_dim


def full_to_local_B(uv_full, box_B, img_size=IMG_SIZE):
    u0, v0, cw, ch = box_B
    return np.stack([(uv_full[:, 0] - u0) * img_size / cw,
                     (uv_full[:, 1] - v0) * img_size / ch], axis=1)


def run_pair(model, scn, fi_A, fi_B, rng, out_dim, sigma_ypr, sigma_t):
    s = make_pair(scn, rng, fi_A, fi_B, IMG_SIZE, 256, sigma_ypr, sigma_t, (128, 256))
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

    res = ba_recover(s, mu, Sigma, delta_d_pred=delta_d, sigma_d_pred=sigma_d)
    if res is None: return None
    theta, _ = res
    dT = _ypr_t_to_mat(theta[:3], theta[3:6])
    T_rec = dT @ s['T_AB_hat']

    # Compute patch-local uv for hyp / gt / rec, using the same box_B-to-local scale
    pts_A_cam = s['P_A_cam'][:N]
    homo = np.concatenate([pts_A_cam, np.ones((N, 1))], axis=1)
    K = s['K']

    def proj_full(T):
        Q = (homo @ T.T)[:, :3]
        uv = (K @ Q.T)[:2] / np.clip(Q[:, 2], 1e-6, None)
        return uv.T, Q[:, 2]

    uv_hat_full, _ = proj_full(s['T_AB_hat'])
    uv_rec_full, _ = proj_full(T_rec)
    uv_gt_full,  _ = proj_full(s['T_AB_gt'])

    box_B = s['box_B']
    uv_hat_l = full_to_local_B(uv_hat_full, box_B)
    uv_rec_l = full_to_local_B(uv_rec_full, box_B)
    uv_gt_l  = full_to_local_B(uv_gt_full,  box_B)

    uv_A_l = s['uvd_A'][:N, :2].numpy()

    # keep points whose hyp & gt & rec are inside the 64×64 local patch
    def in_patch(uv):
        return ((uv[:, 0] >= 0) & (uv[:, 0] < IMG_SIZE) &
                (uv[:, 1] >= 0) & (uv[:, 1] < IMG_SIZE))
    m_in = in_patch(uv_hat_l) & in_patch(uv_rec_l) & in_patch(uv_gt_l) & in_patch(uv_A_l)

    return dict(sample=s, uv_A_l=uv_A_l, uv_hat_l=uv_hat_l, uv_rec_l=uv_rec_l,
                uv_gt_l=uv_gt_l, in_mask=m_in, N=N)


def draw_pair_panel(fig, ax_A, ax_B, r, tag, k_show=8):
    s = r['sample']
    pA = (s['patch_A'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pB = (s['patch_B'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    ax_A.imshow(pA); ax_A.set_xticks([]); ax_A.set_yticks([])
    ax_B.imshow(pB); ax_B.set_xticks([]); ax_B.set_yticks([])

    idx = np.where(r['in_mask'])[0]
    if len(idx) == 0:
        ax_A.set_title(f'{tag} (no co-visible pts)', fontsize=10, loc='left'); return
    if len(idx) > k_show:
        step = max(1, len(idx) // k_show)
        idx = idx[::step][:k_show]

    cmap = cm.get_cmap('tab10')
    for k, i in enumerate(idx):
        c = cmap(k % 10)
        xa, ya = r['uv_A_l'][i]
        xh, yh = r['uv_hat_l'][i]
        xr, yr = r['uv_rec_l'][i]
        xg, yg = r['uv_gt_l'][i]

        # A: numbered filled dot
        ax_A.plot(xa, ya, 'o', color=c, markersize=6, markeredgecolor='white',
                   mew=0.8, zorder=5)
        ax_A.annotate(str(k), (xa, ya), ha='center', va='center',
                       fontsize=7, color='black', weight='bold', zorder=6)

        # cross-panel dotted line A●→B○ (GT correspondence)
        fig.add_artist(ConnectionPatch(
            xyA=(xa, ya), xyB=(xg, yg),
            coordsA='data', coordsB='data',
            axesA=ax_A, axesB=ax_B,
            color=c, lw=0.5, ls=':', alpha=0.45, zorder=3))

        # B: in-panel thin lines showing BA correction direction (× → +) and
        # residual to GT (+ → ○)
        ax_B.plot([xh, xr], [yh, yr], color=c, lw=0.8, alpha=0.6, zorder=4)
        ax_B.plot([xr, xg], [yr, yg], color=c, lw=0.6, alpha=0.45,
                   linestyle=':', zorder=4)
        # markers (small)
        ax_B.plot(xh, yh, 'x', color=c, markersize=5, mew=1.0, alpha=0.65, zorder=5)
        ax_B.plot(xg, yg, 'o', color=c, markersize=4, markeredgecolor=c,
                   markerfacecolor='none', mew=0.9, alpha=0.9, zorder=6)
        ax_B.plot(xr, yr, '+', color=c, markersize=7, mew=1.3, zorder=7)

    # summary errors
    err_hyp = np.linalg.norm(r['uv_hat_l'][idx] - r['uv_gt_l'][idx], axis=-1).mean()
    err_rec = np.linalg.norm(r['uv_rec_l'][idx] - r['uv_gt_l'][idx], axis=-1).mean()
    ax_A.set_title(f'A  {tag}', fontsize=9, loc='left')
    ax_B.set_title(f'B  hyp {err_hyp:.2f} → rec {err_rec:.2f} px   '
                   f'(× hyp, ○ GT, + BA-rec)', fontsize=9, loc='left')


def main(args):
    ckpt = Path(args.ckpt)
    model, out_dim = load_model(ckpt, ckpt_name=args.ckpt_name)
    print(f'loaded {ckpt.name}/{args.ckpt_name}  out_dim={out_dim}')

    import random as _r
    # build (scene, camera) pair pool from --scenes-root; pick split.
    scene_roots = []
    for root_str in str(args.scenes_root).split(','):
        root = Path(root_str.strip())
        for p in sorted(root.iterdir()):
            if p.is_dir() and (p / 'camera').is_dir():
                scene_roots.append(str(p))
    _r.Random(42).shuffle(scene_roots)
    cutoff = int(len(scene_roots) * 0.8)
    if args.split == 'train':
        val_roots = scene_roots[:cutoff] if len(scene_roots) > 5 else scene_roots
    else:
        val_roots = scene_roots[cutoff:] if len(scene_roots) > 5 else scene_roots
    if args.cameras.lower() == 'all':
        def cams_of(sr):
            d = Path(sr) / 'camera'
            return sorted(c.name for c in d.iterdir()
                          if c.is_dir() and (c / 'intrinsics.json').exists())
    else:
        wanted = [c.strip() for c in args.cameras.split(',')]
        def cams_of(sr):
            return [c for c in wanted
                    if (Path(sr) / 'camera' / c / 'intrinsics.json').exists()]
    pairs = [(sr, cam) for sr in val_roots for cam in cams_of(sr)]
    _r.Random(42).shuffle(pairs)
    print(f'viz pool: {len(pairs)} (scene, cam) pairs from {len(val_roots)} val scenes')
    scenes = []
    for sr, cam in pairs[:min(24, len(pairs))]:
        scn = _SceneData(Path(sr), camera_name=cam)
        scn.precompute_all()
        scenes.append(scn)

    rng = np.random.default_rng(args.seed)
    n_rows = args.n_pairs

    # Grid: for each pair, 4 columns:  A_perturbed | B_perturbed | A_identity | B_identity
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4.2 * n_rows), dpi=140)
    fig.patch.set_facecolor('#f6f4ed')
    if n_rows == 1: axes = axes[None, :]

    row = 0; tries = 0
    while row < n_rows and tries < n_rows * 8:
        tries += 1
        scn = scenes[int(rng.integers(len(scenes)))]
        bl = int(rng.integers(args.baseline_min, args.baseline_max + 1)) * int(rng.choice([-1, 1]))
        fi_A = int(rng.integers(5, scn.n_frames - abs(bl) - 5))
        fi_B = fi_A + bl

        rng_p = np.random.default_rng(rng.integers(2**31))
        left = run_pair(model, scn, fi_A, fi_B, rng_p, out_dim,
                         sigma_ypr=1.0, sigma_t=0.20)
        rng_n = np.random.default_rng(rng.integers(2**31))
        right = run_pair(model, scn, fi_A, fi_B, rng_n, out_dim,
                          sigma_ypr=0.0, sigma_t=0.0)
        if left is None or right is None: continue
        if left['in_mask'].sum() < 4 or right['in_mask'].sum() < 4: continue

        draw_pair_panel(fig, axes[row, 0], axes[row, 1], left,
                        tag=f'{scn.scene_id} fi {fi_A}→{fi_B}  σ=1°/0.2m')
        draw_pair_panel(fig, axes[row, 2], axes[row, 3], right,
                        tag=f'(identity σ=0)')
        row += 1

    plt.suptitle(
        f'{ckpt.name} — BA reprojection (patch A | patch B)  '
        f'× hyp   ○ GT   + BA-rec\n'
        f'LEFT 2 cols = perturbed eval.  RIGHT 2 cols = hyp==GT identity '
        f'(any + offset from ○ is the learned dataset bias).',
        fontsize=11, y=0.998)
    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='#f6f4ed')
    print(f'saved → {out}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='experiments/cross_frame_v15_uvd_mix')
    ap.add_argument('--n-pairs', type=int, default=4)
    ap.add_argument('--baseline-min', type=int, default=5)
    ap.add_argument('--baseline-max', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default='experiments/ba_reproject_v15.png')
    ap.add_argument('--scenes-root', default='/mnt/nvme6t/pandaset',
                    help='comma-separated dataset roots to sample val scenes from')
    ap.add_argument('--cameras', default='all',
                    help='"all" or comma-separated camera names')
    ap.add_argument('--split', default='val', choices=['val', 'train'],
                    help='which scene split to sample from (default val).')
    ap.add_argument('--ckpt-name', default='best_model.pt',
                    help='checkpoint filename inside ckpt dir (best_model.pt or last_model.pt).')
    args = ap.parse_args()
    main(args)
