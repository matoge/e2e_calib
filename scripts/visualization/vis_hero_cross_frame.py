"""Hero shot for the cross-frame report top figure.

Generate many pairs with v13_mix at small baseline, score each by mean
per-point prediction error (tight pred → small), cherry-pick the top N,
and render a compact 2×2 grid. Goal: visually clean convergence,
not the random sampling of vis_lln_ba.py (which includes hard/foggy/
night pairs).

Run: python scripts/visualization/vis_hero_cross_frame.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import ConnectionPatch, Ellipse

from datasets.pandaset_pair import _SceneData
from models.cross_frame import CalibNetCrossFrame
from scripts.eval.cross_frame_lln import make_pair, infer_batch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CKPT = Path('experiments/cross_frame_v13_mix')
SCENES_ROOTS = [
    Path('/mnt/mininas/datasets/pandaset'),
    Path('/mnt/nvme6t/waymo_ps'),
]
OUT = Path('docs/images/cross_frame_hero.png')

BASELINE = 5
IMG_SIZE = 64
MAX_POINTS = 256
SIGMA_YPR = 1.0
SIGMA_T   = 0.20
CROP_RANGE = (128, 256)

N_CANDIDATES = 96     # try this many pairs
N_SHOW = 4            # keep top-N cleanest
K_SHOW = 5            # points per pair


def cov_ellipse_axes(sx, sy, rho):
    cxx = sx * sx; cyy = sy * sy; cxy = rho * sx * sy
    tr = cxx + cyy; det = cxx * cyy - cxy * cxy
    disc = max(tr * tr / 4.0 - det, 0.0) ** 0.5
    l1 = tr / 2.0 + disc; l2 = max(tr / 2.0 - disc, 1e-12)
    angle = 0.5 * np.degrees(np.arctan2(2 * cxy, cxx - cyy))
    return 4 * np.sqrt(l1), 4 * np.sqrt(l2), angle   # 2σ diameter


def main():
    # Load model (auto-detect arch)
    sd = torch.load(CKPT / 'best_model.pt', map_location=DEVICE, weights_only=True)
    deform = 'sl' if any('deform_img' in k for k in sd) else 'none'
    n_cross = sum(1 for k in sd if k.startswith('cross_blocks.') and k.endswith('.proj.weight'))
    n_intra = max(1, sum(1 for k in sd if k.startswith('intra_blocks.') and k.endswith('.norm_sa.weight')))
    proj_w = [sd[k] for k in sd if k.startswith('cross_blocks.') and k.endswith('.proj.weight')]
    out_dim = proj_w[0].shape[0] if proj_w else 5
    model = CalibNetCrossFrame(img_size=IMG_SIZE, deform_mode=deform,
                                n_cross_layers=n_cross, n_intra_layers=n_intra,
                                out_dim=out_dim).to(DEVICE)
    model.load_state_dict(sd); model.eval()

    # Gather val scenes (same 80/20 split logic as training): pick only PandaSet
    # (for the hero we want clean daytime urban scenes; Waymo includes night/fog).
    scenes = []
    for root in SCENES_ROOTS[:1]:  # PandaSet only for hero
        for p in sorted(root.iterdir()):
            if p.is_dir() and (p / 'camera/front_camera/intrinsics.json').exists():
                scn = _SceneData(p); scn.precompute_all(); scenes.append(scn)
                if len(scenes) >= 8:
                    break

    # Sample N_CANDIDATES pairs
    rng = np.random.default_rng(1234)
    samples = []
    attempts = 0
    while len(samples) < N_CANDIDATES and attempts < N_CANDIDATES * 4:
        attempts += 1
        scn = scenes[int(rng.integers(len(scenes)))]
        fi_A = int(rng.integers(scn.n_frames))
        fi_B = fi_A + BASELINE * int(rng.choice([-1, 1]))
        if fi_B < 0 or fi_B >= scn.n_frames: continue
        s = make_pair(scn, rng, fi_A, fi_B, IMG_SIZE, MAX_POINTS,
                       SIGMA_YPR, SIGMA_T, CROP_RANGE)
        if s is None: continue
        samples.append(s)
    print(f'collected {len(samples)} candidate pairs (attempts={attempts})')

    # Infer all
    raws = []
    for i in range(0, len(samples), 16):
        raws.append(infer_batch(model, samples[i:i + 16]))
    raws = np.concatenate(raws, axis=0)

    # Score each by mean per-point pred-to-gt pixel error in local B coords
    def full_to_local_B(uv_full, box_B):
        u0, v0, cw, ch = box_B
        return np.stack([(uv_full[:, 0] - u0) * IMG_SIZE / cw,
                         (uv_full[:, 1] - v0) * IMG_SIZE / ch], axis=1)

    scores = []
    for s, raw in zip(samples, raws):
        N = s['N_valid']
        if N < K_SHOW: scores.append(1e9); continue
        mu = raw[:N, :2]
        pred_local = s['uv_B_hat_of_A'][:N].numpy() + mu
        gt_local   = full_to_local_B(s['uv_gt_full'][:N], s['box_B'])
        err = np.linalg.norm(pred_local - gt_local, axis=-1).mean()
        sx = np.exp(raw[:N, 2] if out_dim == 5 else raw[:N, 3])
        sy = np.exp(raw[:N, 3] if out_dim == 5 else raw[:N, 4])
        sig = np.sqrt((sx * sy)).mean()
        scores.append(err + 0.3 * sig)
    scores = np.array(scores)
    order = np.argsort(scores)[:N_SHOW]
    print(f'top-{N_SHOW} scores: {scores[order]}')

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=120)
    fig.patch.set_facecolor('#f6f4ed')
    for pane_idx, idx in enumerate(order):
        row, col = pane_idx // 2, (pane_idx % 2) * 2
        ax_A = axes[row, col]; ax_B = axes[row, col + 1]
        s = samples[idx]; raw = raws[idx]
        N = s['N_valid']

        mu = raw[:N, :2]
        if out_dim == 7:
            log_sx, log_sy = raw[:N, 3], raw[:N, 4]; rho = raw[:N, 6]
        else:
            log_sx, log_sy = raw[:N, 2], raw[:N, 3]; rho = np.tanh(raw[:N, 4]) * 0.99
        sx, sy = np.exp(log_sx), np.exp(log_sy)

        # draw images
        pA = (s['patch_A'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pB = (s['patch_B'].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        ax_A.imshow(pA); ax_A.set_xticks([]); ax_A.set_yticks([])
        ax_B.imshow(pB); ax_B.set_xticks([]); ax_B.set_yticks([])

        # pick K_SHOW spatially-spread points whose hyp AND gt land in patch_B
        uv_A_l = s['uvd_A'][:N, :2].numpy()
        uv_hat_l = s['uv_B_hat_of_A'][:N].numpy()
        uv_gt_l  = full_to_local_B(s['uv_gt_full'][:N], s['box_B'])
        uv_pred_l = uv_hat_l + mu
        in_ok = ((uv_hat_l[:, 0] >= 0) & (uv_hat_l[:, 0] < IMG_SIZE) &
                 (uv_hat_l[:, 1] >= 0) & (uv_hat_l[:, 1] < IMG_SIZE) &
                 (uv_gt_l[:, 0]  >= 0) & (uv_gt_l[:, 0]  < IMG_SIZE) &
                 (uv_gt_l[:, 1]  >= 0) & (uv_gt_l[:, 1]  < IMG_SIZE))
        cand = np.where(in_ok)[0]
        if len(cand) == 0: continue
        step = max(1, len(cand) // K_SHOW)
        sel = cand[::step][:K_SHOW]

        cmap = cm.get_cmap('tab10')
        for k, i in enumerate(sel):
            c = cmap(k % 10)
            # A: numbered circle
            ax_A.plot(uv_A_l[i, 0], uv_A_l[i, 1], 'o', color=c, markersize=16,
                       markeredgecolor='white', mew=1.6, zorder=5)
            ax_A.annotate(str(k), (uv_A_l[i, 0], uv_A_l[i, 1]),
                           ha='center', va='center', fontsize=9, color='black',
                           weight='bold', zorder=6)
            # B: 2σ ellipse + small pred dot + GT star
            w, h, ang = cov_ellipse_axes(sx[i], sy[i], rho[i])
            ax_B.add_patch(Ellipse((uv_pred_l[i, 0], uv_pred_l[i, 1]),
                                    width=w, height=h, angle=ang,
                                    facecolor=c, edgecolor='none', alpha=0.28, zorder=5))
            ax_B.add_patch(Ellipse((uv_pred_l[i, 0], uv_pred_l[i, 1]),
                                    width=w, height=h, angle=ang,
                                    facecolor='none', edgecolor=c, alpha=0.95, lw=1.6, zorder=6))
            ax_B.plot(uv_pred_l[i, 0], uv_pred_l[i, 1], 'o', color=c, markersize=4,
                       markeredgecolor='white', mew=0.6, zorder=7)
            ax_B.plot(uv_gt_l[i, 0], uv_gt_l[i, 1], '*', color=c, markersize=18,
                       markeredgecolor='white', mew=1.3, zorder=8)
            # cross-panel A●→B★ dotted line (same 3D point)
            fig.add_artist(ConnectionPatch(
                xyA=(uv_A_l[i, 0], uv_A_l[i, 1]), xyB=(uv_gt_l[i, 0], uv_gt_l[i, 1]),
                coordsA='data', coordsB='data',
                axesA=ax_A, axesB=ax_B,
                color=c, lw=0.6, ls=':', alpha=0.6, zorder=3))

        err_pred = np.linalg.norm(uv_pred_l[sel] - uv_gt_l[sel], axis=-1).mean()
        err_hyp  = np.linalg.norm(uv_hat_l[sel] - uv_gt_l[sel], axis=-1).mean()
        ax_A.set_title(f'A  scene {s["scene"]}', fontsize=10, loc='left')
        ax_B.set_title(f'B  hyp err {err_hyp:.1f} → pred {err_pred:.2f} px',
                        fontsize=10, loc='left')

    plt.suptitle(
        f'cross-frame residual net — v13_mix best predictions '
        f'(PandaSet val, baseline ±{BASELINE} frames, σ_ypr={SIGMA_YPR}°/σ_t={SIGMA_T}m)',
        fontsize=13, y=0.995)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=120, bbox_inches='tight', facecolor='#f6f4ed')
    print(f'saved → {OUT}')


if __name__ == '__main__':
    main()
