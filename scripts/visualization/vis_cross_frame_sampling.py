"""Sanity-check the cross-frame sampler itself (independent of the model).

For each (fi_A, fi_B) pair, generate one training-style sample and show
patch_A | patch_B side-by-side with:
  - red ★ on patch_A at the pivot's projection (the A-frame center point)
  - red ★ on patch_B at the pivot's HYPOTHESIS projection (patch_B center)
  - blue ★ on patch_B at the pivot's GT projection
  - arrows connecting a few matched LiDAR points (A → its GT location in B),
    colored by depth

If the sampler is working correctly:
  - both stars (red ★ in A, red ★ in B) land on visually-corresponding content
    (same car / building / ground patch)
  - blue ★ is slightly offset from red ★ in B (by the perturbation)
  - arrow endpoints in B land on structures that visually match the arrow
    starting points in A

Output: experiments/cross_frame_SAMPLING_check/sample_grid.png
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

from datasets.pandaset_pair import (
    _SceneData, _ypr_t_to_mat, _invert_mat, _project,
)


def draw_patches(ax, patch, title, img_size=64):
    arr = (patch.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    ax.imshow(arr)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, img_size - 0.5)
    ax.set_ylim(img_size - 0.5, -0.5)
    ax.set_aspect('equal', adjustable='box')
    if title:
        ax.set_title(title, fontsize=8, loc='left', pad=2)


def sample_and_show(scenes, rng, baseline, img_size=64, max_points=256,
                    sigma_ypr=1.0, sigma_t=0.2, crop_range=(64, 192)):
    """Sample one pair and return all info needed for a 2-column plot."""
    for tries in range(50):
        scn = scenes[int(rng.integers(len(scenes)))]
        fi_A = int(rng.integers(scn.n_frames - baseline))
        fi_B = fi_A + baseline
        IW, IH, K = scn.IW, scn.IH, scn.K
        T_w2A = scn.T_w2c[fi_A]; T_w2B = scn.T_w2c[fi_B]
        T_A2w = _invert_mat(T_w2A)
        T_AB_gt = T_w2B @ T_A2w

        pts_w_A, uv_Af, z_Af, in_A = scn.frame_data(fi_A)
        pts_w_B, uv_Bf, z_Bf, in_B = scn.frame_data(fi_B)
        if in_A.sum() < 50:
            continue
        pts_vis = pts_w_A[in_A]
        uv_A_all = uv_Af[in_A]
        z_A_all  = z_Af[in_A]

        ci = int(rng.integers(len(pts_vis)))
        P_c_w = pts_vis[ci]
        uc_A, vc_A = uv_A_all[ci]

        ypr_p = rng.standard_normal(3).astype(np.float32) * sigma_ypr
        t_p   = rng.standard_normal(3).astype(np.float32) * sigma_t
        δT    = _ypr_t_to_mat(ypr_p, t_p)
        T_hat = T_AB_gt @ δT
        P_A   = (T_w2A @ np.append(P_c_w, 1.0))[:3]
        P_Bh  = (T_hat @ np.append(P_A, 1.0))[:3]
        P_Bg  = (T_AB_gt @ np.append(P_A, 1.0))[:3]
        if P_Bh[2] < 1 or P_Bg[2] < 1:
            continue
        uc_B_hat = (K @ P_Bh)[:2] / P_Bh[2]
        uc_B_gt  = (K @ P_Bg)[:2] / P_Bg[2]
        # pivot must land inside actual image under BOTH hat and gt
        if not (0 <= uc_B_hat[0] < IW and 0 <= uc_B_hat[1] < IH): continue
        if not (0 <= uc_B_gt[0]  < IW and 0 <= uc_B_gt[1]  < IH): continue

        CROP = int(rng.integers(crop_range[0], crop_range[1] + 1))
        half = CROP / 2
        # pivot must be INSIDE the image in both A and B (padding is OK when
        # the pivot is near the edge; pivot outside-image → reject).
        if not (0 <= uc_A < IW and 0 <= vc_A < IH):
            continue
        if not (0 <= uc_B_hat[0] < IW and 0 <= uc_B_hat[1] < IH):
            continue
        if not (0 <= uc_B_gt[0] < IW and 0 <= uc_B_gt[1] < IH):
            continue
        u0_A = uc_A - half
        v0_A = vc_A - half
        u0_B = uc_B_hat[0] - half
        v0_B = uc_B_hat[1] - half

        img_A = scn.load_image(fi_A)
        img_B = scn.load_image(fi_B)

        def _pad_crop(img, u0, v0):
            u0i = int(np.floor(u0)); v0i = int(np.floor(v0))
            u1i = u0i + CROP;         v1i = v0i + CROP
            pad_left  = max(0, -u0i)
            pad_top   = max(0, -v0i)
            src_u0 = max(0, u0i); src_v0 = max(0, v0i)
            src_u1 = min(IW, u1i); src_v1 = min(IH, v1i)
            out = np.zeros((CROP, CROP, 3), dtype=img.dtype)
            cw = src_u1 - src_u0; ch = src_v1 - src_v0
            if cw > 0 and ch > 0:
                out[pad_top:pad_top + ch, pad_left:pad_left + cw] = img[src_v0:src_v1, src_u0:src_u1]
            return out

        pA = _pad_crop(img_A, u0_A, v0_A)
        pB = _pad_crop(img_B, u0_B, v0_B)
        patchA = torch.from_numpy(pA).permute(2, 0, 1).float() / 255.0
        patchB = torch.from_numpy(pB).permute(2, 0, 1).float() / 255.0
        u0Ai = int(np.floor(u0_A)); v0Ai = int(np.floor(v0_A))
        u0Bi = int(np.floor(u0_B)); v0Bi = int(np.floor(v0_B))
        from torch.nn import functional as F
        patchA = F.interpolate(patchA.unsqueeze(0), size=(img_size, img_size),
                                mode='bilinear', align_corners=False).squeeze(0)
        patchB = F.interpolate(patchB.unsqueeze(0), size=(img_size, img_size),
                                mode='bilinear', align_corners=False).squeeze(0)

        # map center to patch-local px
        def _to_local(uv_full, box):
            u0, v0, cw, ch = box
            return ((uv_full[0] - u0) * (img_size / cw),
                    (uv_full[1] - v0) * (img_size / ch))

        box_A = (u0Ai, v0Ai, CROP, CROP)
        box_B = (u0Bi, v0Bi, CROP, CROP)
        piv_A = _to_local((uc_A, vc_A), box_A)
        piv_B_hat = _to_local(uc_B_hat, box_B)
        piv_B_gt  = _to_local(uc_B_gt,  box_B)

        # sample a few extra in-patch LiDAR points for arrows
        u0, v0, cw, ch = box_A
        in_box = ((uv_A_all[:, 0] >= u0) & (uv_A_all[:, 0] < u0 + cw) &
                  (uv_A_all[:, 1] >= v0) & (uv_A_all[:, 1] < v0 + ch))
        if in_box.sum() < 4:
            continue
        idx = np.where(in_box)[0]
        if len(idx) > 12:
            idx = rng.choice(idx, size=12, replace=False)
        uv_A_arr = []
        uv_B_arr = []
        depths = []
        for i in idx:
            uA_p = _to_local((uv_A_all[i, 0], uv_A_all[i, 1]), box_A)
            # project the same 3D point through T_AB_gt to B
            P_w_i = pts_vis[i]
            P_A_i = (T_w2A @ np.append(P_w_i, 1.0))[:3]
            P_B_gt_i = (T_AB_gt @ np.append(P_A_i, 1.0))[:3]
            if P_B_gt_i[2] < 1: continue
            uB_full = (K @ P_B_gt_i)[:2] / P_B_gt_i[2]
            uB_p = _to_local(uB_full, box_B)
            if not (0 <= uB_p[0] < img_size and 0 <= uB_p[1] < img_size):
                continue
            uv_A_arr.append(uA_p); uv_B_arr.append(uB_p)
            depths.append(z_A_all[i])

        if len(uv_A_arr) < 3:
            continue

        return dict(
            patchA=patchA, patchB=patchB,
            piv_A=piv_A, piv_B_hat=piv_B_hat, piv_B_gt=piv_B_gt,
            arrows_A=np.array(uv_A_arr), arrows_B=np.array(uv_B_arr),
            depths=np.array(depths),
            scene=scn.root.name, fi_A=fi_A, fi_B=fi_B,
            CROP=CROP, t_pert=t_p, ypr_pert=ypr_p,
        )
    return None


def main(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # val scene set (same split as training, seed=42)
    root = Path(args.scenes_root)
    names = sorted([p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit()])
    shuffled = sorted([str(root / n) for n in names])
    random.Random(42).shuffle(shuffled)
    cutoff = int(len(shuffled) * args.train_frac)
    val_roots = shuffled[cutoff:]
    print(f'val scenes: {[Path(r).name for r in val_roots]}')

    scenes = []
    for sr in val_roots:
        scn = _SceneData(Path(sr))
        scn.precompute_all()
        scenes.append(scn)

    rng = np.random.default_rng(args.seed)

    # grid layout: many small (A, B) pairs side-by-side, multi-column
    n_total = args.n_samples
    per_col_pair = 2   # A, B
    grid_cols = args.grid_cols
    # rows needed: n_total samples × 2 panels / per_col_pair, arranged in `grid_cols` pair-columns
    pair_cols = grid_cols                           # number of (A, B) pair columns per row
    n_rows_grid = (n_total + pair_cols - 1) // pair_cols
    fig, axes = plt.subplots(n_rows_grid, pair_cols * 2,
                              figsize=(4.5 * pair_cols, 2.3 * n_rows_grid), dpi=105)
    fig.patch.set_facecolor('#f6f4ed')
    if n_rows_grid == 1: axes = np.array([axes])
    if pair_cols * 2 == 1: axes = axes[:, None]

    kept = 0
    while kept < n_total:
        bl = int(rng.choice(args.baselines))
        s = sample_and_show(scenes, rng, bl,
                             img_size=args.img_size, max_points=args.max_points,
                             sigma_ypr=args.sigma_ypr, sigma_t=args.sigma_t,
                             crop_range=(args.crop_min, args.crop_max))
        if s is None:
            continue
        row = kept // pair_cols
        col = (kept %  pair_cols) * 2
        ax_l = axes[row, col]
        ax_r = axes[row, col + 1]

        # minimal titles: keep only on FIRST pair of each row
        label_A = f'{s["scene"]}/{s["fi_A"]:02d}' if col == 0 else ''
        label_B = f'→{s["fi_B"]:02d} (Δ{s["fi_B"]-s["fi_A"]:+d})' if col == 0 else ''
        draw_patches(ax_l, s['patchA'], label_A)
        draw_patches(ax_r, s['patchB'], label_B)

        # pivot marker only — no other dots to avoid visual confusion
        ax_l.plot(*s['piv_A'], marker='*', color='#c13c14', markersize=22,
                   markeredgecolor='white', mew=1.5, zorder=10)
        ax_r.plot(*s['piv_B_hat'], marker='*', color='#c13c14', markersize=22,
                   markeredgecolor='white', mew=1.5, zorder=10)
        if args.sigma_t > 0 or args.sigma_ypr > 0:
            ax_r.plot(*s['piv_B_gt'], marker='*', color='#1e6fff', markersize=16,
                       markeredgecolor='white', mew=1.2, zorder=10)

        kept += 1

    plt.tight_layout()
    outpath = out_dir / 'sampling_check.png'
    plt.savefig(outpath, dpi=110, bbox_inches='tight', facecolor='#f6f4ed')
    print(f'saved → {outpath}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes-root', default='/mnt/mininas/datasets/pandaset')
    ap.add_argument('--train-frac', type=float, default=0.80)
    ap.add_argument('--out', default='experiments/cross_frame_SAMPLING_check')
    ap.add_argument('--img-size', type=int, default=64)
    ap.add_argument('--max-points', type=int, default=256)
    ap.add_argument('--n-samples', type=int, default=40)
    ap.add_argument('--grid-cols', type=int, default=4,
                    help='number of (A,B) pair columns per row in the grid')
    ap.add_argument('--baselines', type=int, nargs='+', default=[1, 5, 10, 20])
    ap.add_argument('--sigma-ypr', type=float, default=1.0)
    ap.add_argument('--sigma-t',   type=float, default=0.20)
    ap.add_argument('--crop-min', type=int, default=128)
    ap.add_argument('--crop-max', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    main(args)
