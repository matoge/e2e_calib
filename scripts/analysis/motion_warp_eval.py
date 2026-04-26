"""Motion-warp evaluation: compare model prediction against (a) the dataset's
static-scene GT and (b) a motion-corrected GT obtained by warping each moving-
object query point by its 3D-cuboid rigid transform between frame A and frame B.

Hypothesis (from earlier σ analysis): for points inside MOVING cuboids, the
dataset's "GT" is wrong — the model is trained against where the point would
be IF static, but the point actually moved with its cuboid. The correct
target uv in B is obtained by:
    p_w_in_B = R_b @ R_a^T @ (p_w_in_A_world - c_a) + c_b
    uv_motion = project(p_w_in_B, K, T_w2c_B)
re-projected back into B's local crop.

If the model has any motion-awareness, |pred - uv_motion| should be SMALLER
than |pred - uv_static| for moving points. If not, the model just mostly
predicts "static" and both residuals are about the same — but uv_motion is
the *correct* GT for re-training with motion-aware supervision.

Usage:
    python scripts/analysis/motion_warp_eval.py \
        --ckpt experiments/cross_frame_v92_unified_multi_c4 \
        --scenes-root /mnt/nvme6t/pandaset_39 --cameras front_camera \
        --n-pairs 200 --multi-frame
"""
import argparse
import gzip
import math
import pickle
import sys
import pathlib
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.pandaset_pair import PandaSetCrossFrameDataset, _proj_cam
from scripts.visualization.vis_pred_check import load_model
from scripts.analysis.dynamic_object_variance import (
    load_cuboids, cuboid_box, points_in_box, predict_one,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 64
MOVING_THRESH_M = 0.5


def warp_world_by_box(p_w: np.ndarray, box_a: dict, box_b: dict) -> np.ndarray:
    """Rigid-transform p_w by box A → B motion. p_w (N, 3) — must be inside box_a."""
    rel    = p_w - box_a['c']                  # (N, 3)
    local  = rel @ box_a['R']                  # (N, 3) in box_a frame
    p_w_b  = local @ box_b['R'].T + box_b['c']
    return p_w_b


def world_to_local_B(p_w: np.ndarray, T_w2c_B, K, dist, box_B):
    """World → camera B → image uv → local patch coord."""
    homo = np.column_stack([p_w, np.ones(len(p_w), dtype=p_w.dtype)])
    p_b  = (T_w2c_B @ homo.T)[:3].T            # (N, 3)
    valid = p_b[:, 2] > 0.5
    uv_full = np.zeros((len(p_w), 2), dtype=np.float32)
    uv_full[valid] = _proj_cam(p_b[valid], K, dist).astype(np.float32)
    u0, v0, cw, ch = box_B
    uv_local = np.stack(
        [(uv_full[:, 0] - u0) * IMG_SIZE / cw,
         (uv_full[:, 1] - v0) * IMG_SIZE / ch], axis=1)
    return uv_local, valid


def categorize_with_warp(pts_w_a, boxes_a, boxes_b_by_uuid,
                         T_w2c_B, K, dist, box_B):
    """Same as dynamic_object_variance.categorize, plus emits warped-GT uv
    (motion-corrected target in B's local patch) for moving points only.
    """
    VEHICLE = {'Car', 'Pickup Truck', 'Medium-sized Truck', 'Semi-truck',
               'Towed Object', 'Motorcycle', 'Bus', 'Motorized Scooter'}
    MOTION_TO_CAT = {'Parked': 1, 'Stopped': 2, 'Moving': 3}
    N = pts_w_a.shape[0]
    cat = np.zeros(N, dtype=np.int8)
    disp = np.zeros(N, dtype=np.float32)
    uv_warped_local = np.full((N, 2), np.nan, dtype=np.float32)
    uv_warped_valid = np.zeros(N, dtype=bool)
    for box in boxes_a:
        if box['label'] not in VEHICLE:
            continue
        in_b = points_in_box(pts_w_a, box)
        if not in_b.any():
            continue
        box_b = boxes_b_by_uuid.get(box['uuid'])
        d = float(np.linalg.norm(box['c'] - box_b['c'])) if box_b else 0.0
        c = MOTION_TO_CAT.get(box['motion'])
        if c is None:
            c = 3 if d > MOVING_THRESH_M else 1
        idx = np.where(in_b)[0]
        for i in idx:
            if cat[i] == 0:
                cat[i] = c
                disp[i] = d
        if c == 3 and box_b is not None:
            # warp the points in this box and store
            p_warp = warp_world_by_box(pts_w_a[idx], box, box_b)
            uv_loc, vmask = world_to_local_B(p_warp, T_w2c_B, K, dist, box_B)
            for k, i in enumerate(idx):
                if cat[i] == 3 and vmask[k]:
                    uv_warped_local[i] = uv_loc[k]
                    uv_warped_valid[i] = True
    return cat, disp, uv_warped_local, uv_warped_valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--ckpt-name', default='best_model.pt')
    ap.add_argument('--scenes-root', required=True)
    ap.add_argument('--cameras', default='front_camera')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n-pairs', type=int, default=200)
    ap.add_argument('--baseline-min', type=int, default=1)
    ap.add_argument('--baseline-max', type=int, default=20)
    ap.add_argument('--sigma-ypr', type=float, default=1.0)
    ap.add_argument('--sigma-t', type=float, default=0.2)
    ap.add_argument('--multi-frame', action='store_true')
    ap.add_argument('--out', default='/tmp/motion_warp.png')
    args = ap.parse_args()

    ckpt = pathlib.Path(args.ckpt)
    if ckpt.is_dir():
        model, out_dim = load_model(ckpt, args.ckpt_name)
    else:
        model, out_dim = load_model(ckpt.parent, ckpt.name)
    print(f'loaded {args.ckpt} (out_dim={out_dim})')

    ds = PandaSetCrossFrameDataset(
        scenes_root=args.scenes_root, split=args.split, train_frac=0.8,
        cameras=args.cameras,
        img_size=IMG_SIZE, max_points=256,
        baseline_range=(args.baseline_min, args.baseline_max),
        sigma_ypr=args.sigma_ypr, sigma_t=args.sigma_t,
        crop_range=(128, 256),
        triplet=args.multi_frame,
        virtual_epoch_len=args.n_pairs, seed=20260427,
    )

    cuboid_cache = {}
    def get_boxes(scene_root, fi):
        key = (str(scene_root), fi)
        if key not in cuboid_cache:
            df = load_cuboids(scene_root, fi)
            cuboid_cache[key] = [cuboid_box(row) for _, row in df.iterrows()]
        return cuboid_cache[key]

    # buckets: {bg, parked, stopped, moving, mv-warp-corrected (subset)}
    err_static = {'bg': [], 'parked': [], 'stopped': [], 'moving': []}
    err_warped = {'moving': []}     # only computable for moving with valid warp
    sigma     = {'bg': [], 'parked': [], 'stopped': [], 'moving': []}
    delta_motion_static = []        # for moving: how far the warp shifts GT
    cat_names = ['bg', 'parked', 'stopped', 'moving']

    n_skip = 0
    for i in range(len(ds)):
        sample = ds[i]
        if sample is None:
            n_skip += 1
            continue
        if 'pts_w_A_query' not in sample:
            n_skip += 1
            continue
        scene_root = pathlib.Path(sample['scene_root'])
        fi_a = int(sample['fi_A'])
        fi_b = int(sample['fi_B'])
        pts_w_a = sample['pts_w_A_query'].numpy()

        boxes_a = get_boxes(scene_root, fi_a)
        boxes_b_uuid = {b['uuid']: b for b in get_boxes(scene_root, fi_b)}

        cat, disp, uv_warp, uv_warp_valid = categorize_with_warp(
            pts_w_a, boxes_a, boxes_b_uuid,
            sample['T_w2c_B'].numpy(), sample['K'].numpy(),
            sample['dist'].numpy(), sample['box_B'].numpy().tolist())

        # model predict
        raw = predict_one(model, sample, multi_frame=args.multi_frame)
        log_sx = raw[:, 2 if out_dim == 5 else 3]
        log_sy = raw[:, 3 if out_dim == 5 else 4]
        sigma_px = 0.5 * (np.exp(log_sx) + np.exp(log_sy))
        delta_uv = raw[:, :2]
        uv_hat_local = sample['uv_B_hat_of_A'].numpy()
        uv_gt_static = sample['uv_B_gt_of_A'].numpy()
        pad = sample['pad_A'].numpy()
        uv_pred = uv_hat_local + delta_uv

        err_s = np.linalg.norm(uv_pred - uv_gt_static, axis=-1)        # vs static GT
        err_w = np.linalg.norm(uv_pred - uv_warp, axis=-1)             # vs warped GT (NaN where invalid)
        d_static_warp = np.linalg.norm(uv_warp - uv_gt_static, axis=-1)

        valid = ~pad
        for c, name in enumerate(cat_names):
            mask = valid & (cat == c)
            if mask.any():
                err_static[name].extend(err_s[mask].tolist())
                sigma[name].extend(sigma_px[mask].tolist())
        # warped only for moving, where uv_warp valid
        m_warp = valid & (cat == 3) & uv_warp_valid
        if m_warp.any():
            err_warped['moving'].extend(err_w[m_warp].tolist())
            delta_motion_static.extend(d_static_warp[m_warp].tolist())

        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(ds)}] bg={len(err_static["bg"])} '
                  f'mv={len(err_static["moving"])} mv-warp={len(err_warped["moving"])}')

    print(f'\nskipped {n_skip} (no pts_w_A_query)')
    print(f'\n=== |pred - GT_static| (px), per category ===')
    print(f'  cat        N        median    mean    p90')
    for name in cat_names:
        if not err_static[name]:
            continue
        arr = np.array(err_static[name])
        print(f'  {name:<10s} {len(arr):<8d} {np.median(arr):.2f}     {arr.mean():.2f}    {np.percentile(arr, 90):.2f}')

    print(f'\n=== |pred - GT_warped| for moving only (motion-corrected target) ===')
    if err_warped['moving']:
        arr_w = np.array(err_warped['moving'])
        print(f'  moving     {len(arr_w):<8d} {np.median(arr_w):.2f}     {arr_w.mean():.2f}    {np.percentile(arr_w, 90):.2f}')
        # Direct comparison
        # paired comparison: same N as moving-with-warp
        # err_static for moving on the SAME points isn't easily indexable here, so just
        # report shift between targets:
        arr_d = np.array(delta_motion_static)
        print(f'\n=== |GT_warped - GT_static| (how much warp shifts the target) ===')
        print(f'  moving     {len(arr_d):<8d} median={np.median(arr_d):.2f}  mean={arr_d.mean():.2f}  p90={np.percentile(arr_d, 90):.2f}')

    def _safe_log10(arr):
        arr = np.asarray(arr, dtype=np.float64)
        return np.log10(arr[arr > 1e-3])

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    colors = {'bg': '#888', 'parked': '#3a8', 'stopped': '#28b', 'moving': '#d33'}
    for name in cat_names:
        if not err_static[name]:
            continue
        ax[0].hist(_safe_log10(err_static[name]), bins=40, alpha=0.4,
                   color=colors[name], label=f'{name} (n={len(err_static[name])})', density=True)
    if err_warped['moving']:
        ax[0].hist(_safe_log10(err_warped['moving']), bins=40,
                   color='#000', label=f'moving warped GT (n={len(err_warped["moving"])})',
                   histtype='step', linewidth=2)
    ax[0].set_xlabel('log10 |pred - GT| (px)'); ax[0].set_ylabel('density')
    ax[0].set_title('Residual to static vs warped GT')
    ax[0].legend()
    if delta_motion_static:
        ax[1].hist(_safe_log10(delta_motion_static), bins=40, color='#d33', alpha=0.5,
                   label=f'moving (n={len(delta_motion_static)})', density=True)
    ax[1].set_xlabel('log10 |GT_warped - GT_static| (px)'); ax[1].set_ylabel('density')
    ax[1].set_title('How much motion shifts the target')
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f'\nsaved → {args.out}')


if __name__ == '__main__':
    main()
