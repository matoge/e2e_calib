"""Compare per-point model σ for points belonging to:
    - background (not inside any 3D box)
    - static cars (cuboid `stationary=True` or position diff small)
    - moving cars (cuboid position diff > threshold between frame A and B)

Tests the hypothesis: the cross-frame residual model implicitly inflates σ
on points whose underlying surface is moving (the model can't predict their
B-frame position from A-frame appearance + static-scene assumption).

Bonus pass (`--motion-warp`): for moving-car points, warp world coords by
the box's measured A→B rigid transform (translation + Δyaw), re-project,
re-evaluate. Residual should drop, σ should drop. Confirms σ inflation is
explained by motion (not appearance ambiguity).

Usage:
    python scripts/analysis/dynamic_object_variance.py \
        --ckpt experiments/cross_frame_v92_unified_multi_c4 \
        --scenes-root /mnt/nvme6t/pandaset_39 --cameras front_camera \
        --n-pairs 200 --out /tmp/dyn_var.png
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

from datasets.pandaset_pair import PandaSetCrossFrameDataset
from scripts.visualization.vis_pred_check import load_model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 64

# motion threshold (m) over the (A, B) interval to call a box "moving"
MOVING_THRESH_M = 0.5


def load_cuboids(scene_root: pathlib.Path, fi: int):
    """Return a DataFrame for cuboid annotations of frame fi."""
    p = scene_root / 'annotations' / 'cuboids' / f'{fi:02d}.pkl.gz'
    with gzip.open(p, 'rb') as f:
        return pickle.load(f)


def cuboid_box(row):
    """row: pandas Series → dict with center, size, yaw, R."""
    cy, sy = math.cos(row['yaw']), math.sin(row['yaw'])
    R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return dict(
        c=np.array([row['position.x'], row['position.y'], row['position.z']], dtype=np.float64),
        s=np.array([row['dimensions.x'], row['dimensions.y'], row['dimensions.z']], dtype=np.float64),
        R=R,
        uuid=row['uuid'],
        label=row['label'],
        stationary=bool(row.get('stationary', False)),
        motion=str(row.get('attributes.object_motion', '')),
    )


def points_in_box(pts_w: np.ndarray, box: dict) -> np.ndarray:
    """pts_w (N, 3), box from cuboid_box → (N,) bool mask."""
    rel = pts_w - box['c']
    local = rel @ box['R']                       # rotate into box frame
    half = box['s'] * 0.5
    return ((np.abs(local[:, 0]) < half[0]) &
            (np.abs(local[:, 1]) < half[1]) &
            (np.abs(local[:, 2]) < half[2]))


def categorize(pts_w_a: np.ndarray, boxes_a: list, boxes_b_by_uuid: dict):
    """Tag each point: 0=background, 1=parked, 2=stopped, 3=moving.

    Uses the annotator's `attributes.object_motion` field directly
    (Moving / Stopped / Parked). Falls back to box-displacement for
    rows missing the motion attribute (NaN).
    Vehicle-only (Car, Pickup, Truck, Bus, Motorcycle).
    """
    VEHICLE_LABELS = {'Car', 'Pickup Truck', 'Medium-sized Truck', 'Semi-truck',
                       'Towed Object', 'Motorcycle', 'Bus', 'Motorized Scooter'}
    MOTION_TO_CAT = {'Parked': 1, 'Stopped': 2, 'Moving': 3}
    N = pts_w_a.shape[0]
    cat = np.zeros(N, dtype=np.int8)
    motion_disp = np.zeros(N, dtype=np.float32)
    for box in boxes_a:
        if box['label'] not in VEHICLE_LABELS:
            continue
        in_b = points_in_box(pts_w_a, box)
        if not in_b.any():
            continue
        box_b = boxes_b_by_uuid.get(box['uuid'])
        disp = float(np.linalg.norm(box['c'] - box_b['c'])) if box_b else 0.0
        c = MOTION_TO_CAT.get(box['motion'])
        if c is None:
            # missing motion attr — fall back to displacement-based label
            c = 3 if disp > MOVING_THRESH_M else 1
        for i in np.where(in_b)[0]:
            if cat[i] == 0:
                cat[i] = c
                motion_disp[i] = disp
    return cat, motion_disp


@torch.no_grad()
def predict_one(model, sample, multi_frame=False):
    """Run model, return raw[..., 5] for A→B."""
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
    return raw_AB[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='experiment dir or .pt')
    ap.add_argument('--ckpt-name', default='best_model.pt')
    ap.add_argument('--scenes-root', required=True)
    ap.add_argument('--cameras', default='front_camera')
    ap.add_argument('--split', default='val')
    ap.add_argument('--n-pairs', type=int, default=200)
    ap.add_argument('--baseline-min', type=int, default=1)
    ap.add_argument('--baseline-max', type=int, default=20)
    ap.add_argument('--sigma-ypr', type=float, default=1.0)
    ap.add_argument('--sigma-t', type=float, default=0.2)
    ap.add_argument('--multi-frame', action='store_true',
                     help='set if model trained with --multi-frame')
    ap.add_argument('--out', default='/tmp/dyn_var.png')
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

    # cuboid frame-cache per scene
    cuboid_cache = {}

    def get_boxes(scene_root, fi):
        key = (str(scene_root), fi)
        if key not in cuboid_cache:
            df = load_cuboids(scene_root, fi)
            cuboid_cache[key] = [cuboid_box(row) for _, row in df.iterrows()]
        return cuboid_cache[key]

    sigmas = {0: [], 1: [], 2: [], 3: []}    # bg / parked / stopped / moving
    errs   = {0: [], 1: [], 2: [], 3: []}
    tags = ['background', 'parked', 'stopped', 'moving']
    # for moving cars: bucket by per-pair displacement (m)
    moving_buckets = [(0.0, 1.0), (1.0, 3.0), (3.0, 10.0), (10.0, 1e9)]
    bucket_tags = ['mv 0-1m', 'mv 1-3m', 'mv 3-10m', 'mv 10+m']
    sigmas_mv = {i: [] for i in range(len(moving_buckets))}
    errs_mv   = {i: [] for i in range(len(moving_buckets))}
    disp_per_pt = []                            # parallel to moving rows
    n_skip = 0
    for i in range(len(ds)):
        sample = ds[i]
        if sample is None:
            n_skip += 1
            continue
        # need world coords of query points in frame A. Dataset doesn't
        # expose them directly, but `uvd_A` is the local-patch (u, v, d).
        # Recover world coords from `pts_w_A_query` if dataset emits it,
        # else fall back to uvd_full. Dataset already loads the full
        # world-coord array — we expose it here via a side channel.
        scene_root = pathlib.Path(sample['scene_root'])
        fi_a = int(sample['fi_A'])
        fi_b = int(sample['fi_B'])
        # world coords per query point are embedded in sample['pts_w_A']
        # (added below by patching dataset). For now compute from K, T.
        if 'pts_w_A_query' in sample:
            pts_w_a = sample['pts_w_A_query'].numpy()
        else:
            n_skip += 1
            continue

        # Cuboids
        boxes_a = get_boxes(scene_root, fi_a)
        boxes_b_uuid = {b['uuid']: b for b in get_boxes(scene_root, fi_b)}
        cat, disp = categorize(pts_w_a, boxes_a, boxes_b_uuid)

        # model predict
        raw = predict_one(model, sample, multi_frame=args.multi_frame)
        # raw: (max_pts, 5+) — log_sx, log_sy at indices 2,3 for out_dim=5
        log_sx = raw[:, 2 if out_dim == 5 else 3]
        log_sy = raw[:, 3 if out_dim == 5 else 4]
        sigma_px = 0.5 * (np.exp(log_sx) + np.exp(log_sy))

        # Residual error per point (in local 64-px crop)
        delta_uv = raw[:, :2]
        uv_hat_local = sample['uv_B_hat_of_A'].numpy()           # (P, 2)
        uv_gt_local  = sample['uv_B_gt_of_A'].numpy()
        pad = sample['pad_A'].numpy()
        err_px = np.linalg.norm(uv_hat_local + delta_uv - uv_gt_local, axis=-1)

        valid = ~pad
        for c in (0, 1, 2, 3):
            mask = valid & (cat == c)
            if mask.any():
                sigmas[c].extend(sigma_px[mask].tolist())
                errs[c].extend(err_px[mask].tolist())
        # bucket moving (cat==3) by displacement
        mv_mask = valid & (cat == 3)
        for k, (lo, hi) in enumerate(moving_buckets):
            m = mv_mask & (disp >= lo) & (disp < hi)
            if m.any():
                sigmas_mv[k].extend(sigma_px[m].tolist())
                errs_mv[k].extend(err_px[m].tolist())

        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(ds)}]  bg={len(sigmas[0])} static={len(sigmas[1])} moving={len(sigmas[2])}')

    print(f'\nskipped {n_skip} samples (no pts_w_A_query field — needs dataset patch)')
    def _summ(name, table, ks, label_list):
        print(f'\n=== {name} ===')
        print(f'  bucket            N         median   mean    p90')
        for k in ks:
            arr = table.get(k)
            if arr is None or len(arr) == 0:
                continue
            arr = np.array(arr)
            label = label_list[k]
            print(f'  {label:<15s} {len(arr):<8d} {np.median(arr):.2f}    {arr.mean():.2f}    {np.percentile(arr, 90):.2f}')

    _summ('σ_pred (px) by motion-attr', sigmas, [0, 1, 2, 3], tags)
    _summ('|Δuv| residual (px) by motion-attr', errs, [0, 1, 2, 3], tags)
    _summ('σ_pred (px) by moving-disp bucket', sigmas_mv, list(range(len(moving_buckets))), bucket_tags)
    _summ('|Δuv| residual (px) by moving-disp bucket', errs_mv, list(range(len(moving_buckets))), bucket_tags)

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    colors = ['#888', '#3a8', '#d33']
    for c in (0, 1, 2):
        if not sigmas[c]:
            continue
        ax[0].hist(np.log10(sigmas[c]), bins=40, alpha=0.5,
                    color=colors[c], label=f'{tags[c]} (n={len(sigmas[c])})', density=True)
        ax[1].hist(np.log10(errs[c]), bins=40, alpha=0.5,
                    color=colors[c], label=f'{tags[c]} (n={len(errs[c])})', density=True)
    ax[0].set_xlabel('log10 σ_pred (px)'); ax[0].set_ylabel('density')
    ax[0].set_title('Predicted σ by category'); ax[0].legend()
    ax[1].set_xlabel('log10 |Δuv| residual (px)'); ax[1].set_ylabel('density')
    ax[1].set_title('Actual residual by category'); ax[1].legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f'\nsaved → {args.out}')


if __name__ == '__main__':
    main()
