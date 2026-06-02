"""Sweep N samples from PandaSet pair_mode dataset, dump stats on
true_uvd_B / dist_uvd_B / target = (true - dist) so we can see WHERE the
1e9-scale uv comes from. No model, no DDP, just dataset.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets.pandaset_full import PandaSetCalibDatasetFull  # noqa: E402

CACHE = '/mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full'
N = 100


def main() -> int:
    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE, img_size=128, oversample=1,
        max_offset_m=0.20, max_rot_deg=1.0,
        pair_mode=True, pair_stride=10, grid_n=16,
        same_frame_self_sup=False, split='train',
    )
    rng = np.random.default_rng(0)
    n_seen = 0
    n_skip = 0
    out_lines = []
    for k in range(N * 4):
        idx = int(rng.integers(0, len(ds)))
        try:
            sample = ds[idx]
        except Exception as e:
            n_skip += 1; continue
        if sample is None:
            n_skip += 1; continue
        if isinstance(sample, list):
            if not sample: n_skip += 1; continue
            sample = sample[0]
        built_A, built_B, dpose = sample
        true_A = built_A[1].numpy()  # (N, 5)
        dist_A = built_A[2].numpy()
        true_B = built_B[1].numpy()
        dist_B = built_B[2].numpy()
        # stats
        tB_max = float(np.abs(true_B[:, :2]).max())
        dB_max = float(np.abs(dist_B[:, :2]).max())
        gt = true_B[:, :2] - dist_B[:, :2]
        gt_max = float(np.abs(gt).max())
        # how many points have abs uv > 1000 (= z<=small projection blowup)
        n_blowup_t = int((np.abs(true_B[:, :2]) > 1000).any(axis=-1).sum())
        n_blowup_d = int((np.abs(dist_B[:, :2]) > 1000).any(axis=-1).sum())
        N_pts = true_B.shape[0]
        out_lines.append(
            f"idx={idx:6d}  N={N_pts:4d}  "
            f"|true_B|max={tB_max:.2e}  |dist_B|max={dB_max:.2e}  "
            f"|gt|max={gt_max:.2e}  blow_t={n_blowup_t}/{N_pts}  blow_d={n_blowup_d}/{N_pts}  "
            f"dpose t=({dpose[0]:.2f},{dpose[1]:.2f},{dpose[2]:.2f})m "
            f"ypr=({dpose[3]:.2f},{dpose[4]:.2f},{dpose[5]:.2f})°"
        )
        n_seen += 1
        if n_seen >= N: break

    # sort by gt_max descending
    parsed = []
    for line in out_lines:
        gt_max = float(line.split('|gt|max=')[1].split()[0])
        parsed.append((gt_max, line))
    parsed.sort(reverse=True)
    print(f'collected {n_seen} samples (skipped {n_skip})\n')
    print('=== TOP 20 by |gt|max ===')
    for _, line in parsed[:20]:
        print(line)
    print('\n=== BOTTOM 5 by |gt|max ===')
    for _, line in parsed[-5:]:
        print(line)

    # aggregate
    all_gt_max = [g for g, _ in parsed]
    print(f'\n|gt|max stats: min={min(all_gt_max):.2e} median={np.median(all_gt_max):.2e} '
          f'max={max(all_gt_max):.2e}')
    n_extreme = sum(1 for g, _ in parsed if g > 1e6)
    print(f'samples with |gt|max > 1e6: {n_extreme}/{n_seen}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
