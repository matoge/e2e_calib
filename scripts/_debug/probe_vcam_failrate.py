"""Smoke: how often does PandaSetCalibDatasetFull.__getitem__ end up
recursing under pose_frame='vcam' vs 'orig'?

We bypass the recursion fallback by patching __getitem__ to return
None on max_tries failure, then count.
"""
import sys
sys.path.insert(0, '/home/hfunaya/git/e2e_calib')

import random
random.seed(0)
import numpy as np
np.random.seed(0)

from datasets.pandaset_full import PandaSetCalibDatasetFull

CACHES = [
    '/home/hfunaya/cache_v4/kamikado_v3_tiled',
    '/home/hfunaya/cache_v4/woven_v3_tile',
    '/home/hfunaya/clearml/data/cache/waymo_v3_tiled_i',
]

# Monkey-patch fallback to count fails instead of recursing.
_orig = PandaSetCalibDatasetFull.__getitem__
def _no_recurse(self, idx):
    try:
        return _orig(self, idx)
    except RecursionError:
        return None
PandaSetCalibDatasetFull.__getitem__ = _no_recurse

for cp in CACHES:
    for pf in ('orig', 'vcam'):
        ds = PandaSetCalibDatasetFull(
            cp, split='train', img_size=128,
            max_offset_m=0.6, max_rot_deg=1.5,
            min_crop_px=256, max_crop_px=512,
            oversample=1, grid_n=16, pose_frame=pf,
        )
        N = 200
        ok = fail = 0
        for j in range(N):
            i = random.randint(0, len(ds) - 1)
            try:
                s = ds[i]
                if s is None:
                    fail += 1
                else:
                    ok += 1
            except Exception:
                fail += 1
        print(f'  {cp.split("/")[-1]:>22s}  pose_frame={pf}  ok={ok}/{N} fail={fail}')
