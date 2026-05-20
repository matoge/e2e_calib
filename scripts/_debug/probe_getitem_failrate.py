"""Direct measurement of how often __getitem__ falls through to the
random-fallback path under pose_frame='vcam' vs 'orig'. We monkey-patch
the inner crop loop counter.
"""
import sys
sys.path.insert(0, '/home/hfunaya/git/e2e_calib')

import random; random.seed(0)
import numpy as np; np.random.seed(0)

from datasets.pandaset_full import PandaSetCalibDatasetFull

# Stub fallback so we can count it instead of recursing forever.
fail_count = {'orig': 0, 'vcam': 0}

class _Patched(PandaSetCalibDatasetFull):
    def __getitem__(self, idx):
        # Suppress the recursion-fallback so we can count fall-throughs.
        try:
            return super().__getitem__(idx)
        except RecursionError:
            return None

CACHES = [
    '/home/hfunaya/cache_v4/kamikado_v3_tiled',
    '/home/hfunaya/cache_v4/woven_v3_tile',
]

for cp in CACHES:
    for pf in ('orig', 'vcam'):
        ds = _Patched(
            cp, split='train', img_size=128,
            max_offset_m=0.6, max_rot_deg=1.5,
            min_crop_px=256, max_crop_px=512,
            oversample=1, grid_n=16, pose_frame=pf,
        )
        N = 200
        # Monkey-patch the random fallback to be a no-op None so we can
        # see how often the max_tries crop loop fails to produce.
        def _no_fallback(self, idx, _orig=PandaSetCalibDatasetFull.__getitem__):
            # Save and restore _in_random_fallback to skip recursion path.
            self._in_random_fallback = True
            try:
                got = _orig(self, idx)
            finally:
                self._in_random_fallback = False
            return got
        ok = fail = 0
        for j in range(N):
            i = random.randint(0, len(ds) - 1)
            got = _no_fallback(ds, i)
            if got is None: fail += 1
            else: ok += 1
        print(f'  {cp.split("/")[-1]:>22s}  pose_frame={pf}  '
              f'ok={ok}/{N}  fail={fail} ({100*fail/N:.1f}%)')
