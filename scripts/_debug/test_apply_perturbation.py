"""Refactor guard: extract a public `apply_perturbation(tile_inst, t, ypr)`
from PandaSetCalibDatasetFull.__getitem__ without changing the random-
sampling __getitem__'s output.

This test:
  1. Calls `__getitem__(idx)` with seed=K once → record sample_A.
  2. After the refactor, calls __getitem__(idx) again with seed=K → must
     equal sample_A byte-for-byte.
  3. Then calls the new `apply_perturbation(tile_inst, t_known, ypr_known)`
     with the same (t, ypr) the seeded run produced → must also equal A.

If (3) holds, the public path can be reused at eval/BA time with any
chosen (t, ypr).
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full

CACHE = '/cache/kamikado_v3_tiled'
DS_KW = dict(img_size=128, min_crop_px=512, max_crop_px=512,
              max_rot_deg=1.5, max_offset_m=0.6, grid_n=16, oversample=1)
SEED = 12345
IDX = 0


def _capture_seed_perturbation(seed):
    """Re-derive the (t_delta, ypr) that the seeded __getitem__ would draw,
    by replaying the SAME sequence of np.random calls leading up to the
    perturbation block. This must match what's in datasets/pandaset_full.py
    or this test is invalid; the prefix calls happen at:
      - pivot pick: 1 randint or 2 randints
      - cs pick:    1 randint
      - zoom_aug:   skipped (False)
    Since pivots / cs are integer-randint draws, predicting them needs the
    full instance state. Easiest path: monkey-patch np.random.rand to log
    arguments while __getitem__ runs, capture the (t, ypr) call.
    """
    captured = {}
    orig_rand = np.random.rand
    def spy_rand(*a):
        out = orig_rand(*a)
        if len(a) == 1 and a[0] == 3:
            captured.setdefault('rand3', []).append(out.copy())
        return out
    np.random.seed(seed)
    np.random.rand = spy_rand
    try:
        ds = PandaSetCalibDatasetFull(CACHE, split='val', **DS_KW)
        sample_A = ds[IDX]
    finally:
        np.random.rand = orig_rand
    # In __getitem__ pose_frame='orig', the two rand(3) calls inside the
    # perturbation block are t_delta then ypr (max_offset / max_rot_deg
    # multiplied later).
    rand3 = captured.get('rand3', [])
    if len(rand3) < 2:
        raise RuntimeError(f'expected >=2 rand(3) calls, got {len(rand3)}')
    t_delta = (rand3[-2] * 2 - 1) * DS_KW['max_offset_m']
    ypr     = (rand3[-1] * 2 - 1) * DS_KW['max_rot_deg']
    return sample_A, t_delta.astype(np.float32), ypr.astype(np.float32)


def _bytes_eq(a, b, name):
    if torch.is_tensor(a):
        ok = torch.equal(a, b)
    else:
        ok = (a == b) if not hasattr(a, 'shape') else bool(np.array_equal(a, b))
    print(f'  {name:14s} {"OK" if ok else "MISMATCH"}')
    return ok


def main():
    # Pass 1: seeded __getitem__
    A, t1, ypr1 = _capture_seed_perturbation(SEED)
    print(f'seeded perturbation: t={t1.round(4)} ypr={ypr1.round(4)}')

    # Pass 2: re-seeded __getitem__ (must reproduce A)
    np.random.seed(SEED)
    ds = PandaSetCalibDatasetFull(CACHE, split='val', **DS_KW)
    B = ds[IDX]
    all_ok = True
    for nm, x, y in zip(
            ['imgs', 'true_uvd', 'dist_uvd', 'pad_mask', 'vfp', 'b_uvd', 'b_v', 'pert_vec'],
            A, B):
        all_ok &= _bytes_eq(x, y, nm)
    print(f'PASS-2 (reseeded) : {"OK" if all_ok else "FAIL"}')

    # Pass 3: would call ds.apply_perturbation(idx, t1, ypr1) once that
    # method exists. Check that the new method does NOT exist yet:
    ds = PandaSetCalibDatasetFull(CACHE, split='val', **DS_KW)
    has = hasattr(ds, 'apply_perturbation')
    print(f'apply_perturbation present? {has}  (expected False before refactor)')


if __name__ == '__main__':
    main()
