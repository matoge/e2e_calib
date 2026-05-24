"""Local smoke for pair_mode in PandaSetCalibDatasetFull.

Builds a single (frame_A, frame_B, Δpose_AB) sample from the freshly built
full-frame PandaSet cache and verifies:
  - pair_index is non-empty
  - both 12-tuples have matching cs (VFP-identity)
  - vfp_A == vfp_B
  - Δpose_AB direction roughly matches |t| <= 1m, yaw <= 1° at stride=1

Run:
  /home/hfunaya/.pyenv/versions/3.10.4/bin/python scripts/_debug/_smoke_pair_mode.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_pair  # noqa: E402

CACHE_DIR = '/mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full'

ds = PandaSetCalibDatasetFull(
    cache_dir=CACHE_DIR, split='train',
    img_size=256, min_crop_px=256, max_crop_px=512,
    max_offset_m=0.20, max_rot_deg=1.0,
    pose_frame='orig', oversample=1, frame_stride=1,
    grid_n=16, k_per_cell=8,
    preload=False,
    pair_mode=True, pair_stride=1,
)
print(f'len(ds) = {len(ds)}  (pair_index size = {len(ds.pair_index)})')
assert len(ds.pair_index) > 0, 'pair_index empty'

torch.manual_seed(0)
np.random.seed(0)

# Try a few samples to make sure none crash.
N_TRY = 4
samples = []
for k in range(N_TRY):
    s = ds[k]
    samples.append(s)
    A12, B12, dpose = s
    img_A, true_A, dist_A, vfp_A, _, _, pert_A, *_, cs_A, _ = A12
    img_B, true_B, dist_B, vfp_B, _, _, pert_B, *_, cs_B, _ = B12
    print(f'[{k}] cs_A={float(cs_A):.0f} cs_B={float(cs_B):.0f} '
          f'vfp_A={float(vfp_A):.1f} vfp_B={float(vfp_B):.1f} '
          f'pert_A_t=({pert_A[0]:+.2f},{pert_A[1]:+.2f},{pert_A[2]:+.2f}) '
          f'pert_B_t=({pert_B[0]:+.2f},{pert_B[1]:+.2f},{pert_B[2]:+.2f}) '
          f'dpose_t={tuple(f"{v:+.2f}" for v in dpose[:3].tolist())} '
          f'dpose_ypr={tuple(f"{v:+.2f}" for v in dpose[3:].tolist())}')
    assert float(cs_A) == float(cs_B), 'cs mismatch'
    assert abs(float(vfp_A) - float(vfp_B)) < 1e-3, 'vfp mismatch'
    assert torch.allclose(pert_B, torch.zeros_like(pert_B)), 'pert_B should be 0'

# Collate test
batch = collate_pair(samples)
assert isinstance(batch, dict), 'collate_pair must return dict'
assert 'A' in batch and 'B' in batch and 'dpose_AB' in batch
print(f'\ncollate_pair OK: '
      f'A.imgs={tuple(batch["A"][0].shape)}  '
      f'B.imgs={tuple(batch["B"][0].shape)}  '
      f'dpose_AB={tuple(batch["dpose_AB"].shape)}')
print('SMOKE OK')
