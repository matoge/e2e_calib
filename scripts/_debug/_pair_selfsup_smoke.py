"""Smoke test for same_frame_self_sup pair builder.

Verifies that with same_frame_self_sup=True:
  * pair_index[i] = (i, i) — works on caches without ego-pose (kamikado)
  * dpose_AB_gt ≈ 0 (identity pose)
  * B-side dist_uvd is the closed-form δ-shift on top of the SAME uv_gt
    that A sees (since pts/K/R_gt are shared, projection is exact)
  * delta1_se3_A == delta1_se3_B == POSE_HAT (= 0 ⊕ δ = δ)
  * Photometric jitter makes A and B images visually distinguishable
  * collate_pair batches correctly

Saves: scripts/_debug/_outputs/pair_selfsup_correspondence.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_pair
from scripts.util.pair_vis import render_pair_correspondence

CACHE = Path('/home/hfunaya/cache_v5/kamikado_v3_full')
OUT = Path(__file__).parent / '_outputs'
OUT.mkdir(exist_ok=True)


def main():
    np.random.seed(0); torch.manual_seed(0)
    ds = PandaSetCalibDatasetFull(
        CACHE, split='val', img_size=128,
        min_crop_px=256, max_crop_px=512,
        max_offset_m=0.0, max_rot_deg=0.5,
        grid_n=16, preload=False,
        pair_mode=True,
        same_frame_self_sup=True,
        photometric_jitter=0.30,
        oversample=1,
    )
    print(f'[ds] same_frame_self_sup  pair_index={len(ds.pair_index)}  total={len(ds)}')
    if len(ds.pair_index) == 0:
        raise RuntimeError('no insts in val')
    print(f'  pair_index[0..4]: {ds.pair_index[:5]}')

    # Pull one pair. ds[k] returns a list (oversample shape) of (A, B, dpose).
    sample = None
    for k in range(20):
        s = ds[k]
        if s:
            sample = s[0] if isinstance(s, list) else s
            break
    if sample is None:
        raise RuntimeError('first 20 pairs all returned None')
    A, B, dpose_AB_gt = sample
    img_A, true_A, dist_A, vfp_A, b_uvd_A, b_val_A, pert_A, pco_A, duv_A, K_A, cs_A, d1_A = A
    img_B, true_B, dist_B, vfp_B, b_uvd_B, b_val_B, pert_B, pco_B, duv_B, K_B, cs_B, d1_B = B

    print()
    print(f'  cs_A={float(cs_A):.0f}  cs_B={float(cs_B):.0f}')
    print(f'  dpose_AB_gt    = {dpose_AB_gt.tolist()}')
    print(f'    |dpose|       = {float(dpose_AB_gt.abs().max()):.4e}  (expect 0 → same frame)')
    print(f'  pert_vec_B (δ) = {pert_B.tolist()}')
    print(f'  delta1_se3_A   = {d1_A.tolist()}')
    print(f'  delta1_se3_B   = {d1_B.tolist()}')
    print(f'  same? {torch.allclose(d1_A, d1_B)}')
    # POSE_HAT expected = δ (since GT_AB = 0)
    expected = np.concatenate([pert_B[:3].numpy(), pert_B[3:6].numpy()]).astype(np.float32)
    diff = float(np.abs(d1_B.numpy() - expected).max())
    print(f'  POSE_HAT == δ? max|Δ|={diff:.4e}')

    print()
    print('=== A side ===')
    print(f'  duv_orig_A: shape={tuple(duv_A.shape)}  '
          f'max|Δ|={duv_A.abs().max().item():.4e} px')
    print('=== B side (perturbed, expect closed-form δ-shift) ===')
    print(f'  duv_orig_B: shape={tuple(duv_B.shape)}  '
          f'max|Δ|={duv_B.abs().max().item():.4e} px  '
          f'mean|Δ|={duv_B.abs().mean().item():.4e} px')

    # Photometric: A and B should differ
    img_diff = (img_A.float() - img_B.float()).abs()
    print()
    print(f'  img A vs B diff: max={img_diff.max().item():.0f}/255  '
          f'mean={img_diff.mean().item():.4f}/255  '
          f'(>0 means jitter is acting)')

    # Render correspondence — multiple samples.
    # FIX: reseed before EACH ds[k] so δ (sampled inside __getitem__ via
    # np.random) is identical across all 6 samples. Without this, every
    # PNG uses a different δ and you can't compare them.
    for k_idx in range(6):
        idx = k_idx + 100
        # Seed the global RNG with idx so __getitem__'s internal np.random
        # calls (crop, δ, etc.) are deterministic per-idx. Re-running this
        # script must produce byte-identical PNGs.
        np.random.seed(idx); torch.manual_seed(idx)
        s = ds[idx]
        if not s:
            continue
        A_k, B_k, dp_k = s[0] if isinstance(s, list) else s
        pert_Bk = A_k[6] if False else B_k[6]
        out_corr = OUT / f'pair_selfsup_correspondence_{k_idx:02d}.png'
        render_pair_correspondence(
            A_k, B_k, dp_k,
            title=(f'self-sup A=B  idx={k_idx+100}  '
                   f'δ_t=[{pert_Bk[0]:.3f},{pert_Bk[1]:.3f},{pert_Bk[2]:.3f}]m  '
                   f'δ_ypr=[{pert_Bk[3]:.3f},{pert_Bk[4]:.3f},{pert_Bk[5]:.3f}]deg'),
            out_path=out_corr,
        )
        print(f'[saved] {out_corr.absolute()}')

    # collate
    print()
    print('=== collate_pair smoke (batch=4) ===')
    batch = []
    for k in range(4):
        s = ds[k]
        if s:
            batch.append(s[0] if isinstance(s, list) else s)
    if not batch:
        print('  no valid pair samples in first 4 idx — skipping collate')
    else:
        col = collate_pair(batch)
        print(f"  A imgs shape  : {tuple(col['A'][0].shape)}")
        print(f"  B imgs shape  : {tuple(col['B'][0].shape)}")
        print(f"  dpose_AB shape: {tuple(col['dpose_AB'].shape)}  "
              f"max|Δ|={col['dpose_AB'].abs().max().item():.4e} (expect 0)")


if __name__ == '__main__':
    main()
