"""Local smoke for CalibNetDepth.forward_pair.

Builds 1 pair sample from the freshly built full-frame PandaSet cache, runs
a single forward_pair, and asserts:
  1. Shapes are sane: pred_A, pred_B, pred_A_to_B all (B, N, 5).
  2. PoseEmb identity (gate): with dpose_AB = 0 and identical KV/Q wiring,
     pred_A_to_B forward (image_B as KV, distorted_uvd_A as Q, pose_emb=0)
     equals a plain forward(image_B, distorted_uvd_A, pose_emb=0).
  3. Backward through the sum runs without error (params receive grads).

Run:
  /home/hfunaya/.pyenv/versions/3.10.4/bin/python scripts/_debug/_smoke_forward_pair.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_pair  # noqa: E402
from models.model_depth import CalibNetDepth                              # noqa: E402

CACHE_DIR = '/mnt/fsx/tmp/hfunaya/e2e_calib_cache/pandaset_v3_full'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    torch.manual_seed(0)
    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE_DIR, split='train',
        img_size=128, min_crop_px=128, max_crop_px=256,
        max_offset_m=0.20, max_rot_deg=1.0,
        pose_frame='orig', oversample=1, frame_stride=1,
        grid_n=16, k_per_cell=8,
        preload=False,
        pair_mode=True, pair_stride=1,
    )
    samples = [ds[k] for k in range(2)]
    batch = collate_pair(samples)
    A = batch['A']; B = batch['B']; dpose = batch['dpose_AB'].to(DEVICE)
    # collate_full unpack: imgs, true, dist, pad, vfps, b_uvds, b_valids, pert, pts_orig, duv_orig, K_orig, cs, d1
    img_A, true_A, dist_A, pad_A, vfp_A, b_uvd_A, b_val_A, *_ = A
    img_B, true_B, dist_B, pad_B, vfp_B, b_uvd_B, b_val_B, *_ = B
    img_A = img_A.to(DEVICE).float() / 255.0
    img_B = img_B.to(DEVICE).float() / 255.0
    dist_A = dist_A.to(DEVICE)[..., :4]   # (u, v, d, intensity) for use_intensity=True
    dist_B = dist_B.to(DEVICE)[..., :4]
    pad_A = pad_A.to(DEVICE); pad_B = pad_B.to(DEVICE)
    vfp_A = vfp_A.to(DEVICE); vfp_B = vfp_B.to(DEVICE)
    b_uvd_A = b_uvd_A.to(DEVICE); b_val_A = b_val_A.to(DEVICE)
    b_uvd_B = b_uvd_B.to(DEVICE); b_val_B = b_val_B.to(DEVICE)
    print(f'imgs: A={tuple(img_A.shape)}, B={tuple(img_B.shape)}, '
          f'dist_A={tuple(dist_A.shape)}, vfp={vfp_A.tolist()}')

    model = CalibNetDepth(
        img_size=128, in_channels=3, n_layers=2,
        use_intensity=True, use_frustum=True, frustum_grid_n=16,
        use_pose_emb=True,
    ).to(DEVICE)
    model.eval()

    with torch.no_grad():
        pred_A, pred_B, pred_A_to_B = model.forward_pair(
            image_A=img_A, distorted_uvd_A=dist_A,
            image_B=img_B, distorted_uvd_B=dist_B,
            dpose_AB=dpose,
            key_padding_mask_A=pad_A, key_padding_mask_B=pad_B,
            vfp_A=vfp_A, vfp_B=vfp_B,
            bucket_uvd_A=b_uvd_A, bucket_valid_A=b_val_A,
            bucket_uvd_B=b_uvd_B, bucket_valid_B=b_val_B,
            delta1_A=None, delta1_B=None,
        )
    print(f'pred_A={tuple(pred_A.shape)}, pred_B={tuple(pred_B.shape)}, '
          f'pred_A_to_B={tuple(pred_A_to_B.shape)}')
    # pred_A and pred_A_to_B share Q from A (N_A); pred_B has N_B (different
    # padding per frame). All three have a feature dim of 5.
    assert pred_A.shape[-1] == pred_B.shape[-1] == pred_A_to_B.shape[-1] == 5
    assert pred_A.shape[:2] == pred_A_to_B.shape[:2]

    # Gate: dpose=0 → pred_A_to_B reproduces forward(B, A, pose_emb=0).
    with torch.no_grad():
        ref = model(img_B, dist_A, key_padding_mask=pad_A, vfp=vfp_B,
                    bucket_uvd=b_uvd_B, bucket_valid=b_val_B,
                    pose_emb_se3=torch.zeros_like(dpose))
        zero_pred = model.forward_pair(
            image_A=img_A, distorted_uvd_A=dist_A,
            image_B=img_B, distorted_uvd_B=dist_B,
            dpose_AB=torch.zeros_like(dpose),
            key_padding_mask_A=pad_A, key_padding_mask_B=pad_B,
            vfp_A=vfp_A, vfp_B=vfp_B,
            bucket_uvd_A=b_uvd_A, bucket_valid_A=b_val_A,
            bucket_uvd_B=b_uvd_B, bucket_valid_B=b_val_B,
            delta1_A=None, delta1_B=None,
        )[2]
    diff = (zero_pred - ref).abs().max().item()
    print(f'gate max|pred_A_to_B(dpose=0) - forward(B,A,pose=0)| = {diff:.2e}')
    assert diff < 1e-5, f'gate FAILED: diff={diff}'

    # Backward sanity: any param gets a grad after a tiny dummy MSE.
    model.train()
    pred_A, pred_B, pred_A_to_B = model.forward_pair(
        image_A=img_A, distorted_uvd_A=dist_A,
        image_B=img_B, distorted_uvd_B=dist_B,
        dpose_AB=dpose,
        key_padding_mask_A=pad_A, key_padding_mask_B=pad_B,
        vfp_A=vfp_A, vfp_B=vfp_B,
        bucket_uvd_A=b_uvd_A, bucket_valid_A=b_val_A,
        bucket_uvd_B=b_uvd_B, bucket_valid_B=b_val_B,
    )
    loss = (pred_A.pow(2).sum() + pred_B.pow(2).sum() + pred_A_to_B.pow(2).sum())
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f'backward OK; params with non-zero grad = {n_with_grad}')
    assert n_with_grad > 0
    print('SMOKE OK')


if __name__ == '__main__':
    main()
