"""Offline val eval: load a ckpt, run model on the val split, dump
mean nll / mse stats. Match these against `train.log`'s val nll line for
the same epoch / ckpt to confirm the offline path is equivalent to
training-time eval.

Usage:
    docker exec ... python3 scripts/_debug/eval_val_offline.py \\
        --exp km_wv_wm_dgx2_n2_v4 \\
        --val-size 800
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader, ConcatDataset

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from scripts.inference.infer_calib import load_calib_model
from scripts.inference.infer_pipeline import _load_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='km_wv_wm_dgx2_n2_v4')
    ap.add_argument('--val-size', type=int, default=800)
    ap.add_argument('--cache',
                    default='/cache/kamikado_v3_tiled,/cache/woven_v3_tile,/cache/waymo_v3_tiled_i')
    args = ap.parse_args()

    c = _load_cfg(args.exp)
    model = load_calib_model(args.exp).eval()
    print(f'ckpt: {args.exp}')

    ds_kw = dict(
        max_offset_m = c.get('max_offset_m', 0.20),
        max_rot_deg  = c.get('max_rot_deg', 0.5),
        min_crop_px  = c.get('min_crop_px', 128),
        max_crop_px  = c.get('max_crop_px', 384),
        grid_n       = c.get('grid_n', 16),
        oversample   = 1,
    )
    print(f'ds_kw = {ds_kw}')
    cache_paths = [p.strip() for p in args.cache.split(',') if p.strip()]
    parts = []
    for cp in cache_paths:
        kw = dict(ds_kw)
        if 'waymo' in cp.lower(): kw['oversample'] = 1
        d = PandaSetCalibDatasetFull(cp, split='val', **kw)
        parts.append(d)
        print(f'  [{cp}] val={len(d)}')
    val_full = ConcatDataset(parts) if len(parts) > 1 else parts[0]
    val_subset = Subset(val_full, list(range(args.val_size)))
    print(f'val total={len(val_full)}  using first {args.val_size}')

    loader = DataLoader(val_subset, batch_size=64, num_workers=4,
                         collate_fn=collate_full, shuffle=False, pin_memory=True)
    device = torch.device('cuda')
    nlls_obj = []; nlls_bg = []; nlls_all = []; mses = []
    n_seen = 0
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
        for batch in loader:
            img, true_uvd, dist_uvd, pad_mask_collate, vfp, bucket_uvd, bucket_valid = batch[:7]
            img = img.to(device).float().div_(255.0)
            true_uvd = true_uvd.to(device); dist_uvd = dist_uvd.to(device)
            vfp = vfp.to(device); bucket_uvd = bucket_uvd.to(device)
            bucket_valid = bucket_valid.to(device)
            valid = ~((dist_uvd[..., 0] == 0) & (dist_uvd[..., 1] == 0))
            pad_mask = ~valid
            use_intensity = bool(getattr(model, 'use_intensity', False))
            point_in = (torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
                          if use_intensity else dist_uvd[..., :3])
            out = model(img, point_in, key_padding_mask=pad_mask, vfp=vfp,
                         bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
            per_pt = out[0] if isinstance(out, tuple) else out
            # NLL over Gaussian2D in tile-local px (matches training loss).
            from scripts.training.train_ps_v3_ddp import gaussian2d_nll
            res_uv = (true_uvd[..., :2] - dist_uvd[..., :2])  # the model's target Δ
            log_sx, log_sy, rho = per_pt[..., 2], per_pt[..., 3], torch.tanh(per_pt[..., 4])
            mu = per_pt[..., :2]
            # nll formula in train: -log p(res | mu, σ, ρ)
            sx = torch.exp(log_sx); sy = torch.exp(log_sy)
            d = res_uv - mu
            zx = d[..., 0] / sx
            zy = d[..., 1] / sy
            r2 = zx * zx + zy * zy - 2 * rho * zx * zy
            denom = 1 - rho * rho
            nll_pt = 0.5 * r2 / denom + log_sx + log_sy + 0.5 * torch.log(1 - rho * rho)
            mse_pt = (d ** 2).sum(-1)
            is_obj_b = (true_uvd[..., 3] > 0.5) & valid
            is_bg_b = (~is_obj_b) & valid
            if valid.any():
                nlls_all.append(nll_pt[valid].cpu().float().numpy())
                mses.append(mse_pt[valid].cpu().float().numpy())
            if is_obj_b.any():
                nlls_obj.append(nll_pt[is_obj_b].cpu().float().numpy())
            if is_bg_b.any():
                nlls_bg.append(nll_pt[is_bg_b].cpu().float().numpy())
            n_seen += img.shape[0]
            if n_seen >= args.val_size: break
    all_nll = np.concatenate(nlls_all) if nlls_all else np.array([0.0])
    all_obj = np.concatenate(nlls_obj) if nlls_obj else np.array([0.0])
    all_bg  = np.concatenate(nlls_bg)  if nlls_bg  else np.array([0.0])
    all_mse = np.concatenate(mses) if mses else np.array([0.0])
    print(f'\noffline val on {n_seen} batches:')
    print(f'  nll all = {all_nll.mean():+.4f}')
    print(f'  nll obj = {all_obj.mean():+.4f}')
    print(f'  nll bg  = {all_bg.mean():+.4f}')
    print(f'  mse     = {all_mse.mean():.3f}')


if __name__ == '__main__':
    main()
