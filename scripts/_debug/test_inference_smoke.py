"""Inference smoke / regression test.

Goal: confirm offline inference reproduces train.log's val_nll within a
tolerance, no human eyeballing. Run before pushing any inference change.

Steps per ckpt:
  1. Replay the trainer's val pass on the FIRST 800 inst of the cache via
     PandaSetCalibDatasetFull → DataLoader(collate_full) → model forward
     → gaussian2d_nll. Pass if |nll - train.log_best| ≤ tol.
  2. Run infer_tiles + solve_dofs on one cache tile. Pass if no NaN, σ
     ranges and δ shape are sane.

Usage:
    docker exec caaas python3 /workspace/scripts/_debug/test_inference_smoke.py \\
        --exp km_wv_wm_dgx1_n4_v4_resume --cache /cache/kamikado_v3_tiled
"""
import argparse
import io
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from PIL import Image

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from scripts.inference.infer_calib import load_calib_model
from models.model_cov import gaussian2d_nll


def _trainlog_best_val(exp_dir: Path) -> float | None:
    log = exp_dir / 'train.log'
    if not log.is_file():
        return None
    txt = log.read_text(errors='ignore')
    m = re.findall(r'saved \(val_nll=(\d+\.\d+)\)', txt)
    if not m:
        return None
    return float(m[-1])  # last saved best


def replay_val(exp: str, cache: str, val_size: int = 800) -> float:
    ds_kw = dict(max_offset_m=0.20, max_rot_deg=0.5,
                  min_crop_px=128, max_crop_px=384, oversample=1)
    ds = PandaSetCalibDatasetFull(cache, split='val', **ds_kw)
    val_subset = Subset(ds, list(range(val_size)))
    loader = DataLoader(val_subset, batch_size=64, num_workers=4,
                         collate_fn=collate_full, shuffle=False)
    model = load_calib_model(exp).eval()
    device = torch.device('cuda')
    nll_s = 0.0; mse_s = 0.0; n = 0
    with torch.no_grad():
        for batch in loader:
            imgs, true_uvd, dist_uvd, pad_mask, vfp, b_uvd, b_v = batch[:7]
            imgs = imgs.to(device).float().div_(255.0)
            true_uvd = true_uvd.to(device); dist_uvd = dist_uvd.to(device)
            pad_mask = pad_mask.to(device); vfp = vfp.to(device)
            b_uvd = b_uvd.to(device); b_v = b_v.to(device)
            gt = true_uvd[..., :2] - dist_uvd[..., :2]
            use_intensity = bool(getattr(model, 'use_intensity', False))
            pin = (torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], -1)
                    if use_intensity else dist_uvd[..., :3])
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                params = model(imgs, pin, key_padding_mask=pad_mask, vfp=vfp,
                                bucket_uvd=b_uvd, bucket_valid=b_v)
            valid = ~pad_mask
            if valid.any():
                nll_s += gaussian2d_nll(params[valid], gt[valid]).item()
                mse_s += (params[valid][..., :2].float() - gt[valid]).norm(dim=-1).mean().item()
                n += 1
    return nll_s / max(n, 1), mse_s / max(n, 1)


def smoke_infer_tiles(exp: str, cache: str) -> dict:
    from scripts.ba.ba_multicam_corr import infer_tiles, solve_dofs, _DOF_PRESETS
    ds = PandaSetCalibDatasetFull(cache, split='val',
                                    max_offset_m=0.20, max_rot_deg=0.5,
                                    min_crop_px=128, max_crop_px=384,
                                    oversample=1)
    inst = ds._load_inst(0)
    img = np.asarray(Image.open(io.BytesIO(bytes(inst['jpg_bytes']))).convert('RGB'))
    uv = inst['uv_full'].numpy().astype(np.float32)
    z  = inst['z_cam'].numpy().astype(np.float32)
    intensity = inst['intensity'].numpy().astype(np.float32)
    K = inst['K_full'].numpy().astype(np.float32)
    tu0, tv0 = int(inst.get('tile_u0', 0)), int(inst.get('tile_v0', 0))
    uv = uv - np.array([tu0, tv0], dtype=np.float32)
    K = K.copy(); K[0, 2] -= tu0; K[1, 2] -= tv0
    H, W = img.shape[:2]
    keep = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z > 0)
    uv = uv[keep]; z = z[keep]; intensity = np.clip(intensity[keep], 0, 1).astype(np.float32)

    model = load_calib_model(exp).eval()
    ba_cfg = dict(tile_size=384, model_input_size=128,
                  max_pts_per_tile=256, min_pts_per_tile=8, tile_stride=320)
    res = infer_tiles(model, img, uv, z, K, ba_cfg, torch.device('cuda'),
                       intensity=intensity)
    if res is None:
        return dict(ok=False, reason='infer_tiles returned None')
    uv_pool, par_pool, z_pool = res
    sx, sy = par_pool[:, 2], par_pool[:, 3]
    if not np.all(np.isfinite(par_pool)):
        return dict(ok=False, reason='non-finite in par_pool')
    if sx.min() <= 0 or sy.min() <= 0:
        return dict(ok=False, reason=f'σ ≤ 0 (sx_min={sx.min()}, sy_min={sy.min()})')
    delta = solve_dofs(uv_pool, par_pool, z_pool, K,
                        _DOF_PRESETS['6dof_ext'], damping=1e-3)
    if not np.all(np.isfinite(delta)):
        return dict(ok=False, reason='non-finite in delta')
    return dict(ok=True, n_pool=len(uv_pool),
                sigma_med=(float(np.median(sx)), float(np.median(sy))),
                delta=delta.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--cache', default='/cache/kamikado_v3_tiled')
    ap.add_argument('--val-size', type=int, default=800)
    ap.add_argument('--nll-tol', type=float, default=0.20,
                    help='allowed |offline_nll - train.log_best|')
    args = ap.parse_args()
    exp_dir = REPO_ROOT / 'experiments' / args.exp
    print(f'== {args.exp} ==')
    expected = _trainlog_best_val(exp_dir)
    print(f'  train.log best val_nll: {expected}')

    # Step 1: replay val
    print(f'  → offline replay val (val_size={args.val_size}) ...')
    nll, mse = replay_val(args.exp, args.cache, args.val_size)
    print(f'    offline nll = {nll:+.4f}  mse(=mean px err) = {mse:.3f}')
    diff = abs(nll - expected) if expected is not None else None
    if expected is None:
        print('    [WARN] no train.log best to compare against')
        ok1 = True
    elif diff > args.nll_tol:
        print(f'    [FAIL] |nll - log| = {diff:.4f} > tol {args.nll_tol}')
        ok1 = False
    else:
        print(f'    [PASS] |nll - log| = {diff:.4f} ≤ tol {args.nll_tol}')
        ok1 = True

    # Step 2: infer_tiles smoke
    print(f'  → infer_tiles + solve_dofs on cache[0] ...')
    s = smoke_infer_tiles(args.exp, args.cache)
    if not s['ok']:
        print(f'    [FAIL] {s["reason"]}')
        ok2 = False
    else:
        print(f'    [PASS] n_pool={s["n_pool"]}  σ_med={s["sigma_med"]}  '
              f'δ={[round(x,3) for x in s["delta"]]}')
        ok2 = True

    if ok1 and ok2:
        print('\nALL PASS')
        sys.exit(0)
    print('\nFAIL')
    sys.exit(1)


if __name__ == '__main__':
    main()
