"""Rank ZOD train samples by model prediction error and render the top-N worst.

Usage:
    python -m scripts.visualization.vis_top_err \
        --exp zod_20260513_clean8k_pixonly \
        --cache /mnt/nvme6t/e2e_calib_cache/zod_v3_tiled_clean \
        --split train --n 100 --out experiments/<exp>/vis_train_top100

Reuses render_eval_samples for the actual matplotlib layout — only the index
selection differs (sorted by pixel error instead of random + obj filter).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from datasets.pandaset_full import PandaSetCalibDatasetFull
from scripts.inference.infer_calib import load_calib_model
from scripts.visualization.vis_eval import render_eval_samples


def rank_by_error(model, ds, device, batch_size, workers, log=print):
    """One forward pass over the dataset; return (idxs sorted by descending err, errs[idx])."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, persistent_workers=False)
    errs = np.full(len(ds), np.nan, dtype=np.float32)
    n_done = 0
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            img, true_uvd, dist_uvd, vfp, bucket_uvd_v, bucket_valid_v = batch[:6]
            B, Nmax = true_uvd.shape[:2]
            pad = torch.zeros(B, Nmax, dtype=torch.bool, device=device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                _out = model(img.to(device).float().div_(255.0),
                              dist_uvd.to(device)[..., :3],
                              key_padding_mask=pad,
                              vfp=vfp.to(device),
                              bucket_uvd=bucket_uvd_v.to(device),
                              bucket_valid=bucket_valid_v.to(device))
            per_pt = _out[0] if isinstance(_out, tuple) else _out
            p = per_pt.float().cpu().numpy()
            true_uv = true_uvd[..., :2].numpy()
            dist_uv = dist_uvd[..., :2].numpy()
            pred_uv = dist_uv + p[..., :2]
            valid = (true_uvd[..., 3].numpy() >= 0)  # any point is valid; filter by has-pt mask
            # mean per-sample pixel error
            for b in range(B):
                idx_global = n_done + b
                # use all valid pts (not just obj since ZOD is_obj is all False)
                err_b = np.linalg.norm(pred_uv[b] - true_uv[b], axis=-1)
                # mask out padded points: a quick proxy is where dist_uv is exactly 0,0 with no obj signal
                mask = ~((dist_uv[b, :, 0] == 0) & (dist_uv[b, :, 1] == 0))
                if mask.sum() > 0:
                    errs[idx_global] = float(err_b[mask].mean())
            n_done += B
            if (batch_idx % 50) == 0:
                el = time.time() - t0
                log(f'  rank: {n_done}/{len(ds)} ({el:.0f}s, {n_done/max(el,1e-3):.0f} samples/sec)')
    log(f'  rank done: {n_done} samples in {time.time()-t0:.0f}s')
    valid_mask = ~np.isnan(errs)
    order = np.argsort(-errs)  # descending
    order = order[valid_mask[order]]
    return order, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True,
                    help='Experiment name under experiments/ (loads config.py + best_model.pt)')
    ap.add_argument('--cache', required=True, help='LMDB-packed cache dir')
    ap.add_argument('--split', choices=('train', 'val'), default='train')
    ap.add_argument('--n', type=int, default=100, help='top-N to visualize')
    ap.add_argument('--out', default=None, help='output dir (default: experiments/<exp>/vis_<split>_top<N>)')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    log = print
    log(f'load model: {args.exp}')
    model = load_calib_model(args.exp, device=args.device)
    model.eval()

    # config-matched dataset (matches training perturbation range)
    cfg_path = REPO_ROOT / 'experiments' / args.exp / 'config.py'
    import importlib.util
    spec = importlib.util.spec_from_file_location('_cfg', cfg_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    c = mod.CFG

    ds = PandaSetCalibDatasetFull(
        args.cache, split=args.split,
        img_size=c['img_size'],
        min_crop_px=c.get('min_crop_px', 128),
        max_crop_px=c.get('max_crop_px', 384),
        max_offset_m=c.get('max_offset_m', 0.6),
        max_rot_deg=c.get('max_rot_deg', 1.5),
        oversample=1,
    )
    log(f'dataset: {len(ds)} samples ({args.split})')

    log(f'ranking by prediction error...')
    order, errs = rank_by_error(model, ds, args.device, args.batch_size, args.workers, log=log)
    top = order[:args.n].tolist()
    log(f'top-{args.n} err range: {errs[top[0]]:.2f} → {errs[top[-1]]:.2f} px')

    out = Path(args.out) if args.out else (REPO_ROOT / 'experiments' / args.exp / f'vis_{args.split}_top{args.n}')
    saved = render_eval_samples(
        model=model, ds=ds, out_dir=out,
        img_size=c['img_size'], device=args.device,
        n=args.n, sample_idxs=top, epoch=999, log=log,
    )
    log(f'wrote {saved} → {out}')
    # also dump idx + err json for downstream filtering
    import json
    (out / 'rank_meta.json').write_text(json.dumps({
        'idxs': top,
        'errs': [float(errs[i]) for i in top],
        'split': args.split,
        'exp': args.exp,
        'cache': args.cache,
    }, indent=2))


if __name__ == '__main__':
    main()
