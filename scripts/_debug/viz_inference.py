"""Render N samples from the val cache through the trainer's exact path
(__getitem__ → model forward) and dump PNGs with input/GT/pred points
overlaid on the tile image. So we can SEE what the model is doing.

Output: experiments/{exp}/_vis_smoke/idx{idx:05d}.png

Usage:
    docker exec caaas python3 /workspace/scripts/_debug/viz_inference.py \\
        --exp km_wv_wm_dgx1_n4_v4_resume --n 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.inference.infer_pipeline import infer_one, make_ds, render_red_to_green
from scripts.inference.infer_calib import load_calib_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='km_wv_wm_dgx1_n4_v4_resume')
    ap.add_argument('--cache', default='/cache/kamikado_v3_tiled')
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else (REPO_ROOT / 'experiments' / args.exp /
                                              '_vis_smoke')
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob('idx*.png'):
        old.unlink()

    ds, c = make_ds(args.exp, args.cache, split='val', oversample=1)
    model = load_calib_model(args.exp).eval()
    print(f'cache: {args.cache}  n_val={len(ds)}  ckpt={args.exp}')

    for i in range(args.n):
        idx = i * (len(ds) // max(args.n, 1))
        r = infer_one(model, ds, idx, seed=idx)
        v = r['valid']
        n_v = int(v.sum())
        if n_v < 8:
            print(f'  idx={idx}  N_valid={n_v}  (skip — too few)')
            continue
        err_pre  = float(np.linalg.norm(r['hyp_uv'][v]  - r['true_uv'][v], axis=1).mean())
        err_post = float(np.linalg.norm(r['pred_uv'][v] - r['true_uv'][v], axis=1).mean())
        title = f'  idx={idx}  N={n_v}  err_pre={err_pre:.2f}px  err_post={err_post:.2f}px'
        png = out / f'idx{idx:05d}.png'
        render_red_to_green(r, png, top_k=-1, title_extra=title)
        print(f'  saved {png}  | {title}')

    print(f'\nout dir: {out}')


if __name__ == '__main__':
    main()
