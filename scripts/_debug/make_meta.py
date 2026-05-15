"""Reconstruct meta.pt for a tile cache that lost it (build was killed mid-flush).
Takes inst/*.pt fnames, splits 85/15 by hash so train/val is deterministic.
"""
import argparse, hashlib, sys
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--cache-dir', required=True)
ap.add_argument('--val-frac', type=float, default=0.15)
args = ap.parse_args()

cache = Path(args.cache_dir)
inst = cache / 'inst'
meta_path = cache / 'meta.pt'

fnames = sorted([p.name for p in inst.iterdir() if p.suffix == '.pt'])
print(f'inst fnames: {len(fnames)}', flush=True)

# Stable split via hash mod
def is_val(fn: str) -> bool:
    h = int(hashlib.md5(fn.encode()).hexdigest()[:8], 16)
    return (h % 1000) < int(args.val_frac * 1000)

val = [fn for fn in fnames if is_val(fn)]
train = [fn for fn in fnames if not is_val(fn)]
print(f'split: train={len(train)} val={len(val)}', flush=True)

torch.save({'train': train, 'val': val}, meta_path)
print(f'saved {meta_path}', flush=True)
