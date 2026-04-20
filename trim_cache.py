"""Trim a multicrop cache to a random sample of N instances.

Usage: python trim_cache.py <cache.pt> <N> [--flat]
    flat = cache is a flat list (waymo); default = dict with train/val keys.
"""
import sys, random, torch
from pathlib import Path

def main():
    path = Path(sys.argv[1])
    N    = int(sys.argv[2])
    flat = '--flat' in sys.argv
    random.seed(42)

    print(f"loading {path}...", flush=True)
    data = torch.load(path, weights_only=False)
    if flat:
        random.shuffle(data)
        data = data[:N]
        out  = data
    else:
        tr = data['train']; va = data['val']
        ratio = len(tr) / max(1, (len(tr)+len(va)))
        n_tr = int(N * ratio); n_va = N - n_tr
        random.shuffle(tr); random.shuffle(va)
        out = {'train': tr[:n_tr], 'val': va[:n_va]}
        print(f"  kept train={len(out['train'])} val={len(out['val'])}", flush=True)
    print(f"saving {path}...", flush=True)
    torch.save(out, path)
    print(f"done. new size:")
    import os
    print(f"  {os.path.getsize(path)/1e9:.2f}GB", flush=True)

if __name__ == '__main__':
    main()
