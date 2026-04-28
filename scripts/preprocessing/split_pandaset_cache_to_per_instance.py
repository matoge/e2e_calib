"""Split monolithic PandaSet calib cache (28GB pickle) → per-instance .pt files.

Produces:
  /mnt/nvme6t/e2e_calib_cache/pandaset_mc_s64_lazy/
    meta.pt              # tiny: {'train': [{idx, file}, ...], 'val': [...]}
    inst/000000.pt       # one file per instance (~170KB)
    inst/000001.pt
    ...

Use with PandaSetCalibDatasetLazy: __init__ loads only meta.pt, __getitem__
torch.load's the per-instance file. Memory stays at <1GB regardless of cache
size; OS page cache handles repeated reads.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import argparse, time
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--src', default='/mnt/nvme6t/e2e_calib_cache/pandaset_mc_s64_cache.pt')
ap.add_argument('--dst', default='/mnt/nvme6t/e2e_calib_cache/pandaset_mc_s64_lazy')
args = ap.parse_args()

src = Path(args.src)
dst = Path(args.dst)
inst_dir = dst / 'inst'
inst_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading {src} ({src.stat().st_size / 1e9:.1f}GB)...", flush=True)
t0 = time.time()
data = torch.load(src, weights_only=False)
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

meta = {'train': [], 'val': []}
gid = 0
for split in ('train', 'val'):
    insts = data[split]
    print(f"  splitting {split} ({len(insts)} instances)...", flush=True)
    for i, inst in enumerate(insts):
        fname = f"{gid:07d}.pt"
        out = inst_dir / fname
        torch.save(inst, out)
        meta[split].append(fname)
        gid += 1
        if (i + 1) % 5000 == 0:
            print(f"    {split} {i+1}/{len(insts)}", flush=True)

torch.save(meta, dst / 'meta.pt')
print(f"Done. meta train={len(meta['train'])} val={len(meta['val'])} → {dst}", flush=True)
