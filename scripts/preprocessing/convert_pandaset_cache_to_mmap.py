"""Convert in-memory PandaSet calib cache (48GB pickle) to disk-backed format.

Splits the monolithic /tmp/pandaset_cache.pt into:
  - meta.pt          : small (~MB) per-instance metadata only
  - images.bin       : concatenated uint8 image bytes, mmap-friendly
  - lidar.bin        : concatenated lidar point bytes, mmap-friendly

Each instance keeps offsets/lengths into the .bin files.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import argparse
import time
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--src', default='/tmp/pandaset_cache.pt')
ap.add_argument('--dst', default='/tmp/pandaset_mmap')
args = ap.parse_args()

src = Path(args.src)
dst = Path(args.dst)
dst.mkdir(parents=True, exist_ok=True)

print(f"Loading {src} ({src.stat().st_size / 1e9:.1f}GB)...", flush=True)
t0 = time.time()
data = torch.load(src, weights_only=False)
print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

img_path  = dst / 'images.bin'
lid_path  = dst / 'lidar.bin'
meta_path = dst / 'meta.pt'

img_off_running = 0
lid_off_running = 0
meta = {}

with open(img_path, 'wb') as f_img, open(lid_path, 'wb') as f_lid:
    for split in ('train', 'val'):
        insts = data[split]
        meta[split] = []
        for i, inst in enumerate(insts):
            new_inst = dict(inst)  # shallow copy

            # img_cache: torch.Tensor uint8 (3, H, W). Save bytes, replace with offset.
            if 'img_cache' in new_inst:
                img = new_inst.pop('img_cache')
                if isinstance(img, torch.Tensor):
                    img = img.numpy()
                arr = np.ascontiguousarray(img.astype(np.uint8))
                f_img.write(arr.tobytes())
                new_inst['_img_offset'] = img_off_running
                new_inst['_img_shape']  = tuple(arr.shape)   # (3, H, W)
                img_off_running += arr.nbytes

            # pts_world: (N, 3) float32. Save as (N, 3) bytes, store offset+N.
            if 'pts_world' in new_inst:
                pts = new_inst.pop('pts_world')
                if isinstance(pts, torch.Tensor):
                    pts = pts.numpy()
                arr = np.ascontiguousarray(pts.astype(np.float32))
                f_lid.write(arr.tobytes())
                new_inst['_lid_offset'] = lid_off_running
                new_inst['_lid_n']      = arr.shape[0]
                lid_off_running += arr.nbytes

            meta[split].append(new_inst)

            if (i + 1) % 1000 == 0:
                print(f"  {split} {i+1}/{len(insts)}  imgB={img_off_running/1e9:.1f}G  lidB={lid_off_running/1e9:.1f}G",
                      flush=True)
        print(f"  {split}: {len(insts)} instances done", flush=True)

print(f"Writing meta...", flush=True)
torch.save(meta, meta_path)

print(f"Done.\n  meta:   {meta_path} ({meta_path.stat().st_size / 1e6:.1f}MB)\n"
      f"  images: {img_path} ({img_path.stat().st_size / 1e9:.1f}GB)\n"
      f"  lidar:  {lid_path} ({lid_path.stat().st_size / 1e9:.1f}GB)", flush=True)
