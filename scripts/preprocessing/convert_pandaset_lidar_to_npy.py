"""Convert PandaSet lidar/*.pkl(.gz) → lidar_npy/*.npy (xyz float32, Pandar64 only).

The pandas DataFrame in the .pkl carries (x,y,z,d,...) columns; runtime
loading + filter + .values copy was profiled at ~60 ms/call (57% of total
data-loading cost). A flat xyz npy file mmaps in <1 ms.

Output:
  /mnt/nvme6t/<dataset_root>/<scene>/lidar_npy/{fi:02d}.npy   (N, 3) float32
"""
from __future__ import annotations
import argparse, gzip, pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np


def _load_pickle(path: Path):
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    return pickle.load(open(path, 'rb'))


def process_scene(scene_dir: Path):
    ld_in  = scene_dir / 'lidar'
    ld_out = scene_dir / 'lidar_npy'
    if not ld_in.is_dir():
        return f'[skip-no-lidar] {scene_dir.name}'
    ld_out.mkdir(exist_ok=True)
    n_done = 0
    files = sorted(list(ld_in.glob('*.pkl')) + list(ld_in.glob('*.pkl.gz')))
    # dedupe by stem so we don't redo .pkl twice when both exist
    by_stem = {}
    for p in files:
        stem = p.name.replace('.pkl.gz', '').replace('.pkl', '')
        if stem not in by_stem or p.suffix == '.pkl':
            by_stem[stem] = p
    for stem, p in sorted(by_stem.items()):
        out = ld_out / f'{stem}.npy'
        if out.exists():
            n_done += 1; continue
        df = _load_pickle(p)
        if 'd' in df.columns:
            df = df[df['d'] == 0]
        xyz = df[['x', 'y', 'z']].values.astype(np.float32)
        np.save(out, xyz, allow_pickle=False)
        n_done += 1
    return f'[ok] {scene_dir.name}  frames={n_done}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/mnt/nvme6t/pandaset')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    root = Path(args.root)
    scenes = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f'processing {len(scenes)} scenes with {args.workers} workers')
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_scene, sc) for sc in scenes]
        for i, f in enumerate(as_completed(futs)):
            print(f'  [{i+1}/{len(futs)}] {f.result()}', flush=True)


if __name__ == '__main__':
    main()
