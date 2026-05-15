"""Upload tile caches to DGX2 self-hosted ClearML as Datasets, so DGX1/3/4
trainers can do `Dataset.get(name='kamikado_v3_tile_i')` + get_local_copy()
without manual rsync.
"""
import argparse, sys
from pathlib import Path
from clearml import Dataset, Task

PROJECT = 'e2e_calib/datasets'

DATASETS = [
    ('kamikado_v3_tile_i',  '/home/hfunaya/cache/kamikado_v3_tiled'),
    ('woven_v3_tile_i',     '/home/hfunaya/cache/woven_v3_tile'),
    ('waymo_v3_tile_i',     '/home/hfunaya/clearml/data/cache/waymo_v3_tiled_i'),
    # ZOD は再ビルド完了後に追加
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', help='Upload only this name (default: all)')
    ap.add_argument('--include-lmdb', action='store_true',
                    help='Include data.lmdb (huge, ~50-500 GB). Off by default; '
                         'only upload inst/ + meta.pt and let downloader rebuild lmdb.')
    args = ap.parse_args()

    for name, path in DATASETS:
        if args.name and args.name != name:
            continue
        p = Path(path)
        if not p.is_dir():
            print(f'[skip] {name}: {path} not found', flush=True)
            continue
        print(f'\n=== {name}  src={path} ===', flush=True)
        ds = Dataset.create(dataset_name=name, dataset_project=PROJECT,
                            description=f'tile cache (intensity, 4ch). src: {path}')
        # Add inst/ and meta.pt
        files = []
        files.append(str(p / 'meta.pt'))
        if args.include_lmdb and (p / 'data.lmdb').is_dir():
            files.append(str(p / 'data.lmdb'))
        # inst/ as wildcard
        ds.add_files(str(p / 'inst'), wildcard='*.pt', verbose=False)
        for f in files:
            ds.add_files(f, verbose=False)
        ds.upload(show_progress=True, verbose=True)
        ds.finalize()
        print(f'[done] {name}: id={ds.id}', flush=True)

if __name__ == '__main__':
    main()
