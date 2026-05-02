"""Upload tile caches (PS / NS / Waymo) as ClearML Datasets.

Each cache becomes one ClearML Dataset under project `e2e_calib/data`.
Remote workers grab a local copy via `Dataset.get(name=...).get_local_copy()`.

Usage:
  python scripts/preprocessing/upload_tile_caches.py            # all available
  python scripts/preprocessing/upload_tile_caches.py --only ps  # one
"""
import argparse
from pathlib import Path
from clearml import Dataset

CACHES = [
    ('pandaset_v3_tiled', '/mnt/nvme6t/e2e_calib_cache/pandaset_v3_tiled'),
    ('nuscenes_v3_tiled', '/mnt/nvme6t/e2e_calib_cache/nuscenes_v3_tiled'),
    ('waymo_v3_tiled',    '/mnt/nvme6t/e2e_calib_cache/waymo_v3_tiled'),
]


def upload(name: str, path: str, project: str = 'e2e_calib/data',
            tags: list[str] | None = None):
    p = Path(path)
    if not p.exists():
        print(f'[skip] {name}: {path} not found'); return
    print(f'\n=== {name}: {path} ===', flush=True)
    ds = Dataset.create(dataset_name=name, dataset_project=project,
                         dataset_tags=(tags or ['tile-mode', 'v3']))
    ds.add_files(str(p), local_base_folder=str(p.parent))
    ds.upload(verbose=True, show_progress=True)
    ds.finalize(verbose=True)
    print(f'   id={ds.id}  url=https://clearml.budda.site/datasets/simple/{project}/experiments/{ds.id}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['ps', 'ns', 'wm'], help='upload only one')
    ap.add_argument('--project', default='e2e_calib/data')
    args = ap.parse_args()
    keys = {'ps': 'pandaset', 'ns': 'nuscenes', 'wm': 'waymo'}
    for name, path in CACHES:
        if args.only and keys[args.only] not in name:
            continue
        upload(name, path, project=args.project)


if __name__ == '__main__':
    main()
