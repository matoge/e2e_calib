"""Upload V3 LMDB tile caches as ClearML Datasets (under project e2e_calib/data).

One dataset per cache; version tag = date (YYYYMMDD) so the ClearML UI sorts
runs chronologically. The dataset content is just the two LMDB files
(`data.lmdb/data.mdb` + `data.lmdb/lock.mdb`) plus `meta.pt` — the .pt
inst/*.pt directory is intentionally NOT uploaded (LMDB superset, and
deletion-safe after consumers switch over).

Usage:
    # Upload everything missing
    python scripts/clearml/upload_lmdb_datasets.py
    # Or filter:
    python scripts/clearml/upload_lmdb_datasets.py --cache pandaset_v3_tiled
"""
import argparse, datetime, sys
from pathlib import Path
from clearml import Dataset

CACHE_ROOT = Path('/mnt/nvme6t/e2e_calib_cache')
PROJECT = 'e2e_calib/data'

# (cache_dir_name, ClearML dataset name, description tag)
TARGETS = [
    ('pandaset_v3_tiled',                 'ps_v3_tiled_front_lmdb',          'PS front-cam only, 70399 train + 12000 val tiles, 11 GB LMDB'),
    ('pandaset_v3_tiled_multicam_corr',   'ps_v3_tiled_multicam_corr_lmdb',  'PS 6-cam, BA-corrected extrinsics, 128k train + 23k val, 21 GB'),
    ('nuscenes_v3_tiled_6cam',            'ns_v3_tiled_6cam_lmdb',           'nuScenes 6-cam, 1.6M tiles, 107 GB'),
    ('waymo_v3_tiled',                    'wm_v3_tiled_1cam_lmdb',           'Waymo front-cam via lcp, 638k tiles, 76 GB'),
    ('waymo_v3_tiled_5cam',               'wm_v3_tiled_5cam_lmdb',           'Waymo 5-cam, post-rebuild meta.pt, ~200 GB'),
    ('zod_v3_tiled_clean',                'zod_v3_tiled_clean8k_lmdb',       'ZOD curated 8975 frames (yaw<1 / accel<2 / spd 5-20), 40 GB'),
]


def upload_one(cache_name: str, ds_name: str, desc: str, *, version_tag: str):
    cache = CACHE_ROOT / cache_name
    lmdb_dir = cache / 'data.lmdb'
    meta_pt  = cache / 'meta.pt'
    if not (lmdb_dir.is_dir() and meta_pt.is_file()):
        print(f'[SKIP] {cache_name}: missing data.lmdb dir or meta.pt')
        return None

    # Date-stamped ClearML dataset version, e.g. ps_v3_tiled_front_lmdb @ 20260513
    ds = Dataset.create(
        dataset_name=ds_name,
        dataset_project=PROJECT,
        dataset_version=version_tag,
        description=desc,
    )
    # Add the two LMDB env files + the small meta.pt index. Use add_files
    # (upload-pending state); finalize() commits + actually pushes to the
    # fileserver.
    ds.add_files(str(lmdb_dir / 'data.mdb'))
    ds.add_files(str(lmdb_dir / 'lock.mdb'))
    ds.add_files(str(meta_pt))
    print(f'[{cache_name}] adding files done; uploading...')
    ds.upload(show_progress=True)
    ds.finalize()
    print(f'[{cache_name}] DONE  id={ds.id}  url={ds.get_logger().get_default_upload_destination()}')
    return ds.id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=None,
                    help='filter to one cache (e.g. pandaset_v3_tiled). default: all')
    ap.add_argument('--version', default=datetime.date.today().strftime('%Y%m%d'),
                    help='dataset version tag (default = today YYYYMMDD)')
    args = ap.parse_args()

    todo = [t for t in TARGETS if args.cache is None or t[0] == args.cache]
    if not todo:
        print(f'no targets match --cache={args.cache}', file=sys.stderr)
        sys.exit(2)

    for cache, name, desc in todo:
        upload_one(cache, name, desc, version_tag=args.version)


if __name__ == '__main__':
    main()
