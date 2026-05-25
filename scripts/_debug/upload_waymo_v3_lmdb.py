"""Upload waymo_v3_tiled_i (LMDB + meta.pt only, no inst/) as a ClearML
Dataset under e2e_calib/data so the blog can quote a stable id.

Cache:    /home/hfunaya/cache_v4/waymo_v3_tiled_i
  data.lmdb/data.mdb  ~234 GB
  data.lmdb/lock.mdb  ~8 KB
  meta.pt             ~37 MB
  inst/               EXCLUDED (LMDB superset; consumers don't need it)
"""
from __future__ import annotations
import sys
from pathlib import Path

from clearml import Dataset

PROJECT  = 'e2e_calib/data'
DS_NAME  = 'waymo_v3_tiled_i'
DESC     = ('Waymo Open v1.4.x front+side cams tiled to 512² LMDB '
            '(intensity 4-ch). data.lmdb + meta.pt only — inst/ omitted '
            '(LMDB superset). Used by km_wv_wm n4 img128 σ-head training.')
SRC      = Path('/home/hfunaya/cache_v4/waymo_v3_tiled_i')


def main():
    if not SRC.is_dir():
        sys.exit(f'src not found: {SRC}')
    if not (SRC / 'data.lmdb' / 'data.mdb').is_file():
        sys.exit(f'no data.lmdb/data.mdb under {SRC}')
    if not (SRC / 'meta.pt').is_file():
        sys.exit(f'no meta.pt under {SRC}')

    ds = Dataset.create(
        dataset_name=DS_NAME,
        dataset_project=PROJECT,
        description=DESC,
    )
    ds.add_files(str(SRC / 'data.lmdb' / 'data.mdb'), verbose=False)
    ds.add_files(str(SRC / 'data.lmdb' / 'lock.mdb'), verbose=False)
    ds.add_files(str(SRC / 'meta.pt'),                 verbose=False)
    print(f'[{DS_NAME}] uploading 234 GB to ClearML...', flush=True)
    ds.upload(show_progress=True, verbose=True)
    ds.finalize()
    print(f'[{DS_NAME}] DONE  id={ds.id}', flush=True)
    print(f'  blog table id  →  {ds.id}')


if __name__ == '__main__':
    main()
