"""Write a list of tile inst dicts → V3 LMDB cache (data.lmdb + meta.pt).

This is the dataset-agnostic counterpart of build_*_v3's "loop over tiles
and torch.save them, then convert_tile_cache_to_lmdb pack them" path.
The packing logic itself lives in
scripts.preprocessing.convert_tile_cache_to_lmdb._pack_inst — we import
it directly so the on-disk layout stays byte-identical with anything
the existing reader (datasets.pandaset_full LMDB path) can already
consume.
"""
import struct
from pathlib import Path
from typing import Iterable

import torch
import lmdb

from scripts.preprocessing.convert_tile_cache_to_lmdb import (
    _pack_inst, _pack_cubs, HDR_LEN_FMT, HDR_LEN_SIZE, CUBS_KEY_PREFIX,
)


_DEFAULT_MAP_SIZE_GB = 60


def write_lmdb_cache(out_dir: Path | str,
                      tiles_with_split: Iterable[tuple[str, dict]],
                      *,
                      cam: str = '',
                      is_fisheye: bool = False,
                      map_size_gb: int = _DEFAULT_MAP_SIZE_GB,
                      overwrite: bool = False) -> dict:
    """Write tile inst dicts → V3 LMDB cache compatible with PandaSetCalibDatasetFull.

    Args:
        out_dir: cache root (will contain data.lmdb/ + meta.pt).
        tiles_with_split: iterable of (split, inst_dict) pairs where split is
            'train' or 'val'. inst_dict matches what tile_cutter.frame_to_tiles
            produces (jpg_bytes, pts, uv_full, z_cam, intensity, K_full,
            tile_u0/v0/id, scene, frame, ...).
        cam: dataset-level camera name (e.g. 'fcm') stored in meta.pt.
        is_fisheye: dataset-level fisheye flag stored in meta.pt.
        map_size_gb: LMDB map_size budget. 60 GB covers kamikado.
        overwrite: if True nuke an existing data.lmdb first.

    Returns:
        {'n_train': int, 'n_val': int, 'n_cubs_keys': int,
         'lmdb_path': str, 'meta_path': str}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lmdb_path = out_dir / 'data.lmdb'
    meta_path = out_dir / 'meta.pt'
    if lmdb_path.exists():
        if not overwrite:
            raise FileExistsError(f'{lmdb_path} exists; pass overwrite=True')
        import shutil; shutil.rmtree(lmdb_path)

    env = lmdb.open(
        str(lmdb_path),
        map_size=map_size_gb * (1 << 30),
        subdir=True, writemap=True, sync=False, meminit=False, max_dbs=0,
    )

    train_fnames: list[str] = []
    val_fnames: list[str] = []
    seen_cubs: set[tuple[str, int]] = set()
    n_cubs_written = 0

    txn = env.begin(write=True)
    n_written = 0
    for split, inst in tiles_with_split:
        scene = str(inst.get('scene', ''))
        frame = int(inst.get('frame', -1))
        tile_id = int(inst.get('tile_id', 0))
        # Match the legacy fname format <gid:08d>_t<tile_id>.pt that the
        # existing reader / meta.pt indexes on. We use a content-derived
        # gid: short_hash(scene)+frame so the same (scene, frame) always
        # maps to the same prefix across reruns (ClearML caching wants
        # determinism).
        import hashlib
        scene_short = hashlib.md5(scene.encode()).hexdigest()[:4]
        gid = f'{scene_short}{frame:04d}'
        fname = f'{gid}_t{tile_id}.pt'

        blob = _pack_inst(inst)
        txn.put(fname.encode(), blob)
        n_written += 1
        if split == 'train':
            train_fnames.append(fname)
        else:
            val_fnames.append(fname)

        # Cuboids: dedupe by (scene, frame).
        cubs = inst.get('cuboids') or []
        if cubs and (scene, frame) not in seen_cubs:
            cubs_key = f'{CUBS_KEY_PREFIX}{scene}/{frame}'.encode()
            txn.put(cubs_key, _pack_cubs(cubs))
            seen_cubs.add((scene, frame))
            n_cubs_written += 1

        if n_written % 2000 == 0:
            txn.commit(); txn = env.begin(write=True)
    txn.commit()
    env.sync(); env.close()

    meta = {
        'train': sorted(train_fnames),
        'val':   sorted(val_fnames),
        'cam':   cam,
        'is_fisheye': bool(is_fisheye),
    }
    torch.save(meta, meta_path)
    return dict(
        n_train=len(train_fnames),
        n_val=len(val_fnames),
        n_cubs_keys=n_cubs_written,
        lmdb_path=str(lmdb_path),
        meta_path=str(meta_path),
    )
