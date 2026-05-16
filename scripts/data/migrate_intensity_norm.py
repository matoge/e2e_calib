"""Migrate an existing V3 LMDB cache: rewrite per-inst intensity bytes
in-place so they're already normalised to [0,1].

Why: the new CalibFrame contract says cache stores normalised intensity,
adapter applies the per-sensor divisor once, dataset reader does NOT
re-normalise. We don't want to rebuild the cache from raw just to flip
intensity scaling, so this script walks every existing inst blob,
patches the intensity body slice with `np.clip(intensity / divisor, 0, 1)`,
and writes the patched blob back under the same key.

Usage:
    python -m scripts.data.migrate_intensity_norm \\
        --cache /home/hfunaya/cache/kamikado_v3_tiled \\
        --divisor 128 --in-place

  Or write a copy:
    python -m scripts.data.migrate_intensity_norm \\
        --cache /home/hfunaya/cache/kamikado_v3_tiled \\
        --out   /raid/home/hfunaya/cache_v4/kamikado_v3_tiled \\
        --divisor 128
"""
import argparse
import pickle
import shutil
import struct
import sys
from pathlib import Path

import lmdb
import numpy as np


HDR_LEN_FMT = '<Q'
HDR_LEN_SIZE = struct.calcsize(HDR_LEN_FMT)


def _patch_intensity(blob: bytes, divisor: float) -> tuple[bytes, bool]:
    """Return (new_blob, changed). Apply  np.clip(arr / divisor, 0, 1)
    unconditionally (so divisor=1 == clip-only, useful for waymo whose
    raw intensity is already mostly in [0,1] but has rare spikes)."""
    hdr_len = struct.unpack(HDR_LEN_FMT, blob[:HDR_LEN_SIZE])[0]
    header = pickle.loads(blob[HDR_LEN_SIZE:HDR_LEN_SIZE + hdr_len])
    offsets = header.get('offsets', {})
    spec = offsets.get('intensity')
    if spec is None:
        return blob, False
    off, length, dtype_str, shape = spec
    body_start = HDR_LEN_SIZE + hdr_len
    arr = np.frombuffer(blob, dtype=np.dtype(dtype_str), count=length // 4,
                         offset=body_start + off).copy()
    new = np.clip(arr / float(divisor), 0.0, 1.0).astype(np.float32)
    if np.array_equal(arr, new):
        return blob, False
    out = bytearray(blob)
    out[body_start + off:body_start + off + length] = new.tobytes()
    return bytes(out), True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True,
                    help='cache root containing data.lmdb/ + meta.pt')
    ap.add_argument('--out', default=None,
                    help='copy → patch → write here. Omit for in-place.')
    ap.add_argument('--divisor', type=float, default=128.0)
    ap.add_argument('--in-place', action='store_true',
                    help='patch the original LMDB directly (writes back).')
    ap.add_argument('--map-size-gb', type=int, default=200)
    args = ap.parse_args()

    src = Path(args.cache)
    if not (src / 'data.lmdb').is_dir():
        sys.exit(f'no data.lmdb in {src}')

    if args.in_place and args.out:
        sys.exit('--in-place and --out are mutually exclusive')
    if not args.in_place and not args.out:
        sys.exit('pass --in-place or --out')

    if args.out:
        dst = Path(args.out)
        if dst.exists():
            sys.exit(f'output exists: {dst} (rm first)')
        print(f'copying {src} → {dst} (this can take a while for large LMDBs)')
        shutil.copytree(src, dst, copy_function=shutil.copy2)
        target_lmdb = dst / 'data.lmdb'
    else:
        target_lmdb = src / 'data.lmdb'

    env = lmdb.open(str(target_lmdb), readonly=False, lock=False, subdir=True,
                     map_size=args.map_size_gb * (1 << 30),
                     writemap=True, sync=False, meminit=False, max_dbs=0)
    # Pass 1: collect all keys (read-only) so we can iterate without
    # keeping a write txn open across commits (LMDB cursors get
    # invalidated when the txn commits).
    keys: list[bytes] = []
    with env.begin() as t:
        for k, _ in t.cursor():
            keys.append(bytes(k))
    print(f'  enumerated {len(keys)} keys; patching...')

    n_total = 0; n_patched = 0; n_skipped = 0
    txn = env.begin(write=True)
    for k in keys:
        n_total += 1
        if k.startswith(b'__cubs__/'):
            continue
        v = txn.get(k)
        if v is None:
            continue
        new_blob, changed = _patch_intensity(bytes(v), args.divisor)
        if changed:
            txn.put(k, new_blob)
            n_patched += 1
        else:
            n_skipped += 1
        if (n_total % 5000) == 0:
            txn.commit(); txn = env.begin(write=True)
            print(f'  {n_total} processed  ({n_patched} patched, {n_skipped} skipped)',
                  flush=True)
    txn.commit(); env.sync(); env.close()
    print(f'done: total={n_total}  patched={n_patched}  '
           f'already-normalised/no-intensity={n_skipped}')


if __name__ == '__main__':
    main()
