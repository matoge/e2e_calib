"""Verify that an intensity-migrated cache differs from the source ONLY
in the per-tile intensity field.

For every shared LMDB key:
  - parse header → assert offsets dict identical (so all other arrays
    live at the same body byte ranges)
  - byte-compare body slice for every field EXCEPT intensity
  - assert new intensity ∈ [0,1] and old intensity / divisor matches
    new intensity to within fp32 tolerance.

Then runs the model from one ckpt on a handful of (scene, frame, tile)
samples through both caches (推論での比較は infer_tiles を畳んだ 2026-10-07 に削除), and reports the par delta
(should be small — only intensity changed in the input).

Usage:
    python -m scripts.data.verify_intensity_migration \\
        --src /home/hfunaya/cache/kamikado_v3_tiled \\
        --new /raid/home/hfunaya/cache_v4/kamikado_v3_tiled \\
        --divisor 128 \\
        --exp km_wv_wm_dgx2_n2_img128_v2 \\
        --n-spotcheck 5
"""
import argparse
import io
import pickle
import struct
import sys
from pathlib import Path

import numpy as np
import lmdb

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


HDR_LEN_FMT = '<Q'
HDR_LEN_SIZE = struct.calcsize(HDR_LEN_FMT)


def _split(blob: bytes):
    hdr_len = struct.unpack(HDR_LEN_FMT, blob[:HDR_LEN_SIZE])[0]
    header = pickle.loads(blob[HDR_LEN_SIZE:HDR_LEN_SIZE + hdr_len])
    body = blob[HDR_LEN_SIZE + hdr_len:]
    return header, body


def cmd_compare(src: Path, new: Path, divisor: float, max_keys: int):
    """Walk every shared key, compare bodies field-by-field."""
    s_env = lmdb.open(str(src / 'data.lmdb'), readonly=True, lock=False, subdir=True)
    n_env = lmdb.open(str(new / 'data.lmdb'), readonly=True, lock=False, subdir=True)
    n_total = 0
    n_ok = 0
    n_diff_intensity = 0
    n_diff_other = 0
    n_missing_new = 0
    err_intensity_max = 0.0
    with s_env.begin() as st, n_env.begin() as nt:
        for k, sv in st.cursor():
            if k.startswith(b'__cubs__/'):
                continue
            n_total += 1
            if max_keys and n_total > max_keys:
                break
            nv = nt.get(k)
            if nv is None:
                n_missing_new += 1
                continue
            sh, sb = _split(bytes(sv))
            nh, nb = _split(bytes(nv))
            if sh.get('offsets') != nh.get('offsets'):
                n_diff_other += 1
                if n_diff_other <= 2:
                    print(f'  OFFSETS DIFFER key={k.decode()}')
                continue
            offsets = sh['offsets']
            # Check every field
            field_diff = []
            for name, (off, length, dtype_str, shape) in offsets.items():
                if sb[off:off + length] != nb[off:off + length]:
                    field_diff.append(name)
            if field_diff == ['intensity']:
                # Confirm new[i] == clip(old[i] / divisor, 0, 1)
                off, length, dtype_str, shape = offsets['intensity']
                old = np.frombuffer(sb, dtype=np.dtype(dtype_str),
                                     count=length // 4, offset=off)
                cur = np.frombuffer(nb, dtype=np.dtype(dtype_str),
                                     count=length // 4, offset=off)
                expected = np.clip(old.astype(np.float32) / divisor, 0.0, 1.0)
                err = float(np.abs(cur - expected).max()) if cur.size else 0.0
                err_intensity_max = max(err_intensity_max, err)
                if cur.size and (cur.min() < 0 - 1e-6 or cur.max() > 1 + 1e-6):
                    n_diff_other += 1
                    if n_diff_other <= 2:
                        print(f'  intensity OOR key={k.decode()} '
                               f'min={cur.min()} max={cur.max()}')
                    continue
                n_diff_intensity += 1
                n_ok += 1
            elif not field_diff:
                # nothing changed (cache was already normalised, skip path)
                n_ok += 1
            else:
                n_diff_other += 1
                if n_diff_other <= 3:
                    print(f'  OTHER FIELDS DIFFER key={k.decode()} '
                           f'fields={field_diff}')
    s_env.close(); n_env.close()
    print()
    print(f'compared keys: {n_total}')
    print(f'  OK (intensity-only or unchanged): {n_ok}')
    print(f'  diffs in intensity field: {n_diff_intensity}')
    print(f'  diffs in OTHER fields:    {n_diff_other}')
    print(f'  missing in new:           {n_missing_new}')
    print(f'  max abs (cur - clip(old/div,0,1)): {err_intensity_max:.3e}')
    if n_diff_other == 0 and err_intensity_max < 1e-5:
        print('PASS — migration is intensity-only and matches the divisor.')
        return True
    print('FAIL — see above.')
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='legacy cache root')
    ap.add_argument('--new', required=True, help='migrated cache root')
    ap.add_argument('--divisor', type=float, required=True)
    ap.add_argument('--max-keys', type=int, default=0,
                    help='limit comparison to first N keys (0 = all)')
    args = ap.parse_args()

    ok = cmd_compare(Path(args.src), Path(args.new), args.divisor,
                      args.max_keys)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
