"""Verify that an intensity-migrated cache differs from the source ONLY
in the per-tile intensity field.

For every shared LMDB key:
  - parse header → assert offsets dict identical (so all other arrays
    live at the same body byte ranges)
  - byte-compare body slice for every field EXCEPT intensity
  - assert new intensity ∈ [0,1] and old intensity / divisor matches
    new intensity to within fp32 tolerance.

Then runs the model from one ckpt on a handful of (scene, frame, tile)
samples through both caches via infer_tiles, and reports the par delta
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


def cmd_spotcheck(src: Path, new: Path, exp: str, n: int, divisor: float):
    """Run model on N tiles. For both old and new cache feed the SAME
    pre-normalised intensity (clip(intensity / divisor, 0, 1)) so the only
    thing being checked is whether the migration touched anything besides
    intensity. par diff should be ~ float32 round-off."""
    import torch
    from PIL import Image
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.inference.infer_calib import load_calib_model
    from scripts.ba.ba_multicam_corr import infer_tiles
    from scripts.inference.infer_pipeline import make_ds

    model = load_calib_model(exp).eval()
    ds_old, c = make_ds(exp, str(src), split='val', oversample=1)
    ds_new, _ = make_ds(exp, str(new), split='val', oversample=1)
    ba_cfg = dict(tile_size=512, model_input_size=c['img_size'],
                  max_pts_per_tile=256, min_pts_per_tile=8, tile_stride=384)

    diffs = []
    for idx in range(min(n, len(ds_old))):
        par_old = par_new = None
        for label, ds, scale_intensity in [
                ('old', ds_old, True),   # raw → /divisor → clip
                ('new', ds_new, False),  # already normalised → clip only
            ]:
            inst = ds._load_inst(idx)
            img = np.asarray(Image.open(io.BytesIO(bytes(inst['jpg_bytes'])))
                              .convert('RGB'))
            uv = inst['uv_full'].numpy().astype(np.float32)
            z  = inst['z_cam'].numpy().astype(np.float32)
            intensity = (inst['intensity'].numpy().astype(np.float32)
                          if 'intensity' in inst else None)
            K = inst['K_full'].numpy().astype(np.float32)
            tu0, tv0 = int(inst.get('tile_u0', 0)), int(inst.get('tile_v0', 0))
            uv = uv - np.array([tu0, tv0], dtype=np.float32)
            K = K.copy(); K[0, 2] -= tu0; K[1, 2] -= tv0
            H, W = img.shape[:2]
            keep = ((uv[:, 0] >= 0) & (uv[:, 0] < W)
                    & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z > 0))
            uv = uv[keep]; z = z[keep]
            if intensity is not None:
                intensity = intensity[keep]
                if scale_intensity:
                    intensity = intensity / divisor
                intensity = np.clip(intensity, 0.0, 1.0).astype(np.float32)
            res = infer_tiles(model, img, uv, z, K, ba_cfg,
                               torch.device('cuda'),
                               intensity=intensity)
            if res is None:
                continue
            if label == 'old':
                par_old = res[1]
            else:
                par_new = res[1]
        if par_old is not None and par_new is not None:
            d = float(np.abs(par_old - par_new).max())
            diffs.append(d)
            print(f'  idx={idx}  par max abs diff: {d:.3e}')
    if diffs:
        print(f'\nSpotcheck: max par diff across {len(diffs)} tiles: '
               f'{max(diffs):.3e}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='legacy cache root')
    ap.add_argument('--new', required=True, help='migrated cache root')
    ap.add_argument('--divisor', type=float, required=True)
    ap.add_argument('--max-keys', type=int, default=0,
                    help='limit comparison to first N keys (0 = all)')
    ap.add_argument('--spotcheck', action='store_true',
                    help='also run inference on a few tiles via both caches')
    ap.add_argument('--exp', default='km_wv_wm_dgx2_n2_img128_v2')
    ap.add_argument('--n-spotcheck', type=int, default=5)
    args = ap.parse_args()

    ok = cmd_compare(Path(args.src), Path(args.new), args.divisor,
                      args.max_keys)
    if args.spotcheck:
        print()
        cmd_spotcheck(Path(args.src), Path(args.new), args.exp,
                       args.n_spotcheck, args.divisor)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
