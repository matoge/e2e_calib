"""Golden test: infer_tiles output must not change across refactors.

Run mode 1 (record):  --record   → snapshots one val frame per dataset to
                                    scripts/_debug/_golden/infer_tiles/<cache>.npz
Run mode 2 (verify):  (default)  → re-runs the same input through current code
                                    and asserts the (uv_pool, par_pool, z_pool)
                                    triple matches the snapshot bit-for-bit.

Use this BEFORE and AFTER any change to infer_tiles / build_window /
projection helper / model loader. If verify passes you can be confident
the refactor is numerically a no-op for tile inference.

Usage (from inside the caaas container; the host python lacks torch):
    docker exec caaas python3 /workspace/scripts/_debug/test_infer_tiles_golden.py --record
    docker exec caaas python3 /workspace/scripts/_debug/test_infer_tiles_golden.py
"""
from __future__ import annotations
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.infer_pipeline import make_ds
from scripts.inference.infer_calib import load_calib_model
from scripts.ba.ba_multicam_corr import infer_tiles


EXP = 'km_wv_wm_dgx2_n2_img128_v2'
CACHES = [
    ('kamikado', '/cache/kamikado_v3_tiled'),
    ('woven',    '/cache/woven_v3_tile'),
    ('waymo',    '/cache/waymo_v3_tiled_i'),
]
GOLDEN_DIR = REPO_ROOT / 'scripts' / '_debug' / '_golden' / 'infer_tiles'
ATOL = 1e-5
RTOL = 0.0


def _load_one(cache_path: str, idx: int = 0):
    """Pull (img, uv_full, z, K, intensity) for cache[idx] in tile-local
    coords — i.e. the same prep CaaaS uses to feed infer_tiles."""
    ds, c = make_ds(EXP, cache_path, split='val', oversample=1)
    inst = ds._load_inst(idx)
    full_jpg = bytes(inst['jpg_bytes'])
    img = np.asarray(Image.open(io.BytesIO(full_jpg)).convert('RGB'))
    uv = inst['uv_full'].numpy().astype(np.float32)
    z  = inst['z_cam'].numpy().astype(np.float32)
    intensity = (inst['intensity'].numpy().astype(np.float32)
                 if 'intensity' in inst else None)
    K = inst['K_full'].numpy().astype(np.float32)
    tu0 = int(inst.get('tile_u0', 0)); tv0 = int(inst.get('tile_v0', 0))
    if tu0 or tv0:
        uv = uv - np.array([tu0, tv0], dtype=np.float32)
        K = K.copy(); K[0, 2] -= tu0; K[1, 2] -= tv0
    H, W = img.shape[:2]
    keep = ((uv[:, 0] >= 0) & (uv[:, 0] < W) &
            (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z > 0))
    uv = uv[keep]; z = z[keep]
    if intensity is not None:
        intensity = np.clip(intensity[keep] / 128.0, 0.0, 1.0).astype(np.float32)
    return img, uv, z, K, intensity, c


def _run(cache_path: str, model, idx: int = 0):
    img, uv, z, K, intensity, c = _load_one(cache_path, idx=idx)
    ba_cfg = dict(tile_size=512, model_input_size=c['img_size'],
                  max_pts_per_tile=256, min_pts_per_tile=8,
                  tile_stride=384)
    # Determinism: this test is a refactor invariant check, NOT a "matches
    # production numerics" check. Seed everything AND patch torch.autocast
    # in infer_tiles to a no-op so the forward runs full fp32 — the only
    # mode that's reliably bit-identical run-to-run on V100. Production
    # inference still uses the fp16 autocast inside infer_tiles.
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import contextlib
    real_autocast = torch.autocast
    torch.autocast = lambda *a, **kw: contextlib.nullcontext()
    try:
        res = infer_tiles(model, img, uv, z, K, ba_cfg,
                           torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                           intensity=intensity)
    finally:
        torch.autocast = real_autocast
    if res is None:
        raise RuntimeError(f'infer_tiles returned None for {cache_path}')
    uv_pool, par_pool, z_pool = res
    return uv_pool, par_pool, z_pool


def cmd_record():
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    model = load_calib_model(EXP).eval()
    for name, path in CACHES:
        uv_pool, par_pool, z_pool = _run(path, model)
        out = GOLDEN_DIR / f'{name}.npz'
        np.savez(out, uv=uv_pool, par=par_pool, z=z_pool)
        print(f'[record] {name:10s} N={len(uv_pool):>4d}  '
              f'par mean={par_pool.mean():+.4f}  → {out}')


def cmd_verify():
    if not GOLDEN_DIR.is_dir():
        print(f'NO GOLDEN at {GOLDEN_DIR}. Run --record first.')
        sys.exit(2)
    model = load_calib_model(EXP).eval()
    fail = 0
    for name, path in CACHES:
        gp = GOLDEN_DIR / f'{name}.npz'
        if not gp.is_file():
            print(f'[skip] {name}: no snapshot at {gp}')
            continue
        ref = np.load(gp)
        uv_pool, par_pool, z_pool = _run(path, model)
        ok = (uv_pool.shape == ref['uv'].shape
              and par_pool.shape == ref['par'].shape
              and z_pool.shape == ref['z'].shape
              and np.allclose(uv_pool, ref['uv'], atol=ATOL, rtol=RTOL)
              and np.allclose(par_pool, ref['par'], atol=ATOL, rtol=RTOL)
              and np.allclose(z_pool, ref['z'], atol=ATOL, rtol=RTOL))
        if ok:
            print(f'[PASS] {name:10s} N={len(uv_pool):>4d}')
        else:
            fail += 1
            print(f'[FAIL] {name:10s}')
            print(f'  shape now : uv={uv_pool.shape} par={par_pool.shape} z={z_pool.shape}')
            print(f'  shape ref : uv={ref["uv"].shape} par={ref["par"].shape} z={ref["z"].shape}')
            if uv_pool.shape == ref['uv'].shape:
                print(f'  uv max abs diff : {np.abs(uv_pool - ref["uv"]).max():.3e}')
            if par_pool.shape == ref['par'].shape:
                print(f'  par max abs diff: {np.abs(par_pool - ref["par"]).max():.3e}')
            if z_pool.shape == ref['z'].shape:
                print(f'  z max abs diff  : {np.abs(z_pool - ref["z"]).max():.3e}')
    if fail:
        sys.exit(1)
    print(f'\nALL {len(CACHES)} datasets passed (atol={ATOL}).')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--record', action='store_true',
                    help='snapshot current infer_tiles output as the new golden')
    ap.add_argument('--out-dir', default=None,
                    help='override golden dir (default: scripts/_debug/_golden/'
                         'infer_tiles). Use this when /workspace is mounted ro.')
    args = ap.parse_args()
    if args.out_dir:
        global GOLDEN_DIR
        GOLDEN_DIR = Path(args.out_dir)
    if args.record:
        cmd_record()
    else:
        cmd_verify()


if __name__ == '__main__':
    main()
