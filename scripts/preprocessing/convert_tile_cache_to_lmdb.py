"""Convert tile cache (inst/*.pt) → packed LMDB, no per-call torch.load.

The current .pt path costs ~2.4 ms/sample in `torch.load` + tensor restore
(~47% of __getitem__ CPU). LMDB-of-.pt-bytes alone does NOT remove that —
the read-side still has to pickle.loads + rebuild every torch.Tensor.

This converter explodes each inst into:
  header_dict (small pickle, < 1KB)  — scalars, dtypes, shapes, byte offsets
  body bytes                          — raw jpg + raw array bytes, concatenated

Reader (datasets/pandaset_full.py LMDB path) does:
  1. pickle.loads(header) once  (~10 μs)
  2. np.frombuffer(memoryview, ...) per array  (zero-copy)
  3. torch.from_numpy(view) wrapper for the few sites that call .numpy()

→ zero pickle of big tensors, zero file open per sample.

Usage:
    python scripts/preprocessing/convert_tile_cache_to_lmdb.py \\
        --cache-dir /mnt/nvme6t/e2e_calib_cache/pandaset_v3_tiled_multicam_corr \\
        --map-size-gb 60
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import argparse, pickle, struct, time
from pathlib import Path
import numpy as np
import lmdb
import torch


# Header pickle is preceded by 8-byte little-endian uint64 = header length.
# Body bytes follow immediately after the header pickle.
HDR_LEN_FMT = '<Q'
HDR_LEN_SIZE = struct.calcsize(HDR_LEN_FMT)


def _as_np(t):
    """torch.Tensor or numpy.ndarray → numpy.ndarray (no copy if possible)."""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


# Cuboids are world-coordinate object boxes — identical across all tiles
# (and cameras) of a given (scene, frame). The legacy .pt cache replicates
# them per tile (~9 KB × 30 tiles × N cams = 1.4 GB wasted on a 151k-tile
# cache). Hoist them out into one shared key per (scene, frame).
CUBS_KEY_PREFIX = '__cubs__/'      # b'__cubs__/<scene>/<frame>'


def _pack_cubs(cubs: list) -> bytes:
    """Pack a cuboids list into a small dict of stacked numpy arrays."""
    if not cubs:
        return pickle.dumps({'M': 0}, protocol=pickle.HIGHEST_PROTOCOL)
    M = len(cubs)
    return pickle.dumps({
        'M': M,
        'pos':  np.stack([_as_np(c['pos']).astype(np.float32)  for c in cubs]),
        'dims': np.stack([_as_np(c['dims']).astype(np.float32) for c in cubs]),
        'yaw':  np.fromiter((float(c['yaw']) for c in cubs),
                             dtype=np.float32, count=M),
    }, protocol=pickle.HIGHEST_PROTOCOL)


def _pack_inst(inst: dict) -> bytes:
    """Pack one inst dict into the LMDB value layout. Cuboids are deduped to a
    separate `__cubs__/<scene>/<frame>` key — NOT included in the tile blob."""
    # ─── scalar fields ───────────────────────────────────────────────
    if 'jpg_bytes' in inst:
        jpg = inst['jpg_bytes']
        IH, IW = int(inst['IH']), int(inst['IW'])
    else:
        raise ValueError("inst missing 'jpg_bytes' (legacy uint8 'img' not supported)")
    if not isinstance(jpg, (bytes, bytearray)):
        jpg = bytes(jpg)

    pts = _as_np(inst['pts']).astype(np.float32, copy=False)
    N = int(pts.shape[0])
    K_full = _as_np(inst['K_full']).astype(np.float32, copy=False)
    cam_pos = _as_np(inst['cam_pos']).astype(np.float32, copy=False)
    R_gt = _as_np(inst['R_gt']).astype(np.float32, copy=False)

    # ─── optional fields ─────────────────────────────────────────────
    T_gt = _as_np(inst['T_gt']).astype(np.float32, copy=False) if 'T_gt' in inst else None
    uv_full = _as_np(inst['uv_full']).astype(np.float32, copy=False) if 'uv_full' in inst else None
    z_cam = _as_np(inst['z_cam']).astype(np.float32, copy=False) if 'z_cam' in inst else None
    is_obj = (_as_np(inst['is_obj']).astype(np.uint8, copy=False)
              if 'is_obj' in inst else None)
    in_box = (_as_np(inst['in_box']).astype(np.uint8, copy=False)
              if 'in_box' in inst else None)
    distortion = (_as_np(inst['distortion']).astype(np.float32, copy=False)
                  if 'distortion' in inst else None)
    is_fisheye = bool(inst.get('is_fisheye', False))

    # ─── build body: concatenate raw bytes in deterministic order ────
    chunks = []
    offsets = {}
    cursor = 0
    def _append(name, buf, dtype_str=None, shape=None):
        nonlocal cursor
        b = buf if isinstance(buf, (bytes, bytearray)) else bytes(np.ascontiguousarray(buf).data)
        chunks.append(b)
        offsets[name] = (cursor, len(b), dtype_str, shape)
        cursor += len(b)

    _append('jpg', jpg, None, None)
    _append('K_full',  K_full,  'f4', (3, 3))
    _append('cam_pos', cam_pos, 'f4', (3,))
    _append('R_gt',    R_gt,    'f4', (3, 3))
    if T_gt is not None:
        _append('T_gt', T_gt, 'f4', T_gt.shape)
    if distortion is not None:
        _append('distortion', distortion, 'f4', distortion.shape)
    _append('pts', pts, 'f4', (N, 3))
    if uv_full is not None: _append('uv_full', uv_full, 'f4', (N, 2))
    if z_cam   is not None: _append('z_cam',   z_cam,   'f4', (N,))
    if is_obj  is not None: _append('is_obj',  is_obj,  'u1', (N,))
    if in_box  is not None: _append('in_box',  in_box,  'u1', (N,))

    header = {
        'v': 2,                       # bumped: cuboids hoisted out
        'IH': IH, 'IW': IW,
        'tile_u0': int(inst.get('tile_u0', 0)),
        'tile_v0': int(inst.get('tile_v0', 0)),
        'N_pts': N,
        'is_fisheye': is_fisheye,
        'offsets': offsets,
        'scene': str(inst.get('scene', '')),
        'frame': int(inst.get('frame', -1)),
    }
    hdr_blob = pickle.dumps(header, protocol=pickle.HIGHEST_PROTOCOL)
    out = bytearray()
    out += struct.pack(HDR_LEN_FMT, len(hdr_blob))
    out += hdr_blob
    for c in chunks:
        out += c
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', required=True)
    ap.add_argument('--map-size-gb', type=int, default=200)
    ap.add_argument('--commit-every', type=int, default=2000)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    inst_dir  = cache_dir / 'inst'
    meta_path = cache_dir / 'meta.pt'
    lmdb_path = cache_dir / 'data.lmdb'

    assert inst_dir.is_dir(),  f'inst dir not found: {inst_dir}'
    assert meta_path.is_file(), f'meta.pt not found: {meta_path}'
    if lmdb_path.exists():
        if not args.overwrite:
            raise SystemExit(f'output exists: {lmdb_path}  (pass --overwrite to nuke)')
        import shutil; shutil.rmtree(lmdb_path)

    meta = torch.load(meta_path, weights_only=False)
    fnames_all = list(meta['train']) + list(meta['val'])
    N = len(fnames_all)
    print(f'cache:     {cache_dir}', flush=True)
    print(f'instances: {N}  (train={len(meta["train"])}, val={len(meta["val"])})', flush=True)
    print(f'output:    {lmdb_path}  (map_size={args.map_size_gb}GB)', flush=True)

    env = lmdb.open(
        str(lmdb_path),
        map_size=args.map_size_gb * (1 << 30),
        subdir=True, writemap=True, sync=False, meminit=False, max_dbs=0,
    )

    t0 = time.time()
    txn = env.begin(write=True)
    nb = 0
    nb_cubs = 0
    missing = 0
    seen_cubs_keys = set()      # (scene, frame) → already written
    for i, fn in enumerate(fnames_all):
        p = inst_dir / fn
        try:
            inst = torch.load(p, weights_only=False)
        except FileNotFoundError:
            missing += 1
            if missing <= 5:
                print(f'  WARN missing inst: {fn}', flush=True)
            continue
        # Cuboids dedup: first tile encountered for a (scene, frame) writes,
        # the rest just reference the same key. Viz / fallback paths read by
        # constructing the same key from the per-tile header.
        scene = str(inst.get('scene', ''))
        frame = int(inst.get('frame', -1))
        ck = (scene, frame)
        if ck not in seen_cubs_keys:
            cubs_key = f'{CUBS_KEY_PREFIX}{scene}/{frame}'.encode()
            cubs_blob = _pack_cubs(inst.get('cuboids', []) or [])
            txn.put(cubs_key, cubs_blob)
            nb_cubs += len(cubs_blob)
            seen_cubs_keys.add(ck)
        blob = _pack_inst(inst)
        txn.put(fn.encode(), blob)
        nb += len(blob)
        if (i + 1) % args.commit_every == 0:
            txn.commit()
            txn = env.begin(write=True)
        if (i + 1) % 5000 == 0 or (i + 1) == N:
            dt = time.time() - t0
            sps = (i + 1) / max(dt, 1e-6)
            print(f'  {i+1:>7}/{N}  tiles={nb/1e9:6.1f}GB  cubs={nb_cubs/1e6:6.1f}MB '
                  f'({len(seen_cubs_keys)} frames)  sps={sps:6.0f}  elapsed={dt:6.0f}s',
                  flush=True)
    txn.commit()
    env.sync()
    env.close()
    print(f'\ndone in {time.time()-t0:.0f}s; '
          f'tiles={nb/1e9:.1f}GB, cubs={nb_cubs/1e6:.1f}MB ({len(seen_cubs_keys)} unique frames); '
          f'missing={missing}', flush=True)


if __name__ == '__main__':
    main()
