"""Shared LMDB writer for v3 build scripts.

Workers each write to their own shard lmdb to avoid IPC overhead + serial-write
bottleneck of single-LMDB. After all workers done, master merges shards into
final data.lmdb.

Pack format is identical to convert_tile_cache_to_lmdb.py — reader code is
unchanged.
"""
from __future__ import annotations
import pickle, struct
import numpy as np
import torch
from pathlib import Path

HDR_LEN_FMT = '<Q'
HDR_LEN_SIZE = struct.calcsize(HDR_LEN_FMT)
CUBS_KEY_PREFIX = '__cubs__/'


def _as_np(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def pack_cubs(cubs):
    """Pack cuboid list → bytes (per-frame shared key)."""
    if cubs is None or len(cubs) == 0:
        return pickle.dumps({'N': 0}, protocol=pickle.HIGHEST_PROTOCOL)
    M = len(cubs)
    return pickle.dumps({
        'N': M,
        'cls': np.asarray([str(c.get('label', '')) for c in cubs], dtype='O'),
        'center': np.stack([np.asarray(c['center'], dtype=np.float32) for c in cubs]),
        'size':   np.stack([np.asarray(c['size'],   dtype=np.float32) for c in cubs]),
        'yaw':    np.fromiter((float(c['yaw']) for c in cubs),
                              dtype=np.float32, count=M),
    }, protocol=pickle.HIGHEST_PROTOCOL)


def pack_inst(inst: dict) -> bytes:
    """Pack one inst dict → LMDB value bytes. Excludes cuboids (those go in
    __cubs__/<scene>/<frame> shared key)."""
    if 'jpg_bytes' not in inst:
        raise ValueError("inst missing 'jpg_bytes'")
    jpg = inst['jpg_bytes']
    if not isinstance(jpg, (bytes, bytearray)):
        jpg = bytes(jpg)
    IH, IW = int(inst['IH']), int(inst['IW'])

    pts = _as_np(inst['pts']).astype(np.float32, copy=False)
    N = int(pts.shape[0])
    K_full = _as_np(inst['K_full']).astype(np.float32, copy=False)
    cam_pos = _as_np(inst['cam_pos']).astype(np.float32, copy=False)
    R_gt = _as_np(inst['R_gt']).astype(np.float32, copy=False)
    T_gt = _as_np(inst['T_gt']).astype(np.float32, copy=False) if 'T_gt' in inst else None

    uv_full = _as_np(inst['uv_full']).astype(np.float32, copy=False) if 'uv_full' in inst else None
    z_cam   = _as_np(inst['z_cam']).astype(np.float32, copy=False)   if 'z_cam'   in inst else None
    is_obj  = _as_np(inst['is_obj']).astype(np.uint8, copy=False)    if 'is_obj'  in inst else None
    in_box  = _as_np(inst['in_box']).astype(np.uint8, copy=False)    if 'in_box'  in inst else None
    intensity  = _as_np(inst['intensity']).astype(np.float32, copy=False)  if 'intensity'  in inst else None
    distortion = _as_np(inst['distortion']).astype(np.float32, copy=False) if 'distortion' in inst else None
    is_fisheye = bool(inst.get('is_fisheye', False))

    chunks = []
    offsets = {}
    cursor = 0
    def _add(name, buf, dtype_str=None, shape=None):
        nonlocal cursor
        b = buf if isinstance(buf, (bytes, bytearray)) else bytes(np.ascontiguousarray(buf).data)
        chunks.append(b)
        offsets[name] = (cursor, len(b), dtype_str, shape)
        cursor += len(b)

    _add('jpg', jpg, None, None)
    _add('K_full',  K_full,  'f4', (3, 3))
    _add('cam_pos', cam_pos, 'f4', (3,))
    _add('R_gt',    R_gt,    'f4', (3, 3))
    if T_gt is not None: _add('T_gt', T_gt, 'f4', T_gt.shape)
    if distortion is not None: _add('distortion', distortion, 'f4', distortion.shape)
    _add('pts', pts, 'f4', (N, 3))
    if uv_full is not None: _add('uv_full', uv_full, 'f4', (N, 2))
    if z_cam   is not None: _add('z_cam',   z_cam,   'f4', (N,))
    if is_obj  is not None: _add('is_obj',  is_obj,  'u1', (N,))
    if in_box  is not None: _add('in_box',  in_box,  'u1', (N,))
    if intensity is not None: _add('intensity', intensity, 'f4', (N,))

    header = {
        'v': 2,
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


class ShardWriter:
    """Per-worker LMDB shard. Each worker writes its own file; master merges.

    Usage in worker:
        sw = ShardWriter(shard_path, map_size_gb=80)
        for inst in frames:
            sw.put_inst(key, inst)  # auto packs + handles cuboids
        sw.close()
    """
    def __init__(self, path, map_size_gb=80, commit_every=2000):
        import lmdb
        path = Path(path)
        if path.exists():
            import shutil; shutil.rmtree(path)
        self.env = lmdb.open(
            str(path), map_size=map_size_gb * (1 << 30),
            subdir=True, writemap=True, sync=False, meminit=False, max_dbs=0,
        )
        self.txn = self.env.begin(write=True)
        self._n = 0
        self._commit_every = commit_every
        self._seen_cubs = set()  # (scene, frame) → skip duplicate cubs write

    def put_inst(self, key: str, inst: dict):
        """Pack & write one inst. Cuboids deduped per (scene, frame)."""
        scene = str(inst.get('scene', ''))
        frame = int(inst.get('frame', -1))
        ck = (scene, frame)
        if ck not in self._seen_cubs:
            cubs_blob = pack_cubs(inst.get('cuboids', []) or [])
            self.txn.put(f'{CUBS_KEY_PREFIX}{scene}/{frame}'.encode(), cubs_blob)
            self._seen_cubs.add(ck)
        blob = pack_inst(inst)
        self.txn.put(key.encode(), blob)
        self._n += 1
        if self._n % self._commit_every == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self):
        self.txn.commit()
        self.env.sync()
        self.env.close()


def merge_shards(shard_paths, final_path, map_size_gb=800):
    """Read all shard lmdb files → write single final lmdb. Dedup cuboid keys."""
    import lmdb
    final_path = Path(final_path)
    if final_path.exists():
        import shutil; shutil.rmtree(final_path)
    out_env = lmdb.open(
        str(final_path), map_size=map_size_gb * (1 << 30),
        subdir=True, writemap=True, sync=False, meminit=False, max_dbs=0,
    )
    out_txn = out_env.begin(write=True)
    n_written = 0
    n_cubs = 0
    seen_cubs = set()
    for sp in shard_paths:
        env = lmdb.open(str(sp), readonly=True, lock=False,
                        subdir=True, max_dbs=0)
        with env.begin() as txn:
            cur = txn.cursor()
            for k, v in cur:
                if k.startswith(CUBS_KEY_PREFIX.encode()):
                    if k in seen_cubs: continue
                    out_txn.put(k, v)
                    seen_cubs.add(k)
                    n_cubs += 1
                else:
                    out_txn.put(k, v)
                    n_written += 1
                if (n_written + n_cubs) % 10000 == 0:
                    out_txn.commit()
                    out_txn = out_env.begin(write=True)
        env.close()
    out_txn.commit()
    out_env.sync()
    out_env.close()
    return n_written, n_cubs
