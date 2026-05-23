"""V3 full-image lazy dataset.

Per __getitem__:
  - Load instance (full image + lidar + cuboids).
  - Pick random crop_size ∈ [min_crop, max_crop] full-image px.
  - Pick random (u0, v0) such that crop fits, AND crop has >= min_pts visible lidar.
  - Compute is_obj per point via 3D box-membership against ALL cuboids in inst.
  - Return (img_crop_64, true_uvd, dist_uvd) like the V1 lazy dataset.
"""
import random
from pathlib import Path
import numpy as np

# ── numpy 1.x ↔ 2.x pickle compat shim ────────────────────────────────────
# V3 cache (waymo_v3_tiled, pandaset_v3_full) was pickled on a host with
# numpy 2.x. The payload references `numpy._core.*` (private module introduced
# in numpy 2.0). When we unpickle on the training container (nvcr 24.02 →
# numpy 1.24.4, path not bumpable because torch 2.3.0a0 was built against the
# numpy 1.x ABI), unpickle fails with `ModuleNotFoundError: numpy._core`.
# Register `numpy._core` as an alias for `numpy.core` at import time so
# `torch.load(..., weights_only=False)` finds the expected callables.
# Restricted to the modules that actually appear in our cache payloads
# (multiarray / umath / numeric) — other submodules lazily resolve if/when
# a future cache version pickles new types.
import sys as _sys
try:
    import numpy.core as _np_core
    _sys.modules.setdefault("numpy._core", _np_core)
    for _sub in ("multiarray", "umath", "numeric", "_multiarray_umath",
                 "fromnumeric", "_methods"):
        try:
            _m = __import__(f"numpy.core.{_sub}", fromlist=[_sub])
            _sys.modules.setdefault(f"numpy._core.{_sub}", _m)
        except Exception:
            pass
except Exception:
    pass

import io
import os
import pickle
import struct
import warnings
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

try:
    import lmdb as _lmdb
    _HAVE_LMDB = True
except ImportError:
    _lmdb = None
    _HAVE_LMDB = False

# LMDB packed-inst layout: 8-byte LE uint64 header_len, pickle header, raw body.
# Header carries offsets+dtype+shape per array; body is jpg + arrays concatenated.
_LMDB_HDR_LEN_FMT = '<Q'
_LMDB_HDR_LEN_SIZE = struct.calcsize(_LMDB_HDR_LEN_FMT)


def _unpack_lmdb_inst(blob: bytes, cubs_map: dict | None = None) -> dict:
    """Decode one LMDB value → dict shaped like a torch-loaded inst.pt.

    Zero pickle.loads of large tensors: arrays are np.frombuffer views over
    `blob`, wrapped via torch.from_numpy (zero-copy, read-only).

    Cuboids are NOT stored inline (v2 layout): they live under a separate
    `__cubs__/<scene>/<frame>` key, shared across all tiles of the same
    frame. `cubs_map` is a {(scene, frame): [cuboids list]} preloaded at
    Dataset.__init__ time (~7 MB total for PS multicam_corr).
    """
    hdr_len = struct.unpack_from(_LMDB_HDR_LEN_FMT, blob, 0)[0]
    header = pickle.loads(blob[_LMDB_HDR_LEN_SIZE:_LMDB_HDR_LEN_SIZE + hdr_len])
    body_off = _LMDB_HDR_LEN_SIZE + hdr_len
    offsets = header['offsets']

    def _arr(name):
        off, length, dtype_str, shape = offsets[name]
        a = np.frombuffer(blob, dtype=np.dtype(dtype_str),
                          count=length // np.dtype(dtype_str).itemsize,
                          offset=body_off + off)
        return a.reshape(shape)

    def _bytes(name):
        off, length, _dt, _sh = offsets[name]
        return bytes(blob[body_off + off:body_off + off + length])

    inst = {
        'IH': header['IH'], 'IW': header['IW'],
        'tile_u0': header['tile_u0'], 'tile_v0': header['tile_v0'],
        'jpg_bytes': _bytes('jpg'),
        'scene': header.get('scene', ''),
        'frame': header.get('frame', -1),
    }
    with warnings.catch_warnings():
        # torch.from_numpy on non-writable buffer-backed arrays warns; we
        # only read these tensors, so the warning is noise.
        warnings.simplefilter('ignore', UserWarning)
        for k in ('K_full', 'cam_pos', 'R_gt', 'T_gt', 'distortion',
                  'pts', 'uv_full', 'z_cam', 'is_obj', 'in_box', 'intensity'):
            if k in offsets:
                inst[k] = torch.from_numpy(_arr(k))
    if header.get('is_fisheye', False):
        inst['is_fisheye'] = True
    if cubs_map is not None:
        inst['cuboids'] = cubs_map.get((inst['scene'], inst['frame']), [])
    return inst
try:
    import turbojpeg as _tj
    # PyTurboJPEG class API. One instance per worker process (not thread-safe
    # across workers, but torch DataLoader forks one process per worker so
    # each gets its own). Bench on dgx2 (Xeon 8168): PIL full decode 14.5ms,
    # TJ full decode 6.9ms, TJ crop+decode (384px) 4.5ms → 3.2× vs PIL.
    _TJ_INST = _tj.TurboJPEG()
    _TJ_PF_RGB = _tj.TJPF_RGB
    _HAVE_TJ = True
except Exception:
    _tj = None
    _TJ_INST = None
    _TJ_PF_RGB = None
    _HAVE_TJ = False
# MCU block size for JPEG; used for crop-aligned partial decode.
# 16px matches 4:2:0 chroma subsampling. Crop x,y must be multiples of
# _MCU; width/height are trimmed to image bounds by libjpeg-turbo.
_MCU = 16


def decode_inst_img(inst: dict) -> torch.Tensor:
    """Return (3, H, W) uint8 torch tensor regardless of cache schema.
    New schema: inst['jpg_bytes'] + inst['IH']/['IW'].  Legacy: inst['img']."""
    if 'jpg_bytes' in inst:
        if _HAVE_TJ:
            arr = np.asarray(_TJ_INST.decode(inst['jpg_bytes'], pixel_format=_TJ_PF_RGB))
        else:
            import io
            from PIL import Image as _PIL
            arr = np.asarray(_PIL.open(io.BytesIO(inst['jpg_bytes'])).convert('RGB'),
                             dtype=np.uint8)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return inst['img']


def _is_obj_per_point(pts_sel: np.ndarray, cuboids: list) -> np.ndarray:
    """Return (N,) float32 mask: 1 if a point lies inside ANY cuboid (yaw-rotated AABB).

    Vectorized across both points and cuboids: one einsum + two reductions,
    no Python loop over cuboids.
    """
    N = len(pts_sel)
    if not cuboids or N == 0:
        return np.zeros(N, dtype=np.float32)
    M = len(cuboids)
    poss = np.stack([np.asarray(c['pos'],  dtype=np.float32) for c in cuboids])  # (M,3)
    dims = np.stack([np.asarray(c['dims'], dtype=np.float32) for c in cuboids])  # (M,3)
    yaws = np.fromiter((float(c['yaw']) for c in cuboids), dtype=np.float32, count=M)
    cy, sy = np.cos(yaws), np.sin(yaws)
    # Per-cuboid 3x3 rotation: yaw about z, with the same convention as the loop version
    R = np.zeros((M, 3, 3), dtype=np.float32)
    R[:, 0, 0] = cy;  R[:, 0, 1] = sy
    R[:, 1, 0] = -sy; R[:, 1, 1] = cy
    R[:, 2, 2] = 1.0
    # delta = pts (N,3) - pos (M,3) → (M, N, 3); local = R @ delta.T per cuboid
    delta = pts_sel.astype(np.float32, copy=False)[None, :, :] - poss[:, None, :]
    local = np.einsum('mij,mnj->mni', R, delta)                                 # (M,N,3)
    half  = (dims * 0.5)[:, None, :]                                              # (M,1,3)
    inside = np.all(np.abs(local) <= half, axis=-1)                               # (M,N)
    return inside.any(axis=0).astype(np.float32)                                  # (N,)


class PandaSetCalibDatasetFull(Dataset):
    """Full-image V3 lazy dataset.

    Args:
        cache_dir: contains meta.pt + inst/*.pt (with full-frame images)
        split: 'train' or 'val'
        img_size: model input side (px), default 64
        min_crop_px: minimum random-crop side in full-image px
        max_crop_px: maximum random-crop side in full-image px
        max_offset_m: extrinsic translation perturbation half-range
        max_rot_deg:  extrinsic YPR perturbation half-range
        min_pts: minimum number of lidar pts to keep a sample
        max_tries: re-sample random crop up to this many times before fallback
    """
    def __init__(self,
                 cache_dir: str | Path,
                 split: str = 'train',
                 img_size: int = 64,
                 min_crop_px: int = 128,
                 max_crop_px: int = 512,
                 max_offset_m: float = 0.20,
                 max_rot_deg: float = 0.5,
                 max_fx_pct: float = 0.0,
                 max_fy_pct: float = 0.0,
                 pose_frame: str = 'orig',
                 split_pert: bool = False,
                 min_pts: int = 8,
                 max_tries: int = 8,
                 oversample: int = 12,
                 frame_stride: int = 1,
                 grid_n: int = 16,
                 n_full: int = 1024,
                 k_per_cell: int = 8,
                 zoom_aug: bool = False,
                 rep_strategy: str = 'cell_center',
                 center_band: float = 0.0,
                 fixed_center_crop: bool = False,
                 preload: bool = True):
        self.cache_dir = Path(cache_dir)
        self.inst_dir  = self.cache_dir / 'inst'
        # LMDB packed path: if <cache_dir>/data.lmdb exists, use the packed
        # converter output (raw bytes, zero torch.load per sample). The .pt
        # path stays as a fallback.
        self.lmdb_path = self.cache_dir / 'data.lmdb'
        # E2E_NO_LMDB=1 forces the legacy inst/*.pt path even when LMDB exists
        # (used by bench to A/B the two paths).
        self._use_lmdb = bool(_HAVE_LMDB and self.lmdb_path.is_dir()
                              and os.environ.get('E2E_NO_LMDB') != '1')
        self._lmdb_env = None  # lazy-opened per worker after fork
        # Cuboids (world-coord 3D boxes) are deduped to per-frame keys in the
        # LMDB v2 layout. Preload the whole {(scene, frame): [cubs]} map at
        # __init__ — total ~7 MB even on PS multicam_corr (843 frames × ~9KB).
        # This is shared via fork copy-on-write across DataLoader workers so
        # the hot-path `__getitem__` never queries the cuboid key.
        self._cubs_map: dict | None = None
        meta = torch.load(self.cache_dir / 'meta.pt', weights_only=False)
        assert split in ('train', 'val')
        self.fnames    = list(meta[split])
        if frame_stride > 1:
            self.fnames = self.fnames[::frame_stride]
        self.img_size  = int(img_size)
        self.min_crop_px = int(min_crop_px)
        self.max_crop_px = int(max_crop_px)
        self.max_offset_m = float(max_offset_m)
        self.max_rot_deg  = float(max_rot_deg)
        # Multiplicative fx/fy perturbation half-range (e.g. 0.02 = ±2%).
        # When non-zero, sampled δ_fx, δ_fy ∈ [-pct, +pct] are folded into the
        # projection K used for `dist_uvd` and appended to the per-sample pert
        # vec as dims 6,7 (so the optional CLS frame-pose head with n_dof=8 can
        # regress Δfx_pct, Δfy_pct alongside the 6-DoF SE3 perturbation).
        self.max_fx_pct = float(max_fx_pct)
        self.max_fy_pct = float(max_fy_pct)
        # pose_frame: 'orig' → pert_vec is in original camera frame (legacy).
        # 'vcam' → pert is converted to the tile's virtual-camera frame whose
        # optical axis is the ray through the tile center. This makes the
        # 6-DoF label crop-position-AGNOSTIC: two tiles at left vs right edges
        # with identical observations get identical labels. roll dim is left
        # as 0 in vcam mode (rotation around VCAM optical axis is mostly
        # confounded with t_x/t_y in tile-local space — explicit user advice).
        # Downstream BA aggregates per-tile (μ_vcam, Σ_vcam) via Jacobian
        # J_i = R_orig→vcam_i to recover orig-frame δ.
        assert pose_frame in ('orig', 'vcam'), f'bad pose_frame={pose_frame}'
        self.pose_frame = pose_frame
        # split_pert: self-supervised pose_emb mode. Sample δ = δ1 + δ2 (each
        # half-range, composed as R_δ1 @ R_δ2; t_δ1 + t_δ2). δ1 is exposed via
        # pert_vec[8:14] so the model can fold it into pose_emb as a known hint;
        # the network's job becomes regressing only δ2's reproj residual.
        # true_uvd is set to uv_pre = (uv after applying ONLY δ1) so that
        # target = true_uvd - dist_uvd = -δ2-induced shift. δ1=0 degenerates
        # exactly to the calibration objective. orig-frame only.
        self.split_pert = bool(split_pert)
        if self.split_pert:
            assert self.pose_frame == 'orig', \
                'split_pert is implemented for pose_frame="orig" only'
        self.min_pts   = int(min_pts)
        self.max_tries = int(max_tries)
        self.oversample = int(oversample)
        self.grid_n     = int(grid_n)
        self.n_full     = int(n_full)
        self.k_per_cell = int(k_per_cell)
        # depth-dependent zoom-in aug. When True, far pivots (z>=20m) randomly
        # shrink cs by up to scale_max(z): 1.0 at 20m → 2.0 at 100m+. This
        # synthesizes "telephoto view of distant objects" without needing
        # wider source crops — fills the high-resolution far-object regime
        # that PS / Waymo data lacks at native lens.
        self.zoom_aug  = bool(zoom_aug)
        # rep_strategy: which point per occupied cell becomes the model query.
        #   'cell_center' — point closest to the cell's geometric center in uv
        #                    (legacy default; existing models trained on this).
        #                    Stable in uniform point clouds but the rep can flip
        #                    foreground↔background under small motion since the
        #                    "closest to center" choice doesn't honor depth.
        #   'nearest_cam' — point with the smallest depth (closest to the ego
        #                    camera) in each cell. Foreground points are
        #                    occlusion-stable (a road point can't suddenly
        #                    "win" the cell when a pole edge moves a few px).
        #                    Requires retraining; old cell_center checkpoints
        #                    will see slightly out-of-distribution inputs.
        assert rep_strategy in ('cell_center', 'nearest_cam'), \
            f'bad rep_strategy={rep_strategy}'
        self.rep_strategy = rep_strategy
        # Restrict pivot rows to the central horizontal band of the image. 0
        # = disabled (full height); 0.5 = central 50% of rows (v ∈ [0.25H,
        # 0.75H]). Useful for val: top/bottom slivers are sky / road close-up
        # with few lidar points, hard to interpret. Affects pivot pick only;
        # frustum context still uses every visible point.
        self.center_band = float(center_band)
        # When True, skip pivot/random-crop entirely and just take a cs=
        # max_crop_px square centred on (W/2, cy). Used by val/eval/demo
        # to look at "the middle of the image" instead of stratified bg
        # cells (which often dump sky/road tiles).
        self.fixed_center_crop = bool(fixed_center_crop)
        # Preload every inst .pt into RAM. The full PS cache is ~1 GB, and
        # torch.load per-sample shows up as ~3.8 ms in cProfile — roughly
        # 40% of the remaining __getitem__ cost once TurboJPEG is in place.
        # DataLoader workers spawn/fork after __init__, so the list is shared
        # via copy-on-write / fork semantics and each worker hits RAM directly
        # instead of unpickling from disk every call.
        self._cache = None
        if preload and not self._use_lmdb:
            # LMDB-backed caches are already an OS-mapped shared file; no point
            # duplicating into Python objects per worker.
            self._cache = [
                torch.load(self.inst_dir / fn, weights_only=False)
                for fn in self.fnames
            ]
        if self._use_lmdb and os.environ.get('E2E_PRELOAD_CUBS') == '1':
            # Preload deduped cuboid table once in the parent process. Forked
            # workers inherit it via CoW.
            # OFF by default: any lmdb.open in __init__ leaves an entry in
            # lmdb-py's per-process registry that survives env.close() on at
            # least some library versions, which then conflicts with the
            # workers' open() of the same path. Set E2E_PRELOAD_CUBS=1 only
            # when you trust your lmdb-py version + path doesn't get reopened.
            self._cubs_map = self._preload_cubs_map()

    def __len__(self):
        return len(self.fnames) * self.oversample

    # Process-wide LMDB env cache keyed by (pid, path). lmdb-py rejects a
    # second open() of the same path within one process, so multiple Dataset
    # instances pointing at the same cache (typical: train+val splits, or
    # accel.gather peeking) must SHARE one Environment.
    _LMDB_ENV_CACHE: dict = {}

    def _open_lmdb(self):
        import os as _os
        cur = _os.getpid()
        key = (cur, str(self.lmdb_path))
        env = PandaSetCalibDatasetFull._LMDB_ENV_CACHE.get(key)
        if env is None:
            env = _lmdb.open(
                str(self.lmdb_path), readonly=True, lock=False,
                readahead=False, meminit=False, max_readers=512, subdir=True,
            )
            PandaSetCalibDatasetFull._LMDB_ENV_CACHE[key] = env
        self._lmdb_env = env
        self._lmdb_env_pid = cur

    def __getstate__(self):
        # When DataLoader spawn workers pickle this dataset, drop the lmdb env
        # so the child re-opens its own. Otherwise lmdb's per-process registry
        # in the child rejects open() on the inherited path with
        # 'already open in this process' (since the unpickled handle keeps
        # the path registered).
        st = self.__dict__.copy()
        st['_lmdb_env'] = None
        st['_lmdb_env_pid'] = None
        return st

    def close_lmdb(self):
        """Close the lmdb env. Required after parent-process __getitem__ calls
        (e.g. vis_pretrain, midtrain_vis) — lmdb tracks open paths in a
        process-wide registry, so a subsequent open() of the same path from
        any DataLoader worker forked off this parent will raise
        'already open in this process'.
        """
        if self._lmdb_env is not None:
            try:
                self._lmdb_env.close()
            except Exception:
                pass
            self._lmdb_env = None
            self._lmdb_env_pid = None

    def _preload_cubs_map(self) -> dict:
        env = _lmdb.open(
            str(self.lmdb_path), readonly=True, lock=False,
            readahead=False, meminit=False, max_readers=512, subdir=True,
        )
        out: dict = {}
        prefix = b'__cubs__/'
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            cursor.set_range(prefix)
            for key, val in cursor:
                if not key.startswith(prefix):
                    break
                # key = b'__cubs__/<scene>/<frame>'
                _, scene, frame_s = key.decode().split('/', 2)
                frame = int(frame_s)
                packed = pickle.loads(val)
                M = packed['M']
                if M == 0:
                    out[(scene, frame)] = []
                else:
                    pos = packed['pos']; dims = packed['dims']; yaw = packed['yaw']
                    out[(scene, frame)] = [
                        {'pos': pos[j], 'dims': dims[j], 'yaw': float(yaw[j])}
                        for j in range(M)
                    ]
        env.close()
        return out

    def _load_inst(self, idx: int) -> dict:
        # idx is in [0, len_fnames * oversample); modulo to wrap to file index
        i = idx % len(self.fnames)
        if self._cache is not None:
            return self._cache[i]
        if self._use_lmdb:
            if self._lmdb_env is None:
                self._open_lmdb()
            with self._lmdb_env.begin(write=False) as txn:
                blob = txn.get(self.fnames[i].encode())
            return _unpack_lmdb_inst(blob, self._cubs_map)
        return torch.load(self.inst_dir / self.fnames[i], weights_only=False)

    def __getitem__(self, idx: int):
        # Iterative re-roll on failure. The inner `_build_one_window` returns
        # None when the chosen idx has no valid window after self.max_tries
        # bucket samples. Previously this fallback was a *recursive* call
        # `return self[random.randint(...)]`, which blew the Python stack
        # (~978 deep RecursionError) when many indices in a row were bad.
        N = len(self)
        seen = {idx}
        for _ in range(1024):
            built = self._build_one_window(idx)
            if built is not None:
                return built
            # re-roll an unseen idx
            for _ in range(128):
                j = random.randint(0, N - 1)
                if j not in seen:
                    seen.add(j)
                    idx = j
                    break
            else:
                break
        raise RuntimeError(
            f"no valid window after 1024 re-rolls; "
            f"center_band={self.center_band}, min_pts={self.min_pts}"
        )

    def _build_one_window(self, idx: int):
        inst = self._load_inst(idx)
        # New cache stores jpg_bytes; old cache stored decoded uint8 'img'
        if 'jpg_bytes' in inst:
            IH, IW = int(inst['IH']), int(inst['IW'])
        else:
            IH, IW = int(inst['img'].shape[-2]), int(inst['img'].shape[-1])
        K = inst['K_full'].numpy()
        pts = inst['pts'].numpy()
        cp  = inst['cam_pos'].numpy()
        R_gt = inst['R_gt'].numpy()
        cubs = inst.get('cuboids', [])
        # Intensity is REQUIRED in V3-i caches. Caches built before 2026-05-13
        # do not have this key; rebuild via scripts/preprocessing/build_*_v3.py.
        if 'intensity' not in inst:
            raise KeyError(
                f"inst['intensity'] missing in cache {self.cache_dir.name}. "
                "Rebuild cache with intensity-aware build_*_v3.py."
            )
        intensity = inst['intensity'].numpy() if hasattr(inst['intensity'], 'numpy') else np.asarray(inst['intensity'])
        # Cache contract: intensity is already normalised to [0,1] by
        # scripts/data/migrate_intensity_norm.py. Reader is dataset-agnostic.
        intensity = np.asarray(intensity, dtype=np.float32)
        if intensity.size:
            intensity = np.clip(intensity, 0.0, 1.0).astype(np.float32, copy=False)

        # Cached: uv_full (N,2), z_cam (N,), is_obj (N,) — computed once at build / inject time
        if 'uv_full' in inst and 'z_cam' in inst:
            uv_full = inst['uv_full'].numpy()
            z = inst['z_cam'].numpy()
        else:
            T_gt = inst['T_gt'].numpy()
            homo = np.column_stack([pts, np.ones(len(pts))])
            pts_cam_gt = (T_gt @ homo.T)[:3].T
            z = pts_cam_gt[:, 2].astype(np.float32)
            uv_full = ((K @ pts_cam_gt.T)[:2] / np.maximum(pts_cam_gt[:, 2:].T, 1e-6)).T.astype(np.float32)
        if 'is_obj' in inst:
            is_obj_full = inst['is_obj'].numpy().astype(bool)
        else:
            is_obj_full = _is_obj_per_point(pts, cubs).astype(bool)

        # Tile-mode insts: uv_full is in PARENT-image coords + (tile_u0, tile_v0)
        # is the tile origin. Subtract once → everything downstream operates in
        # tile-local coords (0..tile_w, 0..tile_h). For full-frame insts these
        # keys are absent and we treat (0, 0) as the origin.
        tile_u0 = int(inst.get('tile_u0', 0))
        tile_v0 = int(inst.get('tile_v0', 0))
        if tile_u0 or tile_v0:
            uv_full = uv_full - np.array([tile_u0, tile_v0], dtype=np.float32)
        # in_box: 1.0 if a pt is strictly inside the tile (not in the padding
        # ring). Pivot candidates must be in_box to keep the random crop inside
        # the tile image. The padding-ring pts stay around as frustum context
        # only — if absent (legacy full-frame insts) treat all valid pts as
        # in_box.
        if 'in_box' in inst:
            in_box = inst['in_box'].numpy().astype(bool)
        else:
            in_box = np.ones(len(uv_full), dtype=bool)

        valid_in_image = ((z > 0.5) &
                          (uv_full[:,0] >= 0) & (uv_full[:,0] < IW) &
                          (uv_full[:,1] >= 0) & (uv_full[:,1] < IH))
        # Optional: keep pivots in a vertical band centred on the vanishing
        # point (= principal point cy from K). On a vehicle camera the
        # geometric center 0.5*IH is below the horizon — using cy keeps
        # objects (cars, people, signs) in frame instead of road / hood.
        # Falls back to 0.5*IH if K is missing.
        # NOTE: K is in PARENT-image coords; uv_full above was shifted by
        # (tile_u0, tile_v0) into tile-local. Subtract tile_v0 from cy too,
        # otherwise on tile caches the band sits off-frame.
        if self.center_band > 0.0:
            try:
                cy = float(K[1, 2]) - float(tile_v0)
            except Exception:
                cy = 0.5 * IH
            half = 0.5 * self.center_band * IH
            v_lo = max(0.0, cy - half)
            v_hi = min(float(IH), cy + half)
            in_band = (uv_full[:, 1] >= v_lo) & (uv_full[:, 1] < v_hi)
            valid_in_image = valid_in_image & in_band
        # Pivots: must be valid AND in_box. Frustum context still uses all pts.
        obj_idxs = np.where(is_obj_full & valid_in_image & in_box)[0]
        bg_mask  = (~is_obj_full) & valid_in_image & in_box
        # 10x5 grid for bg-pivot stratification. Drop the top + bottom GV-band
        # rows (sky / hood-and-near-road close-up): with GV=5 keep only rows
        # 1..3 (= mid 60% vertically). Anchored on cy (vanishing point) when
        # available so a sloped horizon doesn't bias the band off-screen.
        GU, GV = 10, 5
        cell_w = IW / GU; cell_h = IH / GV
        cell_u = np.clip((uv_full[:, 0] / cell_w).astype(int), 0, GU-1)
        cell_v = np.clip((uv_full[:, 1] / cell_h).astype(int), 0, GV-1)
        cell_id_full = cell_v * GU + cell_u
        try:
            cy_local = float(K[1, 2]) - float(tile_v0)
            cy_row = int(np.clip(cy_local / cell_h, 0, GV - 1))
        except Exception:
            cy_row = GV // 2
        # keep cy_row ± 1 (3 rows out of 5) → drops top sky and bottom hood/road
        ok_row = (cell_v >= max(0, cy_row - 1)) & (cell_v <= min(GV - 1, cy_row + 1))
        bg_cells = np.unique(cell_id_full[bg_mask & ok_row]) if (bg_mask & ok_row).any() else np.array([], dtype=int)

        S = self.img_size
        # If new cache: defer JPEG decode until crop is chosen (partial decode on the
        # cropped region only). Old cache: image is already decoded as a tensor.
        img_full = None if 'jpg_bytes' in inst else inst['img']  # (3, H, W) uint8 or None

        for _ in range(self.max_tries):
            # Pick pivot first so cs can scale with depth (close pivots need
            # larger crops to capture object edges; far pivots fit fine in small).
            if len(obj_idxs) > 0 and (len(bg_cells) == 0 or np.random.rand() < 0.5):
                i = obj_idxs[np.random.randint(len(obj_idxs))]
                pu, pv = uv_full[i]
            elif len(bg_cells) > 0:
                c = bg_cells[np.random.randint(len(bg_cells))]
                idxs = np.where(bg_mask & (cell_id_full == c))[0]
                i = idxs[np.random.randint(len(idxs))]
                pu, pv = uv_full[i]
            else:
                continue
            # Depth-aware cs sampling: <20m → up to 768 px (keep whole vehicle in frame),
            # else stay narrow [min_crop_px, max_crop_px]
            piv_z = float(z[i])
            if piv_z < 20.0:
                cs_lo, cs_hi = max(self.min_crop_px, 256), min(768, IW, IH)
            else:
                cs_lo, cs_hi = self.min_crop_px, self.max_crop_px
            cs = int(np.random.randint(cs_lo, cs_hi + 1))
            cs = min(cs, IW, IH)
            # Depth-dependent zoom-in aug: shrink cs by up to scale_max(z) so the
            # final image has the same content but VFP-equivalent of "viewing
            # this region from `scale` x farther". scale_max(z) goes from 1.0 at
            # z<20m (no zoom) up to 2.0 at z>=100m. Far pivots only — close-up
            # zoom on near objects creates artifacts (see project_uvemb_query
            # _curriculum stage 4 design rationale).
            if getattr(self, 'zoom_aug', False) and piv_z >= 20.0:
                t = min(1.0, (piv_z - 20.0) / 80.0)         # 0 at 20m, 1 at 100m+
                scale_max = 1.0 + t * 1.0                   # 1.0 → 2.0
                # 20m → 1.0–1.2 (effectively 0.05–0.20), 100m → 1.0–2.0
                cap = 1.0 + t * (scale_max - 1.0)
                scale = float(np.random.uniform(1.0, max(1.0, scale_max)))
                cs = max(self.min_crop_px // 4, int(cs / scale))
            u0 = int(np.clip(pu - cs/2, 0, IW - cs))
            v0 = int(np.clip(pv - cs/2, 0, IH - cs))

            # Pre-filter to crop+10% padding using cached uv_full → cap to n_full
            pad_px = int(cs * 0.10)
            in_pad = ((uv_full[:, 0] >= u0 - pad_px) & (uv_full[:, 0] < u0 + cs + pad_px) &
                      (uv_full[:, 1] >= v0 - pad_px) & (uv_full[:, 1] < v0 + cs + pad_px) &
                      (z > 0.5))
            cand_idx = np.where(in_pad)[0]
            if len(cand_idx) < self.min_pts:
                continue
            if len(cand_idx) > self.n_full:
                cand_idx = np.random.choice(cand_idx, size=self.n_full, replace=False)
            pts_c = pts[cand_idx]                       # (M<=2000, 3)
            intens_c = intensity[cand_idx]              # (M,) per-pt lidar intensity
            uv_gt_c = uv_full[cand_idx]                 # (M, 2) full-image px

            # Perturbation sampling:
            #   pose_frame='orig': sample (t, ypr) uniform in orig camera axes (legacy)
            #   pose_frame='vcam': sample uniform in tile's VCAM frame so labels are
            #     crop-position-AGNOSTIC (identical observations → identical labels
            #     regardless of tile location). Convert to orig for projection.
            if self.pose_frame == 'vcam':
                u_c = u0 + 0.5 * cs
                v_c = v0 + 0.5 * cs
                ray = np.array([(u_c - K[0,2]) / K[0,0],
                                 (v_c - K[1,2]) / K[1,1],
                                 1.0], dtype=np.float64)
                r_i = ray / (np.linalg.norm(ray) + 1e-12)
                z_ax = np.array([0., 0., 1.])
                axis = np.cross(r_i, z_ax)
                an = np.linalg.norm(axis)
                if an < 1e-9:
                    R_o_v = np.eye(3) if r_i[2] > 0 else -np.eye(3)
                else:
                    axis = axis / an
                    angle = float(np.arccos(np.clip(r_i @ z_ax, -1.0, 1.0)))
                    R_o_v = Rotation.from_rotvec(axis * angle).as_matrix()
                # Sample in VCAM frame: t_v ∈ R³, (yaw_v, pitch_v) ∈ R² (roll_v=0)
                t_vcam = (np.random.rand(3) * 2 - 1) * self.max_offset_m
                ypr_vcam = np.zeros(3, dtype=np.float64)
                ypr_vcam[0] = (np.random.rand() * 2 - 1) * self.max_rot_deg  # yaw
                ypr_vcam[1] = (np.random.rand() * 2 - 1) * self.max_rot_deg  # pitch
                # roll_vcam left 0 (confounded with t_x/t_y per tile)
                # Convert to orig for projection
                R_pert_vcam = Rotation.from_euler('zyx', ypr_vcam, degrees=True).as_matrix()
                R_pert_orig = R_o_v.T @ R_pert_vcam @ R_o_v
                t_delta = R_o_v.T @ t_vcam
                ypr = Rotation.from_matrix(R_pert_orig).as_euler('zyx', degrees=True)
                # split_pert is asserted-off for vcam; keep the variables defined.
                t_delta1 = np.zeros(3, dtype=np.float64)
                ypr1     = np.zeros(3, dtype=np.float64)
                R_pre    = None
                cp_pre   = None
            else:
                if self.split_pert:
                    # Split sampling: each of (δ1, δ2) drawn uniformly in
                    # ±max_*. The composed total can exceed max_* by up to 2×
                    # but the *delivered hint* δ1 is bounded by max_*, which
                    # is exactly the regime the deployed pose_emb will see at
                    # cross-frame inference time.
                    t_delta1 = (np.random.rand(3) * 2 - 1) * self.max_offset_m
                    ypr1     = (np.random.rand(3) * 2 - 1) * self.max_rot_deg
                    t_delta2 = (np.random.rand(3) * 2 - 1) * self.max_offset_m
                    ypr2     = (np.random.rand(3) * 2 - 1) * self.max_rot_deg
                    R_d1 = Rotation.from_euler('zyx', ypr1, degrees=True).as_matrix()
                    R_d2 = Rotation.from_euler('zyx', ypr2, degrees=True).as_matrix()
                    # Compose: δ1 first (the "known hint"), then δ2 (residual).
                    R_off_local = R_d1 @ R_d2
                    ypr = Rotation.from_matrix(R_off_local).as_euler('zyx', degrees=True)
                    t_delta = t_delta1 + t_delta2
                    # R_pre / cp_pre = pose after applying ONLY δ1 (no δ2).
                    R_pre  = R_gt @ R_d1
                    cp_pre = cp + t_delta1
                else:
                    t_delta = (np.random.rand(3) * 2 - 1) * self.max_offset_m
                    ypr     = (np.random.rand(3) * 2 - 1) * self.max_rot_deg
                    t_delta1 = np.zeros(3, dtype=np.float64)
                    ypr1     = np.zeros(3, dtype=np.float64)
                    R_pre  = None
                    cp_pre = None
            R_off = R_gt @ Rotation.from_euler('zyx', ypr, degrees=True).as_matrix()
            cp_off = cp + t_delta
            # Intrinsic fx/fy multiplicative perturbation (independent — left/right
            # PS cams show fx/fy don't drift together, so train them separately).
            dfx_pct = float(np.random.uniform(-self.max_fx_pct, self.max_fx_pct)) \
                       if self.max_fx_pct > 0 else 0.0
            dfy_pct = float(np.random.uniform(-self.max_fy_pct, self.max_fy_pct)) \
                       if self.max_fy_pct > 0 else 0.0
            K_pert = K.copy()
            if dfx_pct != 0.0: K_pert[0, 0] = K[0, 0] * (1.0 + dfx_pct)
            if dfy_pct != 0.0: K_pert[1, 1] = K[1, 1] * (1.0 + dfy_pct)
            # 8-vec perturbation label for the optional CLS frame-pose head.
            # Layout depends on pose_frame:
            #   'orig': (tx, ty, tz, yaw_deg, pitch_deg, roll_deg, dfx_pct, dfy_pct)
            #           — model frame_pose_dof=8 takes full 6-DoF SE3 + fx/fy.
            #   'vcam': (tx_v, ty_v, tz_v, yaw_v_deg, pitch_v_deg, dfx_pct, dfy_pct, 0)
            #           — roll_v dropped (slot 7 unused), fx/fy moved to slots 5,6 so
            #           model frame_pose_dof=5 cleanly targets 5-DoF pose, dof=7 picks
            #           up fx/fy too. The CLS head learns one frame-agnostic mapping
            #           (input → VCAM 5/7-DoF); downstream BA combines per-tile
            #           (μ_v, Σ_v) via J_i = R_orig→vcam_i.
            # Translation in meters, ypr in DEGREES, fx/fy as fractional percent.
            if self.pose_frame == 'vcam':
                pert_vec = np.array([t_vcam[0], t_vcam[1], t_vcam[2],
                                      ypr_vcam[0], ypr_vcam[1],
                                      dfx_pct, dfy_pct, 0.0], dtype=np.float32)
            else:
                pert_vec = np.array([t_delta[0], t_delta[1], t_delta[2],
                                      ypr[0], ypr[1], ypr[2],
                                      dfx_pct, dfy_pct], dtype=np.float32)
            # δ1 6-vec (tx, ty, tz, yaw_deg, pitch_deg, roll_deg) — the "hint"
            # consumed by pose_emb. Zero when split_pert is off (pose_emb sees
            # SE3=0, i.e. legacy calib mode).
            delta1_se3 = np.array([t_delta1[0], t_delta1[1], t_delta1[2],
                                    ypr1[0], ypr1[1], ypr1[2]], dtype=np.float32)
            built = self.build_window(
                inst, pts_c, intens_c, uv_gt_c, cand_idx, is_obj_full,
                u0, v0, cs, K, R_off, cp_off, K_pert, cp, pert_vec,
                tile_u0, tile_v0, img_full, IW, IH,
                R_pre=R_pre, cp_pre=cp_pre, delta1_se3=delta1_se3,
            )
            if built is None:
                continue
            return built

        # No valid window for this idx after max_tries. Signal to the caller
        # so the outer loop can re-roll a different idx without recursing
        # (the previous form `return self[random.randint(...)]` blew the
        # Python stack on tile caches where many indices in a row had empty
        # bg_cells, ~978 deep RecursionError).
        return None

    # ─── Explicit-perturbation API for eval / BA / multi-tile demos ──────
    # Same projection / crop / bucketing as __getitem__, but caller supplies
    # (t_delta, ypr_deg) so multiple tiles of the same frame can share one
    # rig-level perturbation (= what BA needs). Crop position is the tile's
    # full extent (u0=v0=0, cs=tile_size) so this works on tile_cutter outputs.
    def apply_perturbation_explicit(self, idx: int,
                                     t_delta: np.ndarray,
                                     ypr_deg: np.ndarray,
                                     dfx_pct: float = 0.0,
                                     dfy_pct: float = 0.0):
        inst = self._load_inst(idx)
        if 'jpg_bytes' in inst:
            IH, IW = int(inst['IH']), int(inst['IW'])
        else:
            IH, IW = int(inst['img'].shape[-2]), int(inst['img'].shape[-1])
        K = inst['K_full'].numpy()
        pts = inst['pts'].numpy()
        cp  = inst['cam_pos'].numpy()
        R_gt = inst['R_gt'].numpy()
        cubs = inst.get('cuboids', [])
        intensity = inst['intensity'].numpy() if hasattr(inst['intensity'], 'numpy') \
                    else np.asarray(inst['intensity'])
        intensity = np.clip(np.asarray(intensity, dtype=np.float32), 0.0, 1.0)

        if 'uv_full' in inst and 'z_cam' in inst:
            uv_full = inst['uv_full'].numpy()
            z = inst['z_cam'].numpy()
        else:
            T_gt = inst['T_gt'].numpy()
            homo = np.column_stack([pts, np.ones(len(pts))])
            pts_cam_gt = (T_gt @ homo.T)[:3].T
            z = pts_cam_gt[:, 2].astype(np.float32)
            uv_full = ((K @ pts_cam_gt.T)[:2] / np.maximum(pts_cam_gt[:, 2:].T, 1e-6)).T.astype(np.float32)
        if 'is_obj' in inst:
            is_obj_full = inst['is_obj'].numpy().astype(bool)
        else:
            is_obj_full = _is_obj_per_point(pts, cubs).astype(bool)

        tile_u0 = int(inst.get('tile_u0', 0))
        tile_v0 = int(inst.get('tile_v0', 0))
        if tile_u0 or tile_v0:
            uv_full = uv_full - np.array([tile_u0, tile_v0], dtype=np.float32)
        img_full = None if 'jpg_bytes' in inst else inst['img']

        # Crop = the full tile (no random pivot). Pre-filter pts inside the
        # padded bbox.
        cs = min(IW, IH)
        u0 = (IW - cs) // 2
        v0 = (IH - cs) // 2
        pad_px = int(cs * 0.10)
        in_pad = ((uv_full[:, 0] >= u0 - pad_px) & (uv_full[:, 0] < u0 + cs + pad_px) &
                  (uv_full[:, 1] >= v0 - pad_px) & (uv_full[:, 1] < v0 + cs + pad_px) &
                  (z > 0.5))
        cand_idx = np.where(in_pad)[0]
        if len(cand_idx) < self.min_pts:
            return None
        if len(cand_idx) > self.n_full:
            cand_idx = np.random.choice(cand_idx, size=self.n_full, replace=False)
        pts_c    = pts[cand_idx]
        intens_c = intensity[cand_idx]
        uv_gt_c  = uv_full[cand_idx]

        # Build R_off / cp_off / K_pert from the supplied perturbation.
        ypr = np.asarray(ypr_deg, dtype=np.float64).reshape(3)
        t   = np.asarray(t_delta, dtype=np.float64).reshape(3)
        R_off  = R_gt @ Rotation.from_euler('zyx', ypr, degrees=True).as_matrix()
        cp_off = cp + t
        K_pert = K.copy()
        if dfx_pct != 0.0: K_pert[0, 0] = K[0, 0] * (1.0 + dfx_pct)
        if dfy_pct != 0.0: K_pert[1, 1] = K[1, 1] * (1.0 + dfy_pct)
        pert_vec = np.array([t[0], t[1], t[2], ypr[0], ypr[1], ypr[2],
                              dfx_pct, dfy_pct], dtype=np.float32)
        delta1_se3 = np.zeros(6, dtype=np.float32)
        return self.build_window(
            inst, pts_c, intens_c, uv_gt_c, cand_idx, is_obj_full,
            u0, v0, cs, K, R_off, cp_off, K_pert, cp, pert_vec,
            tile_u0, tile_v0, img_full, IW, IH,
            R_pre=None, cp_pre=None, delta1_se3=delta1_se3,
        )

    # ─── Shared helper used by both training (__getitem__) and inference demos.
    # ─── Takes pre-sampled pts / perturbation / crop, returns the same tuple
    # ─── __getitem__ used to inline-build. Keeping this on the class so the
    # ─── browser demo can call it directly without duplicating projection /
    # ─── bucket-bin / rep-selection logic.
    def build_window(self, inst, pts_c, intens_c, uv_gt_c, cand_idx, is_obj_full,
                       u0, v0, cs, K, R_off, cp_off, K_pert, cp, pert_vec,
                       tile_u0, tile_v0, img_full, IW, IH,
                       R_pre=None, cp_pre=None, delta1_se3=None):
        """Apply a given perturbation+crop to an inst → DataLoader sample tuple.
        Returns None when the crop ends up with < self.min_pts in-view points;
        caller decides whether to retry (training) or report failure (demo).

        split_pert (R_pre, cp_pre, delta1_se3): when R_pre / cp_pre are given,
        the *target* uv (true_uvd[..., :2]) becomes the reprojection AT THE
        δ1-perturbed pose, not the GT pose. Then `gt = true - dist` is the
        residual the network must regress AFTER pose_emb has consumed δ1 as a
        hint. delta1_se3 (6-vec, tx/ty/tz + yaw/pitch/roll deg) flows through
        as a per-sample tensor for the model forward.
        """
        S = self.img_size
        R_inv = R_off.T.astype(np.float32)
        t_inv = (-(R_off.T @ cp_off)).astype(np.float32)
        pts_cam_off = pts_c @ R_inv.T + t_inv       # (M, 3)
        z_off = pts_cam_off[:, 2]
        if inst.get('is_fisheye', False) and 'distortion' in inst:
            # Kannala-Brandt fisheye (e.g. ZOD). Pinhole projection would
            # diverge at the edges where theta > arctan(image_diagonal/2 fx).
            # Re-project via the same lens model the cache used at build time.
            _dist = inst['distortion'].numpy() if hasattr(inst['distortion'], 'numpy') \
                    else np.asarray(inst['distortion'], dtype=np.float32)
            _x, _y, _z = pts_cam_off[:, 0], pts_cam_off[:, 1], pts_cam_off[:, 2]
            _r = np.sqrt(_x * _x + _y * _y)
            _theta = np.arctan2(_r, np.maximum(_z, 1e-6))
            _t2 = _theta * _theta
            _td = _theta * (1.0 + _dist[0] * _t2 + _dist[1] * _t2 ** 2
                                + _dist[2] * _t2 ** 3 + _dist[3] * _t2 ** 4)
            _r_safe = np.where(_r > 1e-9, _r, 1.0)
            _u = K_pert[0, 0] * (_td * _x / _r_safe) + K_pert[0, 2]
            _v = K_pert[1, 1] * (_td * _y / _r_safe) + K_pert[1, 2]
            uv_off_c = np.stack([_u, _v], axis=-1).astype(np.float32)
        else:
            uv_off_c = (pts_cam_off[:, :2] * (np.array([K_pert[0,0], K_pert[1,1]], dtype=np.float32))) / \
                       np.maximum(z_off[:, None], 1e-6) + np.array([K_pert[0,2], K_pert[1,2]], dtype=np.float32)
        # Tile mode: K_full is unchanged (parent coords) so the freshly-projected
        # uv_off_c lives in parent image coords. Subtract the tile origin so it
        # matches the already-tile-local uv_full / u0 / v0.
        if tile_u0 or tile_v0:
            uv_off_c = uv_off_c - np.array([tile_u0, tile_v0], dtype=np.float32)

        # split_pert path: project the SAME pts via R_pre / cp_pre (= GT⊕δ1)
        # using the *unperturbed* intrinsics K (δ1 is a pose-only hint; intrinsic
        # perturbation goes through K_pert→δ2 only). This produces uv_pre — the
        # network's NEW target. The uv_gt path is unused when uv_pre is set so
        # we just overwrite it below for symmetry with the existing uv_gt_loc
        # consumer (`true_uvd[..., :2] - dist_uvd[..., :2]` becomes
        # `uv_pre_loc - uv_off_loc` = -δ2-induced shift in tile-local px).
        uv_pre_c = None
        if R_pre is not None and cp_pre is not None:
            R_pre_inv = R_pre.T.astype(np.float32)
            t_pre_inv = (-(R_pre.T @ cp_pre)).astype(np.float32)
            pts_cam_pre = pts_c @ R_pre_inv.T + t_pre_inv      # (M, 3)
            z_pre = pts_cam_pre[:, 2]
            if inst.get('is_fisheye', False) and 'distortion' in inst:
                _dist = inst['distortion'].numpy() if hasattr(inst['distortion'], 'numpy') \
                        else np.asarray(inst['distortion'], dtype=np.float32)
                _x, _y, _z = pts_cam_pre[:, 0], pts_cam_pre[:, 1], pts_cam_pre[:, 2]
                _r = np.sqrt(_x * _x + _y * _y)
                _theta = np.arctan2(_r, np.maximum(_z, 1e-6))
                _t2 = _theta * _theta
                _td = _theta * (1.0 + _dist[0] * _t2 + _dist[1] * _t2 ** 2
                                    + _dist[2] * _t2 ** 3 + _dist[3] * _t2 ** 4)
                _r_safe = np.where(_r > 1e-9, _r, 1.0)
                _u = K[0, 0] * (_td * _x / _r_safe) + K[0, 2]
                _v = K[1, 1] * (_td * _y / _r_safe) + K[1, 2]
                uv_pre_c = np.stack([_u, _v], axis=-1).astype(np.float32)
            else:
                uv_pre_c = (pts_cam_pre[:, :2] * (np.array([K[0,0], K[1,1]], dtype=np.float32))) / \
                           np.maximum(z_pre[:, None], 1e-6) + np.array([K[0,2], K[1,2]], dtype=np.float32)
            if tile_u0 or tile_v0:
                uv_pre_c = uv_pre_c - np.array([tile_u0, tile_v0], dtype=np.float32)

        in_crop_off = ((uv_off_c[:, 0] >= u0) & (uv_off_c[:, 0] < u0 + cs) &
                       (uv_off_c[:, 1] >= v0) & (uv_off_c[:, 1] < v0 + cs) &
                       (z_off > 0.5))
        if in_crop_off.sum() < self.min_pts:
            return None

        # 16x16 sub-grid representative selection — fully vectorized
        scale = S / cs
        uv_local = np.stack([(uv_off_c[in_crop_off, 0] - u0) * scale,
                             (uv_off_c[in_crop_off, 1] - v0) * scale], axis=1)
        grid_n = self.grid_n
        cell_S = float(S) / grid_n
        ci_u = np.clip((uv_local[:, 0] / cell_S).astype(int), 0, grid_n - 1)
        ci_v = np.clip((uv_local[:, 1] / cell_S).astype(int), 0, grid_n - 1)
        cell_id = ci_v * grid_n + ci_u
        if self.rep_strategy == 'nearest_cam':
            # Tiebreak by depth: smallest-z (closest to camera) wins per cell.
            # Stable against fg/bg flips when a cell straddles an object edge.
            z_in_crop = z_off[in_crop_off].astype(np.float32)
            score = z_in_crop                                     # min-first
        else:
            # Legacy: distance to cell center in uv plane.
            cu_c = (ci_u + 0.5) * cell_S
            cv_c = (ci_v + 0.5) * cell_S
            score = (uv_local[:, 0] - cu_c) ** 2 + (uv_local[:, 1] - cv_c) ** 2
        order = np.lexsort((score, cell_id))           # primary cell_id, secondary score
        _, first_pos = np.unique(cell_id[order], return_index=True)
        sel = order[first_pos]                        # one rep per occupied cell
        sub_idx = np.where(in_crop_off)[0][sel]      # idx into cand_idx
        pts_sel = pts_c[sub_idx]                     # (Nrep, 3)
        intens_sel = intens_c[sub_idx].astype(np.float32)  # (Nrep,)
        uv_gt_sel  = uv_gt_c[sub_idx]
        uv_off_sel = uv_off_c[sub_idx]

        uv_gt_loc  = ((uv_gt_sel  - np.array([u0, v0], dtype=np.float32)) * scale).astype(np.float32)
        uv_off_loc = ((uv_off_sel - np.array([u0, v0], dtype=np.float32)) * scale).astype(np.float32)
        # split_pert: target uv is the δ1-perturbed projection (uv_pre). Reduces
        # the network's regression target to the δ2-induced shift only.
        if uv_pre_c is not None:
            uv_pre_sel = uv_pre_c[sub_idx]
            uv_target_loc = ((uv_pre_sel - np.array([u0, v0], dtype=np.float32)) * scale).astype(np.float32)
        else:
            uv_target_loc = uv_gt_loc
        dist_m = (np.linalg.norm(pts_sel - cp, axis=1) / 100.0).astype(np.float32)
        is_obj = is_obj_full[cand_idx[sub_idx]].astype(np.float32)

        # (N, 5): [u, v, d, is_obj, intensity]
        # idx 0-3 stay for backward compat; intensity is the new 5th channel.
        # uv_target_loc == uv_gt_loc when split_pert is off (legacy calib),
        # uv_pre_loc when split_pert is on (target = -δ2 residual after pose_emb).
        true_uvd = np.concatenate([uv_target_loc, dist_m[:, None], is_obj[:, None], intens_sel[:, None]], axis=1)
        dist_uvd = np.concatenate([uv_off_loc, dist_m[:, None], is_obj[:, None], intens_sel[:, None]], axis=1)

        # Original-camera-frame solver inputs:
        #   pts_cam_orig : cam-frame XYZ (m), pre-perturbation. J(X,Y,Z,K) consumes
        #                  this directly so units stay metric and original-camera.
        #   duv_orig     : Δuv at original-camera resolution (px). Corresponds to
        #                  uv_gt_sel - uv_off_sel before any local-128px scaling.
        #   K_orig       : original intrinsics (3, 3). The solver uses fx_orig etc.
        #                  so J's pixel-side stays in original-camera px.
        # Network input (img/dist_uvd/vfp) stays local 128px; the solver path is a
        # parallel exit from build_window. The W (or Σ_uv) the network predicts in
        # local px is converted to original-camera px via (cs/S)² downstream.
        pts_cam_orig = pts_sel.astype(np.float32)                      # (Nrep, 3) m
        duv_orig     = (uv_gt_sel - uv_off_sel).astype(np.float32)     # (Nrep, 2) px (orig)
        # K_orig: PARENT-camera intrinsics, exactly as stored in the cache.
        # The GN solver only needs (fx, fy) for J = ∂(u,v)/∂δ — cx, cy don't
        # enter J. duv_orig is tile-invariant (uv_gt - uv_off cancels tile_u0,v0)
        # so the solver runs purely in parent-camera SE3 units.
        K_orig = K.astype(np.float32)

        if img_full is None:
            # TurboJPEG partial decode of just the crop region (~4.5ms for 384px MCU-aligned).
            # DDAD stores PNG bytes in 'jpg_bytes' — TJ chokes on those, fall back to PIL.
            blob = inst['jpg_bytes']
            is_jpeg = (len(blob) > 2 and blob[0] == 0xff and blob[1] == 0xd8)
            ju0 = (u0 // _MCU) * _MCU
            jv0 = (v0 // _MCU) * _MCU
            ju1 = min(IW, ((u0 + cs + _MCU - 1) // _MCU) * _MCU)
            jv1 = min(IH, ((v0 + cs + _MCU - 1) // _MCU) * _MCU)
            jw, jh = ju1 - ju0, jv1 - jv0
            if _HAVE_TJ and is_jpeg:
                cropped = _TJ_INST.crop(blob, ju0, jv0, jw, jh, preserve=False)
                arr = np.asarray(_TJ_INST.decode(cropped, pixel_format=_TJ_PF_RGB))[:jh, :jw]
            else:
                import io
                from PIL import Image as _PILImage
                full = np.asarray(_PILImage.open(io.BytesIO(blob)).convert('RGB'),
                                  dtype=np.uint8)
                arr = full[jv0:jv1, ju0:ju1]
            # Slice to exact (cs, cs) inside the MCU-padded region
            arr = arr[v0 - jv0:v0 - jv0 + cs, u0 - ju0:u0 - ju0 + cs]
            img_crop = torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous().float().unsqueeze(0)
        else:
            img_crop = img_full[:, v0:v0+cs, u0:u0+cs].float().unsqueeze(0)
        img_crop = F.interpolate(img_crop, size=(S, S), mode='bilinear',
                                  align_corners=False).squeeze(0)
        # uint8 で渡す → IPC 4x 軽量化、GPU 側で .float()/255.0 する
        img_crop = img_crop.clamp_(0, 255).to(torch.uint8)

        vfp = float(K[0, 0]) * S / cs

        # Dense raw point cloud for frustum encoder per-cell context
        # (uv_off in local crop px, depth normalized /100 to match dist_uvd convention).
        uv_full_loc = np.stack([(uv_off_c[in_crop_off, 0] - u0) * scale,
                                 (uv_off_c[in_crop_off, 1] - v0) * scale], axis=1).astype(np.float32)
        d_full = (z_off[in_crop_off] / 100.0).astype(np.float32)
        intens_full = intens_c[in_crop_off].astype(np.float32)
        # bucket_uvd is 4D [u, v, d, intensity] — both point query and frustum
        # KV consume the same 4-ch view so intensity participates in cross-attn.
        uvd_full_raw = np.concatenate([uv_full_loc, d_full[:, None],
                                       intens_full[:, None]], axis=1)

        # ── Bucketed cell layout: pre-bin pts into a fixed (G², K) grid so
        # the model can read 3×3 neighbor cells per query in O(K)·O(9) work
        # instead of brute-force O(Nkv) per query. K is fixed per cell so
        # collate is a vanilla stack — no per-batch ragged padding.
        G = int(self.grid_n)
        K_per_cell = int(getattr(self, 'k_per_cell', 8))
        cell_S = float(S) / G
        cu = np.clip((uv_full_loc[:, 0] / cell_S).astype(np.int32), 0, G - 1)
        cv = np.clip((uv_full_loc[:, 1] / cell_S).astype(np.int32), 0, G - 1)
        cell_id = cv * G + cu                                    # (n_raw,)
        n_raw = uvd_full_raw.shape[0]
        # Random shuffle so that "first K per cell" picks K random pts when
        # the cell is over-full. Cheap O(n_raw) permutation.
        shuf = np.random.permutation(n_raw)
        sorted_idx = shuf[np.argsort(cell_id[shuf], kind='stable')]
        sorted_uvd = uvd_full_raw[sorted_idx]                    # (n_raw, 3)
        sorted_cid = cell_id[sorted_idx]                         # (n_raw,)
        # within-cell rank: 0,1,2,... per cell. Take only those <K.
        counts = np.bincount(sorted_cid, minlength=G * G)
        cell_starts = np.zeros(G * G + 1, dtype=np.int64)
        cell_starts[1:] = counts.cumsum()
        intra = np.arange(n_raw, dtype=np.int64) - cell_starts[sorted_cid]
        keep_mask = intra < K_per_cell
        slots = intra[keep_mask]
        cells = sorted_cid[keep_mask]
        bucket_uvd  = np.zeros((G * G, K_per_cell, 4), dtype=np.float32)
        bucket_valid = np.zeros((G * G, K_per_cell), dtype=bool)
        bucket_uvd[cells, slots]  = sorted_uvd[keep_mask]
        bucket_valid[cells, slots] = True

        self._last_crop = dict(u0=int(u0), v0=int(v0), cs=int(cs),
                                scene=inst.get('scene'), frame=int(inst.get('frame', -1)))
        if delta1_se3 is None:
            delta1_se3 = np.zeros(6, dtype=np.float32)
        return (img_crop, torch.from_numpy(true_uvd), torch.from_numpy(dist_uvd),
                torch.tensor(vfp, dtype=torch.float32),
                torch.from_numpy(bucket_uvd), torch.from_numpy(bucket_valid),
                torch.from_numpy(pert_vec),
                torch.from_numpy(pts_cam_orig),
                torch.from_numpy(duv_orig),
                torch.from_numpy(K_orig),
                torch.tensor(float(cs), dtype=torch.float32),
                torch.from_numpy(delta1_se3))


def collate_full(batch):
    """Stack img/vfp + (G², K, 3) bucketed lidar grid + per-sample pert vec.
    Per-batch ragged padding only on the per-pivot true/dist tensors; the
    lidar bucket is fixed-size. pert_vec is per-sample (B, N_DOF) — 6-DoF SE3
    perturbation + (when N_DOF=8) Δfx_pct, Δfy_pct for the CLS frame-pose head."""
    # Tolerate older 6-tuple samples (no pert) for backward compat — pad with zeros.
    imgs    = torch.stack([s[0] for s in batch])
    trues   = [s[1] for s in batch]
    dists   = [s[2] for s in batch]
    vfps    = torch.stack([s[3] for s in batch])
    b_uvds  = torch.stack([s[4] for s in batch])
    b_valids= torch.stack([s[5] for s in batch])
    if len(batch[0]) >= 7:
        # Each sample's pert may be 6-vec (legacy) or 8-vec (with fx/fy). Pad to 8.
        v0 = batch[0][6]
        n_dof = v0.shape[-1] if hasattr(v0, 'shape') else len(v0)
        if n_dof < 8:
            pad = torch.zeros(8 - n_dof, dtype=torch.float32)
            pert_6vec = torch.stack([torch.cat([s[6], pad]) for s in batch])
        else:
            pert_6vec = torch.stack([s[6] for s in batch])     # (B, 8)
    else:
        pert_6vec = torch.zeros(len(batch), 8, dtype=torch.float32)
    Nmax = max(t.shape[0] for t in trues)
    B = len(trues)
    Cdim = trues[0].shape[1]
    true_p = torch.zeros(B, Nmax, Cdim)
    dist_p = torch.zeros(B, Nmax, Cdim)
    pad    = torch.ones(B, Nmax, dtype=torch.bool)
    for k, (t, d) in enumerate(zip(trues, dists)):
        n = t.shape[0]
        true_p[k, :n] = t
        dist_p[k, :n] = d
        pad[k, :n] = False

    # Original-camera-frame solver inputs (None when the dataset still emits the
    # old 7-tuple sample). pts_cam_orig and duv_orig share the (Nmax) padding of
    # true_p/dist_p so the same `pad` mask applies. K_orig and cs are per-sample.
    has_orig = len(batch[0]) >= 11
    if has_orig:
        pts_cam_orig = torch.zeros(B, Nmax, 3, dtype=torch.float32)
        duv_orig     = torch.zeros(B, Nmax, 2, dtype=torch.float32)
        for k, s in enumerate(batch):
            n = s[7].shape[0]
            pts_cam_orig[k, :n] = s[7]
            duv_orig[k, :n]     = s[8]
        K_orig = torch.stack([s[9] for s in batch])                 # (B, 3, 3)
        cs_t   = torch.stack([s[10] for s in batch])                # (B,)
    else:
        pts_cam_orig = duv_orig = K_orig = cs_t = None

    # δ1 hint (split_pert): per-sample 6-vec [tx, ty, tz, yaw, pitch, roll deg]
    # consumed by pose_emb. Always emitted in the 12-tuple; zeros when split_pert
    # is off (network sees SE3=0 → backward-compat with legacy calib mode).
    has_d1 = len(batch[0]) >= 12
    if has_d1:
        delta1_se3 = torch.stack([s[11] for s in batch])            # (B, 6)
    else:
        delta1_se3 = torch.zeros(B, 6, dtype=torch.float32)

    return (imgs, true_p, dist_p, pad, vfps, b_uvds, b_valids, pert_6vec,
            pts_cam_orig, duv_orig, K_orig, cs_t, delta1_se3)
