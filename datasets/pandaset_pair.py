"""PandaSet cross-frame pair dataset — multi-scene, scene-level split.

Flow:
  1. Index all scenes under `scenes_root` (e.g. /mnt/mininas/datasets/pandaset).
  2. Shuffle (deterministic seed) → first `train_frac` go to train, rest to val.
     This is SCENE-level split; val scenes are never seen in training.
  3. For each scene in this split, precompute per-frame (uv, z, in_view_mask)
     in the scene's own camera.
  4. __getitem__ picks a random scene → random fi_A → nearby fi_B within baseline,
     crops patches, samples perturbation, gathers in-patch LiDAR, projects
     to the other frame via T_AB_gt and T_AB_hat.

Everything in SI units: ypr in degrees, t in meters.
"""
import json, os, random, zipfile
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

try:
    import turbojpeg as _tj
    _HAVE_TJ = True
except ImportError:
    _HAVE_TJ = False


# ─── small geometry utilities ────────────────────────────────────────────────

def _quat_pos_to_mat(heading, position):
    """world→cam transform (inverse of the pose_quat provided by PandaSet)."""
    q = [heading['x'], heading['y'], heading['z'], heading['w']]
    R_c2w = Rotation.from_quat(q).as_matrix()
    t_c2w = np.array([position['x'], position['y'], position['z']])
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R_w2c
    M[:3, 3]  = t_w2c
    return M


def _proj_cam(pts_cam, K, dist=None):
    """Project camera-frame points to pixel uv, optionally with OpenCV radial
    distortion (k1, k2, k3). Accepts (N,3) or (3,) single point and returns
    (N,2) or (2,) accordingly.

    Distortion model (same as OpenCV pinhole + radial, no tangential):
        r² = x_n² + y_n²     where x_n = X/Z, y_n = Y/Z
        f  = 1 + k1 r² + k2 r⁴ + k3 r⁶
        u  = fx · (x_n · f) + cx
        v  = fy · (y_n · f) + cy
    dist=None (or all-zero) falls through to pure pinhole — bit-identical to
    the legacy path for existing PandaSet / NuScenes intrinsics.
    """
    single = (pts_cam.ndim == 1)
    if single:
        pts_cam = pts_cam[None, :]
    z = pts_cam[:, 2]
    zc = np.clip(z, 1e-6, None)
    x_n = pts_cam[:, 0] / zc
    y_n = pts_cam[:, 1] / zc
    if dist is not None and np.any(dist):
        r2 = x_n * x_n + y_n * y_n
        f = 1.0 + dist[0] * r2 + dist[1] * r2 * r2 + dist[2] * r2 * r2 * r2
        x_n = x_n * f
        y_n = y_n * f
    u = K[0, 0] * x_n + K[0, 2]
    v = K[1, 1] * y_n + K[1, 2]
    uv = np.stack([u, v], axis=1).astype(np.float32)
    return uv[0] if single else uv


def _project(pts_world, T_w2c, K, dist=None):
    """Project world points into camera. Returns (uv (N,2), z (N,)) in pixel coords."""
    N = pts_world.shape[0]
    homo = np.concatenate([pts_world, np.ones((N, 1), dtype=pts_world.dtype)], axis=1)
    pts_cam = (T_w2c @ homo.T)[:3].T  # (N, 3)
    z = pts_cam[:, 2].astype(np.float32)
    uv = _proj_cam(pts_cam, K, dist)
    return uv, z


def _ypr_t_to_mat(ypr_deg, t):
    R = Rotation.from_euler('zyx', ypr_deg, degrees=True).as_matrix()
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3]  = t
    return M


def _mat_to_ypr_t(M):
    ypr = Rotation.from_matrix(M[:3, :3]).as_euler('zyx', degrees=True).astype(np.float32)
    t   = M[:3, 3].astype(np.float32)
    return ypr, t


def _invert_mat(M):
    R = M[:3, :3]; t = M[:3, 3]
    Minv = np.eye(4, dtype=M.dtype)
    Minv[:3, :3] = R.T
    Minv[:3, 3]  = -R.T @ t
    return Minv


# ─── per-scene container ─────────────────────────────────────────────────────

class _SceneData:
    """Holds metadata + precomputed per-frame projections for ONE (scene, camera) pair."""

    def __init__(self, scene_root: Path, camera_name: str = 'front_camera'):
        self.root = scene_root
        self.camera_name = camera_name
        # identifier used by collate/debug: "<scene>/<cam>"
        self.scene_id = f'{scene_root.name}/{camera_name}'

        cam_dir = scene_root / 'camera' / camera_name

        # intrinsics (pinhole K + optional OpenCV radial distortion k1,k2,k3)
        intr = json.loads((cam_dir / 'intrinsics.json').read_text())
        self.K = np.array([[intr['fx'], 0, intr['cx']],
                           [0, intr['fy'], intr['cy']],
                           [0,          0,          1]], dtype=np.float32)
        self.dist = np.array([intr.get('k1', 0.0),
                              intr.get('k2', 0.0),
                              intr.get('k3', 0.0)], dtype=np.float32)

        # poses (world→cam for THIS camera)
        poses = json.loads((cam_dir / 'poses.json').read_text())
        self.n_frames = len(poses)
        self.T_w2c = np.stack([_quat_pos_to_mat(p['heading'], p['position']) for p in poses])

        # image/lidar paths (lidar is shared across cameras in a scene)
        self.image_paths = [cam_dir / f'{fi:02d}.jpg' for fi in range(self.n_frames)]
        self.lidar_paths = []
        for fi in range(self.n_frames):
            p = scene_root / f'lidar/{fi:02d}.pkl'
            if not p.exists():
                pgz = scene_root / f'lidar/{fi:02d}.pkl.gz'
                if pgz.exists():
                    p = pgz
            self.lidar_paths.append(p)

        # image size (assume all same in scene) from first image
        with Image.open(self.image_paths[0]) as im:
            self.IW, self.IH = im.size

    # ------------------------------------------------------------------ helpers
    # On-disk projection cache: precompute once per (scene, camera) to an npz
    # file; frame_data() mmap-reads it every call so the OS page cache handles
    # residency and worker RAM never accumulates.

    img_scale_div: int = 1
    _MCU: int = 16   # libjpeg-turbo crop alignment (4:2:0 → 16-px MCU)

    def load_image(self, fi):
        with Image.open(self.image_paths[fi]) as im:
            arr = np.array(im.convert('RGB'))
        if self.img_scale_div > 1:
            H, W = arr.shape[:2]
            h = H // self.img_scale_div
            w = W // self.img_scale_div
            arr = np.array(Image.fromarray(arr).resize((w, h), Image.BILINEAR))
        return arr

    def load_patch(self, fi, u0_full, v0_full, s_full):
        """Return an (s, s, 3) uint8 patch at full-res coords (u0, v0) with
        size s. Regions outside the image are zero-padded. Uses libjpeg-turbo
        lossless transform + SIMD decompress so only the enclosing MCU tiles
        are decoded (≈9× faster than full decode)."""
        u0, v0, s = int(u0_full), int(v0_full), int(s_full)
        out = np.zeros((s, s, 3), dtype=np.uint8)
        IW, IH = self.IW, self.IH
        # intersect with image
        su0 = max(0, u0); sv0 = max(0, v0)
        su1 = min(IW, u0 + s); sv1 = min(IH, v0 + s)
        if su1 <= su0 or sv1 <= sv0:
            return out
        if _HAVE_TJ:
            MCU = self._MCU
            ju0 = (su0 // MCU) * MCU
            jv0 = (sv0 // MCU) * MCU
            ju1 = min(IW, ((su1 + MCU - 1) // MCU) * MCU)
            jv1 = min(IH, ((sv1 + MCU - 1) // MCU) * MCU)
            jw, jh = ju1 - ju0, jv1 - jv0
            with open(self.image_paths[fi], 'rb') as f:
                blob = f.read()
            cropped_jpeg = _tj.transform(blob, crop=True,
                                          x=ju0, y=jv0, w=jw, h=jh,
                                          perfect=False)
            img = np.asarray(_tj.decompress(cropped_jpeg,
                                            pixelformat=_tj.PF.RGB))
            # img shape may exceed (jh, jw) if turbojpeg rounded; clamp.
            img = img[:jh, :jw]
            out[sv0 - v0:sv1 - v0, su0 - u0:su1 - u0] = \
                img[sv0 - jv0:sv1 - jv0, su0 - ju0:su1 - ju0]
        else:
            # PIL fallback: crop-before-decode (partial IDCT via libjpeg)
            with Image.open(self.image_paths[fi]) as im:
                sub = np.array(im.crop((su0, sv0, su1, sv1)).convert('RGB'))
            out[sv0 - v0:sv1 - v0, su0 - u0:su1 - u0] = sub
        return out

    def _proj_cache_path(self, fi):
        # packed binary: [N:uint32 LE][pts_w: N*12][uv: N*8][z: N*4][in_view: N]
        # → single file, single memmap, zero-copy reads.
        return self.root / '_proj_cache' / self.camera_name / f'{fi:02d}.bin'

    def _build_proj(self, fi):
        df = pd.read_pickle(self.lidar_paths[fi])
        pts_w = df[['x', 'y', 'z']].values.astype(np.float32)
        uv, z = _project(pts_w, self.T_w2c[fi], self.K, self.dist)
        in_view = ((z > 1.0) &
                   (uv[:, 0] > 0) & (uv[:, 0] < self.IW) &
                   (uv[:, 1] > 0) & (uv[:, 1] < self.IH))
        return pts_w, uv.astype(np.float32), z.astype(np.float32), in_view

    def _read_packed(self, p):
        m = np.memmap(p, dtype=np.uint8, mode='r')
        if m.size < 4:
            raise ValueError(f'cache too small: {p}')
        N = int(m[:4].view(np.uint32)[0])
        expect = 4 + N * (12 + 8 + 4 + 1)
        if m.size != expect:
            raise ValueError(f'cache size mismatch: {p} ({m.size} vs {expect} for N={N})')
        off = 4
        pts_w = m[off:off + N*12].view(np.float32).reshape(N, 3); off += N*12
        uv    = m[off:off + N*8 ].view(np.float32).reshape(N, 2); off += N*8
        z     = m[off:off + N*4 ].view(np.float32);               off += N*4
        in_view = m[off:off + N ].view(np.bool_)
        return pts_w, uv, z, in_view

    def _write_packed(self, p, pts_w, uv, z, in_view):
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f'.{os.getpid()}.tmp')
        N = pts_w.shape[0]
        with open(tmp, 'wb') as f:
            f.write(np.uint32(N).tobytes())
            f.write(np.ascontiguousarray(pts_w, dtype=np.float32).tobytes())
            f.write(np.ascontiguousarray(uv,    dtype=np.float32).tobytes())
            f.write(np.ascontiguousarray(z,     dtype=np.float32).tobytes())
            f.write(np.ascontiguousarray(in_view, dtype=np.bool_).tobytes())
        tmp.rename(p)

    def frame_data(self, fi):
        """Return (pts_world, uv_own_cam, z_own_cam, in_view_mask).
        Memmap-reads the packed binary cache; builds it on first access.
        Corrupt cache files are deleted and rebuilt transparently."""
        p = self._proj_cache_path(fi)
        if p.exists():
            try:
                return self._read_packed(p)
            except (ValueError, OSError):
                try: p.unlink()
                except OSError: pass
        pts_w, uv, z, in_view = self._build_proj(fi)
        try:
            self._write_packed(p, pts_w, uv, z, in_view)
        except OSError:
            pass
        return pts_w, uv, z, in_view

    def precompute_all(self, n_workers: int = 8, **_ignored):
        """Build the on-disk projection cache for every frame. Idempotent:
        skips frames whose cache file already exists. Runs in parent only.
        Does NOT hold results in memory — cache is the output."""
        from concurrent.futures import ThreadPoolExecutor
        missing = [fi for fi in range(self.n_frames)
                   if not self._proj_cache_path(fi).exists()]
        if not missing:
            return
        def _build_and_drop(fi):
            self.frame_data(fi)  # side-effect: writes npz
            return None
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for _ in ex.map(_build_and_drop, missing):
                pass


# ─── main dataset ────────────────────────────────────────────────────────────

class PandaSetCrossFrameDataset(Dataset):
    """Multi-scene PandaSet cross-frame pair dataset.

    Call with either:
        scenes_root='/mnt/mininas/datasets/pandaset' (auto-discover all scenes)
    or
        scene_roots=['/mnt/.../015', '/mnt/.../018', ...] (explicit list)

    Legacy single-scene call `scene_root='/mnt/.../015'` is also accepted
    and wrapped into a 1-element list (no train/val split then — whole scene
    used for whichever split is requested).
    """

    def __init__(self,
                 scenes_root: Union[str, None] = None,
                 scene_roots: Union[List[str], None] = None,
                 scene_root: Union[str, None] = None,          # legacy, single scene
                 split: str = 'train',
                 train_frac: float = 0.80,
                 img_size: int = 64,
                 baseline_range = (1, 5),
                 sigma_ypr: float = 0.3,
                 sigma_t:   float = 0.15,
                 max_points: int = 96,
                 crop_range = (128, 256),
                 virtual_epoch_len: int = 2000,
                 frame_train_frac: float = 1.0,
                 cameras: Union[List[str], str, None] = None,
                 n_frames: int = 2,        # how many frames the stacked path emits
                 use_stacked: bool = False, # True → _try_one_nframe (new path).
                                           # False → legacy _try_one (v51-v55 path).
                 triplet: bool = False,    # legacy: append M as aux KV in the
                                           # named-field pair output (used by v51-v55)
                 seed: int = 42):
        super().__init__()
        if n_frames not in (2, 3):
            raise NotImplementedError(f'n_frames={n_frames} not yet supported')
        self.img_size = img_size
        self.baseline_range = baseline_range
        self.sigma_ypr = sigma_ypr
        self.sigma_t   = sigma_t
        self.max_points = max_points
        self.crop_range = crop_range
        self.virtual_epoch_len = virtual_epoch_len
        self.split = split
        self.n_frames = n_frames
        # `triplet` keeps the legacy v51-v55 behaviour (named fields + M aux KV).
        # `use_stacked` opts into the new stacked-array path (any N >= 2).
        self.triplet = triplet
        self.use_stacked = use_stacked
        self.rng = np.random.default_rng(seed)

        # cameras: single str, comma-separated str, list, or None (→ front_camera).
        # Special value 'all' = use every camera that exists under each scene.
        if cameras is None:
            cameras = ['front_camera']
        elif isinstance(cameras, str):
            cameras = [c.strip() for c in cameras.split(',') if c.strip()]
        self.cameras = cameras

        # ── resolve scene list ──
        # scenes_root can be a single path or a comma-separated list of roots
        # (for mixing PandaSet + Waymo-converted scenes).
        if scene_roots is None and scenes_root is not None:
            roots = [Path(r) for r in str(scenes_root).split(',')]
            scene_roots = []
            for root in roots:
                # A scene is any direct child dir that has at least one camera.
                # (Relaxed from `.name.isdigit()` so Waymo segment names work.)
                for p in sorted(root.iterdir()):
                    if not (p.is_dir() and (p / 'camera').is_dir()):
                        continue
                    if self.cameras == ['all']:
                        # any subdir with intrinsics counts as a valid camera
                        has_cam = any((d / 'intrinsics.json').exists()
                                      for d in (p / 'camera').iterdir() if d.is_dir())
                    else:
                        has_cam = any((p / 'camera' / c / 'intrinsics.json').exists()
                                      for c in self.cameras)
                    if has_cam:
                        scene_roots.append(str(p))
        elif scene_roots is None and scene_root is not None:
            # legacy single scene: use whole scene for requested split
            scene_roots = [scene_root]
            train_frac = 1.0 if split == 'train' else 0.0
        elif scene_roots is None:
            raise ValueError('Must provide scenes_root, scene_roots, or scene_root')

        # ── scene-level split ──
        # CRITICAL: split at the physical-scene level (not scene×camera), so the
        # same physical scene's cameras never straddle train/val.
        if len(scene_roots) > 1 and 0.0 < train_frac < 1.0:
            shuffled = sorted(scene_roots)
            random.Random(seed).shuffle(shuffled)
            cutoff = int(len(shuffled) * train_frac)
            if split == 'train':
                split_roots = shuffled[:cutoff]
            elif split == 'val':
                split_roots = shuffled[cutoff:]
            else:
                split_roots = shuffled
        else:
            split_roots = scene_roots

        # ── expand (scene × camera) pairs, filtering out missing cameras ──
        scene_cam_pairs: List[tuple] = []
        for sr in split_roots:
            sp = Path(sr)
            if self.cameras == ['all']:
                cam_list = sorted(
                    d.name for d in (sp / 'camera').iterdir()
                    if d.is_dir() and (d / 'intrinsics.json').exists()
                )
            else:
                cam_list = [c for c in self.cameras
                            if (sp / 'camera' / c / 'intrinsics.json').exists()]
            for c in cam_list:
                scene_cam_pairs.append((sr, c))

        from concurrent.futures import ThreadPoolExecutor
        print(f'[PandaSetCrossFrameDataset/{split}] loading {len(split_roots)} scenes '
              f'× {len(self.cameras) if self.cameras != ["all"] else "all"} cams '
              f'= {len(scene_cam_pairs)} (scene, camera) pairs…',
              flush=True)

        def _load_scene(sr_cam):
            sr, cam = sr_cam
            scn = _SceneData(Path(sr), camera_name=cam)
            scn.precompute_all(n_workers=4)   # within-scene threads
            all_fi = list(range(scn.n_frames))
            if frame_train_frac < 1.0:
                cutoff = int(scn.n_frames * frame_train_frac)
                scn.fi_pool = (all_fi[:cutoff] if split == 'train'
                                else all_fi[cutoff:] if split == 'val'
                                else all_fi)
            else:
                scn.fi_pool = all_fi
            return scn

        with ThreadPoolExecutor(max_workers=8) as ex:
            self.scenes: List[_SceneData] = list(ex.map(_load_scene, scene_cam_pairs))

        for scn in self.scenes:
            print(f'  · {scn.scene_id}: {scn.n_frames} frames, pool={len(scn.fi_pool)}',
                  flush=True)

        print(f'[PandaSetCrossFrameDataset/{split}] ready. '
              f'{len(self.scenes)} (scene, cam) pairs, '
              f'{sum(len(s.fi_pool) for s in self.scenes)} total frames',
              flush=True)

    # ------------------------------------------------------------------ helpers

    def _resize_patch(self, patch_img, u0_full, v0_full, s_full):
        """Take a pre-cropped (s, s, 3) uint8 patch (already zero-padded for
        out-of-image by load_patch), resize to img_size, return (tensor, box)."""
        s = int(s_full)
        t = torch.from_numpy(patch_img).permute(2, 0, 1).float() / 255.0
        t = F.interpolate(t.unsqueeze(0),
                          size=(self.img_size, self.img_size),
                          mode='bilinear', align_corners=False).squeeze(0)
        return t, (int(u0_full), int(v0_full), s, s)

    def _crop_patch(self, img_cached, u0_full, v0_full, s_full, IW, IH, scale_div):
        """Pivot stays at exact patch center — pad outside-FOV with zeros so
        pose_emb's "pivot = patch center" invariant holds. Patch box returned
        in FULL-res coords (may include out-of-image area, which is zero-padded).

        img_cached is at full_res / scale_div; u0/v0/s are in FULL-res px.
        """
        s_full_i = int(s_full)
        u0i = int(u0_full); v0i = int(v0_full)
        u1i = u0i + s_full_i; v1i = v0i + s_full_i
        # cache-resolution target box
        u0c = u0i // scale_div; v0c = v0i // scale_div
        u1c = u1i // scale_div; v1c = v1i // scale_div
        cw = u1c - u0c; ch = v1c - v0c
        if cw < 2 or ch < 2:
            return None
        # clip cache box to valid image area
        src_u0 = max(0, u0c); src_v0 = max(0, v0c)
        src_u1 = min(img_cached.shape[1], u1c); src_v1 = min(img_cached.shape[0], v1c)
        out = np.zeros((ch, cw, img_cached.shape[2]), dtype=img_cached.dtype)
        inner_w = src_u1 - src_u0; inner_h = src_v1 - src_v0
        if inner_w > 0 and inner_h > 0:
            out_pad_left = src_u0 - u0c
            out_pad_top  = src_v0 - v0c
            out[out_pad_top:out_pad_top + inner_h,
                out_pad_left:out_pad_left + inner_w] = img_cached[src_v0:src_v1, src_u0:src_u1]
        t = torch.from_numpy(out).permute(2, 0, 1).float() / 255.0
        t = F.interpolate(t.unsqueeze(0), size=(self.img_size, self.img_size),
                           mode='bilinear', align_corners=False).squeeze(0)
        return t, (u0i, v0i, s_full_i, s_full_i)

    def _uv_to_patch_local(self, uv_full, crop_box):
        u0, v0, cw, ch = crop_box
        scale_u = self.img_size / cw
        scale_v = self.img_size / ch
        uv_local = np.stack([(uv_full[:, 0] - u0) * scale_u,
                              (uv_full[:, 1] - v0) * scale_v], axis=1)
        return uv_local.astype(np.float32)

    # -------------------------------------------------------------- main getter

    def __len__(self):
        return self.virtual_epoch_len

    def __getitem__(self, idx):
        for retry in range(30):
            if self.use_stacked:
                sample = self._try_one_nframe(idx + retry * 9973)
            else:
                sample = self._try_one(idx + retry * 9973)
            if sample is not None:
                return sample
        raise RuntimeError(
            f'could not form a valid {self.n_frames}-frame sample in 30 tries')

    # ─────────────────────── stacked-tensor reformatter ────────────────────
    # Wraps `_try_one()` and repacks its named-field output into per-frame
    # `(N, ...)` and per-pair `(N, N, ...)` tensors so the model's stacked
    # forward path can consume any N >= 2. For N=2 this is byte-equivalent
    # to the legacy pair output (no semantic divergence) — the data path is
    # the same physical sample, just reshaped. For N=3 we additionally
    # add M-as-query directions (M->A, M->B) on top of legacy triplet.

    def _try_one_nframe(self, idx):
        N = self.n_frames
        # N=3 stacked → full P1: 3 frames as queries, iid per-pair pose
        # perturbation, all 6 directions supervised. Built from scratch
        # (not via legacy reformat) because legacy treats pose_AM as GT.
        if N >= 3:
            return self._try_one_p1(idx)

        prev_triplet = self.triplet
        self.triplet = False               # N=2: pure pair from legacy
        try:
            s = self._try_one(idx)
        finally:
            self.triplet = prev_triplet
        if s is None:
            return None

        max_pts = self.max_points
        N_FULL  = s['uvd_A_full'].shape[0]
        IMG     = self.img_size

        # --- per-frame stacked tensors ---
        if N == 2:
            patches  = torch.stack([s['patch_A'].float(), s['patch_B'].float()], dim=0)
            uvd      = torch.stack([s['uvd_A'],          s['uvd_B']         ], dim=0)
            pad      = torch.stack([s['pad_A'],          s['pad_B']         ], dim=0)
            uvd_full = torch.stack([s['uvd_A_full'],     s['uvd_B_full']    ], dim=0)
            pad_full = torch.stack([s['pad_A_full'],     s['pad_B_full']    ], dim=0)
            fis      = torch.tensor([s['fi_A'], s['fi_B']], dtype=torch.long)
        elif N == 3:
            patches  = torch.stack([s['patch_A'].float(), s['patch_M'].float(),
                                    s['patch_B'].float()], dim=0)
            uvd      = torch.stack([s['uvd_A'],     s['uvd_M'],     s['uvd_B']    ], dim=0)
            pad      = torch.stack([s['pad_A'],     s['pad_M'],     s['pad_B']    ], dim=0)
            uvd_full = torch.stack([s['uvd_A_full'], s['uvd_M_full'], s['uvd_B_full']], dim=0)
            pad_full = torch.stack([s['pad_A_full'], s['pad_M_full'], s['pad_B_full']], dim=0)
            fis      = torch.tensor([s['fi_A'], s['fi_M'], s['fi_B']], dtype=torch.long)
        else:
            raise NotImplementedError(N)

        # --- per-pair (N, N) tensors ---
        pose_hat = torch.zeros(N, N, 6, dtype=torch.float32)
        uv_hat_arr = torch.zeros(N, N, max_pts, 2, dtype=torch.float32)
        uv_gt_arr  = torch.zeros(N, N, max_pts, 2, dtype=torch.float32)
        pad_dir    = torch.ones (N, N, max_pts,    dtype=torch.bool)
        # depth fields are diagnostic only when not in --uvd mode
        d_hat_arr  = torch.zeros(N, N, max_pts, dtype=torch.float32)
        d_gt_arr   = torch.zeros(N, N, max_pts, dtype=torch.float32)

        # A↔B (always present)
        pose_hat[0, N - 1] = s['pose_AB_6dof']
        pose_hat[N - 1, 0] = s['pose_BA_6dof']
        uv_hat_arr[0, N - 1] = s['uv_B_hat_of_A']
        uv_gt_arr [0, N - 1] = s['uv_B_gt_of_A']
        pad_dir   [0, N - 1] = s['pad_A']            # legacy filter already in/out checked
        uv_hat_arr[N - 1, 0] = s['uv_A_hat_of_B']
        uv_gt_arr [N - 1, 0] = s['uv_A_gt_of_B']
        pad_dir   [N - 1, 0] = s['pad_B']
        if 'd_B_hat_of_A' in s:
            d_hat_arr[0, N - 1] = s['d_B_hat_of_A']
            d_gt_arr [0, N - 1] = s['d_B_gt_of_A']
            d_hat_arr[N - 1, 0] = s['d_A_hat_of_B']
            d_gt_arr [N - 1, 0] = s['d_A_gt_of_B']

        if N == 3:
            # A↔M (legacy: pose_AM treated GT, so uv_M_hat_of_A == uv_M_gt_of_A.
            # A→M residual is structurally zero in PoC; loss masks would still
            # push σ→0. Mark pad_dir all True for these directions so step_n
            # skips them entirely until pose_AM perturbation is added.)
            pose_hat[0, 1] = s['pose_AM_6dof']
            pose_hat[1, 0] = -s['pose_AM_6dof']     # approx inverse; unused for loss
            uv_hat_arr[0, 1] = s['uv_M_hat_of_A']
            uv_gt_arr [0, 1] = s['uv_M_hat_of_A']   # = GT in PoC
            pad_dir   [0, 1] = True                 # skip loss (zero-residual trap)
            pad_dir   [1, 0] = True
            # B↔M analogous — pose_BM derived from pose_AB perturbation, so
            # the residual IS non-trivial here. But legacy doesn\'t store
            # M-as-query data, so we'd need to recompute. Defer: mark skipped.
            pad_dir[1, 2] = True
            pad_dir[2, 1] = True

        return dict(
            patches      = patches,
            uvd          = uvd,
            pad          = pad,
            uvd_full     = uvd_full,
            pad_full     = pad_full,
            pose_hat_6dof= pose_hat,
            pose_gt_6dof = pose_hat.clone(),        # we only store hat; gt unused for loss
            uv_hat       = uv_hat_arr,
            uv_gt        = uv_gt_arr,
            d_hat        = d_hat_arr,
            d_gt         = d_gt_arr,
            pad_dir      = pad_dir,
            fi           = fis,
            scene        = s['scene'],
        )

    @staticmethod
    def _pack_full_local(uv_local, z, n_full, img_size, rng):
        """Pack (n_full, 3) = uv_local + z/50, padded. Stratified subsample."""
        uvd = np.concatenate([uv_local, (z / 50.0)[:, None]], axis=1).astype(np.float32)
        n = len(uvd)
        if n > n_full:
            G = int(np.sqrt(n_full))
            cell = img_size / G
            cj = np.clip((uv_local[:, 0] / cell).astype(np.int32), 0, G - 1)
            ck = np.clip((uv_local[:, 1] / cell).astype(np.int32), 0, G - 1)
            cid = ck * G + cj
            du = uv_local[:, 0] - (cj + 0.5) * cell
            dv = uv_local[:, 1] - (ck + 0.5) * cell
            d2 = du * du + dv * dv
            sel = []
            for u_c in np.unique(cid):
                mem = np.where(cid == u_c)[0]
                sel.append(int(mem[np.argmin(d2[mem])]))
            kept = np.array(sorted(set(sel)), dtype=np.int64)
            if len(kept) < n_full:
                rest = np.setdiff1d(np.arange(n), kept, assume_unique=False)
                if len(rest):
                    fill = rng.choice(rest, size=min(n_full - len(kept), len(rest)),
                                      replace=False)
                    kept = np.concatenate([kept, fill])[:n_full]
            uvd = uvd[kept]
            n = len(uvd)
        pad = np.zeros(n_full, dtype=bool)
        if n < n_full:
            pad[n:] = True
            uvd = np.concatenate([uvd, np.zeros((n_full - n, 3), dtype=np.float32)],
                                 axis=0)
        return uvd, pad

    # ─────────────────────── full P1: 3 frames, 6 directions ────────────────
    # Independent iid perturbation per pair (i, j). Patches centered on HAT
    # projection of the anchor pivot. Per-frame query points = LiDAR in box_i.
    # Per-direction (i, j): project frame_i's 3D points to frame_j with
    # both T_hat[i,j] and T_gt[i,j], pad_dir = (~in_box_j_under_both).
    # Loss runs on every (i, j) with i != j → all 3 frames get gradient as Q.

    def _try_one_p1(self, idx):
        N = self.n_frames
        if N != 3:
            raise NotImplementedError(f'_try_one_p1 currently N=3 only, got {N}')
        rng = np.random.default_rng((idx + 1) * 2654435761 & 0xFFFFFFFF)
        scn: _SceneData = self.scenes[int(rng.integers(len(self.scenes)))]
        IW, IH, K = scn.IW, scn.IH, scn.K

        # 1. fi_anchor + fi_far (legacy direction-first)
        if not scn.fi_pool:
            return None
        fi_A = int(rng.choice(scn.fi_pool))
        bmin, bmax = self.baseline_range
        cn = scn.camera_name.lower()
        is_fb = (('front' in cn or 'back' in cn or 'rear' in cn)
                 and 'left' not in cn and 'right' not in cn and 'side' not in cn)
        direction = int(rng.choice([-1, 1]))
        if direction > 0:
            room = scn.n_frames - 1 - fi_A
        else:
            room = fi_A
        eff_max = room if is_fb else min(bmax, room)
        if eff_max < bmin:
            direction = -direction
            room = scn.n_frames - 1 - fi_A if direction > 0 else fi_A
            eff_max = room if is_fb else min(bmax, room)
            if eff_max < bmin:
                return None
        delta = int(rng.integers(bmin, eff_max + 1)) * direction
        fi_B = fi_A + delta
        if fi_B < 0 or fi_B >= scn.n_frames:
            return None
        fi_M = int(round(0.5 * (fi_A + fi_B)))
        if fi_M == fi_A or fi_M == fi_B:
            return None
        fis = [fi_A, fi_M, fi_B]

        # 2. GT relative poses
        T_w2 = [scn.T_w2c[fi].astype(np.float32) for fi in fis]
        T_w2inv = [_invert_mat(T) for T in T_w2]
        T_gt = np.zeros((N, N, 4, 4), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                T_gt[i, j] = T_w2[j] @ T_w2inv[i]

        # 3. iid per-pair perturbation (full sigma per pair, no scaling)
        T_hat = np.zeros_like(T_gt)
        for i in range(N):
            T_hat[i, i] = np.eye(4, dtype=np.float32)
        for i in range(N):
            for j in range(i + 1, N):
                ypr = rng.standard_normal(3).astype(np.float32) * self.sigma_ypr
                t   = rng.standard_normal(3).astype(np.float32) * self.sigma_t
                T_hat[i, j] = T_gt[i, j] @ _ypr_t_to_mat(ypr, t)
                T_hat[j, i] = _invert_mat(T_hat[i, j])

        # 4. anchor pivot from frame 0 LiDAR (stratified spatial sampling)
        pts_w_0, uv_0_full, z_0_full, in_0 = scn.frame_data(fis[0])
        if in_0.sum() < 50:
            return None
        pts_w_0_in = pts_w_0[in_0]
        uv_0_all   = uv_0_full[in_0]
        H_BINS, W_BINS = 4, 6
        u_bin = np.clip((uv_0_all[:, 0] * W_BINS / IW).astype(int), 0, W_BINS - 1)
        v_bin = np.clip((uv_0_all[:, 1] * H_BINS / IH).astype(int), 0, H_BINS - 1)
        flat_bin = v_bin * W_BINS + u_bin
        occupied = np.unique(flat_bin)
        b = int(rng.choice(occupied))
        in_bin = np.where(flat_bin == b)[0]
        ci = int(rng.choice(in_bin))
        P_center_w = pts_w_0_in[ci]
        uc_full = np.zeros((N, 2), dtype=np.float32)
        uc_full[0] = uv_0_all[ci]

        # 5. project pivot to each other frame using HAT pose
        P_center_in0 = (T_w2[0] @ np.append(P_center_w, 1.0))[:3].astype(np.float32)
        for i in range(1, N):
            P_in_i = (T_hat[0, i] @ np.append(P_center_in0, 1.0))[:3]
            if P_in_i[2] < 1.0:
                return None
            uvi = _proj_cam(P_in_i, K, scn.dist).astype(np.float32)
            if not (0 <= uvi[0] < IW and 0 <= uvi[1] < IH):
                return None
            uc_full[i] = uvi
            # also gate on GT projection in-image
            P_in_i_gt = (T_gt[0, i] @ np.append(P_center_in0, 1.0))[:3]
            if P_in_i_gt[2] < 1.0:
                return None
            uv_gt_i = _proj_cam(P_in_i_gt, K, scn.dist)
            if not (0 <= uv_gt_i[0] < IW and 0 <= uv_gt_i[1] < IH):
                return None

        # 6. crop patches with shared random size, centered on hat projection
        CROP = int(rng.integers(self.crop_range[0], self.crop_range[1] + 1))
        if CROP < 2:
            return None
        half = CROP / 2
        boxes = []
        patches = []
        for i in range(N):
            u0_i = float(uc_full[i, 0]) - half
            v0_i = float(uc_full[i, 1]) - half
            patch_img = scn.load_patch(fis[i], u0_i, v0_i, CROP)
            pi = self._resize_patch(patch_img, u0_i, v0_i, CROP)
            if pi is None:
                return None
            patch_i, box_i = pi
            patches.append(patch_i)
            boxes.append(box_i)

        max_pts = self.max_points
        N_FULL  = 2048
        IMG     = self.img_size

        # 7. per-frame query points: in-patch LiDAR, intersected with
        # "projects to ALL other frames under both hat & gt" — so the same
        # padded set works for every (i, j) loss direction. Same intersection
        # discipline as legacy A↔B (good_qa + inb), generalised to N targets.
        pts_3d_in_cam = np.zeros((N, max_pts, 3), dtype=np.float32)
        uvd_arr       = np.zeros((N, max_pts, 3), dtype=np.float32)
        pad_arr       = np.ones((N, max_pts), dtype=bool)
        uvd_full_arr  = np.zeros((N, N_FULL, 3), dtype=np.float32)
        pad_full_arr  = np.ones((N, N_FULL), dtype=bool)
        # per-direction projection results; padded inputs project to junk but
        # we mask them via pad_dir below.
        uv_hat_arr = np.zeros((N, N, max_pts, 2), dtype=np.float32)
        uv_gt_arr  = np.zeros((N, N, max_pts, 2), dtype=np.float32)
        d_hat_arr  = np.zeros((N, N, max_pts),    dtype=np.float32)
        d_gt_arr   = np.zeros((N, N, max_pts),    dtype=np.float32)
        pad_dir_arr= np.ones ((N, N, max_pts),    dtype=bool)

        for i in range(N):
            box_i = boxes[i]
            if i == 0:
                pts_w_i, uv_i_full, z_i_full, in_i = pts_w_0, uv_0_full, z_0_full, in_0
            else:
                pts_w_i, uv_i_full, z_i_full, in_i = scn.frame_data(fis[i])
                if in_i.sum() < 4:
                    return None
            uv_in    = uv_i_full[in_i]
            z_in     = z_i_full[in_i]
            pts_w_in = pts_w_i[in_i]
            in_box = ((uv_in[:, 0] >= box_i[0]) & (uv_in[:, 0] < box_i[0] + box_i[2]) &
                      (uv_in[:, 1] >= box_i[1]) & (uv_in[:, 1] < box_i[1] + box_i[3]) &
                      (z_in > 1.0))
            if in_box.sum() < 4:
                return None
            pts_w_q   = pts_w_in[in_box]
            uv_q_full = uv_in[in_box]
            z_q       = z_in[in_box].astype(np.float32)
            uv_q_local_self = self._uv_to_patch_local(uv_q_full, box_i)

            # 3D in cam_i frame, homogeneous for downstream T_hat[i, j] @ P_in_i
            homo_w = np.column_stack([pts_w_q, np.ones(len(pts_w_q), dtype=np.float32)])
            P_in_i = (T_w2[i] @ homo_w.T)[:3].T.astype(np.float32)
            homo_i = np.column_stack([P_in_i, np.ones(len(P_in_i), dtype=np.float32)])
            n_raw = len(P_in_i)

            # Per-direction projection + per-direction validity mask.
            # Query points are filtered ONLY by in-box_i (no intersection
            # across targets — too restrictive for 3+ frames). Each (i, j)
            # direction has its own pad_dir marking points whose projection
            # lands outside box_j or behind the camera; loss masks those out.
            per_target_uv_hat_local = {}
            per_target_uv_gt_local  = {}
            per_target_d_hat        = {}
            per_target_d_gt         = {}
            per_target_ok           = {}
            keep = np.ones(n_raw, dtype=bool)            # accept all in-box_i pts
            for j in range(N):
                if j == i:
                    continue
                P_j_hat = (T_hat[i, j] @ homo_i.T)[:3].T
                P_j_gt  = (T_gt [i, j] @ homo_i.T)[:3].T
                d_hat = P_j_hat[:, 2]
                d_gt  = P_j_gt [:, 2]
                ok_d = (d_hat > 0.5) & (d_gt > 0.5)
                uv_full_hat = np.zeros((n_raw, 2), dtype=np.float32)
                uv_full_gt  = np.zeros((n_raw, 2), dtype=np.float32)
                if ok_d.any():
                    uv_full_hat[ok_d] = _proj_cam(P_j_hat[ok_d], K, scn.dist).astype(np.float32)
                    uv_full_gt [ok_d] = _proj_cam(P_j_gt [ok_d], K, scn.dist).astype(np.float32)
                uv_local_hat = self._uv_to_patch_local(uv_full_hat, boxes[j])
                uv_local_gt  = self._uv_to_patch_local(uv_full_gt,  boxes[j])
                in_patch_hat = ((uv_local_hat[:, 0] >= 0) & (uv_local_hat[:, 0] < IMG) &
                                (uv_local_hat[:, 1] >= 0) & (uv_local_hat[:, 1] < IMG))
                in_patch_gt  = ((uv_local_gt [:, 0] >= 0) & (uv_local_gt [:, 0] < IMG) &
                                (uv_local_gt [:, 1] >= 0) & (uv_local_gt [:, 1] < IMG))
                per_target_uv_hat_local[j] = uv_local_hat
                per_target_uv_gt_local [j] = uv_local_gt
                per_target_d_hat       [j] = d_hat
                per_target_d_gt        [j] = d_gt
                per_target_ok          [j] = ok_d & in_patch_hat & in_patch_gt
            kept_idx = np.where(keep)[0]
            n_kept = len(kept_idx)

            # full frustum context (use ALL in-box points, before keep filter)
            uvd_full_arr[i], pad_full_arr[i] = self._pack_full_local(
                uv_q_local_self, z_q, N_FULL, IMG, rng)

            # subsample kept points to max_pts (stratified spatial-grid pick)
            if n_kept > max_pts:
                G = int(np.sqrt(max_pts))
                cell = IMG / G
                cj = np.clip((uv_q_local_self[kept_idx, 0] / cell).astype(np.int32), 0, G - 1)
                ck = np.clip((uv_q_local_self[kept_idx, 1] / cell).astype(np.int32), 0, G - 1)
                cid = ck * G + cj
                du = uv_q_local_self[kept_idx, 0] - (cj + 0.5) * cell
                dv = uv_q_local_self[kept_idx, 1] - (ck + 0.5) * cell
                d2 = du * du + dv * dv
                sel = []
                for u_c in np.unique(cid):
                    mem = np.where(cid == u_c)[0]
                    sel.append(int(mem[np.argmin(d2[mem])]))
                pick_local = np.array(sorted(set(sel)), dtype=np.int64)
                pick = kept_idx[pick_local]
            else:
                pick = kept_idx
            n_use = len(pick)

            # populate per-frame fields
            uvd_arr[i, :n_use, 0] = uv_q_local_self[pick, 0]
            uvd_arr[i, :n_use, 1] = uv_q_local_self[pick, 1]
            uvd_arr[i, :n_use, 2] = z_q[pick] / 50.0
            pts_3d_in_cam[i, :n_use] = P_in_i[pick]
            pad_arr[i, :n_use] = False

            # populate per-direction projections + per-direction pads.
            # Each direction (i, j) marks pad_dir True for pts that don't
            # project into box_j or fall behind cam_j; loss masks them out.
            # A↔B (= 0↔N-1) almost always has valid pts (pivot anchored
            # there) → that's the always-supervised "pair baseline" inside
            # the 6-dir loss. M-related directions only contribute where
            # geometry permits → natural curriculum.
            for j in range(N):
                if j == i:
                    continue
                uv_hat_arr[i, j, :n_use] = per_target_uv_hat_local[j][pick]
                uv_gt_arr [i, j, :n_use] = per_target_uv_gt_local [j][pick]
                d_hat_arr [i, j, :n_use] = per_target_d_hat       [j][pick]
                d_gt_arr  [i, j, :n_use] = per_target_d_gt        [j][pick]
                pad_dir_arr[i, j, :n_use] = ~per_target_ok[j][pick]

        # 8. assemble pose (N, N, 6) ypr+t
        pose_hat_arr = np.zeros((N, N, 6), dtype=np.float32)
        pose_gt_arr  = np.zeros((N, N, 6), dtype=np.float32)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                ypr_h, t_h = _mat_to_ypr_t(T_hat[i, j])
                ypr_g, t_g = _mat_to_ypr_t(T_gt [i, j])
                pose_hat_arr[i, j] = np.concatenate([ypr_h, t_h])
                pose_gt_arr [i, j] = np.concatenate([ypr_g, t_g])

        return dict(
            patches      = torch.stack(patches, dim=0).float(),
            uvd          = torch.from_numpy(uvd_arr),
            pad          = torch.from_numpy(pad_arr),
            uvd_full     = torch.from_numpy(uvd_full_arr),
            pad_full     = torch.from_numpy(pad_full_arr),
            pose_hat_6dof= torch.from_numpy(pose_hat_arr),
            pose_gt_6dof = torch.from_numpy(pose_gt_arr),
            uv_hat       = torch.from_numpy(uv_hat_arr),
            uv_gt        = torch.from_numpy(uv_gt_arr),
            d_hat        = torch.from_numpy(d_hat_arr),
            d_gt         = torch.from_numpy(d_gt_arr),
            pad_dir      = torch.from_numpy(pad_dir_arr),
            fi           = torch.tensor(fis, dtype=torch.long),
            scene        = scn.scene_id,
        )


    def _try_one(self, idx):
        rng = np.random.default_rng((idx + 1) * 2654435761 & 0xFFFFFFFF)

        # 0. pick scene
        scn: _SceneData = self.scenes[int(rng.integers(len(self.scenes)))]
        IW, IH, K = scn.IW, scn.IH, scn.K

        # 1. pick fi_A
        if not scn.fi_pool:
            return None
        fi_A = int(rng.choice(scn.fi_pool))
        # 2. fi_B: nearby frame. baseline_max is per-camera:
        #   front/back (pure, no left/right/side) → unlimited (scene length)
        #   side / front_left / rear_right / etc.                → baseline_max
        # Pick direction first, then delta bounded by how much room is left
        # on that side of fi_A — keeps the sampling uniform within a valid
        # range instead of throwing away invalid draws.
        bmin, bmax = self.baseline_range
        cn = scn.camera_name.lower()
        is_fb = (('front' in cn or 'back' in cn or 'rear' in cn)
                 and 'left' not in cn and 'right' not in cn and 'side' not in cn)
        direction = int(rng.choice([-1, 1]))
        if direction > 0:
            room = scn.n_frames - 1 - fi_A
        else:
            room = fi_A
        eff_max = room if is_fb else min(bmax, room)
        if eff_max < bmin:
            # try the other direction
            direction = -direction
            room = scn.n_frames - 1 - fi_A if direction > 0 else fi_A
            eff_max = room if is_fb else min(bmax, room)
            if eff_max < bmin:
                return None
        delta = int(rng.integers(bmin, eff_max + 1)) * direction
        fi_B = fi_A + delta
        if fi_B < 0 or fi_B >= scn.n_frames:
            return None

        # 3. poses
        T_w2A = scn.T_w2c[fi_A]
        T_w2B = scn.T_w2c[fi_B]
        T_A2w = _invert_mat(T_w2A)
        T_A_to_B_gt = T_w2B @ T_A2w

        # 4. load LiDAR and project (cached per frame)
        pts_w_A, uv_Af, z_Af, in_A = scn.frame_data(fi_A)
        pts_w_B, uv_Bf, z_Bf, in_B = scn.frame_data(fi_B)
        if in_A.sum() < 50 or in_B.sum() < 50:
            return None

        pts_w_A_in = pts_w_A[in_A]
        uv_A_all   = uv_Af[in_A]
        z_A_all    = z_Af[in_A]

        # 6. pick center 3D point from A via spatial-bin stratified sampling.
        # Uniform rng.integers over all in-view points heavily biases toward
        # ground (LiDAR density much higher on near/horizontal surfaces), so
        # the σ head over-specialises to the ground. Instead, bin uv_A_all
        # into a coarse image grid, pick an occupied cell uniformly, then pick
        # a point inside that cell — cell-level uniform, not point-level.
        H_BINS, W_BINS = 4, 6
        u_bin = np.clip((uv_A_all[:, 0] * W_BINS / scn.IW).astype(int), 0, W_BINS - 1)
        v_bin = np.clip((uv_A_all[:, 1] * H_BINS / scn.IH).astype(int), 0, H_BINS - 1)
        flat_bin = v_bin * W_BINS + u_bin
        occupied = np.unique(flat_bin)
        b = int(rng.choice(occupied))
        in_bin = np.where(flat_bin == b)[0]
        ci = int(rng.choice(in_bin))
        P_center_w = pts_w_A_in[ci]
        uc_A, vc_A = uv_A_all[ci]

        # 7. sample perturbation → T_AB_hat
        ypr_pert = rng.standard_normal(3).astype(np.float32) * self.sigma_ypr
        t_pert   = rng.standard_normal(3).astype(np.float32) * self.sigma_t
        δT       = _ypr_t_to_mat(ypr_pert, t_pert)
        T_A_to_B_hat = T_A_to_B_gt @ δT

        # 8. project center into B under both hat and gt
        P_center_A   = (T_w2A @ np.append(P_center_w, 1.0))[:3]
        P_center_Bh  = (T_A_to_B_hat @ np.append(P_center_A, 1.0))[:3]
        P_center_Bg  = (T_A_to_B_gt @ np.append(P_center_A, 1.0))[:3]
        if P_center_Bh[2] < 1.0 or P_center_Bg[2] < 1.0:
            return None
        uc_B_hat = _proj_cam(P_center_Bh, K, scn.dist)
        uc_B_gt  = _proj_cam(P_center_Bg, K, scn.dist)
        uc_B_hat = uc_B_hat.astype(np.float32)
        uc_B_gt  = uc_B_gt.astype(np.float32)
        # pivot projection must land INSIDE the actual image in B under both
        # hypothesis and ground-truth poses (otherwise the pair is degenerate:
        # the pivot's "true" location is outside any image we have).
        if not (0 <= uc_B_hat[0] < IW and 0 <= uc_B_hat[1] < IH):
            return None
        if not (0 <= uc_B_gt[0]  < IW and 0 <= uc_B_gt[1]  < IH):
            return None

        # 9. crop patches with random shared size
        # pivot stays at EXACT patch center — padding (inside _crop_patch)
        # handles any out-of-image area so pose_emb semantics are preserved.
        CROP = int(rng.integers(self.crop_range[0], self.crop_range[1] + 1))
        half = CROP / 2
        u0_A = uc_A - half
        v0_A = vc_A - half
        u0_B = uc_B_hat[0] - half
        v0_B = uc_B_hat[1] - half
        if CROP < 2:
            return None
        patch_A_img = scn.load_patch(fi_A, u0_A, v0_A, CROP)
        patch_B_img = scn.load_patch(fi_B, u0_B, v0_B, CROP)
        pa = self._resize_patch(patch_A_img, u0_A, v0_A, CROP)
        pb = self._resize_patch(patch_B_img, u0_B, v0_B, CROP)
        if pa is None or pb is None:
            return None
        patch_A, box_A = pa
        patch_B, box_B = pb

        # 9b. triplet — pick fi_M between fi_A and fi_B, project pivot into M
        if self.triplet:
            fi_M = int(round(0.5 * (fi_A + fi_B)))
            if fi_M == fi_A or fi_M == fi_B:
                return None
            T_w2M = scn.T_w2c[fi_M]
            P_center_M = (T_w2M @ np.append(P_center_w, 1.0))[:3]
            if P_center_M[2] < 1.0:
                return None
            uc_M_arr = _proj_cam(P_center_M, K, scn.dist).astype(np.float32)
            uc_M, vc_M = float(uc_M_arr[0]), float(uc_M_arr[1])
            if not (0 <= uc_M < IW and 0 <= vc_M < IH):
                return None
            u0_M = uc_M - half
            v0_M = vc_M - half
            patch_M_img = scn.load_patch(fi_M, u0_M, v0_M, CROP)
            pm = self._resize_patch(patch_M_img, u0_M, v0_M, CROP)
            if pm is None:
                return None
            patch_M, box_M = pm
            T_A_to_M_gt = T_w2M @ T_A2w  # auxiliary; treated as exact in PoC
            ypr_AM_gt, t_AM_gt = _mat_to_ypr_t(T_A_to_M_gt)
            pts_w_M, uv_Mf, z_Mf, in_M = scn.frame_data(fi_M)
            in_box_M = ((uv_Mf[:, 0] >= box_M[0]) & (uv_Mf[:, 0] < box_M[0] + box_M[2]) &
                        (uv_Mf[:, 1] >= box_M[1]) & (uv_Mf[:, 1] < box_M[1] + box_M[3]) &
                        (z_Mf > 1.0))
            if in_box_M.sum() < 4:
                return None
            uv_M_patch = self._uv_to_patch_local(uv_Mf[in_box_M], box_M)
            z_M_patch  = z_Mf[in_box_M].astype(np.float32)

        # 10. A-query points = A's in-patch LiDAR
        u0, v0, cw, ch = box_A
        in_box_A = ((uv_A_all[:, 0] >= u0) & (uv_A_all[:, 0] < u0 + cw) &
                    (uv_A_all[:, 1] >= v0) & (uv_A_all[:, 1] < v0 + ch))
        if in_box_A.sum() < 4:
            return None
        pts_w_QA   = pts_w_A_in[in_box_A]
        uv_A_patch = self._uv_to_patch_local(uv_A_all[in_box_A], box_A)
        z_A_patch  = z_A_all[in_box_A]

        homo = np.concatenate([pts_w_QA, np.ones((len(pts_w_QA), 1), dtype=np.float32)], axis=1)
        P_QA_in_A     = (T_w2A @ homo.T)[:3].T
        P_QA_in_B_gt  = (np.column_stack([P_QA_in_A, np.ones(len(P_QA_in_A))]) @ T_A_to_B_gt.T)[:, :3]
        P_QA_in_B_hat = (np.column_stack([P_QA_in_A, np.ones(len(P_QA_in_A))]) @ T_A_to_B_hat.T)[:, :3]
        good_qa = (P_QA_in_B_gt[:, 2] > 0.5) & (P_QA_in_B_hat[:, 2] > 0.5)
        if good_qa.sum() < 4:
            return None
        uv_B_gt_full_A  = _proj_cam(P_QA_in_B_gt[good_qa],  K, scn.dist)
        uv_B_hat_full_A = _proj_cam(P_QA_in_B_hat[good_qa], K, scn.dist)
        uv_B_gt_local_A  = self._uv_to_patch_local(uv_B_gt_full_A,  box_B)
        uv_B_hat_local_A = self._uv_to_patch_local(uv_B_hat_full_A, box_B)

        uv_A_patch    = uv_A_patch[good_qa]
        z_A_patch     = z_A_patch[good_qa]
        inb = ((uv_B_hat_local_A[:, 0] >= 0) & (uv_B_hat_local_A[:, 0] < self.img_size) &
               (uv_B_hat_local_A[:, 1] >= 0) & (uv_B_hat_local_A[:, 1] < self.img_size) &
               (uv_B_gt_local_A[:, 0]  >= 0) & (uv_B_gt_local_A[:, 0]  < self.img_size) &
               (uv_B_gt_local_A[:, 1]  >= 0) & (uv_B_gt_local_A[:, 1]  < self.img_size))
        if inb.sum() < 4:
            return None
        uv_A_patch       = uv_A_patch[inb]
        z_A_patch        = z_A_patch[inb]
        uv_B_gt_local_A  = uv_B_gt_local_A[inb]
        uv_B_hat_local_A = uv_B_hat_local_A[inb]

        # 11. B-query points = B's in-patch LiDAR (symmetric)
        in_box_B = ((uv_Bf[:, 0] >= box_B[0]) & (uv_Bf[:, 0] < box_B[0] + box_B[2]) &
                    (uv_Bf[:, 1] >= box_B[1]) & (uv_Bf[:, 1] < box_B[1] + box_B[3]) &
                    (z_Bf > 1.0))
        uv_B_patch = self._uv_to_patch_local(uv_Bf[in_box_B], box_B)
        z_B_patch  = z_Bf[in_box_B]
        pts_w_QB   = pts_w_B[in_box_B]
        if len(pts_w_QB) < 4:
            return None
        homoB = np.concatenate([pts_w_QB, np.ones((len(pts_w_QB), 1), dtype=np.float32)], axis=1)
        P_QB_in_B = (T_w2B @ homoB.T)[:3].T
        T_B_to_A_gt  = _invert_mat(T_A_to_B_gt)
        T_B_to_A_hat = _invert_mat(T_A_to_B_hat)
        P_QB_in_A_gt  = (np.column_stack([P_QB_in_B, np.ones(len(P_QB_in_B))]) @ T_B_to_A_gt.T)[:, :3]
        P_QB_in_A_hat = (np.column_stack([P_QB_in_B, np.ones(len(P_QB_in_B))]) @ T_B_to_A_hat.T)[:, :3]
        good_qb = (P_QB_in_A_gt[:, 2] > 0.5) & (P_QB_in_A_hat[:, 2] > 0.5)
        if good_qb.sum() < 4:
            return None
        uv_A_gt_full_B  = _proj_cam(P_QB_in_A_gt[good_qb],  K, scn.dist)
        uv_A_hat_full_B = _proj_cam(P_QB_in_A_hat[good_qb], K, scn.dist)
        uv_A_gt_local_B  = self._uv_to_patch_local(uv_A_gt_full_B,  box_A)
        uv_A_hat_local_B = self._uv_to_patch_local(uv_A_hat_full_B, box_A)
        uv_B_patch = uv_B_patch[good_qb]
        z_B_patch  = z_B_patch[good_qb]
        inb2 = ((uv_A_hat_local_B[:, 0] >= 0) & (uv_A_hat_local_B[:, 0] < self.img_size) &
                (uv_A_hat_local_B[:, 1] >= 0) & (uv_A_hat_local_B[:, 1] < self.img_size) &
                (uv_A_gt_local_B[:, 0]  >= 0) & (uv_A_gt_local_B[:, 0]  < self.img_size) &
                (uv_A_gt_local_B[:, 1]  >= 0) & (uv_A_gt_local_B[:, 1]  < self.img_size))
        if inb2.sum() < 4:
            return None
        uv_B_patch       = uv_B_patch[inb2]
        z_B_patch        = z_B_patch[inb2]
        uv_A_gt_local_B  = uv_A_gt_local_B[inb2]
        uv_A_hat_local_B = uv_A_hat_local_B[inb2]

        # depth in target camera frame (B for A-query, A for B-query)
        d_B_hat_of_A = P_QA_in_B_hat[good_qa][inb, 2].astype(np.float32)
        d_B_gt_of_A  = P_QA_in_B_gt [good_qa][inb, 2].astype(np.float32)
        d_A_hat_of_B = P_QB_in_A_hat[good_qb][inb2, 2].astype(np.float32)
        d_A_gt_of_B  = P_QB_in_A_gt [good_qb][inb2, 2].astype(np.float32)

        # 11b. triplet — project filtered A/B query points into M frame
        # (treated as exact pose: uv_M_hat == uv_M_gt in PoC)
        if self.triplet:
            # A-query → M
            P_QA_filt = P_QA_in_A[good_qa][inb]              # (N_A, 3) in A-cam
            P_QA_in_M = (np.column_stack([P_QA_filt, np.ones(len(P_QA_filt))]) @ T_A_to_M_gt.T)[:, :3]
            uv_M_full_A  = _proj_cam(P_QA_in_M, K, scn.dist)
            uv_M_local_A = self._uv_to_patch_local(uv_M_full_A, box_M)
            d_M_of_A     = P_QA_in_M[:, 2].astype(np.float32)
            # B-query → M (via T_B→A → T_A→M)
            T_B_to_M_gt  = T_A_to_M_gt @ _invert_mat(T_A_to_B_gt)
            P_QB_filt = P_QB_in_B[good_qb][inb2]
            P_QB_in_M = (np.column_stack([P_QB_filt, np.ones(len(P_QB_filt))]) @ T_B_to_M_gt.T)[:, :3]
            uv_M_full_B  = _proj_cam(P_QB_in_M, K, scn.dist)
            uv_M_local_B = self._uv_to_patch_local(uv_M_full_B, box_M)
            d_M_of_B     = P_QB_in_M[:, 2].astype(np.float32)

        # 12. spatial-grid stratified subsample → pad to max_points
        # G×G grid in patch-local space, keep one point per cell (the one
        # closest to its cell center). Replaces the legacy uniform-random
        # pick which biased toward dense regions (ground / nearby cars).
        # img=64,  max_points=256  → G=16, cell=4×4 px
        # img=128, max_points=256  → G=16, cell=8×8 px
        def _pad(uv_query, z_query, uv_hat, uv_gt, d_hat, d_gt):
            N = len(uv_query)
            if N > self.max_points:
                G = int(np.sqrt(self.max_points))
                cell = self.img_size / G
                cj = np.clip((uv_query[:, 0] / cell).astype(np.int32), 0, G - 1)
                ci = np.clip((uv_query[:, 1] / cell).astype(np.int32), 0, G - 1)
                cid = ci * G + cj
                du  = uv_query[:, 0] - (cj + 0.5) * cell
                dv  = uv_query[:, 1] - (ci + 0.5) * cell
                d2  = du * du + dv * dv
                sel = []
                for u_c in np.unique(cid):
                    mem = np.where(cid == u_c)[0]
                    sel.append(int(mem[np.argmin(d2[mem])]))
                pick = np.array(sorted(set(sel)), dtype=np.int64)
            else:
                pick = np.arange(N, dtype=np.int64)
            N_use = len(pick)
            uv_query = uv_query[pick]; z_query = z_query[pick]
            uv_hat = uv_hat[pick];     uv_gt = uv_gt[pick]
            d_hat = d_hat[pick];       d_gt  = d_gt[pick]
            pad = np.zeros((self.max_points,), dtype=bool)
            if N_use < self.max_points:
                pad[N_use:] = True
                pad_n = self.max_points - N_use
                uv_query = np.concatenate([uv_query, np.zeros((pad_n, 2), np.float32)])
                z_query  = np.concatenate([z_query,  np.zeros(pad_n, np.float32)])
                uv_hat   = np.concatenate([uv_hat,   np.zeros((pad_n, 2), np.float32)])
                uv_gt    = np.concatenate([uv_gt,    np.zeros((pad_n, 2), np.float32)])
                d_hat    = np.concatenate([d_hat,    np.zeros(pad_n, np.float32)])
                d_gt     = np.concatenate([d_gt,     np.zeros(pad_n, np.float32)])
            return uv_query, z_query, uv_hat, uv_gt, d_hat, d_gt, pad

        # ── snapshot the FULL in-box sets BEFORE subsampling, for frustum
        # local-encoder lookup (model side runs on GPU, no CPU cost concern).
        # Pad to N_FULL=2048 with mask; if more, stratify-truncate to keep
        # spatial coverage (avoids dense-region bias).
        N_FULL = 2048
        def _pack_full(uv, z):
            uvd = np.concatenate([uv, (z / 50.0)[:, None]], axis=1).astype(np.float32)
            N = len(uvd)
            if N > N_FULL:
                G_full = int(np.sqrt(N_FULL))   # ~45
                cell_full = self.img_size / G_full
                cj = np.clip((uv[:, 0] / cell_full).astype(np.int32), 0, G_full - 1)
                ci = np.clip((uv[:, 1] / cell_full).astype(np.int32), 0, G_full - 1)
                cid = ci * G_full + cj
                du = uv[:, 0] - (cj + 0.5) * cell_full
                dv = uv[:, 1] - (ci + 0.5) * cell_full
                d2 = du * du + dv * dv
                sel = []
                for u_c in np.unique(cid):
                    mem = np.where(cid == u_c)[0]
                    sel.append(int(mem[np.argmin(d2[mem])]))
                # if grid yields fewer than N_FULL (sparse cells), fill rest randomly
                kept = np.array(sorted(set(sel)), dtype=np.int64)
                if len(kept) < N_FULL:
                    rest = np.setdiff1d(np.arange(N), kept, assume_unique=False)
                    fill = self.rng.choice(rest, size=min(N_FULL - len(kept), len(rest)),
                                           replace=False) if len(rest) > 0 else np.empty(0, dtype=np.int64)
                    kept = np.concatenate([kept, fill])[:N_FULL]
                uvd = uvd[kept]
                N = len(uvd)
            pad = np.zeros(N_FULL, dtype=bool)
            if N < N_FULL:
                pad[N:] = True
                uvd = np.concatenate([uvd, np.zeros((N_FULL - N, 3), dtype=np.float32)], axis=0)
            return uvd, pad

        uvd_A_full, pad_A_full = _pack_full(uv_A_patch, z_A_patch)
        uvd_B_full, pad_B_full = _pack_full(uv_B_patch, z_B_patch)
        if self.triplet:
            uvd_M_full, pad_M_full = _pack_full(uv_M_patch, z_M_patch)
            # stratify M down to max_points (no GT/hat correspondence — auxiliary)
            def _pad_m(uv_q, z_q):
                N = len(uv_q)
                if N > self.max_points:
                    G = int(np.sqrt(self.max_points))
                    cell = self.img_size / G
                    cj = np.clip((uv_q[:, 0] / cell).astype(np.int32), 0, G - 1)
                    ci = np.clip((uv_q[:, 1] / cell).astype(np.int32), 0, G - 1)
                    cid = ci * G + cj
                    du  = uv_q[:, 0] - (cj + 0.5) * cell
                    dv  = uv_q[:, 1] - (ci + 0.5) * cell
                    d2  = du * du + dv * dv
                    sel = []
                    for u_c in np.unique(cid):
                        mem = np.where(cid == u_c)[0]
                        sel.append(int(mem[np.argmin(d2[mem])]))
                    pick = np.array(sorted(set(sel)), dtype=np.int64)
                else:
                    pick = np.arange(N, dtype=np.int64)
                N_use = len(pick)
                uv_q = uv_q[pick]; z_q = z_q[pick]
                pad = np.zeros(self.max_points, dtype=bool)
                if N_use < self.max_points:
                    pad[N_use:] = True
                    pad_n = self.max_points - N_use
                    uv_q = np.concatenate([uv_q, np.zeros((pad_n, 2), np.float32)])
                    z_q  = np.concatenate([z_q,  np.zeros(pad_n, np.float32)])
                return uv_q, z_q, pad
            uv_M_patch, z_M_patch, pad_M = _pad_m(uv_M_patch, z_M_patch)

        if self.triplet:
            # extend _pad to also stratify uv_M_local_*; share the same pick.
            def _pad8(uv_q, z_q, uv_hat, uv_gt, d_hat, d_gt, uv_M, d_M):
                N = len(uv_q)
                if N > self.max_points:
                    G = int(np.sqrt(self.max_points))
                    cell = self.img_size / G
                    cj = np.clip((uv_q[:, 0] / cell).astype(np.int32), 0, G - 1)
                    ci = np.clip((uv_q[:, 1] / cell).astype(np.int32), 0, G - 1)
                    cid = ci * G + cj
                    du  = uv_q[:, 0] - (cj + 0.5) * cell
                    dv  = uv_q[:, 1] - (ci + 0.5) * cell
                    d2  = du * du + dv * dv
                    sel = []
                    for u_c in np.unique(cid):
                        mem = np.where(cid == u_c)[0]
                        sel.append(int(mem[np.argmin(d2[mem])]))
                    pick = np.array(sorted(set(sel)), dtype=np.int64)
                else:
                    pick = np.arange(N, dtype=np.int64)
                arrs = [uv_q[pick], z_q[pick], uv_hat[pick], uv_gt[pick],
                        d_hat[pick], d_gt[pick], uv_M[pick], d_M[pick]]
                N_use = len(pick)
                pad = np.zeros(self.max_points, dtype=bool)
                if N_use < self.max_points:
                    pad[N_use:] = True
                    pad_n = self.max_points - N_use
                    arrs[0] = np.concatenate([arrs[0], np.zeros((pad_n, 2), np.float32)])
                    arrs[1] = np.concatenate([arrs[1], np.zeros(pad_n, np.float32)])
                    arrs[2] = np.concatenate([arrs[2], np.zeros((pad_n, 2), np.float32)])
                    arrs[3] = np.concatenate([arrs[3], np.zeros((pad_n, 2), np.float32)])
                    arrs[4] = np.concatenate([arrs[4], np.zeros(pad_n, np.float32)])
                    arrs[5] = np.concatenate([arrs[5], np.zeros(pad_n, np.float32)])
                    arrs[6] = np.concatenate([arrs[6], np.zeros((pad_n, 2), np.float32)])
                    arrs[7] = np.concatenate([arrs[7], np.zeros(pad_n, np.float32)])
                return (*arrs, pad)
            (uv_A_patch, z_A_patch, uv_B_hat_local_A, uv_B_gt_local_A, d_B_hat_of_A,
             d_B_gt_of_A, uv_M_local_A, d_M_of_A, pad_A) = _pad8(
                uv_A_patch, z_A_patch, uv_B_hat_local_A, uv_B_gt_local_A,
                d_B_hat_of_A, d_B_gt_of_A, uv_M_local_A, d_M_of_A)
            (uv_B_patch, z_B_patch, uv_A_hat_local_B, uv_A_gt_local_B, d_A_hat_of_B,
             d_A_gt_of_B, uv_M_local_B, d_M_of_B, pad_B) = _pad8(
                uv_B_patch, z_B_patch, uv_A_hat_local_B, uv_A_gt_local_B,
                d_A_hat_of_B, d_A_gt_of_B, uv_M_local_B, d_M_of_B)
        else:
            uv_A_patch, z_A_patch, uv_B_hat_local_A, uv_B_gt_local_A, d_B_hat_of_A, d_B_gt_of_A, pad_A = _pad(
                uv_A_patch, z_A_patch, uv_B_hat_local_A, uv_B_gt_local_A, d_B_hat_of_A, d_B_gt_of_A)
            uv_B_patch, z_B_patch, uv_A_hat_local_B, uv_A_gt_local_B, d_A_hat_of_B, d_A_gt_of_B, pad_B = _pad(
                uv_B_patch, z_B_patch, uv_A_hat_local_B, uv_A_gt_local_B, d_A_hat_of_B, d_A_gt_of_B)

        z_A_norm = z_A_patch / 50.0
        z_B_norm = z_B_patch / 50.0

        ypr_AB_hat, t_AB_hat = _mat_to_ypr_t(T_A_to_B_hat)
        ypr_BA_hat, t_BA_hat = _mat_to_ypr_t(_invert_mat(T_A_to_B_hat))

        out = dict(
            patch_A = patch_A.float(),
            patch_B = patch_B.float(),
            uvd_A   = torch.from_numpy(
                np.concatenate([uv_A_patch, z_A_norm[:, None]], axis=1)).float(),
            uvd_B   = torch.from_numpy(
                np.concatenate([uv_B_patch, z_B_norm[:, None]], axis=1)).float(),
            uv_B_hat_of_A = torch.from_numpy(uv_B_hat_local_A).float(),
            uv_B_gt_of_A  = torch.from_numpy(uv_B_gt_local_A).float(),
            uv_A_hat_of_B = torch.from_numpy(uv_A_hat_local_B).float(),
            uv_A_gt_of_B  = torch.from_numpy(uv_A_gt_local_B).float(),
            d_B_hat_of_A  = torch.from_numpy(d_B_hat_of_A).float(),
            d_B_gt_of_A   = torch.from_numpy(d_B_gt_of_A).float(),
            d_A_hat_of_B  = torch.from_numpy(d_A_hat_of_B).float(),
            d_A_gt_of_B   = torch.from_numpy(d_A_gt_of_B).float(),
            pose_AB_6dof  = torch.from_numpy(
                np.concatenate([ypr_AB_hat, t_AB_hat]).astype(np.float32)),
            pose_BA_6dof  = torch.from_numpy(
                np.concatenate([ypr_BA_hat, t_BA_hat]).astype(np.float32)),
            pad_A = torch.from_numpy(pad_A),
            pad_B = torch.from_numpy(pad_B),
            uvd_A_full = torch.from_numpy(uvd_A_full),       # (N_FULL, 3) frustum-context
            uvd_B_full = torch.from_numpy(uvd_B_full),
            pad_A_full = torch.from_numpy(pad_A_full),       # (N_FULL,) bool
            pad_B_full = torch.from_numpy(pad_B_full),
            fi_A = fi_A, fi_B = fi_B,
            scene = scn.scene_id,
        )
        if self.triplet:
            z_M_norm = z_M_patch / 50.0
            out['patch_M']      = patch_M.float()
            out['uvd_M']        = torch.from_numpy(
                np.concatenate([uv_M_patch, z_M_norm[:, None]], axis=1)).float()
            out['pad_M']        = torch.from_numpy(pad_M)
            out['uvd_M_full']   = torch.from_numpy(uvd_M_full)
            out['pad_M_full']   = torch.from_numpy(pad_M_full)
            out['pose_AM_6dof'] = torch.from_numpy(
                np.concatenate([ypr_AM_gt, t_AM_gt]).astype(np.float32))
            # A-query → M reference (= GT in PoC since pose_AM is treated exact)
            out['uv_M_hat_of_A'] = torch.from_numpy(uv_M_local_A).float()
            out['uv_M_hat_of_B'] = torch.from_numpy(uv_M_local_B).float()
            out['fi_M']         = fi_M
        return out


# ─── smoke test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time
    t0 = time.time()
    ds = PandaSetCrossFrameDataset(
        scenes_root='/mnt/mininas/datasets/pandaset',
        split='train', train_frac=0.80,
        img_size=64, max_points=256,
        baseline_range=(1, 20), sigma_ypr=1.0, sigma_t=0.2,
    )
    print(f'init: {time.time()-t0:.1f}s')
    s = ds[0]
    print(f'sample scene={s["scene"]} fi_A={s["fi_A"]} fi_B={s["fi_B"]} '
          f'valid_A={(~s["pad_A"]).sum().item()} valid_B={(~s["pad_B"]).sum().item()}')

    t0 = time.time()
    N = 200
    for i in range(N):
        _ = ds[i]
    print(f'{N} samples: {time.time()-t0:.2f}s '
          f'→ {(time.time()-t0)/N*1000:.1f} ms/sample')
