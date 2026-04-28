"""Cache-less PandaSet calib dataset using libjpeg-turbo partial JPEG decode.

Drop-in replacement for PandaSetCalibDatasetLazyVFP. Reproduces ps_v9_lazy
semantics (uniform perturbation, cuboid-anchored crops, 2D bbox-driven crop
size) without the pre-computed lazy cache. Per __getitem__:
  - Pick (scene, frame_idx, cuboid_idx) from index built at __init__
  - Load lidar pickle for that frame (on demand, ~few-ms)
  - Apply uniform [-sigma, +sigma] perturbation in cam-local frame
  - Project pts in WORLD frame through perturbed pose to get HAT uv
  - Compute crop window from cuboid 2D bbox + bbox_scale (full-res px)
  - Partial-decode JPEG at crop window via turbojpeg (≈5 ms vs full decode)
  - Resample to img_size, return (img, true_uvd, dist_uvd, vfp).

Returns:
  img      : (3, S, S) float32 [0,1]
  true_uvd : (N, 4) [u_gt, v_gt, d_norm, is_obj]   in patch-local px coords
  dist_uvd : (N, 4) [u_hat, v_hat, d_norm, is_obj] in patch-local px coords
  vfp      : (,)    fx_full * S / crop_size_full_px
"""
from __future__ import annotations
import gzip, json, pickle, random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

try:
    import turbojpeg as _tj
    _HAVE_TJ = True
except ImportError:
    _HAVE_TJ = False
    _tj = None

# Reuse the v9 helpers untouched.
from datasets.pandaset import (USEFUL_LABELS, _quat_pos_to_mat, _project,
                                _bbox2d_of_cuboid)

_MCU = 16  # libjpeg-turbo MCU alignment for 4:2:0 chroma sub-sampling


def _load_pickle(path: Path):
    """Load .pkl or .pkl.gz transparently."""
    if path.suffix == '.gz':
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    with open(path, 'rb') as f:
        return pickle.load(f)


def _scaled_decode_crop(jpg_path: Path, u0: int, v0: int, s: int, S: int,
                         IW: int, IH: int) -> np.ndarray:
    """Decode JPEG at the smallest libjpeg-turbo scale ≥ S, crop the patch
    region in that scale, then resize to (S, S). PIL's draft() invokes
    libjpeg's scaled IDCT (1/8, 1/4, 1/2, ...), so decompress time scales
    with the *output* resolution rather than the full 1920×1080. ~3 ms vs
    ~10 ms for 1:1 MCU partial decode + resize on this dataset.
    """
    out = np.zeros((S, S, 3), dtype=np.uint8)
    su0 = max(0, u0); sv0 = max(0, v0)
    su1 = min(IW, u0 + s); sv1 = min(IH, v0 + s)
    if su1 <= su0 or sv1 <= sv0:
        return out
    with Image.open(jpg_path) as im:
        # Hint a draft size ≥ S; libjpeg picks the closest 1/2^N factor.
        target = max(S * 2, 256)   # gives 1/8 for 1920px (= 240 px) typically
        im.draft('RGB', (target, target))
        sf_x = im.size[0] / IW
        sf_y = im.size[1] / IH
        cu0 = max(0, int(round(su0 * sf_x)))
        cv0 = max(0, int(round(sv0 * sf_y)))
        cu1 = min(im.size[0], int(round(su1 * sf_x)))
        cv1 = min(im.size[1], int(round(sv1 * sf_y)))
        if cu1 <= cu0 or cv1 <= cv0:
            return out
        sub = im.crop((cu0, cv0, cu1, cv1)).convert('RGB').resize((S, S), Image.BILINEAR)
        out[:] = np.asarray(sub, dtype=np.uint8)
    return out


# Legacy 1:1 partial-decode kept for backward compatibility; unused after the
# scaled-decode switch but referenced elsewhere (e.g., visualization scripts).
def _partial_decode(jpg_path: Path, u0: int, v0: int, s: int, IW: int, IH: int) -> np.ndarray:
    out = np.zeros((s, s, 3), dtype=np.uint8)
    su0 = max(0, u0); sv0 = max(0, v0)
    su1 = min(IW, u0 + s); sv1 = min(IH, v0 + s)
    if su1 <= su0 or sv1 <= sv0:
        return out
    if _HAVE_TJ:
        ju0 = (su0 // _MCU) * _MCU
        jv0 = (sv0 // _MCU) * _MCU
        ju1 = min(IW, ((su1 + _MCU - 1) // _MCU) * _MCU)
        jv1 = min(IH, ((sv1 + _MCU - 1) // _MCU) * _MCU)
        jw, jh = ju1 - ju0, jv1 - jv0
        with open(jpg_path, 'rb') as f:
            blob = f.read()
        cropped = _tj.transform(blob, crop=True, x=ju0, y=jv0, w=jw, h=jh, perfect=False)
        img = np.asarray(_tj.decompress(cropped, pixelformat=_tj.PF.RGB))[:jh, :jw]
        out[sv0 - v0:sv1 - v0, su0 - u0:su1 - u0] = img[sv0 - jv0:sv1 - jv0, su0 - ju0:su1 - ju0]
    else:
        with Image.open(jpg_path) as im:
            sub = np.array(im.crop((su0, sv0, su1, sv1)).convert('RGB'))
        out[sv0 - v0:sv1 - v0, su0 - u0:su1 - u0] = sub
    return out


class PandaSetLiveDataset(Dataset):
    """Cache-less PandaSet calib dataset.

    Args:
        scenes_root:    e.g. '/mnt/mininas/datasets/pandaset' (or the 103-scene
                        '/mnt/nvme6t/pandaset' which uses .pkl.gz)
        cameras:        'front_camera' (legacy) or 'all'
        img_size:       output patch side (px)
        max_offset_m:   uniform translation half-range (m)
        max_rot_deg:    uniform YPR half-range (deg)
        bbox_scale:     crop size = bbox_max × bbox_scale (full-res px)
        min_pts:        skip frames with fewer in-view points
        val_fraction:   fraction of scenes held out for val
        split_seed:     scene-shuffle seed (matches build_cache convention)
        split:          'train' / 'val'
    """
    def __init__(self,
                 scenes_root: str,
                 cameras: str = 'front_camera',
                 img_size: int = 64,
                 max_offset_m: float = 0.20,
                 max_rot_deg: float = 0.5,
                 bbox_scale: float = 3.0,
                 min_pts: int = 8,
                 val_fraction: float = 0.15,
                 split_seed: int = 42,
                 split: str = 'train',
                 virtual_epoch_len: int = None):
        self.root         = Path(scenes_root)
        self.img_size     = int(img_size)
        self.max_offset_m = float(max_offset_m)
        self.max_rot_deg  = float(max_rot_deg)
        self.bbox_scale   = float(bbox_scale)
        self.min_pts      = int(min_pts)

        # Stable scene-level split (shuffle then cut, matches build_cache).
        all_scenes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        rng = random.Random(split_seed); rng.shuffle(all_scenes)
        n_val = max(1, int(len(all_scenes) * val_fraction))
        self.scenes = all_scenes[:n_val] if split == 'val' else all_scenes[n_val:]

        # Camera list per scene (only those that exist).
        if cameras == 'all':
            self.cam_names_pref = ['front_camera', 'front_left_camera', 'front_right_camera',
                                   'left_camera', 'right_camera', 'back_camera']
        else:
            self.cam_names_pref = [cameras]

        # Pre-walk per-scene metadata + (scene, cam, fi, cuboid_idx) index.
        # Lidar / cuboid / image paths are stored as Path; lazy on getitem.
        self._scene_meta: Dict[str, dict] = {}
        self.index: List[Tuple[str, str, int, int]] = []

        for sc_name in self.scenes:
            sc_dir = self.root / sc_name
            cb_dir = sc_dir / 'annotations' / 'cuboids'
            ld_dir = sc_dir / 'lidar'
            ld_npy_dir = sc_dir / 'lidar_npy'
            # Prefer uncompressed .pkl over .pkl.gz when both exist (skips the
            # gzip decompress cost). Dedupe by stem so the frame index stays
            # aligned.
            def _pick_one_per_frame(d):
                seen = {}
                for p in list(d.glob('*.pkl')) + list(d.glob('*.pkl.gz')):
                    stem = p.name.replace('.pkl.gz', '').replace('.pkl', '')
                    if stem not in seen or p.suffix == '.pkl':
                        seen[stem] = p
                return [seen[k] for k in sorted(seen)]
            cuboid_files = _pick_one_per_frame(cb_dir)
            # Lidar: prefer pre-extracted xyz .npy (mmap-friendly, ~60× faster
            # than pandas DataFrame load + filter); fall back to .pkl/.gz.
            if ld_npy_dir.is_dir():
                lidar_files = sorted(ld_npy_dir.glob('*.npy'))
            else:
                lidar_files = _pick_one_per_frame(ld_dir)
            if not cuboid_files or not lidar_files:
                continue
            n_frames = min(len(cuboid_files), len(lidar_files))

            cams_avail = []
            for cam_name in self.cam_names_pref:
                cam_dir = sc_dir / 'camera' / cam_name
                if (cam_dir / 'poses.json').exists() and (cam_dir / 'intrinsics.json').exists():
                    cams_avail.append(cam_name)
            if not cams_avail:
                continue

            # Cache scene meta once.
            cam_metas = {}
            for cam in cams_avail:
                cam_dir = sc_dir / 'camera' / cam
                with open(cam_dir / 'poses.json') as f:    poses = json.load(f)
                with open(cam_dir / 'intrinsics.json') as f: intr = json.load(f)
                K = np.array([[intr['fx'], 0, intr['cx']],
                              [0, intr['fy'], intr['cy']],
                              [0, 0, 1]], dtype=np.float64)
                # PandaSet intrinsics.json doesn't include width/height; the
                # native PandaSet front camera is 1920×1080. Other cams may
                # differ, so derive from the first JPEG if not present.
                W = int(intr.get('width',  1920))
                H = int(intr.get('height', 1080))
                if 'width' not in intr:
                    jpg = next(cam_dir.glob('*.jpg'), None)
                    if jpg is not None:
                        with Image.open(jpg) as im:
                            W, H = im.size
                cam_metas[cam] = dict(poses=poses, K=K, width=W, height=H, cam_dir=cam_dir)
            self._scene_meta[sc_name] = dict(
                cams=cam_metas, lidar_files=lidar_files,
                cuboid_files=cuboid_files, n_frames=n_frames,
            )

            # Pre-index every cuboid in every (cam, frame) where the front
            # camera has a pose. Cuboid filtering by USEFUL_LABELS happens
            # at __getitem__ to keep this list compact.
            for cam in cams_avail:
                for fi in range(min(n_frames, len(cam_metas[cam]['poses']))):
                    self.index.append((sc_name, cam, fi, -1))   # -1 = pick a random cuboid

        if not self.index:
            raise RuntimeError(f'no usable scenes found under {scenes_root}')
        if virtual_epoch_len is not None:
            self._virtual_epoch_len = int(virtual_epoch_len)

    def __len__(self):
        # Virtual epoch size set explicitly to keep per-epoch wall time reasonable
        # and let the cosine LR schedule align with v9_lazy timing.
        return self._virtual_epoch_len if hasattr(self, '_virtual_epoch_len') else len(self.index)

    def _pick(self, idx):
        return self.index[idx % len(self.index)]

    def __getitem__(self, idx):
        for _try in range(20):
            sc_name, cam_name, fi, _ = self._pick(idx + _try)
            out = self._build(sc_name, cam_name, fi)
            if out is not None:
                return out
        # final fallback: random reroll
        return self[random.randint(0, len(self) - 1)]

    def _build(self, sc_name: str, cam_name: str, fi: int):
        meta = self._scene_meta[sc_name]
        cam  = meta['cams'][cam_name]
        IW, IH = cam['width'], cam['height']
        K = cam['K']
        cam_pose = cam['poses'][fi]
        pose_mat = _quat_pos_to_mat(cam_pose['heading'], cam_pose['position'])
        cam_pos = np.array([cam_pose['position']['x'],
                            cam_pose['position']['y'],
                            cam_pose['position']['z']], dtype=np.float32)

        # Load lidar (world frame). Fast path: pre-extracted xyz .npy (mmap);
        # fallback: pandas DataFrame from .pkl(.gz) with d==0 filter.
        lp = meta['lidar_files'][fi]
        if lp.suffix == '.npy':
            pts_world = np.load(lp, mmap_mode='r')
        else:
            lidar_df = _load_pickle(lp)
            if 'd' in lidar_df.columns:
                lidar_df = lidar_df[lidar_df['d'] == 0]
            pts_world = lidar_df[['x', 'y', 'z']].values.astype(np.float32)

        uv_gt_full, z_gt_full = _project(pts_world, pose_mat, K)
        vis = (z_gt_full > 0.5) & \
              (uv_gt_full[:, 0] >= 0) & (uv_gt_full[:, 0] < IW) & \
              (uv_gt_full[:, 1] >= 0) & (uv_gt_full[:, 1] < IH)
        if vis.sum() < self.min_pts:
            return None
        pts_vis = pts_world[vis]

        # Load cuboids; pick one with a valid 2D bbox.
        cuboids = _load_pickle(meta['cuboid_files'][fi])
        if 'label' in cuboids.columns:
            cuboids = cuboids[cuboids['label'].isin(USEFUL_LABELS)]
        cuboids = cuboids.reset_index(drop=True)
        if len(cuboids) == 0:
            return None
        # Sample the perturbation ONCE per __getitem__ (was inside the cuboid
        # loop → redundant re-projection every retry). The δT is independent
        # of which cuboid we anchor on; perturbed UVs only need to be filtered
        # by the chosen crop window after the fact.
        ypr     = (np.random.rand(3) * 2 - 1) * self.max_rot_deg
        t_delta = (np.random.rand(3) * 2 - 1) * self.max_offset_m
        R_gt   = Rotation.from_quat([cam_pose['heading']['x'],
                                     cam_pose['heading']['y'],
                                     cam_pose['heading']['z'],
                                     cam_pose['heading']['w']]).as_matrix().astype(np.float32)
        R_off  = R_gt @ Rotation.from_euler('zyx', ypr, degrees=True).as_matrix()
        cp_off = cam_pos + t_delta.astype(np.float32)
        T_off  = np.eye(4, dtype=np.float32)
        T_off[:3, :3] = R_off.T
        T_off[:3,  3] = -(R_off.T @ cp_off)
        pts_cam_off = T_off[:3, :3] @ pts_vis.T + T_off[:3, 3:]
        z_off       = pts_cam_off[2]
        uv_off      = ((K @ pts_cam_off)[:2] / z_off).T
        # GT projection of all vis pts (deterministic per (scene, cam, fi);
        # candidate for on-disk caching once we wire that in).
        uv_gt_vis = uv_gt_full[vis]
        # cam-frame depths for selected points; used for dist_m.
        # Already have z_gt_full[vis] would do it without re-inv.

        order = np.random.permutation(len(cuboids))

        for ci in order:
            obj = cuboids.iloc[int(ci)]
            pos = np.array([obj['position.x'], obj['position.y'], obj['position.z']])
            dims = np.array([obj['dimensions.x'], obj['dimensions.y'], obj['dimensions.z']])
            yaw = float(obj['yaw'])

            bbox = _bbox2d_of_cuboid(pos, dims, yaw, pose_mat, K)
            if bbox is None:
                continue
            u_min, v_min, u_max, v_max = bbox
            uc, vc = (u_min + u_max) / 2, (v_min + v_max) / 2
            if not (0 <= uc < IW and 0 <= vc < IH):
                continue

            bw, bh = u_max - u_min, v_max - v_min
            base_crop = max(bw, bh) * self.bbox_scale
            base_crop = max(base_crop, 64)
            base_half = base_crop / 2
            base_u0 = float(np.clip(uc - base_half, 0, IW - base_crop))
            base_v0 = float(np.clip(vc - base_half, 0, IH - base_crop))
            S = self.img_size

            # vfp / position augmentation: pick a random sub-window inside the
            # bbox-anchored crop. Mirrors build_sample's _sample_sub on the
            # 192-px lazy cache (s ∈ [img_size, cache_img], cuboid kept inside).
            # Without this the live path would always train at the single vfp
            # implied by bbox*scale, losing the v9 cache's scale augmentation.
            CACHE_REF = 192    # match build_sample's cache scale convention
            s_ref = np.random.randint(S, CACHE_REF + 1)             # [64, 192]
            sub_frac = s_ref / CACHE_REF
            crop_size = float(base_crop * sub_frac)
            # offset window so the cuboid stays roughly inside (matches lazy logic)
            free = base_crop - crop_size
            shift = (np.random.rand(2) - 0.5) * free * 0.66          # ~ [-1/3, +1/3]
            u0 = float(np.clip(base_u0 + free / 2 + shift[0], 0, IW - crop_size))
            v0 = float(np.clip(base_v0 + free / 2 + shift[1], 0, IH - crop_size))

            in_crop = ((uv_off[:, 0] >= u0) & (uv_off[:, 0] < u0 + crop_size) &
                       (uv_off[:, 1] >= v0) & (uv_off[:, 1] < v0 + crop_size) &
                       (z_off > 0.5))
            if in_crop.sum() < self.min_pts:
                continue

            scale = S / crop_size
            uv_d_crop = np.stack([(uv_off[in_crop, 0] - u0) * scale,
                                  (uv_off[in_crop, 1] - v0) * scale], axis=1)

            # Stratified sub-sample on a 16×16 grid (matches build_sample).
            grid, cell = 16, float(S) / 16
            sel = []
            for gi in range(grid):
                for gj in range(grid):
                    d2 = ((uv_d_crop[:, 0] - (gj + 0.5) * cell) ** 2 +
                          (uv_d_crop[:, 1] - (gi + 0.5) * cell) ** 2)
                    sel.append(int(d2.argmin()))
            sel = sorted(set(sel))
            idx_in = np.where(in_crop)[0][sel]
            pts_sel = pts_vis[idx_in]

            # GT projections (already computed once outside the loop).
            uv_gt_sel = uv_gt_vis[idx_in]
            uv_gt_c = np.stack([(uv_gt_sel[:, 0] - u0) * scale,
                                 (uv_gt_sel[:, 1] - v0) * scale], axis=1)
            uv_off_c = uv_d_crop[sel]
            dist_m = (np.linalg.norm(pts_sel - cam_pos, axis=1) / 100.0).astype(np.float32)

            # Per-point obj/bg via the cuboid centroid+yaw geometric test.
            c_y, s_y = np.cos(yaw), np.sin(yaw)
            R_obj = np.array([[c_y, s_y, 0], [-s_y, c_y, 0], [0, 0, 1]], dtype=np.float32)
            pts_local = (R_obj @ (pts_sel - pos.astype(np.float32)).T).T
            half_d = dims.astype(np.float32) / 2.0
            is_obj = ((np.abs(pts_local[:, 0]) <= half_d[0]) &
                      (np.abs(pts_local[:, 1]) <= half_d[1]) &
                      (np.abs(pts_local[:, 2]) <= half_d[2])).astype(np.float32)

            true_uvd = np.concatenate([uv_gt_c.astype(np.float32),
                                        dist_m[:, None], is_obj[:, None]], axis=1)
            dist_uvd = np.concatenate([uv_off_c.astype(np.float32),
                                        dist_m[:, None], is_obj[:, None]], axis=1)

            # libjpeg-turbo MCU partial decode + PIL resize.
            cam_dir = cam['cam_dir']
            jpg_path = cam_dir / f'{fi:02d}.jpg'
            patch = _partial_decode(jpg_path, int(u0), int(v0), int(crop_size), IW, IH)
            patch_resized = np.asarray(
                Image.fromarray(patch).resize((S, S), Image.BILINEAR), dtype=np.uint8)
            img_crop = torch.from_numpy(patch_resized).permute(2, 0, 1).float() / 255.0

            vfp = float(K[0, 0]) * S / crop_size
            return (img_crop,
                    torch.from_numpy(true_uvd),
                    torch.from_numpy(dist_uvd),
                    torch.tensor(vfp, dtype=torch.float32))
        return None


def collate_pandaset_live(batch):
    """5-tuple collate: (imgs, true_t, dist_t, mask, vfp_t). Matches
    collate_pandaset_vfp shape so the existing trainer just slots in."""
    imgs, true_uvds, dist_uvds, vfps = zip(*batch)
    max_n = max(t.shape[0] for t in true_uvds)
    def pad(tensors):
        out  = torch.zeros(len(tensors), max_n, tensors[0].shape[1])
        mask = torch.ones(len(tensors), max_n, dtype=torch.bool)
        for i, t in enumerate(tensors):
            out[i, :len(t)] = t; mask[i, :len(t)] = False
        return out, mask
    true_t, _    = pad(true_uvds)
    dist_t, mask = pad(dist_uvds)
    return torch.stack(imgs), true_t, dist_t, mask, torch.stack(vfps)


if __name__ == '__main__':
    import time
    ds = PandaSetLiveDataset('/mnt/mininas/datasets/pandaset',
                              cameras='front_camera', split='train')
    print(f'instances (virtual): {len(ds)}  underlying frames: {len(ds.index)}')
    t0 = time.time()
    for i in range(8):
        img, t, d, vfp = ds[i]
    print(f'8 samples: {(time.time()-t0)*1000:.0f}ms ({(time.time()-t0)*125:.1f}ms/sample)')
    print(f'img: {img.shape}  true: {t.shape}  vfp: {vfp.item():.1f}')
    print(f'shift mean: {(t[:, :2] - d[:, :2]).norm(dim=1).mean():.2f} px')
    print(f'is_obj count: {(t[:, 3] > 0.5).sum().item()} / {t.shape[0]}')
