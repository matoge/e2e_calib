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
import json, random
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation


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


def _project(pts_world, T_w2c, K):
    """Project world points into camera. Returns (uv (N,2), z (N,)) in pixel coords."""
    N = pts_world.shape[0]
    homo = np.concatenate([pts_world, np.ones((N, 1), dtype=pts_world.dtype)], axis=1)
    pts_cam = (T_w2c @ homo.T)[:3]
    z = pts_cam[2]
    uv = (K @ pts_cam)[:2] / np.clip(z, 1e-6, None)
    return uv.T.astype(np.float32), z.astype(np.float32)


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
    """Holds metadata + precomputed per-frame projections for ONE scene."""

    def __init__(self, scene_root: Path):
        self.root = scene_root

        # intrinsics
        intr = json.loads((scene_root / 'camera/front_camera/intrinsics.json').read_text())
        self.K = np.array([[intr['fx'], 0, intr['cx']],
                           [0, intr['fy'], intr['cy']],
                           [0,          0,          1]], dtype=np.float32)

        # poses
        poses = json.loads((scene_root / 'camera/front_camera/poses.json').read_text())
        self.n_frames = len(poses)
        self.T_w2c = np.stack([_quat_pos_to_mat(p['heading'], p['position']) for p in poses])

        # image/lidar paths
        self.image_paths = [scene_root / f'camera/front_camera/{fi:02d}.jpg' for fi in range(self.n_frames)]
        self.lidar_paths = [scene_root / f'lidar/{fi:02d}.pkl'              for fi in range(self.n_frames)]

        # image size (assume all same in scene) from first image
        with Image.open(self.image_paths[0]) as im:
            self.IW, self.IH = im.size

        # frame-level caches: computed lazily (but always via precompute_all())
        self._img_cache = {}
        self._frame_cache = {}

    # ------------------------------------------------------------------ helpers

    # image cache stores FULL-resolution pre-decoded arrays (default).
    # Set img_scale_div > 1 only when memory pressure forces it; quality is
    # the priority for cross-frame matching.
    img_scale_div: int = 1     # 1 = full res, 2 = 1/2, 4 = 1/4

    def load_image(self, fi):
        if fi not in self._img_cache:
            with Image.open(self.image_paths[fi]) as im:
                arr = np.array(im.convert('RGB'))
            if self.img_scale_div > 1:
                H, W = arr.shape[:2]
                h = H // self.img_scale_div
                w = W // self.img_scale_div
                # fast downsample via PIL (SIMD-optimised)
                arr = np.array(Image.fromarray(arr).resize((w, h), Image.BILINEAR))
            self._img_cache[fi] = arr
        return self._img_cache[fi]

    def frame_data(self, fi):
        """Return (pts_world, uv_own_cam, z_own_cam, in_view_mask)."""
        if fi not in self._frame_cache:
            df = pd.read_pickle(self.lidar_paths[fi])
            pts_w = df[['x', 'y', 'z']].values.astype(np.float32)
            uv, z = _project(pts_w, self.T_w2c[fi], self.K)
            in_view = ((z > 1.0) &
                       (uv[:, 0] > 0) & (uv[:, 0] < self.IW) &
                       (uv[:, 1] > 0) & (uv[:, 1] < self.IH))
            self._frame_cache[fi] = (pts_w, uv, z, in_view)
        return self._frame_cache[fi]

    def precompute_all(self, preload_images: bool = True, n_workers: int = 8):
        """Eagerly compute per-frame projections (and optionally decode images),
        in parallel via ThreadPoolExecutor (numpy + PIL release the GIL during
        decode/projection so threads scale well).

        With DataLoader workers, parent precomputes → workers fork via COW,
        so the dataset is effectively read-only and disk I/O at training
        time drops to zero.
        """
        from concurrent.futures import ThreadPoolExecutor
        fis = list(range(self.n_frames))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(self.frame_data, fis))
            if preload_images:
                list(ex.map(self.load_image, fis))


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
                 seed: int = 42):
        super().__init__()
        self.img_size = img_size
        self.baseline_range = baseline_range
        self.sigma_ypr = sigma_ypr
        self.sigma_t   = sigma_t
        self.max_points = max_points
        self.crop_range = crop_range
        self.virtual_epoch_len = virtual_epoch_len
        self.split = split
        self.rng = np.random.default_rng(seed)

        # ── resolve scene list ──
        if scene_roots is None and scenes_root is not None:
            root = Path(scenes_root)
            scene_names = sorted([p.name for p in root.iterdir()
                                   if p.is_dir() and p.name.isdigit()])
            scene_roots = [str(root / name) for name in scene_names]
        elif scene_roots is None and scene_root is not None:
            # legacy single scene: use whole scene for requested split
            scene_roots = [scene_root]
            train_frac = 1.0 if split == 'train' else 0.0
        elif scene_roots is None:
            raise ValueError('Must provide scenes_root, scene_roots, or scene_root')

        # ── scene-level split ──
        if len(scene_roots) > 1 and 0.0 < train_frac < 1.0:
            # deterministic scene shuffle, then split
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

        from concurrent.futures import ThreadPoolExecutor
        print(f'[PandaSetCrossFrameDataset/{split}] loading {len(split_roots)} scenes (parallel)…',
              flush=True)

        def _load_scene(sr):
            scn = _SceneData(Path(sr))
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

        # ThreadPool over scenes; numpy + PIL release GIL during heavy work.
        with ThreadPoolExecutor(max_workers=8) as ex:
            self.scenes: List[_SceneData] = list(ex.map(_load_scene, split_roots))

        for scn in self.scenes:
            print(f'  · {scn.root.name}: {scn.n_frames} frames, pool={len(scn.fi_pool)}',
                  flush=True)

        print(f'[PandaSetCrossFrameDataset/{split}] ready. '
              f'{len(self.scenes)} scenes, '
              f'{sum(len(s.fi_pool) for s in self.scenes)} total frames',
              flush=True)

    # ------------------------------------------------------------------ helpers

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
            sample = self._try_one(idx + retry * 9973)
            if sample is not None:
                return sample
        raise RuntimeError("could not form a valid cross-frame pair in 30 tries")

    def _try_one(self, idx):
        rng = np.random.default_rng((idx + 1) * 2654435761 & 0xFFFFFFFF)

        # 0. pick scene
        scn: _SceneData = self.scenes[int(rng.integers(len(self.scenes)))]
        IW, IH, K = scn.IW, scn.IH, scn.K

        # 1. pick fi_A
        if not scn.fi_pool:
            return None
        fi_A = int(rng.choice(scn.fi_pool))
        # 2. fi_B: nearby frame
        bmin, bmax = self.baseline_range
        delta = int(rng.integers(bmin, bmax + 1)) * int(rng.choice([-1, 1]))
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

        # 6. pick center 3D point from A
        ci = int(rng.integers(len(pts_w_A_in)))
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
        uc_B_hat = (K @ P_center_Bh)[:2] / P_center_Bh[2]
        uc_B_gt  = (K @ P_center_Bg)[:2] / P_center_Bg[2]
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
        img_A_full = scn.load_image(fi_A)
        img_B_full = scn.load_image(fi_B)
        pa = self._crop_patch(img_A_full, u0_A, v0_A, CROP, IW, IH, scn.img_scale_div)
        pb = self._crop_patch(img_B_full, u0_B, v0_B, CROP, IW, IH, scn.img_scale_div)
        if pa is None or pb is None:
            return None
        patch_A, box_A = pa
        patch_B, box_B = pb

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
        uv_B_gt_full_A  = (K @ P_QA_in_B_gt[good_qa].T)[:2]  / P_QA_in_B_gt[good_qa, 2]
        uv_B_hat_full_A = (K @ P_QA_in_B_hat[good_qa].T)[:2] / P_QA_in_B_hat[good_qa, 2]
        uv_B_gt_local_A  = self._uv_to_patch_local(uv_B_gt_full_A.T.astype(np.float32),  box_B)
        uv_B_hat_local_A = self._uv_to_patch_local(uv_B_hat_full_A.T.astype(np.float32), box_B)

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
        uv_A_gt_full_B  = (K @ P_QB_in_A_gt[good_qb].T)[:2]  / P_QB_in_A_gt[good_qb, 2]
        uv_A_hat_full_B = (K @ P_QB_in_A_hat[good_qb].T)[:2] / P_QB_in_A_hat[good_qb, 2]
        uv_A_gt_local_B  = self._uv_to_patch_local(uv_A_gt_full_B.T.astype(np.float32),  box_A)
        uv_A_hat_local_B = self._uv_to_patch_local(uv_A_hat_full_B.T.astype(np.float32), box_A)
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

        # 12. pad/truncate to max_points
        def _pad(uv_query, z_query, uv_hat, uv_gt, d_hat, d_gt):
            N = len(uv_query)
            N_use = min(N, self.max_points)
            pick = self.rng.choice(N, size=N_use, replace=False) if N > N_use else np.arange(N)
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

        uv_A_patch, z_A_patch, uv_B_hat_local_A, uv_B_gt_local_A, d_B_hat_of_A, d_B_gt_of_A, pad_A = _pad(
            uv_A_patch, z_A_patch, uv_B_hat_local_A, uv_B_gt_local_A, d_B_hat_of_A, d_B_gt_of_A)
        uv_B_patch, z_B_patch, uv_A_hat_local_B, uv_A_gt_local_B, d_A_hat_of_B, d_A_gt_of_B, pad_B = _pad(
            uv_B_patch, z_B_patch, uv_A_hat_local_B, uv_A_gt_local_B, d_A_hat_of_B, d_A_gt_of_B)

        z_A_norm = z_A_patch / 50.0
        z_B_norm = z_B_patch / 50.0

        ypr_AB_hat, t_AB_hat = _mat_to_ypr_t(T_A_to_B_hat)
        ypr_BA_hat, t_BA_hat = _mat_to_ypr_t(_invert_mat(T_A_to_B_hat))

        return dict(
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
            fi_A = fi_A, fi_B = fi_B,
            scene = scn.root.name,
        )


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
