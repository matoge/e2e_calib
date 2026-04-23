"""PandaSet cross-frame pair dataset — short-baseline PoC.

For a chosen scene:
  1. Index all frames: image_path, lidar_path, GT pose, intrinsics.
  2. __getitem__ samples (fi_A, fi_B) with |fi_A - fi_B| ∈ [baseline_min, baseline_max].
  3. Load both frames, project all LiDAR into each camera.
  4. Pick a center 3D point P_center from A's in-view LiDAR.
  5. Crop patch_A around P's projection in A.
  6. Sample pose perturbation δT (ypr, t) → T_AB_hat = T_AB_gt · δT.
  7. Crop patch_B around P's HYPOTHESIS projection in B.
  8. Gather A's LiDAR points projecting into patch_A → query points for A→B.
     Gather B's LiDAR points projecting into patch_B → query points for B→A.
  9. For each query point, compute (uv_hat, uv_gt) on the other frame;
     target Δuv = uv_gt - uv_hat (pixel space, patch-local coords).

Everything in SI units: ypr in degrees, t in meters.
"""
import json, random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation


def _quat_pos_to_mat(heading, position):
    """world→cam transform (inverse of the pose_quat provided by PandaSet).

    PandaSet poses.json gives camera→world (quaternion of camera orientation in world).
    We want world→cam.
    """
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
    pts_cam = (T_w2c @ homo.T)[:3]                                   # (3, N)
    z = pts_cam[2]
    uv = (K @ pts_cam)[:2] / np.clip(z, 1e-6, None)
    return uv.T.astype(np.float32), z.astype(np.float32)


def _ypr_t_to_mat(ypr_deg, t):
    """6DoF (ypr in degrees, t in meters) → 4x4 matrix."""
    R = Rotation.from_euler('zyx', ypr_deg, degrees=True).as_matrix()
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3]  = t
    return M


def _mat_to_ypr_t(M):
    """4x4 → (ypr_deg, t)."""
    ypr = Rotation.from_matrix(M[:3, :3]).as_euler('zyx', degrees=True).astype(np.float32)
    t   = M[:3, 3].astype(np.float32)
    return ypr, t


def _invert_mat(M):
    R = M[:3, :3]; t = M[:3, 3]
    Minv = np.eye(4, dtype=M.dtype)
    Minv[:3, :3] = R.T
    Minv[:3, 3]  = -R.T @ t
    return Minv


class PandaSetCrossFrameDataset(Dataset):
    """One-scene cross-frame pair dataset for PoC.

    Call PandaSetCrossFrameDataset(scene_root=..., train=True/False) to split by frame index.
    """

    def __init__(self,
                 scene_root: str,
                 split: str = 'train',
                 train_frac: float = 0.75,
                 img_size: int = 64,
                 baseline_range = (1, 5),
                 sigma_ypr: float = 0.3,   # degrees
                 sigma_t:   float = 0.15,  # meters
                 max_points: int = 96,
                 crop_range = (64, 192),   # px in full-res, random per sample, same for A & B
                 virtual_epoch_len: int = 2000,
                 seed: int = 42):
        super().__init__()
        root = Path(scene_root)
        self.root = root
        self.img_size = img_size
        self.baseline_range = baseline_range
        self.sigma_ypr = sigma_ypr
        self.sigma_t   = sigma_t
        self.max_points = max_points
        self.crop_range = crop_range
        self.virtual_epoch_len = virtual_epoch_len
        self.split = split
        self.rng = np.random.default_rng(seed)

        # intrinsics
        intr = json.loads((root / 'camera/front_camera/intrinsics.json').read_text())
        self.K = np.array([[intr['fx'], 0, intr['cx']],
                           [0, intr['fy'], intr['cy']],
                           [0,          0,          1]], dtype=np.float32)

        # poses
        poses = json.loads((root / 'camera/front_camera/poses.json').read_text())
        self.n_frames = len(poses)
        self.T_w2c = np.stack([_quat_pos_to_mat(p['heading'], p['position']) for p in poses])

        # image/lidar paths
        self.image_paths = [root / f'camera/front_camera/{fi:02d}.jpg' for fi in range(self.n_frames)]
        self.lidar_paths = [root / f'lidar/{fi:02d}.pkl'              for fi in range(self.n_frames)]

        # image size (assume all same) from first image
        with Image.open(self.image_paths[0]) as im:
            self.IW, self.IH = im.size

        # simple frame split
        cutoff = int(self.n_frames * train_frac)
        all_fi = list(range(self.n_frames))
        if split == 'train':
            self.fi_pool = all_fi[:cutoff]
        elif split == 'val':
            self.fi_pool = all_fi[cutoff:]
        else:
            self.fi_pool = all_fi

        # tiny memory cache (scene is small: ~80 frames × ~2MB lidar + ~500KB jpg)
        self._img_cache    = {}
        self._frame_cache  = {}    # fi → (pts_world, uv_own_cam, z_own_cam, in_view_mask)

        # eagerly precompute per-frame projections: ~150k-point matmul × 80 frames
        # → dominates sample generation cost if done per __getitem__
        print(f'[PandaSetCrossFrameDataset/{split}] precomputing projections for '
              f'{self.n_frames} frames…', flush=True)
        for fi in range(self.n_frames):
            self._frame_data(fi)
        print(f'[PandaSetCrossFrameDataset/{split}] done. '
              f'pool(train/val)={len(self.fi_pool)} frames', flush=True)

    # ------------------------------------------------------------------ helpers

    def _load_image(self, fi):
        if fi not in self._img_cache:
            with Image.open(self.image_paths[fi]) as im:
                self._img_cache[fi] = np.array(im.convert('RGB'))
        return self._img_cache[fi]

    def _frame_data(self, fi):
        """Return (pts_world, uv_own_cam, z_own_cam, in_view_mask) for frame fi.

        Projection computed once per frame and cached; matmul on ~150k LiDAR
        points is the single most expensive operation in __getitem__, so this
        is where all the speedup is.
        """
        if fi not in self._frame_cache:
            df = pd.read_pickle(self.lidar_paths[fi])
            pts_w = df[['x', 'y', 'z']].values.astype(np.float32)
            uv, z = _project(pts_w, self.T_w2c[fi], self.K)
            in_view = ((z > 1.0) &
                       (uv[:, 0] > 0) & (uv[:, 0] < self.IW) &
                       (uv[:, 1] > 0) & (uv[:, 1] < self.IH))
            self._frame_cache[fi] = (pts_w, uv, z, in_view)
        return self._frame_cache[fi]

    def _crop_patch(self, img_full, u0, v0, s):
        """Crop img_full to [u0:u0+s, v0:v0+s] and resize to (img_size, img_size)."""
        u0i, v0i = int(u0), int(v0)
        u1i, v1i = u0i + int(s), v0i + int(s)
        u0i = max(0, u0i); v0i = max(0, v0i)
        u1i = min(self.IW, u1i); v1i = min(self.IH, v1i)
        if u1i - u0i < 2 or v1i - v0i < 2:
            return None
        patch = img_full[v0i:v1i, u0i:u1i]                 # (h, w, 3) uint8
        t = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
        t = F.interpolate(t.unsqueeze(0), size=(self.img_size, self.img_size),
                           mode='bilinear', align_corners=False).squeeze(0)
        # return crop box (u0i, v0i, u1i - u0i, v1i - v0i) for coord conversion
        return t, (u0i, v0i, u1i - u0i, v1i - v0i)

    def _uv_to_patch_local(self, uv_full, crop_box):
        """Map full-image uv to patch-local pixel coords (img_size scale)."""
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
        for _ in range(30):                          # retry until a valid sample forms
            sample = self._try_one(idx + _ * 9973)
            if sample is not None:
                return sample
        raise RuntimeError("could not form a valid cross-frame pair in 30 tries")

    def _try_one(self, idx):
        rng = np.random.default_rng((idx + 1) * 2654435761 & 0xFFFFFFFF)

        # 1. pick fi_A
        fi_A = int(rng.choice(self.fi_pool))
        # 2. fi_B: nearby frame, maybe earlier or later
        bmin, bmax = self.baseline_range
        delta = int(rng.integers(bmin, bmax + 1)) * int(rng.choice([-1, 1]))
        fi_B = fi_A + delta
        if fi_B < 0 or fi_B >= self.n_frames:
            return None

        # 3. poses
        T_w2A = self.T_w2c[fi_A]
        T_w2B = self.T_w2c[fi_B]
        T_A2w = _invert_mat(T_w2A)
        T_A_to_B_gt = T_w2B @ T_A2w                  # A-cam → B-cam

        # 4. load LiDAR and project (cached per frame)
        pts_w_A, uv_Af, z_Af, in_A = self._frame_data(fi_A)
        pts_w_B, uv_Bf, z_Bf, in_B = self._frame_data(fi_B)
        if in_A.sum() < 50 or in_B.sum() < 50:
            return None

        pts_w_A_in = pts_w_A[in_A]
        uv_A_all   = uv_Af[in_A]
        z_A_all    = z_Af[in_A]

        # 6. pick center 3D point from A → defines patch_A center
        ci = int(rng.integers(len(pts_w_A_in)))
        P_center_w = pts_w_A_in[ci]
        uc_A, vc_A = uv_A_all[ci]

        # 7. sample perturbation → T_AB_hat
        ypr_pert = rng.standard_normal(3).astype(np.float32) * self.sigma_ypr
        t_pert   = rng.standard_normal(3).astype(np.float32) * self.sigma_t
        δT       = _ypr_t_to_mat(ypr_pert, t_pert)
        T_A_to_B_hat = T_A_to_B_gt @ δT

        # 8. project center into B with T_hat → patch_B center
        P_center_A   = (T_w2A @ np.append(P_center_w, 1.0))[:3]
        P_center_Bh  = (T_A_to_B_hat @ np.append(P_center_A, 1.0))[:3]
        if P_center_Bh[2] < 1.0:
            return None
        uc_B_hat = (self.K @ P_center_Bh)[:2] / P_center_Bh[2]
        uc_B_hat = uc_B_hat.astype(np.float32)

        # 9. crop patches. random crop size per sample, shared by A & B, resized to img_size
        CROP = int(rng.integers(self.crop_range[0], self.crop_range[1] + 1))
        half = CROP / 2
        u0_A = max(0, min(self.IW - CROP, uc_A - half))
        v0_A = max(0, min(self.IH - CROP, vc_A - half))
        u0_B = max(0, min(self.IW - CROP, uc_B_hat[0] - half))
        v0_B = max(0, min(self.IH - CROP, uc_B_hat[1] - half))
        img_A_full = self._load_image(fi_A)
        img_B_full = self._load_image(fi_B)
        pa = self._crop_patch(img_A_full, u0_A, v0_A, CROP)
        pb = self._crop_patch(img_B_full, u0_B, v0_B, CROP)
        if pa is None or pb is None:
            return None
        patch_A, box_A = pa
        patch_B, box_B = pb

        # 10. gather A's in-patch LiDAR → query points for A→B
        u0, v0, cw, ch = box_A
        in_box_A = ((uv_A_all[:, 0] >= u0) & (uv_A_all[:, 0] < u0 + cw) &
                    (uv_A_all[:, 1] >= v0) & (uv_A_all[:, 1] < v0 + ch))
        if in_box_A.sum() < 4:
            return None
        pts_w_QA      = pts_w_A_in[in_box_A]
        uv_A_patch    = self._uv_to_patch_local(uv_A_all[in_box_A], box_A)
        z_A_patch     = z_A_all[in_box_A]

        # for each QA point, compute uv_B_gt (patch_B coords) and uv_B_hat
        homo = np.concatenate([pts_w_QA, np.ones((len(pts_w_QA), 1), dtype=np.float32)], axis=1)
        P_QA_in_A  = (T_w2A @ homo.T)[:3].T                         # (N,3)  A-cam
        P_QA_in_B_gt  = (np.column_stack([P_QA_in_A, np.ones(len(P_QA_in_A))]) @ T_A_to_B_gt.T)[:, :3]
        P_QA_in_B_hat = (np.column_stack([P_QA_in_A, np.ones(len(P_QA_in_A))]) @ T_A_to_B_hat.T)[:, :3]
        valid_gt  = P_QA_in_B_gt[:, 2]  > 0.5
        valid_hat = P_QA_in_B_hat[:, 2] > 0.5
        good_qa = valid_gt & valid_hat
        if good_qa.sum() < 4:
            return None
        uv_B_gt_full_A  = (self.K @ P_QA_in_B_gt[good_qa].T)[:2]  / P_QA_in_B_gt[good_qa, 2]
        uv_B_hat_full_A = (self.K @ P_QA_in_B_hat[good_qa].T)[:2] / P_QA_in_B_hat[good_qa, 2]
        uv_B_gt_full_A  = uv_B_gt_full_A.T.astype(np.float32)
        uv_B_hat_full_A = uv_B_hat_full_A.T.astype(np.float32)
        # to patch_B coords
        uv_B_gt_local_A  = self._uv_to_patch_local(uv_B_gt_full_A,  box_B)
        uv_B_hat_local_A = self._uv_to_patch_local(uv_B_hat_full_A, box_B)

        uv_A_patch    = uv_A_patch[good_qa]
        z_A_patch     = z_A_patch[good_qa]
        P_QA_in_A     = P_QA_in_A[good_qa]
        # drop queries where either hat or gt lies outside patch_B
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

        # 11. gather B's in-patch LiDAR → query points for B→A  (symmetric)
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
        uv_A_gt_full_B  = (self.K @ P_QB_in_A_gt[good_qb].T)[:2]  / P_QB_in_A_gt[good_qb, 2]
        uv_A_hat_full_B = (self.K @ P_QB_in_A_hat[good_qb].T)[:2] / P_QB_in_A_hat[good_qb, 2]
        uv_A_gt_local_B  = self._uv_to_patch_local(uv_A_gt_full_B.T.astype(np.float32),  box_A)
        uv_A_hat_local_B = self._uv_to_patch_local(uv_A_hat_full_B.T.astype(np.float32), box_A)
        uv_B_patch = uv_B_patch[good_qb]
        z_B_patch  = z_B_patch[good_qb]
        P_QB_in_B  = P_QB_in_B[good_qb]
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

        # 12. pad/truncate to max_points
        def _pad(uv_query, z_query, uv_hat, uv_gt):
            N = len(uv_query)
            N_use = min(N, self.max_points)
            pick = self.rng.choice(N, size=N_use, replace=False) if N > N_use else np.arange(N)
            uv_query = uv_query[pick]; z_query = z_query[pick]
            uv_hat = uv_hat[pick];     uv_gt = uv_gt[pick]
            pad = np.zeros((self.max_points,), dtype=bool)
            if N_use < self.max_points:
                pad[N_use:] = True
                uv_query = np.concatenate([uv_query, np.zeros((self.max_points - N_use, 2), np.float32)])
                z_query  = np.concatenate([z_query,  np.zeros(self.max_points - N_use, np.float32)])
                uv_hat   = np.concatenate([uv_hat,   np.zeros((self.max_points - N_use, 2), np.float32)])
                uv_gt    = np.concatenate([uv_gt,    np.zeros((self.max_points - N_use, 2), np.float32)])
            return uv_query, z_query, uv_hat, uv_gt, pad

        uv_A_patch, z_A_patch, uv_B_hat_local_A, uv_B_gt_local_A, pad_A = _pad(
            uv_A_patch, z_A_patch, uv_B_hat_local_A, uv_B_gt_local_A)
        uv_B_patch, z_B_patch, uv_A_hat_local_B, uv_A_gt_local_B, pad_B = _pad(
            uv_B_patch, z_B_patch, uv_A_hat_local_B, uv_A_gt_local_B)

        # 13. depth normalization (consistent with existing datasets): log(1+d)/log(1+80)
        # keep it simple: just raw depth / 50 for [0, ~1]
        z_A_norm = z_A_patch / 50.0
        z_B_norm = z_B_patch / 50.0

        # 14. pose hypothesis as 6DoF
        ypr_AB_hat, t_AB_hat = _mat_to_ypr_t(T_A_to_B_hat)
        ypr_BA_hat, t_BA_hat = _mat_to_ypr_t(_invert_mat(T_A_to_B_hat))

        return dict(
            patch_A = patch_A.float(),                      # (3, img_size, img_size)
            patch_B = patch_B.float(),
            uvd_A   = torch.from_numpy(
                np.concatenate([uv_A_patch, z_A_norm[:, None]], axis=1)).float(),
            uvd_B   = torch.from_numpy(
                np.concatenate([uv_B_patch, z_B_norm[:, None]], axis=1)).float(),
            uv_B_hat_of_A = torch.from_numpy(uv_B_hat_local_A).float(),
            uv_B_gt_of_A  = torch.from_numpy(uv_B_gt_local_A).float(),
            uv_A_hat_of_B = torch.from_numpy(uv_A_hat_local_B).float(),
            uv_A_gt_of_B  = torch.from_numpy(uv_A_gt_local_B).float(),
            pose_AB_6dof  = torch.from_numpy(
                np.concatenate([ypr_AB_hat, t_AB_hat]).astype(np.float32)),
            pose_BA_6dof  = torch.from_numpy(
                np.concatenate([ypr_BA_hat, t_BA_hat]).astype(np.float32)),
            pad_A = torch.from_numpy(pad_A),
            pad_B = torch.from_numpy(pad_B),
            # scalar meta for debugging
            fi_A = fi_A, fi_B = fi_B,
        )


# ─── smoke test ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ds = PandaSetCrossFrameDataset(
        scene_root='/mnt/mininas/datasets/pandaset/015',
        split='train',
        img_size=64,
        max_points=64,
    )
    s = ds[0]
    for k, v in s.items():
        if hasattr(v, 'shape'):
            print(f'  {k}: {tuple(v.shape)}  dtype={v.dtype}')
        else:
            print(f'  {k}: {v}')
    pad_A = s['pad_A']
    pad_B = s['pad_B']
    print(f'valid A: {(~pad_A).sum().item()} / {len(pad_A)}')
    print(f'valid B: {(~pad_B).sum().item()} / {len(pad_B)}')
    dA = (s['uv_B_gt_of_A'] - s['uv_B_hat_of_A'])[~pad_A]
    print(f'target residual A→B: mean |Δu|={dA[:,0].abs().mean():.2f} px, mean |Δv|={dA[:,1].abs().mean():.2f} px')
