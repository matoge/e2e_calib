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
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation


def _is_obj_per_point(pts_sel: np.ndarray, cuboids: list) -> np.ndarray:
    """Return (N,) float32 mask: 1 if a point lies inside ANY cuboid AABB-rotated box."""
    if not cuboids:
        return np.zeros(len(pts_sel), dtype=np.float32)
    is_obj = np.zeros(len(pts_sel), dtype=bool)
    for cub in cuboids:
        pos  = cub['pos']
        dims = cub['dims']
        yaw  = float(cub['yaw'])
        c_y, s_y = np.cos(yaw), np.sin(yaw)
        R_obj = np.array([[c_y, s_y, 0],
                          [-s_y, c_y, 0],
                          [0,    0,  1]], dtype=np.float32)
        local = (R_obj @ (pts_sel - pos).T).T
        half = dims / 2.0
        inside = ((np.abs(local[:,0]) <= half[0]) &
                  (np.abs(local[:,1]) <= half[1]) &
                  (np.abs(local[:,2]) <= half[2]))
        is_obj |= inside
    return is_obj.astype(np.float32)


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
                 min_pts: int = 8,
                 max_tries: int = 8,
                 oversample: int = 12):
        self.cache_dir = Path(cache_dir)
        self.inst_dir  = self.cache_dir / 'inst'
        meta = torch.load(self.cache_dir / 'meta.pt', weights_only=False)
        assert split in ('train', 'val')
        self.fnames    = list(meta[split])
        self.img_size  = int(img_size)
        self.min_crop_px = int(min_crop_px)
        self.max_crop_px = int(max_crop_px)
        self.max_offset_m = float(max_offset_m)
        self.max_rot_deg  = float(max_rot_deg)
        self.min_pts   = int(min_pts)
        self.max_tries = int(max_tries)
        self.oversample = int(oversample)

    def __len__(self):
        return len(self.fnames) * self.oversample

    def _load_inst(self, idx: int) -> dict:
        # idx is in [0, len_fnames * oversample); modulo to wrap to file index
        return torch.load(self.inst_dir / self.fnames[idx % len(self.fnames)], weights_only=False)

    def __getitem__(self, idx: int):
        inst = self._load_inst(idx)
        IH, IW = int(inst['img'].shape[-2]), int(inst['img'].shape[-1])
        K = inst['K_full'].numpy()
        pts = inst['pts'].numpy()
        cp  = inst['cam_pos'].numpy()
        R_gt = inst['R_gt'].numpy()
        cubs = inst.get('cuboids', [])

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

        valid_in_image = ((z > 0.5) &
                          (uv_full[:,0] >= 0) & (uv_full[:,0] < IW) &
                          (uv_full[:,1] >= 0) & (uv_full[:,1] < IH))
        obj_idxs = np.where(is_obj_full & valid_in_image)[0]
        bg_mask  = (~is_obj_full) & valid_in_image
        # 10x5 grid for bg-pivot stratification
        GU, GV = 10, 5
        cell_w = IW / GU; cell_h = IH / GV
        cell_u = np.clip((uv_full[:, 0] / cell_w).astype(int), 0, GU-1)
        cell_v = np.clip((uv_full[:, 1] / cell_h).astype(int), 0, GV-1)
        cell_id_full = cell_v * GU + cell_u
        bg_cells = np.unique(cell_id_full[bg_mask]) if bg_mask.any() else np.array([], dtype=int)

        S = self.img_size
        # Pre-flatten image for fast cropping
        img_full = inst['img']  # (3, H, W) uint8

        for _ in range(self.max_tries):
            cs = int(np.random.randint(self.min_crop_px, self.max_crop_px + 1))
            cs = min(cs, IW, IH)
            # 50/50 obj/bg pivot
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
            u0 = int(np.clip(pu - cs/2, 0, IW - cs))
            v0 = int(np.clip(pv - cs/2, 0, IH - cs))

            # Pre-filter to crop+10% padding using cached uv_full → cap at 2000 pts
            pad_px = int(cs * 0.10)
            in_pad = ((uv_full[:, 0] >= u0 - pad_px) & (uv_full[:, 0] < u0 + cs + pad_px) &
                      (uv_full[:, 1] >= v0 - pad_px) & (uv_full[:, 1] < v0 + cs + pad_px) &
                      (z > 0.5))
            cand_idx = np.where(in_pad)[0]
            if len(cand_idx) < self.min_pts:
                continue
            if len(cand_idx) > 2000:
                cand_idx = np.random.choice(cand_idx, size=2000, replace=False)
            pts_c = pts[cand_idx]                       # (M<=2000, 3)
            uv_gt_c = uv_full[cand_idx]                 # (M, 2) full-image px

            # Project candidates with perturbed pose
            t_delta = (np.random.rand(3) * 2 - 1) * self.max_offset_m
            ypr     = (np.random.rand(3) * 2 - 1) * self.max_rot_deg
            R_off = R_gt @ Rotation.from_euler('zyx', ypr, degrees=True).as_matrix()
            cp_off = cp + t_delta
            R_inv = R_off.T.astype(np.float32)
            t_inv = (-(R_off.T @ cp_off)).astype(np.float32)
            pts_cam_off = pts_c @ R_inv.T + t_inv       # (M, 3)
            z_off = pts_cam_off[:, 2]
            uv_off_c = (pts_cam_off[:, :2] * (np.array([K[0,0], K[1,1]], dtype=np.float32))) / \
                       np.maximum(z_off[:, None], 1e-6) + np.array([K[0,2], K[1,2]], dtype=np.float32)

            in_crop_off = ((uv_off_c[:, 0] >= u0) & (uv_off_c[:, 0] < u0 + cs) &
                           (uv_off_c[:, 1] >= v0) & (uv_off_c[:, 1] < v0 + cs) &
                           (z_off > 0.5))
            if in_crop_off.sum() < self.min_pts:
                continue

            # 16x16 sub-grid representative selection — fully vectorized
            scale = S / cs
            uv_local = np.stack([(uv_off_c[in_crop_off, 0] - u0) * scale,
                                 (uv_off_c[in_crop_off, 1] - v0) * scale], axis=1)
            grid_n, cell_S = 16, float(S) / 16
            ci_u = np.clip((uv_local[:, 0] / cell_S).astype(int), 0, grid_n - 1)
            ci_v = np.clip((uv_local[:, 1] / cell_S).astype(int), 0, grid_n - 1)
            cell_id = ci_v * grid_n + ci_u
            cu_c = (ci_u + 0.5) * cell_S
            cv_c = (ci_v + 0.5) * cell_S
            d2 = (uv_local[:, 0] - cu_c) ** 2 + (uv_local[:, 1] - cv_c) ** 2
            order = np.lexsort((d2, cell_id))           # primary cell_id, secondary d2
            _, first_pos = np.unique(cell_id[order], return_index=True)
            sel = order[first_pos]                        # one rep per occupied cell
            sub_idx = np.where(in_crop_off)[0][sel]      # idx into cand_idx
            pts_sel = pts_c[sub_idx]                     # (Nrep, 3)
            uv_gt_sel  = uv_gt_c[sub_idx]
            uv_off_sel = uv_off_c[sub_idx]

            uv_gt_loc  = ((uv_gt_sel  - np.array([u0, v0], dtype=np.float32)) * scale).astype(np.float32)
            uv_off_loc = ((uv_off_sel - np.array([u0, v0], dtype=np.float32)) * scale).astype(np.float32)
            dist_m = (np.linalg.norm(pts_sel - cp, axis=1) / 100.0).astype(np.float32)
            is_obj = is_obj_full[cand_idx[sub_idx]].astype(np.float32)

            true_uvd = np.concatenate([uv_gt_loc,  dist_m[:, None], is_obj[:, None]], axis=1)
            dist_uvd = np.concatenate([uv_off_loc, dist_m[:, None], is_obj[:, None]], axis=1)

            img_crop = img_full[:, v0:v0+cs, u0:u0+cs].float().unsqueeze(0)
            img_crop = F.interpolate(img_crop, size=(S, S), mode='bilinear',
                                      align_corners=False).squeeze(0) / 255.0

            vfp = float(K[0, 0]) * S / cs
            return (img_crop, torch.from_numpy(true_uvd), torch.from_numpy(dist_uvd),
                    torch.tensor(vfp, dtype=torch.float32))

        return self[random.randint(0, len(self) - 1)]


def collate_full(batch):
    """Pad ragged uvd tensors and stack img/vfp."""
    imgs, trues, dists, vfps = zip(*batch)
    imgs = torch.stack(imgs)            # (B, 3, S, S)
    vfps = torch.stack(vfps)            # (B,)
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
    return imgs, true_p, dist_p, pad, vfps
