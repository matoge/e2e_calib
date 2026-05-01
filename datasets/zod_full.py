"""ZOD direct-read dataset for calib training (no cache build).

Reads raw ZOD Frames structure on the fly:
  <src>/single_frames/<fid>/{camera_front_dnat/*.jpg, lidar_velodyne/*.npy,
                              calibration.json, ...}

Returns the same (img, true_uvd, dist_uvd, vfp) tuple as PandaSetCalibDatasetFull
so the existing train_ps_v3.py loop works unchanged.

Differences from PandaSetCalibDatasetFull:
  - 8.3 MP single front cam, Kannala-Brandt fisheye distortion
    (perturbation re-projection uses zod.utils.geometry.project_3d_to_2d_kannala
    so the lens curve at the edges is consistent between GT and perturbed UVs)
  - Lidar pts come in cam frame after one-shot transform via official toolkit
  - Image opened via PIL on each __getitem__ — Xeon has the cores, IPC dominates anyway

Args mirror PandaSetCalibDatasetFull where possible.
"""
from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


def _project_kannala(pts_cam: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Inline Kannala-Brandt forward projection (matches zod toolkit).

    pts_cam: (N, 3) in camera frame, z forward.
    K: (3, 4) or (3, 3) intrinsic with fx, fy, cx, cy.
    dist: (4,) k1..k4 — Kannala fisheye coefficients.
    Returns (N, 2) UV in pixels.
    """
    z = np.maximum(pts_cam[:, 2], 1e-6)
    x = pts_cam[:, 0] / z
    y = pts_cam[:, 1] / z
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)
    k1, k2, k3, k4 = dist
    theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4
                          + k3 * theta**6 + k4 * theta**8)
    factor = np.where(r > 1e-9, theta_d / r, 1.0)
    x_d = factor * x
    y_d = factor * y
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return np.stack([fx * x_d + cx, fy * y_d + cy], axis=-1).astype(np.float32)


def _is_obj_per_point(pts_cam: np.ndarray, cuboids: list) -> np.ndarray:
    """Yaw-rotated AABB membership test. Mirrors datasets/pandaset_full.py."""
    N = len(pts_cam)
    if not cuboids or N == 0:
        return np.zeros(N, dtype=np.float32)
    M = len(cuboids)
    poss = np.stack([np.asarray(c['pos'],  dtype=np.float32) for c in cuboids])
    dims = np.stack([np.asarray(c['dims'], dtype=np.float32) for c in cuboids])
    yaws = np.fromiter((float(c['yaw']) for c in cuboids), dtype=np.float32, count=M)
    cy, sy = np.cos(yaws), np.sin(yaws)
    R = np.zeros((M, 3, 3), dtype=np.float32)
    R[:, 0, 0] = cy;  R[:, 0, 1] = sy
    R[:, 1, 0] = -sy; R[:, 1, 1] = cy
    R[:, 2, 2] = 1.0
    delta = pts_cam.astype(np.float32, copy=False)[None, :, :] - poss[:, None, :]
    local = np.einsum('mij,mnj->mni', R, delta)
    half = (dims * 0.5)[:, None, :]
    inside = np.all(np.abs(local) <= half, axis=-1)
    return inside.any(axis=0).astype(np.float32)


class ZODCalibDataset(Dataset):
    """Direct-read ZOD Frames dataset (no cache build).

    Args:
        src: ZOD Frames root (must contain single_frames/<fid>/ + trainval-frames-*.json).
        split: 'train' | 'val' | 'all' (val = first 15% by sorted fid).
        img_size: model input side (px).
        min_crop_px / max_crop_px: random crop side in full-image px.
        max_offset_m / max_rot_deg: extrinsic perturbation half-range.
        oversample: replay each fid this many times per epoch (random crops).
        anonymization: 'dnat' (default — neural face/plate replacement) or 'blur'.
    """
    def __init__(self,
                 src: str | Path = '/mnt/nvme6t/zod/frames',
                 split: str = 'train',
                 img_size: int = 128,
                 min_crop_px: int = 256,
                 max_crop_px: int = 768,
                 max_offset_m: float = 0.20,
                 max_rot_deg: float = 1.0,
                 min_pts: int = 8,
                 max_tries: int = 8,
                 oversample: int = 4,
                 grid_n: int = 16,
                 anonymization: str = 'dnat',
                 edge_margin_frac: float = 0.08):
        try:
            from zod import ZodFrames
            from zod.constants import Anonymization
        except ImportError as e:
            raise ImportError("pip install zod") from e

        self.src = Path(src)
        self._zf_kw = dict(dataset_root=str(src), version='full')
        try:
            zf = ZodFrames(**self._zf_kw)
        except Exception:
            self._zf_kw['version'] = 'mini'
            zf = ZodFrames(**self._zf_kw)
        all_ids = sorted(zf.get_all_ids())

        if split == 'all':
            self.fnames = all_ids
        else:
            n_val = max(1, int(len(all_ids) * 0.15))
            val_set = set(all_ids[:n_val])  # deterministic first-N as val
            if split == 'train':
                self.fnames = [f for f in all_ids if f not in val_set]
            elif split == 'val':
                self.fnames = [f for f in all_ids if f in val_set]
            else:
                raise ValueError(f"split={split}")

        self.img_size  = int(img_size)
        self.min_crop_px = int(min_crop_px)
        self.max_crop_px = int(max_crop_px)
        self.max_offset_m = float(max_offset_m)
        self.max_rot_deg  = float(max_rot_deg)
        self.min_pts   = int(min_pts)
        self.max_tries = int(max_tries)
        self.oversample = int(oversample)
        self.grid_n    = int(grid_n)
        self._anon = Anonymization.DNAT if anonymization == 'dnat' else Anonymization.BLUR
        # ZOD edge calib residual is ~few px at the outermost ring (KB distortion
        # estimation error + lidar mount offset). Excluding 8% margin at each
        # edge keeps training pivots in the area where projection is reliable.
        self.edge_margin_frac = float(edge_margin_frac)

        # Worker-local zod handle (forked workers don't share cleanly)
        self._zf = None

    def __len__(self):
        return len(self.fnames) * self.oversample

    def _ensure_zf(self):
        if self._zf is None:
            from zod import ZodFrames
            self._zf = ZodFrames(**self._zf_kw)
        return self._zf

    def __getitem__(self, idx: int):
        from zod.constants import Camera, Lidar
        from zod.utils.geometry import transform_points, get_points_in_camera_fov
        from zod.utils.compensation import motion_compensate_pointwise

        zf = self._ensure_zf()
        fid = self.fnames[idx % len(self.fnames)]
        frame = zf[fid]

        cam = frame.calibration.cameras[Camera.FRONT]
        lidar_calib = frame.calibration.lidars[Lidar.VELODYNE]

        # Per-shot motion compensation to camera shutter time.
        # The lidar takes ~115 ms to scan 360°. Without per-shot ego-pose
        # interpolation, points captured early in the scan project a few px
        # off (50 km/h × 100 ms = 1.4 m → ~10 px at 30 m for fx=1857).
        # zod.utils.compensation.motion_compensate_scanwise uses ONE
        # `core_timestamp` for the whole scan — that's block compensation,
        # not enough at the edges. `motion_compensate_pointwise` interpolates
        # ego pose at every per-shot timestamp (lidar_data.timestamps), giving
        # alignment quality comparable to Waymo's precomputed LCP.
        cam_ts = frame.info.camera_frames['front_dnat'][0].time.timestamp()
        pc = motion_compensate_pointwise(
            frame.get_lidar()[0],
            frame.ego_motion,
            lidar_calib,
            target_timestamp=cam_ts,
        )
        T_c_l = np.linalg.inv(cam.extrinsics.transform) @ lidar_calib.extrinsics.transform
        pts_cam = transform_points(pc.points, T_c_l).astype(np.float32)
        pts_cam_fov, fov_mask = get_points_in_camera_fov(cam.field_of_view, pts_cam)
        K = np.asarray(cam.intrinsics, dtype=np.float32)
        dist = np.asarray(cam.distortion, dtype=np.float32)
        IW, IH = int(cam.image_dimensions[0]), int(cam.image_dimensions[1])

        # GT projection (Kannala) for full set of in-FOV pts
        uv_full = _project_kannala(pts_cam_fov, K, dist)
        z = pts_cam_fov[:, 2]
        # Drop edge-ring pivots where KB distortion residual is largest
        em = self.edge_margin_frac
        u_lo, u_hi = int(em * IW), int((1 - em) * IW)
        v_lo, v_hi = int(em * IH), int((1 - em) * IH)
        valid_in_image = ((z > 0.5) &
                          (uv_full[:, 0] >= u_lo) & (uv_full[:, 0] < u_hi) &
                          (uv_full[:, 1] >= v_lo) & (uv_full[:, 1] < v_hi))

        # Cuboid is_obj (only on visible pts to save compute)
        cubs = []
        if frame.is_annotated:
            try:
                ann = frame.get_annotation('object_detection')
                # ZOD 3D box format: cuboid has center (3,), dimensions (3,), yaw rad
                for o in getattr(ann, 'objects', []):
                    if not getattr(o, 'box3d', None):
                        continue
                    box = o.box3d
                    cubs.append({
                        'pos':  np.asarray(box.center, dtype=np.float32),
                        'dims': np.asarray(box.size,   dtype=np.float32),
                        'yaw':  float(box.yaw if hasattr(box, 'yaw') else 0.0),
                    })
            except Exception:
                pass
        # cuboid pos is in vehicle frame in ZOD — skip is_obj if convention not yet
        # plumbed (PandaSet trains fine without obj split, just gives cleaner metrics)
        is_obj_full = np.zeros(len(pts_cam_fov), dtype=bool)

        obj_idxs = np.where(is_obj_full & valid_in_image)[0]
        bg_mask  = (~is_obj_full) & valid_in_image
        # 10x5 grid stratification for bg pivot picks
        GU, GV = 10, 5
        cell_w = IW / GU; cell_h = IH / GV
        cell_u = np.clip((uv_full[:, 0] / cell_w).astype(int), 0, GU - 1)
        cell_v = np.clip((uv_full[:, 1] / cell_h).astype(int), 0, GV - 1)
        cell_id_full = cell_v * GU + cell_u
        bg_cells = np.unique(cell_id_full[bg_mask]) if bg_mask.any() else np.array([], dtype=int)

        S = self.img_size
        for _ in range(self.max_tries):
            # Pivot first → depth-aware cs
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
            piv_z = float(z[i])
            if piv_z < 20.0:
                cs_lo, cs_hi = max(self.min_crop_px, 256), min(768, IW, IH)
            else:
                cs_lo, cs_hi = self.min_crop_px, self.max_crop_px
            cs = int(np.random.randint(cs_lo, cs_hi + 1))
            cs = min(cs, IW, IH)
            u0 = int(np.clip(pu - cs / 2, 0, IW - cs))
            v0 = int(np.clip(pv - cs / 2, 0, IH - cs))

            pad_px = int(cs * 0.10)
            in_pad = ((uv_full[:, 0] >= u0 - pad_px) & (uv_full[:, 0] < u0 + cs + pad_px) &
                      (uv_full[:, 1] >= v0 - pad_px) & (uv_full[:, 1] < v0 + cs + pad_px) &
                      (z > 0.5))
            cand_idx = np.where(in_pad)[0]
            if len(cand_idx) < self.min_pts:
                continue
            if len(cand_idx) > 2000:
                cand_idx = np.random.choice(cand_idx, size=2000, replace=False)
            pts_c = pts_cam_fov[cand_idx]
            uv_gt_c = uv_full[cand_idx]

            # Perturbation in cam frame (T_GT for ZOD = identity at this stage)
            t_delta = (np.random.rand(3) * 2 - 1) * self.max_offset_m
            ypr     = (np.random.rand(3) * 2 - 1) * self.max_rot_deg
            R_off   = Rotation.from_euler('zyx', ypr, degrees=True).as_matrix().astype(np.float32)
            pts_cam_off = pts_c @ R_off.T + t_delta.astype(np.float32)
            uv_off_c = _project_kannala(pts_cam_off, K, dist)

            in_crop_off = ((uv_off_c[:, 0] >= u0) & (uv_off_c[:, 0] < u0 + cs) &
                           (uv_off_c[:, 1] >= v0) & (uv_off_c[:, 1] < v0 + cs) &
                           (pts_cam_off[:, 2] > 0.5))
            if in_crop_off.sum() < self.min_pts:
                continue

            # 16x16 sub-grid representative selection
            scale = S / cs
            uv_local = np.stack([(uv_off_c[in_crop_off, 0] - u0) * scale,
                                 (uv_off_c[in_crop_off, 1] - v0) * scale], axis=1)
            grid_n = self.grid_n
            cell_S = float(S) / grid_n
            ci_u = np.clip((uv_local[:, 0] / cell_S).astype(int), 0, grid_n - 1)
            ci_v = np.clip((uv_local[:, 1] / cell_S).astype(int), 0, grid_n - 1)
            cell_id = ci_v * grid_n + ci_u
            cu_c = (ci_u + 0.5) * cell_S
            cv_c = (ci_v + 0.5) * cell_S
            d2 = (uv_local[:, 0] - cu_c) ** 2 + (uv_local[:, 1] - cv_c) ** 2
            order = np.lexsort((d2, cell_id))
            _, first_pos = np.unique(cell_id[order], return_index=True)
            sel = order[first_pos]
            sub_idx = np.where(in_crop_off)[0][sel]
            pts_sel = pts_c[sub_idx]
            uv_gt_sel  = uv_gt_c[sub_idx]
            uv_off_sel = uv_off_c[sub_idx]

            uv_gt_loc  = ((uv_gt_sel  - np.array([u0, v0], dtype=np.float32)) * scale).astype(np.float32)
            uv_off_loc = ((uv_off_sel - np.array([u0, v0], dtype=np.float32)) * scale).astype(np.float32)
            dist_m = (np.linalg.norm(pts_sel, axis=1) / 100.0).astype(np.float32)  # cam-origin dist
            is_obj = is_obj_full[cand_idx[sub_idx]].astype(np.float32)

            true_uvd = np.concatenate([uv_gt_loc,  dist_m[:, None], is_obj[:, None]], axis=1)
            dist_uvd = np.concatenate([uv_off_loc, dist_m[:, None], is_obj[:, None]], axis=1)

            # Decode + crop image (cv2 partial decode would be faster but PIL is fine)
            img_full = frame.get_image(self._anon)  # (H, W, 3) uint8
            arr = img_full[v0:v0 + cs, u0:u0 + cs]
            img_crop = torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous().float().unsqueeze(0)
            img_crop = F.interpolate(img_crop, size=(S, S), mode='bilinear',
                                      align_corners=False).squeeze(0)
            img_crop = img_crop.clamp_(0, 255).to(torch.uint8)

            vfp = float(K[0, 0]) * S / cs
            self._last_crop = dict(u0=int(u0), v0=int(v0), cs=int(cs),
                                    scene=fid, frame=int(fid))
            return (img_crop, torch.from_numpy(true_uvd), torch.from_numpy(dist_uvd),
                    torch.tensor(vfp, dtype=torch.float32))

        return self[random.randint(0, len(self) - 1)]
