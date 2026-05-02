"""WovenSequence cross-frame pair dataset — zero-shot eval target.

Source data (one sequence):
    /home/hfunaya/git/loom/backend/assets/woven_sequence/.../sequence=.../
        tss4_fcm_rect/NNNN_TS.jpg         rectified pinhole images, 3840x2160
        vls128_rear_axle/NNNN_TS.npz      VLS128 points in rear_axle (vehicle) frame
        K_rect.json                        rectified pinhole intrinsic
        setting-ipXXX.json                 recalib (fcm + poslv -> cam<-rear_axle)
        metadata.json                      contains baed_poses_v4_absolute
                                           (world<-rear_axle per frame), etc.

Differences vs PandaSetCrossFrameDataset:
- Native resolution is ~2x training resolution (3840x2160 vs ~1920x1080) so crop
  range defaults to 256..512 to preserve angular FOV.
- Per-frame (uv, z, in_view) computed via pinhole K_rect on points that have
  been transformed rear_axle -> cam via the precomputed R_rear2cam / t_rear2cam
  extrinsic. Distortion is already removed in tss4_fcm_rect.
- World-frame points are synthesized for cross-frame lookup via
  pts_world = poses[fi] @ pts_rear_axle (exact same contract as pandaset_pair).

The loader emits the same sample dict as PandaSetCrossFrameDataset so the
existing CalibNetCrossFrame model and vis scripts can consume it unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation


# The loom tree has the projection helpers we want to reuse; import lazily so
# this module can still be import-inspected without the loom venv.
def _import_projection_utils():
    loom_backend = Path('/home/hfunaya/git/loom/backend')
    if str(loom_backend) not in sys.path:
        sys.path.insert(0, str(loom_backend))
    # stub logger_utils if absent (projection_utils imports it)
    try:
        import logger_utils  # noqa: F401
    except ImportError:
        import types, logging
        m = types.ModuleType('logger_utils')

        def get_logger(name):
            lg = logging.getLogger(name)
            lg.setLevel(logging.WARNING)
            return lg
        m.get_logger = get_logger
        sys.modules['logger_utils'] = m
    from projection_utils import get_poslv_adjusted_calibration
    return get_poslv_adjusted_calibration


# ─── geometry helpers ────────────────────────────────────────────────────────

def _invert_mat(M):
    R = M[:3, :3]; t = M[:3, 3]
    Minv = np.eye(4, dtype=M.dtype)
    Minv[:3, :3] = R.T
    Minv[:3, 3] = -R.T @ t
    return Minv


def _ypr_t_to_mat(ypr_deg, t):
    R = Rotation.from_euler('zyx', ypr_deg, degrees=True).as_matrix()
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _mat_to_ypr_t(M):
    ypr = Rotation.from_matrix(M[:3, :3]).as_euler('zyx', degrees=True).astype(np.float32)
    t = M[:3, 3].astype(np.float32)
    return ypr, t


def _project_pinhole(pts_cam, K):
    """pts_cam: (N, 3) in camera frame (Z forward). Returns (uv (N,2), z (N,))."""
    z = pts_cam[:, 2]
    uv = (K @ pts_cam.T)[:2] / np.clip(z, 1e-6, None)
    return uv.T.astype(np.float32), z.astype(np.float32)


# ─── scene container ─────────────────────────────────────────────────────────

class _WovenSequence:
    """One WovenSequence directory, loaded into memory (poses + per-frame
    LiDAR/image indices)."""

    def __init__(self, seq_root: Path, pose_key: str = 'baed_poses_v4_absolute'):
        self.root = Path(seq_root)
        self.pose_key = pose_key

        # intrinsics (rectified pinhole)
        k_rect = json.loads((self.root / 'K_rect.json').read_text())
        self.K = np.array(k_rect['K_rect'], dtype=np.float32)
        self.IW = int(k_rect['width'])
        self.IH = int(k_rect['height'])

        # extrinsic: camera <- rear_axle
        setting_files = list(self.root.glob('setting-*.json'))
        if not setting_files:
            raise FileNotFoundError(f'no setting-*.json under {self.root}')
        setting_json = json.loads(setting_files[0].read_text())
        get_poslv = _import_projection_utils()
        R_rear2cam, t_rear2cam, _, _ = get_poslv(setting_json)
        self.R_rear2cam = np.asarray(R_rear2cam, dtype=np.float32)
        self.t_rear2cam = np.asarray(t_rear2cam, dtype=np.float32).reshape(3)

        # build T_cam_from_rear (4x4)
        T_cr = np.eye(4, dtype=np.float32)
        T_cr[:3, :3] = self.R_rear2cam
        T_cr[:3, 3] = self.t_rear2cam
        self.T_cam_from_rear = T_cr
        self.T_rear_from_cam = _invert_mat(T_cr)

        # per-frame poses (world <- rear_axle)
        meta = json.loads((self.root / 'metadata.json').read_text())
        if pose_key not in meta:
            raise KeyError(f'{pose_key} not in metadata.json; '
                           f'available keys include: '
                           f'{[k for k in meta if "pose" in k.lower()][:10]}')
        pose_dict = meta[pose_key]

        # index frames by the directory listings, then keep only ones that have
        # a pose AND matching image + lidar files.
        img_dir = self.root / 'tss4_fcm_rect'
        lidar_dir = self.root / 'vls128_rear_axle'
        img_files = {p.stem: p for p in sorted(img_dir.glob('*.jpg'))}
        lidar_files = {p.stem: p for p in sorted(lidar_dir.glob('*.npz'))}

        self.frames: List[dict] = []
        # stems look like "NNNN_TS"; use stem as pose key too
        for stem in sorted(img_files):
            if stem not in lidar_files or stem not in pose_dict:
                continue
            T_wr = np.asarray(pose_dict[stem], dtype=np.float32)
            if T_wr.shape != (4, 4):
                continue
            self.frames.append({
                'stem': stem,
                'img_path': img_files[stem],
                'lidar_path': lidar_files[stem],
                'T_world_from_rear': T_wr,
            })
        self.n_frames = len(self.frames)

        # derive T_world_from_cam and T_cam_from_world for each frame
        self.T_w2c = np.stack([
            _invert_mat(f['T_world_from_rear'] @ self.T_rear_from_cam)
            for f in self.frames
        ])

        # caches
        self._img_cache = {}     # fi -> np.uint8 HxWx3
        self._frame_cache = {}   # fi -> (pts_world, uv, z, in_view)

    # ---------------------------------------------------------------- images

    img_scale_div: int = 1

    def load_image(self, fi):
        if fi not in self._img_cache:
            with Image.open(self.frames[fi]['img_path']) as im:
                arr = np.array(im.convert('RGB'))
            if self.img_scale_div > 1:
                H, W = arr.shape[:2]
                arr = np.array(Image.fromarray(arr).resize(
                    (W // self.img_scale_div, H // self.img_scale_div), Image.BILINEAR))
            self._img_cache[fi] = arr
        return self._img_cache[fi]

    # ---------------------------------------------------------------- lidar

    def _load_pts_world(self, fi):
        """Load VLS128 points, return (pts_world (N,3),) in world frame."""
        d = np.load(self.frames[fi]['lidar_path'])
        # xs, ys, zs are in rear_axle (vehicle) frame.
        pts_rear = np.stack([d['xs'], d['ys'], d['zs']], axis=1).astype(np.float32)
        # simple range cull to keep memory bounded (within 80 m of ego).
        r = np.linalg.norm(pts_rear, axis=1)
        pts_rear = pts_rear[(r > 1.0) & (r < 80.0)]
        # world <- rear_axle
        T_wr = self.frames[fi]['T_world_from_rear']
        homo = np.concatenate([pts_rear, np.ones((len(pts_rear), 1), dtype=np.float32)], axis=1)
        pts_world = (T_wr @ homo.T)[:3].T.astype(np.float32)
        return pts_world

    def frame_data(self, fi):
        if fi not in self._frame_cache:
            pts_world = self._load_pts_world(fi)
            T_w2c = self.T_w2c[fi]
            homo = np.concatenate([pts_world, np.ones((len(pts_world), 1), dtype=np.float32)], axis=1)
            pts_cam = (T_w2c @ homo.T)[:3].T
            uv, z = _project_pinhole(pts_cam, self.K)
            in_view = ((z > 1.0) &
                       (uv[:, 0] > 0) & (uv[:, 0] < self.IW) &
                       (uv[:, 1] > 0) & (uv[:, 1] < self.IH))
            self._frame_cache[fi] = (pts_world, uv, z, in_view)
        return self._frame_cache[fi]

    def precompute_all(self, preload_images: bool = False, n_workers: int = 4):
        from concurrent.futures import ThreadPoolExecutor
        fis = list(range(self.n_frames))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(self.frame_data, fis))
            if preload_images:
                list(ex.map(self.load_image, fis))


# ─── main dataset ────────────────────────────────────────────────────────────

class WovenSequenceCrossFrameDataset(Dataset):
    """Zero-shot cross-frame eval on a WovenSequence sequence.

    Perturbation semantics mirror training: the hypothesis pose is the BA pose
    multiplied by a random 6-DoF perturbation sampled with (sigma_ypr, sigma_t).
    """

    def __init__(self,
                 seq_root: str,
                 img_size: int = 64,
                 baseline_range = (1, 20),
                 sigma_ypr: float = 1.0,
                 sigma_t: float = 0.2,
                 max_points: int = 256,
                 crop_range = (256, 512),   # 2x training (128..256) for 2x resolution
                 virtual_epoch_len: int = 200,
                 pose_key: str = 'baed_poses_v4_absolute',
                 img_scale_div: int = 2,    # cache images at 1/2 res -> 1920x1080
                 seed: int = 42,
                 preload_images: bool = False):
        super().__init__()
        self.img_size = img_size
        self.baseline_range = baseline_range
        self.sigma_ypr = sigma_ypr
        self.sigma_t = sigma_t
        self.max_points = max_points
        self.crop_range = crop_range
        self.virtual_epoch_len = virtual_epoch_len
        self.rng = np.random.default_rng(seed)

        scn = _WovenSequence(Path(seq_root), pose_key=pose_key)
        scn.img_scale_div = img_scale_div
        print(f'[WovenSequence] {scn.root.name}: {scn.n_frames} frames, '
              f'image_cache={scn.IW//img_scale_div}x{scn.IH//img_scale_div} '
              f'(full-res {scn.IW}x{scn.IH})', flush=True)
        scn.precompute_all(preload_images=preload_images, n_workers=4)
        self.scene = scn

    # ------------------------------------------------------------------ helpers

    def _crop_patch(self, img_cached, u0_full, v0_full, s_full, scale_div):
        """Same semantics as pandaset_pair._crop_patch but simpler — the
        cached image is at (full / scale_div), u0/v0/s are FULL-res pixels."""
        s_full_i = int(s_full)
        u0i = int(u0_full); v0i = int(v0_full)
        u1i = u0i + s_full_i; v1i = v0i + s_full_i
        u0c = u0i // scale_div; v0c = v0i // scale_div
        u1c = u1i // scale_div; v1c = v1i // scale_div
        cw = u1c - u0c; ch = v1c - v0c
        if cw < 2 or ch < 2:
            return None
        src_u0 = max(0, u0c); src_v0 = max(0, v0c)
        src_u1 = min(img_cached.shape[1], u1c); src_v1 = min(img_cached.shape[0], v1c)
        out = np.zeros((ch, cw, img_cached.shape[2]), dtype=img_cached.dtype)
        inner_w = src_u1 - src_u0; inner_h = src_v1 - src_v0
        if inner_w > 0 and inner_h > 0:
            out_pad_left = src_u0 - u0c
            out_pad_top = src_v0 - v0c
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
        return np.stack([(uv_full[:, 0] - u0) * scale_u,
                         (uv_full[:, 1] - v0) * scale_v], axis=1).astype(np.float32)

    # ------------------------------------------------------------ main getter

    def __len__(self):
        return self.virtual_epoch_len

    def __getitem__(self, idx):
        for retry in range(30):
            s = self._try_one(idx + retry * 9973)
            if s is not None:
                return s
        raise RuntimeError(f'could not form a valid pair after 30 tries (idx={idx})')

    def _try_one(self, idx):
        rng = np.random.default_rng((idx + 1) * 2654435761 & 0xFFFFFFFF)
        scn = self.scene
        IW, IH, K = scn.IW, scn.IH, scn.K

        # 1. pick fi_A
        fi_A = int(rng.integers(scn.n_frames))
        # 2. fi_B: nearby frame
        bmin, bmax = self.baseline_range
        delta = int(rng.integers(bmin, bmax + 1)) * int(rng.choice([-1, 1]))
        fi_B = fi_A + delta
        if fi_B < 0 or fi_B >= scn.n_frames:
            return None

        T_w2A = scn.T_w2c[fi_A]
        T_w2B = scn.T_w2c[fi_B]
        T_A2w = _invert_mat(T_w2A)
        T_A_to_B_gt = T_w2B @ T_A2w

        # 3. LiDAR, projections
        pts_w_A, uv_Af, z_Af, in_A = scn.frame_data(fi_A)
        pts_w_B, uv_Bf, z_Bf, in_B = scn.frame_data(fi_B)
        if in_A.sum() < 50 or in_B.sum() < 50:
            return None

        pts_w_A_in = pts_w_A[in_A]
        uv_A_all = uv_Af[in_A]
        z_A_all = z_Af[in_A]

        # 4. center pivot
        ci = int(rng.integers(len(pts_w_A_in)))
        P_center_w = pts_w_A_in[ci]
        uc_A, vc_A = uv_A_all[ci]

        # 5. perturbation -> T_AB_hat
        ypr_pert = rng.standard_normal(3).astype(np.float32) * self.sigma_ypr
        t_pert = rng.standard_normal(3).astype(np.float32) * self.sigma_t
        delta_T = _ypr_t_to_mat(ypr_pert, t_pert)
        T_A_to_B_hat = T_A_to_B_gt @ delta_T

        # 6. project pivot under hat/gt
        P_center_A = (T_w2A @ np.append(P_center_w, 1.0))[:3]
        P_center_Bh = (T_A_to_B_hat @ np.append(P_center_A, 1.0))[:3]
        P_center_Bg = (T_A_to_B_gt @ np.append(P_center_A, 1.0))[:3]
        if P_center_Bh[2] < 1.0 or P_center_Bg[2] < 1.0:
            return None
        uc_B_hat = (K @ P_center_Bh)[:2] / P_center_Bh[2]
        uc_B_gt = (K @ P_center_Bg)[:2] / P_center_Bg[2]
        uc_B_hat = uc_B_hat.astype(np.float32)
        uc_B_gt = uc_B_gt.astype(np.float32)
        if not (0 <= uc_B_hat[0] < IW and 0 <= uc_B_hat[1] < IH):
            return None
        if not (0 <= uc_B_gt[0] < IW and 0 <= uc_B_gt[1] < IH):
            return None

        # 7. shared crop size (in FULL-res pixels)
        CROP = int(rng.integers(self.crop_range[0], self.crop_range[1] + 1))
        half = CROP / 2
        u0_A = uc_A - half; v0_A = vc_A - half
        u0_B = uc_B_hat[0] - half; v0_B = uc_B_hat[1] - half
        img_A_full = scn.load_image(fi_A)
        img_B_full = scn.load_image(fi_B)
        pa = self._crop_patch(img_A_full, u0_A, v0_A, CROP, scn.img_scale_div)
        pb = self._crop_patch(img_B_full, u0_B, v0_B, CROP, scn.img_scale_div)
        if pa is None or pb is None:
            return None
        patch_A, box_A = pa
        patch_B, box_B = pb

        # 8. A-query points
        u0, v0, cw, ch = box_A
        in_box_A = ((uv_A_all[:, 0] >= u0) & (uv_A_all[:, 0] < u0 + cw) &
                    (uv_A_all[:, 1] >= v0) & (uv_A_all[:, 1] < v0 + ch))
        if in_box_A.sum() < 4:
            return None
        pts_w_QA = pts_w_A_in[in_box_A]
        uv_A_patch = self._uv_to_patch_local(uv_A_all[in_box_A], box_A)
        z_A_patch = z_A_all[in_box_A]

        homo = np.concatenate([pts_w_QA, np.ones((len(pts_w_QA), 1), dtype=np.float32)], axis=1)
        P_QA_in_A = (T_w2A @ homo.T)[:3].T
        P_QA_in_B_gt = (np.column_stack([P_QA_in_A, np.ones(len(P_QA_in_A))]) @ T_A_to_B_gt.T)[:, :3]
        P_QA_in_B_hat = (np.column_stack([P_QA_in_A, np.ones(len(P_QA_in_A))]) @ T_A_to_B_hat.T)[:, :3]
        good_qa = (P_QA_in_B_gt[:, 2] > 0.5) & (P_QA_in_B_hat[:, 2] > 0.5)
        if good_qa.sum() < 4:
            return None
        uv_B_gt_full_A = (K @ P_QA_in_B_gt[good_qa].T)[:2] / P_QA_in_B_gt[good_qa, 2]
        uv_B_hat_full_A = (K @ P_QA_in_B_hat[good_qa].T)[:2] / P_QA_in_B_hat[good_qa, 2]
        uv_B_gt_local_A = self._uv_to_patch_local(uv_B_gt_full_A.T.astype(np.float32), box_B)
        uv_B_hat_local_A = self._uv_to_patch_local(uv_B_hat_full_A.T.astype(np.float32), box_B)
        uv_A_patch = uv_A_patch[good_qa]
        z_A_patch = z_A_patch[good_qa]
        inb = ((uv_B_hat_local_A[:, 0] >= 0) & (uv_B_hat_local_A[:, 0] < self.img_size) &
               (uv_B_hat_local_A[:, 1] >= 0) & (uv_B_hat_local_A[:, 1] < self.img_size) &
               (uv_B_gt_local_A[:, 0] >= 0) & (uv_B_gt_local_A[:, 0] < self.img_size) &
               (uv_B_gt_local_A[:, 1] >= 0) & (uv_B_gt_local_A[:, 1] < self.img_size))
        if inb.sum() < 4:
            return None
        uv_A_patch = uv_A_patch[inb]
        z_A_patch = z_A_patch[inb]
        uv_B_gt_local_A = uv_B_gt_local_A[inb]
        uv_B_hat_local_A = uv_B_hat_local_A[inb]

        # 9. B-query points (symmetric)
        in_box_B = ((uv_Bf[:, 0] >= box_B[0]) & (uv_Bf[:, 0] < box_B[0] + box_B[2]) &
                    (uv_Bf[:, 1] >= box_B[1]) & (uv_Bf[:, 1] < box_B[1] + box_B[3]) &
                    (z_Bf > 1.0))
        uv_B_patch = self._uv_to_patch_local(uv_Bf[in_box_B], box_B)
        z_B_patch = z_Bf[in_box_B]
        pts_w_QB = pts_w_B[in_box_B]
        if len(pts_w_QB) < 4:
            return None
        homoB = np.concatenate([pts_w_QB, np.ones((len(pts_w_QB), 1), dtype=np.float32)], axis=1)
        P_QB_in_B = (T_w2B @ homoB.T)[:3].T
        T_B_to_A_gt = _invert_mat(T_A_to_B_gt)
        T_B_to_A_hat = _invert_mat(T_A_to_B_hat)
        P_QB_in_A_gt = (np.column_stack([P_QB_in_B, np.ones(len(P_QB_in_B))]) @ T_B_to_A_gt.T)[:, :3]
        P_QB_in_A_hat = (np.column_stack([P_QB_in_B, np.ones(len(P_QB_in_B))]) @ T_B_to_A_hat.T)[:, :3]
        good_qb = (P_QB_in_A_gt[:, 2] > 0.5) & (P_QB_in_A_hat[:, 2] > 0.5)
        if good_qb.sum() < 4:
            return None
        uv_A_gt_full_B = (K @ P_QB_in_A_gt[good_qb].T)[:2] / P_QB_in_A_gt[good_qb, 2]
        uv_A_hat_full_B = (K @ P_QB_in_A_hat[good_qb].T)[:2] / P_QB_in_A_hat[good_qb, 2]
        uv_A_gt_local_B = self._uv_to_patch_local(uv_A_gt_full_B.T.astype(np.float32), box_A)
        uv_A_hat_local_B = self._uv_to_patch_local(uv_A_hat_full_B.T.astype(np.float32), box_A)
        uv_B_patch = uv_B_patch[good_qb]
        z_B_patch = z_B_patch[good_qb]
        inb2 = ((uv_A_hat_local_B[:, 0] >= 0) & (uv_A_hat_local_B[:, 0] < self.img_size) &
                (uv_A_hat_local_B[:, 1] >= 0) & (uv_A_hat_local_B[:, 1] < self.img_size) &
                (uv_A_gt_local_B[:, 0] >= 0) & (uv_A_gt_local_B[:, 0] < self.img_size) &
                (uv_A_gt_local_B[:, 1] >= 0) & (uv_A_gt_local_B[:, 1] < self.img_size))
        if inb2.sum() < 4:
            return None
        uv_B_patch = uv_B_patch[inb2]
        z_B_patch = z_B_patch[inb2]
        uv_A_gt_local_B = uv_A_gt_local_B[inb2]
        uv_A_hat_local_B = uv_A_hat_local_B[inb2]

        d_B_hat_of_A = P_QA_in_B_hat[good_qa][inb, 2].astype(np.float32)
        d_B_gt_of_A = P_QA_in_B_gt[good_qa][inb, 2].astype(np.float32)
        d_A_hat_of_B = P_QB_in_A_hat[good_qb][inb2, 2].astype(np.float32)
        d_A_gt_of_B = P_QB_in_A_gt[good_qb][inb2, 2].astype(np.float32)

        # 10. pad/truncate to max_points
        def _pad(uv_query, z_query, uv_hat, uv_gt, d_hat, d_gt):
            N = len(uv_query)
            N_use = min(N, self.max_points)
            pick = self.rng.choice(N, size=N_use, replace=False) if N > N_use else np.arange(N)
            uv_query = uv_query[pick]; z_query = z_query[pick]
            uv_hat = uv_hat[pick]; uv_gt = uv_gt[pick]
            d_hat = d_hat[pick]; d_gt = d_gt[pick]
            pad = np.zeros((self.max_points,), dtype=bool)
            if N_use < self.max_points:
                pad[N_use:] = True
                pad_n = self.max_points - N_use
                uv_query = np.concatenate([uv_query, np.zeros((pad_n, 2), np.float32)])
                z_query = np.concatenate([z_query, np.zeros(pad_n, np.float32)])
                uv_hat = np.concatenate([uv_hat, np.zeros((pad_n, 2), np.float32)])
                uv_gt = np.concatenate([uv_gt, np.zeros((pad_n, 2), np.float32)])
                d_hat = np.concatenate([d_hat, np.zeros(pad_n, np.float32)])
                d_gt = np.concatenate([d_gt, np.zeros(pad_n, np.float32)])
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
            patch_A=patch_A.float(),
            patch_B=patch_B.float(),
            uvd_A=torch.from_numpy(np.concatenate([uv_A_patch, z_A_norm[:, None]], axis=1)).float(),
            uvd_B=torch.from_numpy(np.concatenate([uv_B_patch, z_B_norm[:, None]], axis=1)).float(),
            uv_B_hat_of_A=torch.from_numpy(uv_B_hat_local_A).float(),
            uv_B_gt_of_A=torch.from_numpy(uv_B_gt_local_A).float(),
            uv_A_hat_of_B=torch.from_numpy(uv_A_hat_local_B).float(),
            uv_A_gt_of_B=torch.from_numpy(uv_A_gt_local_B).float(),
            d_B_hat_of_A=torch.from_numpy(d_B_hat_of_A).float(),
            d_B_gt_of_A=torch.from_numpy(d_B_gt_of_A).float(),
            d_A_hat_of_B=torch.from_numpy(d_A_hat_of_B).float(),
            d_A_gt_of_B=torch.from_numpy(d_A_gt_of_B).float(),
            pose_AB_6dof=torch.from_numpy(np.concatenate([ypr_AB_hat, t_AB_hat]).astype(np.float32)),
            pose_BA_6dof=torch.from_numpy(np.concatenate([ypr_BA_hat, t_BA_hat]).astype(np.float32)),
            pad_A=torch.from_numpy(pad_A),
            pad_B=torch.from_numpy(pad_B),
            fi_A=fi_A, fi_B=fi_B,
            scene=scn.root.name,
        )


# ─── smoke test ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    SEQ = ('/home/hfunaya/git/loom/backend/assets/woven_sequence/llinking_27/tf_long2/'
           'sequence=ip654_1337941440921107425_16943630305775105398_1749030654176-1749030664176')
    ds = WovenSequenceCrossFrameDataset(
        seq_root=SEQ,
        img_size=64, max_points=256,
        baseline_range=(1, 20), sigma_ypr=1.0, sigma_t=0.2,
        crop_range=(256, 512), img_scale_div=2,
        virtual_epoch_len=10, seed=42,
    )
    s = ds[0]
    print(f'sample: fi_A={s["fi_A"]} fi_B={s["fi_B"]} '
          f'scene={s["scene"]} '
          f'valid_A={(~s["pad_A"]).sum().item()} valid_B={(~s["pad_B"]).sum().item()}')
    print(f'uv_B_hat_of_A[0]: {s["uv_B_hat_of_A"][0].tolist()}')
    print(f'uv_B_gt_of_A [0]: {s["uv_B_gt_of_A" ][0].tolist()}')
    print(f'pose_AB_6dof   : {s["pose_AB_6dof"].tolist()}')
