"""Adapter: PandaSetCrossFrameDataset(calib_mode=True, scenes_root=...) →
4-tuple (img, true_uvd, dist_uvd, vfp) compatible with PandaSetCalibDataset
LazyVFP / collate_pandaset_vfp / CalibNetDepth's calib forward.

Why this exists: ps_v9..v11 trained on a single-dataset PandaSet lazy cache.
For full calib (panda + waymo + nuscenes, 6 cams), the data path is the
pair dataset's `_try_one_calib` (it already supports `scenes_root` with
comma-separated dirs, multi-camera, vfl_range, etc., and exposes the same
δT-perturbation calib semantics). This wrapper picks the calib-relevant
fields from its dict output and reshapes them to the calib 4-tuple.
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from datasets.pandaset_pair import PandaSetCrossFrameDataset


class MultiDSCalibDataset(Dataset):
    """Multi-dataset, multi-camera calib (single-frame δT-perturbation only).

    Args:
        scenes_root: comma-separated string OR list of dataset roots
            (e.g., "/mnt/nvme6t/pandaset,/mnt/nvme6t/waymo_ps,/mnt/nvme6t/nuscenes_ps")
        cameras: 'all' or specific camera name
        img_size: output crop side
        max_points: per-sample point budget
        baseline_range: kept for parity (in calib_mode fi_B == fi_A so unused)
        sigma_ypr / sigma_t: extrinsic perturbation sigma
        crop_range: full-res crop side range in pixels
        vfl_range: optional (vfl_min, vfl_max) for vfl-invariant cropping
        virtual_epoch_len: samples per epoch
        split: 'train' / 'val'
        train_frac: scene-level split (only when scenes_root is multi-DS)
    """
    def __init__(self,
                 scenes_root,
                 cameras: str = 'all',
                 img_size: int = 64,
                 max_points: int = 256,
                 baseline_range = (1, 5),
                 sigma_ypr: float = 2.0,    # = ±2 deg per axis
                 sigma_t: float = 0.20,     # = ±20 cm per axis
                 crop_range = (128, 256),
                 vfl_range = None,
                 virtual_epoch_len: int = 10000,
                 split: str = 'train',
                 train_frac: float = 0.8):
        if isinstance(scenes_root, (list, tuple)):
            scenes_root = ','.join(scenes_root)
        self.inner = PandaSetCrossFrameDataset(
            scenes_root      = scenes_root,
            train_frac       = train_frac,
            split            = split,
            cameras          = cameras,
            img_size         = img_size,
            max_points       = max_points,
            baseline_range   = baseline_range,
            sigma_ypr        = sigma_ypr,
            sigma_t          = sigma_t,
            crop_range       = crop_range,
            vfl_range        = vfl_range,
            virtual_epoch_len= virtual_epoch_len,
            n_frames         = 2,
            use_stacked      = False,
            calib_mode       = True,    # → _try_one_calib (no leak)
            calib_legacy     = False,
        )
        self.img_size = img_size

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        s = self.inner[idx]
        img        = s['patch_A']                       # (3, H, W)
        uv_dist    = s['uv_B_hat_of_A']                  # (N, 2)  perturbed UV
        uv_true    = s['uv_B_gt_of_A']                   # (N, 2)  GT UV
        depth      = s['uvd_A'][:, 2:3]                  # (N, 1)  z_norm (= z/50)
        # obj label: pair's uvd_A is (N, 3) [u, v, d_norm] — obj lives in
        # is_obj_A as a separate (N,) bool. Carry it through; fall back to 0
        # only if the field truly isn't there (e.g., older cache).
        if 'is_obj_A' in s:
            is_obj = s['is_obj_A'].to(dtype=torch.float32).unsqueeze(-1)
        elif s['uvd_A'].shape[-1] >= 4:
            is_obj = s['uvd_A'][:, 3:4]
        else:
            is_obj = torch.zeros((uv_dist.shape[0], 1), dtype=torch.float32)
        # 4-channel uvd to match collate_pandaset_vfp expectations:
        # [u, v, d, is_obj]
        true_uvd = torch.cat([uv_true,  depth, is_obj], dim=-1)
        dist_uvd = torch.cat([uv_dist,  depth, is_obj], dim=-1)
        # vfp scalar: pair's _try_one_calib stores it under key 'vfl'
        # (= fx_orig * img_size / CROP). Fall back to img_size if absent.
        if 'vfl' in s:
            vfp_val = s['vfl'].float() if torch.is_tensor(s['vfl']) else torch.tensor(float(s['vfl']))
        else:
            vfp_val = torch.tensor(float(self.img_size))
        return img, true_uvd, dist_uvd, vfp_val.float()
