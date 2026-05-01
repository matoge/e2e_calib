"""Adapter: feed GridDepthDataset's synthetic samples into the unified
frame-token model (CalibNetUnifiedFrame) via the cross-frame batch dict
schema. Frame A and Frame B share the same image and the same HAT-projected
points, pose A↔B is identity — same shape as `_try_one_calib`, but with
synthetic data underneath."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from datasets.synthetic import make_image_and_points_grid_depth


class SyntheticCalibUnified(Dataset):
    """Synthetic calib pair (A==B) for the unified model.

    Each sample dict contains all keys the unified model expects in
    cross-frame mode, with calib semantics: pose=identity, A==B."""

    def __init__(self, length: int = 4000, img_size: int = 64,
                 max_offset: float = 16.0, max_points: int = 256,
                 random_depths: bool = False, base_seed: int = 0,
                 random_each_epoch: bool = False):
        super().__init__()
        self.length = length
        self.img_size = img_size
        self.max_offset = max_offset
        self.max_points = max_points
        self.random_depths = random_depths
        self.base_seed = base_seed
        self.random_each_epoch = random_each_epoch

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        seed = (int(torch.randint(0, 2 ** 30, (1,)).item())
                if self.random_each_epoch else self.base_seed + idx)
        image, true_uvd, dist_uvd = make_image_and_points_grid_depth(
            img_size=self.img_size, max_offset=self.max_offset,
            seed=seed, random_depths=self.random_depths)
        # `true_uvd[:, 2]` is the per-point normalised depth. For obj groups
        # the depth differs from the BG default (=1.0); use that to flag obj.
        bg_depth = float(true_uvd[:, 2].max())
        is_obj = (torch.abs(true_uvd[:, 2] - bg_depth) > 1e-3)

        N = true_uvd.shape[0]
        M = self.max_points
        pad = torch.ones(M, dtype=torch.bool)
        if N > M:
            true_uvd = true_uvd[:M]
            dist_uvd = dist_uvd[:M]
            is_obj   = is_obj[:M]
            N = M
        pad[:N] = False

        def _pad(x, fill_shape):
            if N >= M:
                return x
            extra = torch.zeros(M - N, *fill_shape, dtype=x.dtype) if fill_shape else \
                    torch.zeros(M - N, dtype=x.dtype)
            return torch.cat([x, extra], dim=0)

        true_uvd_p = _pad(true_uvd, (3,))
        dist_uvd_p = _pad(dist_uvd, (3,))
        is_obj_p   = _pad(is_obj.to(torch.bool), ())

        # uvd input to model: (u, v, d_norm) with HAT (= dist) positions.
        uvd_t = dist_uvd_p.clone().float()
        # Supervision positions in patch-local coords (which == image coords here).
        uv_hat_t = dist_uvd_p[:, :2].clone().float()
        uv_gt_t  = true_uvd_p[:, :2].clone().float()
        # Depths: for synth there is no actual cam-frame z, so just zero-fill
        # the z residuals — the trainer's uvd-mode head still works because
        # d_hat - d_gt = 0 → depth term contributes a fixed log-σ_d penalty.
        z_zero = torch.zeros(M, dtype=torch.float32)

        zero_pose = torch.zeros(6, dtype=torch.float32)
        feats = torch.zeros(M, 3, dtype=torch.float32)

        # Frustum-context: the unified encoder takes a denser (N_FULL, 3)
        # set of points + pad mask. Reuse the same dist points truncated/
        # padded to N_FULL=2048; identical content is fine for synth.
        N_FULL = 2048
        full_pad = torch.ones(N_FULL, dtype=torch.bool)
        full = torch.zeros(N_FULL, 3, dtype=torch.float32)
        n_full = min(dist_uvd.shape[0], N_FULL)
        full[:n_full] = dist_uvd[:n_full].float()
        full_pad[:n_full] = False

        out = dict(
            patch_A=image.float(), patch_B=image.float().clone(),
            uvd_A=uvd_t.clone(), uvd_B=uvd_t.clone(),
            uv_B_hat_of_A=uv_hat_t.clone(), uv_B_gt_of_A=uv_gt_t.clone(),
            uv_A_hat_of_B=uv_hat_t.clone(), uv_A_gt_of_B=uv_gt_t.clone(),
            d_B_hat_of_A=z_zero.clone(), d_B_gt_of_A=z_zero.clone(),
            d_A_hat_of_B=z_zero.clone(), d_A_gt_of_B=z_zero.clone(),
            pose_AB_6dof=zero_pose.clone(), pose_BA_6dof=zero_pose.clone(),
            pad_A=pad.clone(), pad_B=pad.clone(),
            feats_A=feats.clone(), feats_B=feats.clone(),
            uvd_A_full=full.clone(), uvd_B_full=full.clone(),
            pad_A_full=full_pad.clone(), pad_B_full=full_pad.clone(),
            is_obj_A=is_obj_p.clone(), is_obj_B=is_obj_p.clone(),
        )
        return out
