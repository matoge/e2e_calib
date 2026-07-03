"""Re-eval pose_opt vs NOPOSE on TRAIN split with masked PSNR.
For pose_opt, apply per-frame pose delta from CameraOptModule (saved in ckpt).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2
import numpy as np
import torch
import imageio.v3 as iio

sys.path.insert(0, '/host_e2e_calib/scripts/splatad_kb')
from woven_parser_pinhole import WovenParserPinhole
from gsplat.rendering import rasterization
from torchmetrics.image import (
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

PANDASET = Path('/raid/home/hfunaya/woven_pandaset_pylon/001_half')
SEQ      = Path('/mnt/ecp-perception/woven_sequence/tss4_calib_raw_01/'
                '20230612_001946/sequence=248_20230612_001946_'
                '1686533186104-1686533191007')
MASKS    = Path('/host_e2e_calib/scripts/webui_kb_fit/_outputs/_masks/'
                'baked_pylon_seq_dynonly_2500x700')
DEVICE   = torch.device('cuda')


def _so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """omega [B,3] -> R [B,3,3] (Rodrigues)."""
    theta = omega.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis = omega / theta
    K = torch.zeros((*omega.shape[:-1], 3, 3),
                     device=omega.device, dtype=omega.dtype)
    K[..., 0, 1] = -axis[..., 2]; K[..., 0, 2] =  axis[..., 1]
    K[..., 1, 0] =  axis[..., 2]; K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]; K[..., 2, 1] =  axis[..., 0]
    eye = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(K)
    sin_t = torch.sin(theta)[..., None]
    cos_t = torch.cos(theta)[..., None]
    return eye + sin_t * K + (1 - cos_t) * (K @ K)


def apply_pose_delta(c2w: torch.Tensor, delta9: torch.Tensor) -> torch.Tensor:
    """gsplat CameraOptModule applies a left perturbation in cam frame.
    Embeds is [N,9]: rot[3], trans[3], extra[3] (ignored). The official
    forward in utils.CameraOptModule applies as:
        c2w_new = c2w @ exp(delta_se3)
    where delta_se3 = (rot, trans). Reproduce here.
    """
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    dR = _so3_exp(delta9[:3].unsqueeze(0))[0]
    dt = delta9[3:6]
    R_new = R @ dR
    t_new = t + R @ dt
    out = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
    out[:3, :3] = R_new
    out[:3, 3] = t_new
    return out


def reeval_train(ckpt_path: Path, parser: WovenParserPinhole, use_pose_delta: bool):
    ck = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    splats = ck['splats']
    means = splats['means'].to(DEVICE)
    quats = splats['quats'].to(DEVICE)
    scales = torch.exp(splats['scales']).to(DEVICE)
    opacities = torch.sigmoid(splats['opacities']).to(DEVICE).squeeze(-1)
    sh0 = splats['sh0'].to(DEVICE)
    shN = splats['shN'].to(DEVICE)
    colors = torch.cat([sh0, shN], 1)

    pose_delta = None
    if use_pose_delta and 'pose_adjust' in ck:
        pose_delta = ck['pose_adjust']['embeds.weight'].to(DEVICE)  # [n_train, 9]

    n = len(parser.image_names)
    test_every = parser.test_every
    train_idx = [i for i in range(n) if i % test_every != 0]
    val_idx = [i for i in range(n) if i % test_every == 0]

    # NOTE: torchmetrics modules accumulate across calls. Use functional
    # PSNR per-frame to avoid mean-of-running-mean confusion.
    from torchmetrics.functional.image import (
        peak_signal_noise_ratio as F_psnr,
    )
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type='alex',
                                                       normalize=True).to(DEVICE)

    K = torch.from_numpy(parser.Ks_dict[0]).float().to(DEVICE).unsqueeze(0)
    W, H = parser.imsize_dict[0]
    src_jpgs = sorted((SEQ / 'tss4_fcm').glob('*.jpg'))

    def render_one(idx_in_full: int, delta_idx: int = -1) -> tuple[float, float, float]:
        img_path = parser.image_paths[idx_in_full]
        gt = iio.imread(img_path)[..., :3].astype(np.float32) / 255.0
        gt_t = torch.from_numpy(gt).to(DEVICE)
        c2w = torch.from_numpy(parser.camtoworlds[idx_in_full]).to(DEVICE).float()
        if pose_delta is not None and delta_idx >= 0:
            c2w = apply_pose_delta(c2w, pose_delta[delta_idx])
        viewmat = torch.linalg.inv(c2w).unsqueeze(0)
        out, _, _ = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=viewmat, Ks=K, width=W, height=H,
            sh_degree=3, packed=False, camera_model='pinhole',
        )
        pred = out[0].clamp(0, 1)
        pred_p = pred.permute(2, 0, 1).unsqueeze(0)
        gt_p = gt_t.permute(2, 0, 1).unsqueeze(0)
        # mask
        src_stem = src_jpgs[idx_in_full].stem
        mask_path = MASKS / f'{src_stem}.png'
        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m_t = torch.from_numpy((m > 127).astype(np.float32)).to(DEVICE)
        m_t = m_t[None, None]
        diff2 = (pred_p - gt_p) ** 2 * m_t
        n_kept = m_t.sum() * 3.0
        mse = diff2.sum() / n_kept.clamp(min=1.0)
        psnr_v = -10.0 * torch.log10(mse.clamp(min=1e-12))
        pred_p_m = pred_p * m_t
        gt_p_m = gt_p * m_t
        ssim_v = ssim_m(pred_p_m, gt_p_m)
        lpips_v = lpips_m(pred_p_m, gt_p_m)
        return float(psnr_v), float(ssim_v), float(lpips_v)

    # train
    train_psnr = []
    for di, idx in enumerate(train_idx):
        p_v, _, _ = render_one(idx, delta_idx=di)
        train_psnr.append(p_v)
    # val without delta
    val_psnr_nopose = []
    for idx in val_idx:
        p_v, _, _ = render_one(idx, delta_idx=-1)
        val_psnr_nopose.append(p_v)
    # val with nearest train delta interpolation (linear in frame index)
    val_psnr_nearest = []
    if pose_delta is not None:
        for idx in val_idx:
            # find two nearest train indices
            train_arr = np.asarray(train_idx)
            diffs = np.abs(train_arr - idx)
            nearest = np.argsort(diffs)[:2]
            # linear weight
            d0, d1 = diffs[nearest[0]], diffs[nearest[1]]
            if d0 + d1 == 0:
                w0, w1 = 1.0, 0.0
            else:
                w0 = d1 / (d0 + d1); w1 = d0 / (d0 + d1)
            interp_delta = pose_delta[nearest[0]] * w0 + pose_delta[nearest[1]] * w1
            # render with this delta (insert temporarily)
            # easier: temporarily save & restore via _temporary attr
            # but render_one expects pose_delta global; let's just inline render
            c2w_orig = torch.from_numpy(parser.camtoworlds[idx]).to(DEVICE).float()
            c2w = apply_pose_delta(c2w_orig, interp_delta)
            viewmat = torch.linalg.inv(c2w).unsqueeze(0)
            out, _, _ = rasterization(
                means=means, quats=quats, scales=scales,
                opacities=opacities, colors=colors,
                viewmats=viewmat, Ks=K, width=W, height=H,
                sh_degree=3, packed=False, camera_model='pinhole',
            )
            pred = out[0].clamp(0, 1)
            gt = iio.imread(parser.image_paths[idx])[..., :3].astype(np.float32) / 255.0
            gt_t = torch.from_numpy(gt).to(DEVICE)
            pred_p = pred.permute(2, 0, 1).unsqueeze(0)
            gt_p = gt_t.permute(2, 0, 1).unsqueeze(0)
            src_stem = src_jpgs[idx].stem
            mask_path = MASKS / f'{src_stem}.png'
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m.shape[:2] != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            m_t = torch.from_numpy((m > 127).astype(np.float32)).to(DEVICE)
            m_t = m_t[None, None]
            diff2 = (pred_p - gt_p) ** 2 * m_t
            n_kept = m_t.sum() * 3.0
            mse = diff2.sum() / n_kept.clamp(min=1.0)
            val_psnr_nearest.append(float(-10 * torch.log10(mse.clamp(min=1e-12))))
    return {
        'train_psnr_mean': float(np.mean(train_psnr)),
        'val_psnr_nopose': float(np.mean(val_psnr_nopose)),
        'val_psnr_nearest_delta': (float(np.mean(val_psnr_nearest))
                                     if val_psnr_nearest else None),
        'pose_delta_mean_norm_deg': (float(np.mean(np.linalg.norm(
                                            np.degrees(pose_delta[:, :3].cpu().numpy()),
                                            axis=1)))
                                       if pose_delta is not None else None),
    }


if __name__ == '__main__':
    parser = WovenParserPinhole(PANDASET, SEQ, masks_dir=None)
    runs = {
        'pose_opt': '/raid/_splat_kb/woven_pinhole_pose_0613_0355/'
                     'ckpts/ckpt_29999_rank0.pt',
        'NOPOSE':   '/raid/_splat_kb/woven_pinhole_NOPOSE_0613_0426/'
                     'ckpts/ckpt_29999_rank0.pt',
    }
    for name, p in runs.items():
        use_delta = (name == 'pose_opt')
        res = reeval_train(Path(p), parser, use_pose_delta=use_delta)
        print(f'\n=== {name} ===')
        for k, v in res.items():
            print(f'  {k}: {v}')
