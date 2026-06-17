"""Re-eval pose_opt vs NOPOSE on TRAIN + VAL with masked PSNR/SSIM/LPIPS.

Fixes:
  - Use torchmetrics functional PSNR (no accumulation between frames)
  - Reset SSIM/LPIPS metric modules between frames
  - For pose_opt, apply per-frame pose delta from CameraOptModule on train
    frames; for val frames apply the nearest-neighbour delta in frame-index
    space (linear interpolation between the two closest train frames).
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import imageio.v3 as iio

sys.path.insert(0, '/host_e2e_calib/scripts/splatad_kb')
from woven_parser_pinhole import WovenParserPinhole
from gsplat.rendering import rasterization
from torchmetrics.functional.image import (
    peak_signal_noise_ratio as F_psnr,
    structural_similarity_index_measure as F_ssim,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

PANDASET = Path('/raid/home/hfunaya/woven_pandaset_pylon/001_half')
SEQ      = Path('/mnt/ecp-perception/woven_sequence/tss4_calib_raw_01/'
                '20230612_001946/sequence=248_20230612_001946_'
                '1686533186104-1686533191007')
MASKS    = Path('/host_e2e_calib/scripts/webui_kb_fit/_outputs/_masks/'
                'baked_pylon_seq_dynonly_2500x700')
DEVICE   = torch.device('cuda')


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt 6D -> 3x3 rotation (matches gsplat utils)."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / b2.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns


_IDENTITY_6D = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def apply_pose_delta(c2w: torch.Tensor, delta9: torch.Tensor) -> torch.Tensor:
    """Match gsplat's CameraOptModule.forward exactly:
        dx, drot = delta9[:3], delta9[3:9]
        rot = rotation_6d_to_matrix(drot + identity_6d)
        T = [[rot, dx], [0, 1]]
        c2w_new = c2w @ T
    """
    dx = delta9[:3]
    drot = delta9[3:9]
    identity = _IDENTITY_6D.to(delta9.device).to(delta9.dtype)
    rot = rotation_6d_to_matrix(drot + identity)        # [3,3]
    T = torch.eye(4, dtype=c2w.dtype, device=c2w.device)
    T[:3, :3] = rot
    T[:3, 3] = dx
    return c2w @ T


def reeval(ckpt_path: Path, parser: WovenParserPinhole, *, has_pose_delta: bool):
    ck = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    splats = ck['splats']
    means = splats['means'].to(DEVICE)
    quats = splats['quats'].to(DEVICE)
    scales = torch.exp(splats['scales']).to(DEVICE)
    opacities = torch.sigmoid(splats['opacities']).to(DEVICE).squeeze(-1)
    sh0 = splats['sh0'].to(DEVICE); shN = splats['shN'].to(DEVICE)
    colors = torch.cat([sh0, shN], 1)
    pose_delta = None
    if has_pose_delta and 'pose_adjust' in ck:
        pose_delta = ck['pose_adjust']['embeds.weight'].to(DEVICE)

    n = len(parser.image_names); te = parser.test_every
    train_idx = [i for i in range(n) if i % te != 0]
    val_idx   = [i for i in range(n) if i % te == 0]

    K = torch.from_numpy(parser.Ks_dict[0]).float().to(DEVICE).unsqueeze(0)
    W, H = parser.imsize_dict[0]
    src_jpgs = sorted((SEQ / 'tss4_fcm').glob('*.jpg'))

    lpips_m = LearnedPerceptualImagePatchSimilarity(
        net_type='alex', normalize=True).to(DEVICE)

    def render_with(idx_full: int, c2w: torch.Tensor) -> torch.Tensor:
        viewmat = torch.linalg.inv(c2w).unsqueeze(0)
        out, _, _ = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=viewmat, Ks=K, width=W, height=H,
            sh_degree=3, packed=False, camera_model='pinhole',
        )
        return out[0].clamp(0, 1)

    def metrics_for(idx_full: int, c2w_corrected: torch.Tensor):
        pred = render_with(idx_full, c2w_corrected)
        pred_p = pred.permute(2, 0, 1).unsqueeze(0)
        gt = iio.imread(parser.image_paths[idx_full])[..., :3].astype(np.float32) / 255.0
        gt_p = torch.from_numpy(gt).to(DEVICE).permute(2, 0, 1).unsqueeze(0)
        # mask
        src_stem = src_jpgs[idx_full].stem
        m = cv2.imread(str(MASKS / f'{src_stem}.png'), cv2.IMREAD_GRAYSCALE)
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        m_t = torch.from_numpy((m > 127).astype(np.float32)).to(DEVICE)
        m_t = m_t[None, None]
        # masked PSNR (manual)
        diff2 = (pred_p - gt_p) ** 2 * m_t
        n_kept = m_t.sum() * 3.0
        mse = diff2.sum() / n_kept.clamp(min=1.0)
        psnr_v = float(-10 * torch.log10(mse.clamp(min=1e-12)))
        # masked SSIM/LPIPS via functional/no-state
        pp = (pred_p * m_t)
        gp = (gt_p * m_t)
        ssim_v = float(F_ssim(pp, gp, data_range=1.0))
        lpips_v = float(lpips_m(pp, gp))
        # unmasked PSNR (functional, no accumulation)
        psnr_u = float(F_psnr(pred_p, gt_p, data_range=1.0))
        return dict(psnr=psnr_v, ssim=ssim_v, lpips=lpips_v,
                     psnr_unmasked=psnr_u)

    # --- train: apply pose_delta if pose_opt run; otherwise raw POSLV pose
    train_metrics = []
    for di, idx in enumerate(train_idx):
        c2w_orig = torch.from_numpy(parser.camtoworlds[idx]).to(DEVICE).float()
        c2w = (apply_pose_delta(c2w_orig, pose_delta[di])
               if pose_delta is not None else c2w_orig)
        train_metrics.append(metrics_for(idx, c2w))

    # --- val: 3 modes: (a) raw POSLV pose, (b) nearest train-delta-interp
    val_raw = []
    val_nearest = []
    for idx in val_idx:
        c2w_orig = torch.from_numpy(parser.camtoworlds[idx]).to(DEVICE).float()
        val_raw.append(metrics_for(idx, c2w_orig))
        if pose_delta is not None:
            train_arr = np.asarray(train_idx)
            diffs = np.abs(train_arr - idx)
            order = np.argsort(diffs)[:2]
            d0, d1 = diffs[order[0]], diffs[order[1]]
            if d0 + d1 == 0:
                w0, w1 = 1.0, 0.0
            else:
                w0 = d1 / (d0 + d1); w1 = d0 / (d0 + d1)
            interp = pose_delta[order[0]] * w0 + pose_delta[order[1]] * w1
            c2w = apply_pose_delta(c2w_orig, interp)
            val_nearest.append(metrics_for(idx, c2w))

    def agg(rows, key):
        return float(np.mean([r[key] for r in rows]))

    return dict(
        train_psnr=agg(train_metrics, 'psnr'),
        train_ssim=agg(train_metrics, 'ssim'),
        train_lpips=agg(train_metrics, 'lpips'),
        train_psnr_unmasked=agg(train_metrics, 'psnr_unmasked'),
        val_raw_psnr=agg(val_raw, 'psnr'),
        val_raw_ssim=agg(val_raw, 'ssim'),
        val_raw_lpips=agg(val_raw, 'lpips'),
        val_raw_psnr_unmasked=agg(val_raw, 'psnr_unmasked'),
        val_nearest_psnr=agg(val_nearest, 'psnr') if val_nearest else None,
        val_nearest_ssim=agg(val_nearest, 'ssim') if val_nearest else None,
        val_nearest_lpips=agg(val_nearest, 'lpips') if val_nearest else None,
        val_nearest_psnr_unmasked=agg(val_nearest, 'psnr_unmasked') if val_nearest else None,
        pose_delta_norm_deg=(float(np.mean(np.linalg.norm(
                                    np.degrees(pose_delta[:, :3].cpu().numpy()),
                                    axis=1))) if pose_delta is not None else None),
    )


if __name__ == '__main__':
    parser = WovenParserPinhole(PANDASET, SEQ, masks_dir=None)
    runs = {
        'pose_opt': '/raid/_splat_kb/woven_pinhole_pose_0613_0355/'
                     'ckpts/ckpt_29999_rank0.pt',
        'NOPOSE':   '/raid/_splat_kb/woven_pinhole_NOPOSE_0613_0426/'
                     'ckpts/ckpt_29999_rank0.pt',
    }
    print(f'{"run":<10} | {"train PSNR(M)":>13} {"PSNR(U)":>9} {"SSIM(M)":>9} {"LPIPS(M)":>10} '
          f'| {"val(raw) PSNR(M)":>17} {"PSNR(U)":>9} {"SSIM":>9} {"LPIPS":>9} '
          f'| {"val(nearest) PSNR(M)":>22} {"PSNR(U)":>9}')
    for name, p in runs.items():
        r = reeval(Path(p), parser, has_pose_delta=(name == 'pose_opt'))
        print(f'{name:<10} | {r["train_psnr"]:>13.2f} {r["train_psnr_unmasked"]:>9.2f} '
              f'{r["train_ssim"]:>9.4f} {r["train_lpips"]:>10.4f} '
              f'| {r["val_raw_psnr"]:>17.2f} {r["val_raw_psnr_unmasked"]:>9.2f} '
              f'{r["val_raw_ssim"]:>9.4f} {r["val_raw_lpips"]:>9.4f} '
              f'| {(r["val_nearest_psnr"] or float("nan")):>22.2f} '
              f'{(r["val_nearest_psnr_unmasked"] or float("nan")):>9.2f}')
