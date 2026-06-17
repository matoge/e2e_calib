"""Re-eval pose_opt vs NOPOSE ckpts with masked PSNR/SSIM/LPIPS.

Run inside e2e-calib-splatkb:v1-examples (gsplat 1.5.3, simple_trainer deps).
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
import cv2
import numpy as np
import torch
import imageio.v3 as iio

sys.path.insert(0, '/host_e2e_calib/scripts/splatad_kb')
from woven_parser_pinhole import WovenParserPinhole
from gsplat.rendering import rasterization
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

PANDASET = Path('/raid/home/hfunaya/woven_pandaset_pylon/001_half')
SEQ      = Path('/mnt/ecp-perception/woven_sequence/tss4_calib_raw_01/'
                '20230612_001946/sequence=248_20230612_001946_'
                '1686533186104-1686533191007')
MASKS    = Path('/host_e2e_calib/scripts/webui_kb_fit/_outputs/_masks/'
                'baked_pylon_seq_dynonly_2500x700')

DEVICE = torch.device('cuda')


def reeval(ckpt_path: Path, parser: WovenParserPinhole, mask_aware: bool):
    ck = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    splats = ck['splats']
    means = splats['means'].to(DEVICE)
    quats = splats['quats'].to(DEVICE)
    scales = torch.exp(splats['scales']).to(DEVICE)
    opacities = torch.sigmoid(splats['opacities']).to(DEVICE).squeeze(-1)
    sh0 = splats['sh0'].to(DEVICE)
    shN = splats['shN'].to(DEVICE)
    colors = torch.cat([sh0, shN], 1)
    n = len(parser.image_names)
    test_every = parser.test_every
    val_idx = [i for i in range(n) if i % test_every == 0]
    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type='alex',
                                                       normalize=True).to(DEVICE)
    psnrs, ssims, lpipss = [], [], []
    K = torch.from_numpy(parser.Ks_dict[0]).float().to(DEVICE).unsqueeze(0)
    W, H = parser.imsize_dict[0]
    for i_val, idx in enumerate(val_idx):
        img_path = parser.image_paths[idx]
        gt = iio.imread(img_path)[..., :3].astype(np.float32) / 255.0
        gt_t = torch.from_numpy(gt).to(DEVICE)
        c2w = torch.from_numpy(parser.camtoworlds[idx]).to(DEVICE).float()
        viewmat = torch.linalg.inv(c2w).unsqueeze(0)
        out, _, _ = rasterization(
            means=means, quats=quats, scales=scales,
            opacities=opacities, colors=colors,
            viewmats=viewmat, Ks=K, width=W, height=H,
            sh_degree=3, packed=False,
            camera_model='pinhole',
        )
        pred = out[0].clamp(0, 1)
        pred_p = pred.permute(2, 0, 1).unsqueeze(0)
        gt_p = gt_t.permute(2, 0, 1).unsqueeze(0)
        if mask_aware:
            stem = parser.image_names[idx]
            src_stem = sorted((SEQ / 'tss4_fcm').glob('*.jpg'))[idx].stem
            mask_path = MASKS / f'{src_stem}.png'
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m.shape[:2] != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            m_t = torch.from_numpy((m > 127).astype(np.float32)).to(DEVICE)
            m_t = m_t[None, None]                      # [1,1,H,W]
            diff2 = (pred_p - gt_p) ** 2 * m_t
            n_kept = m_t.sum() * 3.0
            mse = diff2.sum() / n_kept.clamp(min=1.0)
            psnr_v = -10.0 * torch.log10(mse.clamp(min=1e-12))
            pred_p_m = pred_p * m_t
            gt_p_m = gt_p * m_t
            ssim_v = ssim_m(pred_p_m, gt_p_m)
            lpips_v = lpips_m(pred_p_m, gt_p_m)
        else:
            psnr_v = psnr_m(pred_p, gt_p)
            ssim_v = ssim_m(pred_p, gt_p)
            lpips_v = lpips_m(pred_p, gt_p)
        psnrs.append(float(psnr_v))
        ssims.append(float(ssim_v))
        lpipss.append(float(lpips_v))
    return (float(np.mean(psnrs)), float(np.mean(ssims)),
             float(np.mean(lpipss)))


if __name__ == '__main__':
    parser = WovenParserPinhole(PANDASET, SEQ, masks_dir=None)
    runs = {
        'pose_opt': '/raid/_splat_kb/woven_pinhole_pose_0613_0355/'
                     'ckpts/ckpt_29999_rank0.pt',
        'NOPOSE':   '/raid/_splat_kb/woven_pinhole_NOPOSE_0613_0426/'
                     'ckpts/ckpt_29999_rank0.pt',
    }
    print(f'{"run":<12} {"masked PSNR":>12} {"masked SSIM":>12} '
          f'{"masked LPIPS":>13} {"unmasked PSNR":>14}')
    for name, p in runs.items():
        p_m, s_m, l_m = reeval(Path(p), parser, mask_aware=True)
        p_u, _, _ = reeval(Path(p), parser, mask_aware=False)
        print(f'{name:<12} {p_m:>12.3f} {s_m:>12.4f} {l_m:>13.4f} '
              f'{p_u:>14.3f}')
