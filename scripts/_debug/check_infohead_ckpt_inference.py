"""img256/grid32/info_head ep18-best ckpt の推論健全性チェック.

200ep run は ep18 で val_nll=3.32, BA ω=0.13° まで来てから ep19 で全 NaN
爆発した。best_model.pt (= ep18 saved) を読んで:
  1. forward が NaN 出さないか
  2. info_nll_2d (fp16 / fp32) で det/quad/log_det の hist
  3. W = LLᵀ の対角と off-diag の分布

を dump する。fp16 で det.log() が −∞ に飛んでいないか、quad が大きすぎ
ないかを目視で確認するため。
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from torch.utils.data import DataLoader

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.model_depth import CalibNetDepth
from models.model_cov import gaussian2d_nll, info_nll_2d


CKPT = ('experiments/'
        'km_wv_wm_15deg_20cm_img256_grid32_pe_infohead_200ep_dgx2_12gpu/'
        'best_model.pt')

CFG = dict(
    img_size=256, in_channels=3, n_layers=4,
    use_convnext=True, use_frustum=True, frustum_grid_n=32,
    use_pose_emb=True, use_info_head=True, deform_mode='sl',
)

DS_KW = dict(
    img_size=256, max_offset_m=0.2, max_rot_deg=1.5,
    min_crop_px=128, max_crop_px=384,
    frame_stride=1, grid_n=32, oversample=4,
)


def _stats(t: torch.Tensor, name: str) -> str:
    t = t.detach().float().flatten().cpu()
    if t.numel() == 0:
        return f"{name:<14} N=0"
    finite = torch.isfinite(t)
    n_inf = (~finite).sum().item()
    f = t[finite]
    if f.numel() == 0:
        return f"{name:<14} all non-finite (N={t.numel()})"
    q = torch.quantile(f, torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0]))
    return (f"{name:<14} N={t.numel():>8d}  non-fin={n_inf:>4d}  "
            f"min={q[0]:+.3e}  p1={q[1]:+.3e}  med={q[2]:+.3e}  "
            f"p99={q[3]:+.3e}  max={q[4]:+.3e}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CalibNetDepth(**CFG).to(device)

    sd = torch.load(CKPT, map_location='cpu', weights_only=True)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] {CKPT}")
    print(f"[ckpt] miss={len(miss)} unexp={len(unexp)}")
    if miss:    print(f"        miss[:8]={miss[:8]}")
    if unexp:   print(f"        unexp[:8]={unexp[:8]}")
    model.eval()

    # 1 cache (kami) に絞って val 64 inst
    cache = '/home/hfunaya/cache_v4/kamikado_v3_tiled'
    va_kw = dict(DS_KW); va_kw['center_band'] = 0.5
    ds = PandaSetCalibDatasetFull(cache, split='val', **va_kw)
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0,
                         collate_fn=collate_full)

    print(f"[ds] {cache}  N={len(ds)}")

    n_batches = 8       # 64 inst total
    nan_batches = 0
    all_W_a, all_W_b, all_W_c = [], [], []
    all_det, all_quad, all_logdet = [], [], []
    all_resid = []
    val_nll_fp32, val_nll_fp16, val_mse = 0.0, 0.0, 0.0
    n_used = 0

    for bi, batch in enumerate(loader):
        if bi >= n_batches: break
        imgs, true_uvd, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid = batch[:7]
        imgs = imgs.float().div_(255.0).to(device)
        true_uvd = true_uvd.to(device)
        dist_uvd = dist_uvd.to(device)
        pad_mask = pad_mask.to(device)
        vfp = vfp.to(device)
        bucket_uvd = bucket_uvd.to(device)
        bucket_valid = bucket_valid.to(device)

        gt = true_uvd[..., :2] - dist_uvd[..., :2]
        # use_intensity 既定 True
        point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)

        with torch.no_grad():
            # fp32 path (sanity 基準)
            out = model(imgs, point_in, key_padding_mask=pad_mask, vfp=vfp,
                        bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
            params, W = out[0], out[-1]
            valid = ~pad_mask
            r = (gt[valid] - params[valid][..., :2])
            mu = params[valid][..., :2]
            tg = gt[valid]
            Wv = W[valid]
            nll_fp32 = info_nll_2d(mu, Wv, tg).item()

            # fp16 forward (autocast) — train と同じ条件
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                out16 = model(imgs, point_in, key_padding_mask=pad_mask, vfp=vfp,
                              bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
                params16, W16 = out16[0], out16[-1]
                Wv16 = W16[valid]
                mu16 = params16[valid][..., :2]
                nll_fp16 = info_nll_2d(mu16, Wv16, tg).item()

            mse = (mu.float() - tg).norm(dim=-1).mean().item()

            # det / quad / log_det を fp32 で出す (overflow 観察用に
            # fp16 W も合わせて見る)
            a = Wv[..., 0, 0]; b = Wv[..., 1, 1]; c = Wv[..., 0, 1]
            det = a * b - c * c
            r2  = (tg - mu).unsqueeze(-1)
            quad = (r2.transpose(-1, -2) @ Wv @ r2).squeeze(-1).squeeze(-1)
            logdet = det.clamp(min=1e-30).log()

            all_W_a.append(a); all_W_b.append(b); all_W_c.append(c)
            all_det.append(det); all_quad.append(quad); all_logdet.append(logdet)
            all_resid.append((tg - mu).norm(dim=-1))

            val_nll_fp32 += nll_fp32; val_nll_fp16 += nll_fp16
            val_mse += mse; n_used += 1

            has_nan_p = (~torch.isfinite(params)).any().item()
            has_nan_W = (~torch.isfinite(W)).any().item()
            has_nan_p16 = (~torch.isfinite(params16)).any().item()
            has_nan_W16 = (~torch.isfinite(W16)).any().item()
            if has_nan_p or has_nan_W or has_nan_p16 or has_nan_W16:
                nan_batches += 1
                print(f"[batch {bi}] NaN params={has_nan_p} W={has_nan_W} "
                      f"params16={has_nan_p16} W16={has_nan_W16}")

    print(f"\n=== aggregate over {n_used} batches ({n_used*8} inst) ===")
    print(f"NaN batches:           {nan_batches}/{n_used}")
    print(f"val nll  (fp32):       {val_nll_fp32 / n_used:+.4f}")
    print(f"val nll  (fp16 cast):  {val_nll_fp16 / n_used:+.4f}")
    print(f"val mse  (px):         {val_mse / n_used:.4f}")

    print()
    print(_stats(torch.cat(all_W_a), 'W[0,0] (a)'))
    print(_stats(torch.cat(all_W_b), 'W[1,1] (b)'))
    print(_stats(torch.cat(all_W_c), 'W[0,1] (c)'))
    print(_stats(torch.cat(all_det), 'det W'))
    print(_stats(torch.cat(all_logdet), 'log det W'))
    print(_stats(torch.cat(all_quad), 'r^T W r'))
    print(_stats(torch.cat(all_resid), '|r| px'))


if __name__ == '__main__':
    main()
