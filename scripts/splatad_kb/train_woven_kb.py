"""Minimal Gaussian-Splatting trainer for woven_sequence with:
  - gsplat 1.5.3 native KB4 fisheye (camera_model="fisheye", radial_coeffs)
  - LiDAR-initialized Gaussians (rear_axle frame), dynamic-mask hits dropped
  - per-pixel dynamic mask zeroes the L1+SSIM loss
  - actor / rolling-shutter / motion blur: NONE (mask handles dynamics)

ClearML reports loss/iter and a render+gt+|diff| triptych periodically.

World frame == frame-0 rear_axle (see woven_dataloader.py).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from woven_dataloader import Frame, load_woven_sequence


def _ssim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """SSIM on (B, C, H, W) inputs in [0,1]. Returns scalar."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    win = 11
    pad = win // 2

    def avg(t):
        return F.avg_pool2d(t, win, 1, pad)

    mu_x = avg(x)
    mu_y = avg(y)
    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x = avg(x * x) - mu_x_sq
    sigma_y = avg(y * y) - mu_y_sq
    sigma_xy = avg(x * y) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    )
    return ssim_map.mean()


def _read_image_chw(path: Path, scale: float = 1.0) -> torch.Tensor:
    img = cv2.imread(str(path))
    if scale != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)  # (3,H,W)


def _read_mask(path: Optional[Path], target_hw, scale: float = 1.0) -> torch.Tensor:
    H, W = target_hw
    if path is None or not path.is_file():
        return torch.ones((1, H, W), dtype=torch.float32)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if scale != 1.0:
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    if m.shape[:2] != (H, W):
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
    m = (m.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
    return torch.from_numpy(m).unsqueeze(0)  # (1,H,W)


def _project_lidar_to_image(
    pc_xyz_world: np.ndarray,
    cam_t2w: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    width: int,
    height: int,
):
    """Returns (uv (N,2), z_cam (N,), in-image-mask (N,))."""
    T_w2c = np.linalg.inv(cam_t2w)
    pts_c = (T_w2c[:3, :3] @ pc_xyz_world.T + T_w2c[:3, 3:4]).T
    Z = pts_c[:, 2]
    in_front = Z > 0.5
    pcv = pts_c[in_front].reshape(-1, 1, 3)
    uv, _ = cv2.fisheye.projectPoints(pcv, np.zeros(3), np.zeros(3), K, dist)
    uv = uv.reshape(-1, 2)
    Zv = Z[in_front]
    in_b = ((uv[:, 0] >= 0) & (uv[:, 0] < width)
            & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    uv_full = np.full((len(Z), 2), np.nan, dtype=np.float64)
    z_full = np.full((len(Z),), np.nan, dtype=np.float64)
    mask_full = np.zeros(len(Z), dtype=bool)
    idx_front = np.where(in_front)[0]
    keep = idx_front[in_b]
    uv_full[keep] = uv[in_b]
    z_full[keep] = Zv[in_b]
    mask_full[keep] = True
    return uv_full, z_full, mask_full


def _init_gaussians_from_lidar(
    frames: List[Frame],
    scale: float,
    device: torch.device,
    sample_per_frame: int = 80_000,
) -> dict:
    """Build initial Gaussian set: union of frame-0 LiDAR points (in WORLD = rear_axle
    frame 0) with mask-hit drop and color sampled from the corresponding image."""
    means_list = []
    rgbs_list = []
    rng = np.random.default_rng(0)
    for f in frames:
        pc_lid = f.pc[:, :3].astype(np.float64)
        pts_w = (f.lid_t2w[:3, :3] @ pc_lid.T + f.lid_t2w[:3, 3:4]).T
        # project to image to get (a) mask drop (b) color
        uv, z, ok = _project_lidar_to_image(
            pts_w, f.cam_t2w, f.K, f.dist, f.width, f.height)
        if not ok.any():
            continue
        # sub-sample
        idx_ok = np.where(ok)[0]
        if len(idx_ok) > sample_per_frame:
            idx_ok = rng.choice(idx_ok, sample_per_frame, replace=False)

        # mask drop
        if f.mask_path is not None and f.mask_path.is_file():
            m = cv2.imread(str(f.mask_path), cv2.IMREAD_GRAYSCALE)
            if m.shape[:2] != (f.height, f.width):
                m = cv2.resize(m, (f.width, f.height), cv2.INTER_NEAREST)
            keep = []
            for i in idx_ok:
                u = int(round(uv[i, 0])); v = int(round(uv[i, 1]))
                u = max(0, min(f.width - 1, u))
                v = max(0, min(f.height - 1, v))
                if m[v, u] > 127:
                    keep.append(i)
            idx_ok = np.asarray(keep, dtype=np.int64)

        if len(idx_ok) == 0:
            continue

        # color from image
        img = cv2.imread(str(f.img_path))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        u = np.clip(np.round(uv[idx_ok, 0]).astype(np.int32), 0, f.width - 1)
        v = np.clip(np.round(uv[idx_ok, 1]).astype(np.int32), 0, f.height - 1)
        cols = rgb_img[v, u].astype(np.float32) / 255.0

        means_list.append(pts_w[idx_ok])
        rgbs_list.append(cols)

    means = np.concatenate(means_list, 0).astype(np.float32)
    rgbs = np.concatenate(rgbs_list, 0).astype(np.float32)
    print(f'[init] {len(means)} gaussians from {len(frames)} frames')

    # Initial scale: median nearest-neighbor distance / 3 — quick approximation:
    # use a constant 0.05 m to start (refined by densification anyway).
    n = len(means)
    quats = torch.zeros((n, 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0  # w=1
    scales = torch.full((n, 3), math.log(0.05), dtype=torch.float32, device=device)
    opacities = torch.full((n, 1), torch.logit(torch.tensor(0.1)).item(),
                            dtype=torch.float32, device=device)
    means_t = torch.from_numpy(means).to(device)
    # gsplat uses sh0 = (rgb - 0.5) / 0.28209... formula; simpler: use plain "colors" path
    # by passing colors directly. We'll rasterize w/ colors arg.
    colors = torch.from_numpy(rgbs).to(device)
    return dict(
        means=torch.nn.Parameter(means_t),
        quats=torch.nn.Parameter(quats),
        scales=torch.nn.Parameter(scales),
        opacities=torch.nn.Parameter(opacities),
        colors=torch.nn.Parameter(colors),
    )


def _to_w2c_torch(cam_t2w: np.ndarray, device) -> torch.Tensor:
    T = torch.from_numpy(cam_t2w).to(device).float()
    return torch.linalg.inv(T)


def main():
    from gsplat.rendering import rasterization

    ap = argparse.ArgumentParser()
    ap.add_argument('--seq-dir', type=Path, required=True)
    ap.add_argument('--vehicle', default='248')
    ap.add_argument('--masks-dir', type=Path, default=None)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--iters', type=int, default=5000)
    ap.add_argument('--lr-means', type=float, default=1e-4)
    ap.add_argument('--lr-quats', type=float, default=1e-3)
    ap.add_argument('--lr-scales', type=float, default=5e-3)
    ap.add_argument('--lr-opacities', type=float, default=5e-2)
    ap.add_argument('--lr-colors', type=float, default=2.5e-3)
    ap.add_argument('--ssim-w', type=float, default=0.2)
    ap.add_argument('--scale', type=float, default=0.5,
                    help='image downscale (1.0 = full 3840x1952)')
    ap.add_argument('--init-sample', type=int, default=40_000)
    ap.add_argument('--report-every', type=int, default=200)
    ap.add_argument('--clearml', action='store_true')
    ap.add_argument('--log-every-img', type=int, default=500)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    frames = load_woven_sequence(
        args.seq_dir, vehicle=args.vehicle, masks_dir=args.masks_dir)
    print(f'[frames] {len(frames)}')

    # scale K and image WxH
    s = args.scale
    Ks_list = []
    Ws, Hs = [], []
    dists_list = []
    for f in frames:
        K = f.K.copy()
        K[0, 0] *= s; K[1, 1] *= s
        K[0, 2] = (K[0, 2] + 0.5) * s - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
        Ks_list.append(K)
        Ws.append(int(round(f.width * s)))
        Hs.append(int(round(f.height * s)))
        dists_list.append(f.dist.copy())

    init = _init_gaussians_from_lidar(frames, scale=s, device=device,
                                        sample_per_frame=args.init_sample)
    means = init['means']
    quats = init['quats']
    scales = init['scales']
    opacities = init['opacities']
    colors = init['colors']

    optim = torch.optim.Adam([
        {'params': [means], 'lr': args.lr_means},
        {'params': [quats], 'lr': args.lr_quats},
        {'params': [scales], 'lr': args.lr_scales},
        {'params': [opacities], 'lr': args.lr_opacities},
        {'params': [colors], 'lr': args.lr_colors},
    ])

    task = None
    if args.clearml:
        from clearml import Task
        task = Task.init(
            project_name='e2e_calib/splat_kb',
            task_name=f'woven_kb_{args.seq_dir.name}',
            task_type=Task.TaskTypes.training,
            reuse_last_task_id=False,
            auto_connect_arg_parser=True,
            auto_connect_frameworks=False,
        )
        task.connect(vars(args))

    rng = np.random.default_rng(123)
    image_cache = [None] * len(frames)
    mask_cache = [None] * len(frames)

    def get_img(i):
        if image_cache[i] is None:
            image_cache[i] = _read_image_chw(frames[i].img_path, scale=s).to(device)
        return image_cache[i]

    def get_mask(i):
        if mask_cache[i] is None:
            mask_cache[i] = _read_mask(
                frames[i].mask_path, (Hs[i], Ws[i]), scale=s).to(device)
        return mask_cache[i]

    t_start = time.monotonic()
    for it in range(1, args.iters + 1):
        idx = int(rng.integers(0, len(frames)))
        f = frames[idx]
        K = torch.from_numpy(Ks_list[idx]).to(device).float().unsqueeze(0)
        W, H = Ws[idx], Hs[idx]
        viewmat = _to_w2c_torch(f.cam_t2w, device).unsqueeze(0)  # (1,4,4)

        rgb_target = get_img(idx)               # (3,H,W)
        mask = get_mask(idx)                    # (1,H,W)

        # rasterize
        out, _, _info = rasterization(
            means=means,
            quats=quats / (quats.norm(dim=-1, keepdim=True) + 1e-12),
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities).squeeze(-1),
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=W,
            height=H,
            camera_model='fisheye',
            radial_coeffs=torch.from_numpy(dists_list[idx]).to(device).float()[None],
            with_ut=True,
            with_eval3d=True,
            packed=False,
        )
        # rasterization returns colors in (1, H, W, 3)
        pred = out[0].permute(2, 0, 1)          # (3,H,W)

        diff = (pred - rgb_target).abs()
        l1 = (diff * mask).mean()
        ssim_v = _ssim(pred.unsqueeze(0) * mask.unsqueeze(0),
                       rgb_target.unsqueeze(0) * mask.unsqueeze(0))
        loss = (1 - args.ssim_w) * l1 + args.ssim_w * (1 - ssim_v)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if it % args.report_every == 0 or it == 1:
            with torch.no_grad():
                psnr = -10 * torch.log10(((pred - rgb_target) ** 2).mean() + 1e-12)
            elapsed = time.monotonic() - t_start
            sps = it / max(elapsed, 1e-6)
            print(f'  it={it:6d} loss={loss.item():.4f} '
                  f'l1={l1.item():.4f} ssim={ssim_v.item():.3f} '
                  f'psnr={psnr.item():.2f}dB n_g={means.shape[0]} '
                  f'{sps:.1f}sps')
            if task is not None:
                logger = task.get_logger()
                logger.report_scalar('train', 'loss', loss.item(), it)
                logger.report_scalar('train', 'l1', l1.item(), it)
                logger.report_scalar('train', 'ssim', ssim_v.item(), it)
                logger.report_scalar('train', 'psnr', psnr.item(), it)
                logger.report_scalar('train', 'n_gaussians', means.shape[0], it)
                logger.report_scalar('train', 'sps', sps, it)

        if it % args.log_every_img == 0:
            with torch.no_grad():
                pred_np = (pred.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                gt_np = (rgb_target.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                d_np = np.abs(pred_np.astype(np.int16) - gt_np.astype(np.int16)).clip(0, 255).astype(np.uint8)
                triptych = np.concatenate([pred_np, gt_np, d_np], axis=1)
                triptych_bgr = cv2.cvtColor(triptych, cv2.COLOR_RGB2BGR)
                p = args.out_dir / f'iter_{it:06d}.jpg'
                cv2.imwrite(str(p), triptych_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if task is not None:
                    task.get_logger().report_image('compare', 'last',
                                                    iteration=it, local_path=str(p))

    # save final
    ckpt = args.out_dir / 'final.pt'
    torch.save({
        'means': means.detach().cpu(),
        'quats': quats.detach().cpu(),
        'scales': scales.detach().cpu(),
        'opacities': opacities.detach().cpu(),
        'colors': colors.detach().cpu(),
        'iters': args.iters,
    }, ckpt)
    print(f'[final ckpt] {ckpt}')
    if task is not None:
        task.upload_artifact('final_ckpt', str(ckpt))
        task.flush(wait_for_uploads=True)
        task.close()


if __name__ == '__main__':
    main()
