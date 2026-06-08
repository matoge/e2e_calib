"""Train CalibNet2 on PandaSet full LMDB with BA-based pose loss.

Pipeline:
  1. PS dataloader returns (image, dist_uvd, true_uvd, K, pert_vec)
  2. cnd2 forward → per-point (μ, log_σ_x, log_σ_y, ρ)
  3. Build W = Σ⁻¹ per-point
  4. Solve BA via ba_torch.gn_step → recover 6 DoF pose perturbation
  5. Loss = SE(3) distance between solved and GT pert_vec

Runs natively (no docker).
"""
import sys, os, math, time, argparse
sys.path.insert(0, '/home/hiro/git/e2e_calib')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator

from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.calibnet2 import CalibNet2
from scripts.ba.ba_torch import (
    pinhole_jacobian, project_pinhole, gn_step, make_info_from_sigma_rho,
    _apply_extrinsic, _D2R,
)


def build_W_from_per_pt(per_pt):
    """per_pt: (B, N, 5) [du, dv, log_sx, log_sy, rho] → μ (B,N,2), W (B,N,2,2)."""
    mu = per_pt[..., :2]
    sx = torch.exp(per_pt[..., 2]).clamp(0.1, 50.0)
    sy = torch.exp(per_pt[..., 3]).clamp(0.1, 50.0)
    rho = torch.tanh(per_pt[..., 4]) * 0.95
    W = make_info_from_sigma_rho(sx, sy, rho)
    return mu, W


def unproject(uv, d, K):
    """uv: (B, N, 2) image coord, d: (B, N) depth in m, K: (B, 3, 3)
    → pts_cam: (B, N, 3)."""
    fx = K[..., 0, 0:1]; fy = K[..., 1, 1:2]
    cx = K[..., 0, 2:3]; cy = K[..., 1, 2:3]
    X = (uv[..., 0] - cx) * d / fx
    Y = (uv[..., 1] - cy) * d / fy
    return torch.stack([X, Y, d], dim=-1)


def ba_solve_pose(pts_cam, uv_target, W, K, valid, n_iter=4, damping=1e-3,
                  dof_names=('omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz'),
                  prior_diag=None):
    """Solve for 6-DoF perturbation [ω, t] (deg + m) that best maps pts_cam →
    uv_target under W. Returns δ (B, 6) and pose-cov H (B, 6, 6)."""
    B, N, _ = pts_cam.shape
    omega = torch.zeros(B, 3, device=pts_cam.device, dtype=pts_cam.dtype)
    t = torch.zeros(B, 3, device=pts_cam.device, dtype=pts_cam.dtype)
    delta_acc = torch.zeros(B, 6, device=pts_cam.device, dtype=pts_cam.dtype)
    H_last = None
    for it in range(n_iter):
        pts_t = _apply_extrinsic(pts_cam, omega, t)
        uv_proj = project_pinhole(pts_t, K)
        r = uv_target - uv_proj
        X, Y, Z = pts_t.unbind(-1)
        uv = uv_proj.detach()
        J = pinhole_jacobian(X, Y, Z, K, uv, dof_names)
        delta, H_last = gn_step(J, W, r, valid=valid, damping=damping,
                                prior_diag=prior_diag)
        # split [ω(deg), t(m)] order matches dof_names
        omega = omega + delta[:, :3]
        t = t + delta[:, 3:]
        delta_acc = torch.cat([omega, t], dim=-1)
    return delta_acc, H_last


def pose_loss(delta_solved, pert_vec_gt):
    """delta_solved: (B, 6) [ω_deg, t_m]
    pert_vec_gt: (B, 8) [tx, ty, tz, yaw, pitch, roll, dfx, dfy] from dataset
    → MSE loss combining rotation (deg) + translation (m).
    """
    # Pert_vec ordering: [tx, ty, tz, yaw, pitch, roll] = [t, ypr]
    # delta_solved ordering: [ω_xyz_deg, t_xyz]
    # We compare in the same convention by reordering pert_vec_gt
    t_gt = pert_vec_gt[:, :3]
    ypr_gt = pert_vec_gt[:, 3:6]  # ω in degrees
    t_loss = (delta_solved[:, 3:] - t_gt).pow(2).mean()
    rot_loss = (delta_solved[:, :3] - ypr_gt).pow(2).mean()
    return rot_loss + 100.0 * t_loss  # weight translation up (~m scale)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--cache', default='/mnt/nvme6t/e2e_calib_cache/pandaset_v3_full')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--ba-iter', type=int, default=4)
    p.add_argument('--ba-damping', type=float, default=1e-3)
    p.add_argument('--img-size', type=int, default=128)
    p.add_argument('--grid-n', type=int, default=16)
    p.add_argument('--oversample', type=int, default=4)
    p.add_argument('--rot-deg', type=float, default=1.5)
    p.add_argument('--t-m', type=float, default=0.20)
    args = p.parse_args()

    acc = Accelerator(mixed_precision='no')
    device = acc.device
    print(f'[init] device={device}', flush=True)

    ds_kw = dict(
        img_size=args.img_size, grid_n=args.grid_n,
        min_crop_px=256, max_crop_px=512,
        max_rot_deg=args.rot_deg, max_offset_m=args.t_m,
        oversample=args.oversample, split_pert=False, pair_mode=False,
    )
    tr_ds = PandaSetCalibDatasetFull(args.cache, split='train', u_band=0.0, **ds_kw)
    va_ds = PandaSetCalibDatasetFull(args.cache, split='val',
                                     center_band=0.5, u_band=0.0, **ds_kw)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.workers, collate_fn=collate_full,
                           drop_last=True, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, collate_fn=collate_full,
                           drop_last=False, pin_memory=True)
    print(f'[data] train={len(tr_ds)}  val={len(va_ds)}', flush=True)

    model = CalibNet2(
        d=128, img_size=args.img_size, in_channels=3,
        use_intensity=True, frustum_grid_n=args.grid_n,
        n_iter=2, n_heads=4,
        d_scalar=8, n_type1=40, use_info_head=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model, opt, tr_loader, va_loader = acc.prepare(model, opt, tr_loader, va_loader)

    out_dir = f'/home/hiro/git/e2e_calib/experiments/{args.name}'
    os.makedirs(out_dir, exist_ok=True)
    log_path = f'{out_dir}/train.log'

    # Reasonable prior: 3° / 0.3 m
    prior_diag = torch.tensor([1/9., 1/9., 1/9., 1/0.09, 1/0.09, 1/0.09],
                              device=device)

    step = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for batch in tr_loader:
            imgs, true_uvd, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid = batch[:7]
            pert_vec = batch[7] if len(batch) > 7 else None
            if pert_vec is None:
                continue
            # Need K for projection. Try batch slot 11 (= K_full per sample) or fallback.
            K = batch[11] if len(batch) > 11 else None
            if K is None:
                continue
            imgs = imgs.float().div_(255.0)
            point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1)
            out = model(imgs, point_in,
                        dpose_R=None, vfp=vfp,
                        bucket_uvd=bucket_uvd, bucket_valid=bucket_valid,
                        key_padding_mask=pad_mask)
            per_pt = out[0] if isinstance(out, tuple) else out
            mu, W = build_W_from_per_pt(per_pt)
            valid = ~pad_mask
            # uv_target = dist_uvd's (u, v) + cnd2 predicted mu
            uv_target = dist_uvd[..., :2] + mu
            # 3D points in PERTURBED camera frame: use dist_uvd's depth (already perturbed)
            # depth is in column 2, *100 (cache normalises)
            d = dist_uvd[..., 2] * 100.0
            pts_cam = unproject(dist_uvd[..., :2], d, K)
            delta_solved, H = ba_solve_pose(
                pts_cam, uv_target, W, K, valid,
                n_iter=args.ba_iter, damping=args.ba_damping,
                prior_diag=prior_diag,
            )
            loss = pose_loss(delta_solved, pert_vec)
            opt.zero_grad()
            acc.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if step % 20 == 0 and acc.is_main_process:
                with torch.no_grad():
                    sx = torch.exp(per_pt[..., 2]).mean()
                    rot_err = (delta_solved[:, :3] - pert_vec[:, 3:6]).abs().mean()
                    t_err = (delta_solved[:, 3:] - pert_vec[:, :3]).abs().mean()
                msg = (f'[ep{epoch:2d} step{step:5d}]  loss={loss.item():.4f}  '
                       f'rot_err={rot_err.item():.3f}°  t_err={t_err.item():.3f}m  '
                       f'σ_mean={sx.item():.3f}')
                print(msg, flush=True)
                with open(log_path, 'a') as f:
                    f.write(msg + '\n')
            step += 1
        if acc.is_main_process:
            ckpt = f'{out_dir}/ep{epoch:03d}.pt'
            torch.save(acc.unwrap_model(model).state_dict(), ckpt)
            print(f'  saved {ckpt}  epoch_time={(time.time()-t0)/60:.1f}min', flush=True)


if __name__ == '__main__':
    main()
