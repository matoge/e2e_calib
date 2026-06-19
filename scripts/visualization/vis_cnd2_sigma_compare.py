"""CND2 per-point Σ (information) before/after the GN pose-NLL head.

Loads two CalibNet2 checkpoints (e.g. calib baseline vs ba-nll head), runs
BOTH on the SAME fixed val batch, and renders, per query point:
    ×  pert  (dist_uv, the mis-calibrated projection / GN input)
    ○  GT    (true_uv)
    +  pred  (dist_uv + μ)
    ellipse  2σ from (log_sx, log_sy, rho)   ← the per-point INFORMATION

Two panels (baseline | head) on identical points → see how the head
RE-DISTRIBUTES information (Σ), not just whether μ/MSE moved (that's
bias-variance). Also prints a pose-covariance calibration check: for each
model, solve the GN pose, compare predicted Σ_δ=H⁻¹ against the actual
(δ_solved-δ_gt) over the batch — does the claimed pose uncertainty match
reality (is the NLL honest)?

Run inside the e2e-calib docker on sakurai2:
  python scripts/visualization/vis_cnd2_sigma_compare.py \
    --ckpt-a experiments/cnd2_ps_calib_repro_sk2/best_model.pt --label-a calib \
    --ckpt-b experiments/cnd2_ps_ba_fix_sk2/best_model.pt      --label-b head \
    --cache /data --out /work/experiments/sigma_compare.png
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from torch.utils.data import DataLoader
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from models.calibnet2 import CalibNet2
from scripts.ba.ba_torch import (
    pinhole_jacobian, project_pinhole, gn_step, make_info_from_sigma_rho,
    _apply_extrinsic,
)


def load_model(ckpt, dev):
    m = CalibNet2(d=128, img_size=128, in_channels=3, use_intensity=True,
                  frustum_grid_n=16, n_iter=4, n_heads=4,
                  d_scalar=8, n_type1=40, use_info_head=True).to(dev)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    sd = {k.removeprefix('module.'): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


@torch.no_grad()
def run(model, batch, dev):
    imgs, true_uvd, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid = batch[:7]
    imgs = imgs.float().div(255.0).to(dev)
    point_in = torch.cat([dist_uvd[..., :3], dist_uvd[..., 4:5]], dim=-1).to(dev)
    out = model(imgs, point_in, dpose_R=None, vfp=vfp.to(dev),
                bucket_uvd=bucket_uvd.to(dev), bucket_valid=bucket_valid.to(dev),
                key_padding_mask=pad_mask.to(dev))
    per_pt = out[0] if isinstance(out, tuple) else out
    return per_pt.float().cpu()


def gn_pose(per_pt, dist_uvd, pad_mask, batch, dev):
    """Mirror _ba_pose_loss: GN 6-DoF solve + return δ_solved, Σ_δ=H⁻¹, δ_gt."""
    K_orig = batch[10]; cs_t = batch[11]; u0_t = batch[13]; v0_t = batch[14]
    pert = batch[7]
    B, N, _ = dist_uvd.shape
    valid = (~pad_mask).to(dev)
    img_size = 128.0
    scale = img_size / cs_t
    K = torch.zeros(B, 3, 3)
    K[:, 0, 0] = K_orig[:, 0, 0] * scale; K[:, 1, 1] = K_orig[:, 1, 1] * scale
    K[:, 0, 2] = (K_orig[:, 0, 2] - u0_t) * scale
    K[:, 1, 2] = (K_orig[:, 1, 2] - v0_t) * scale; K[:, 2, 2] = 1.0
    K = K.to(dev)
    per_pt = per_pt.to(dev); dist_uvd = dist_uvd.to(dev)
    sx = per_pt[..., 2].exp().clamp(0.1, 50.0)
    sy = per_pt[..., 3].exp().clamp(0.1, 50.0)
    rho = per_pt[..., 4].tanh() * 0.95
    W = make_info_from_sigma_rho(sx, sy, rho)
    fx = K[..., 0, 0:1]; fy = K[..., 1, 1:2]; cx = K[..., 0, 2:3]; cy = K[..., 1, 2:3]
    d = (dist_uvd[..., 2] * 100.0)
    safe_d = torch.where(valid, d, torch.full_like(d, 10.0)).clamp(min=0.5)
    u, v = dist_uvd[..., 0], dist_uvd[..., 1]
    pts_cam = torch.stack([(u - cx) * safe_d / fx, (v - cy) * safe_d / fy, safe_d], -1)
    uv_target = dist_uvd[..., :2] + per_pt[..., :2]
    dof = ('omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz')
    prior = torch.tensor([1/9., 1/9., 1/9., 1/0.09, 1/0.09, 1/0.09], device=dev)
    omega = torch.zeros(B, 3, device=dev); t = torch.zeros(B, 3, device=dev); H = None
    for _ in range(4):
        pts_t = _apply_extrinsic(pts_cam, omega, t)
        uv_proj = project_pinhole(pts_t, K)
        r = uv_target - uv_proj
        Xp, Yp, Zp = pts_t.unbind(-1)
        J = pinhole_jacobian(Xp, Yp, Zp, K, uv_proj.detach(), dof)
        delta, H = gn_step(J, W, r, valid=valid, damping=1e-3, prior_diag=prior)
        omega = omega + delta[:, :3]; t = t + delta[:, 3:]
    delta_solved = torch.cat([omega, t], -1)
    Sigma = torch.linalg.inv(H)
    ypr = pert[:, 3:6].to(dev); tgt = pert[:, :3].to(dev)
    delta_gt = torch.cat([ypr.flip(-1), tgt], -1)
    return delta_solved.cpu(), Sigma.cpu(), delta_gt.cpu()


def draw(ax, true_uvd, dist_uvd, per_pt, valid, k_show, title):
    idx = np.where(valid)[0][:k_show]
    for i in idx:
        xp = dist_uvd[i, :2]; gt = true_uvd[i, :2]; mu = xp + per_pt[i, :2]
        ax.plot(xp[0], xp[1], 'x', c='tab:red', ms=6, mew=1.5)
        ax.plot(gt[0], gt[1], 'o', mfc='none', mec='tab:green', ms=9, mew=1.5)
        ax.plot(mu[0], mu[1], '+', c='tab:blue', ms=8, mew=1.5)
        sx = math.exp(per_pt[i, 2]); sy = math.exp(per_pt[i, 3]); rho = math.tanh(per_pt[i, 4])
        cov = np.array([[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]])
        ev, evec = np.linalg.eigh(cov)
        ang = math.degrees(math.atan2(evec[1, 1], evec[0, 1]))
        for ns in (1, 2):
            ax.add_patch(Ellipse(mu, 2*ns*math.sqrt(max(ev[1], 1e-6)),
                                 2*ns*math.sqrt(max(ev[0], 1e-6)), angle=ang,
                                 fill=False, ec='tab:blue', alpha=0.5, lw=1))
    ax.set_title(title, fontsize=11); ax.set_aspect('equal'); ax.invert_yaxis()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt-a', required=True); p.add_argument('--label-a', default='A')
    p.add_argument('--ckpt-b', required=True); p.add_argument('--label-b', default='B')
    p.add_argument('--cache', required=True)
    p.add_argument('--out', default='sigma_compare.png')
    p.add_argument('--sample', type=int, default=0)
    p.add_argument('--k-show', type=int, default=14)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    ds = PandaSetCalibDatasetFull(args.cache, split='val', center_band=0.5, u_band=0.0,
                                  img_size=128, grid_n=16, min_crop_px=128, max_crop_px=512,
                                  max_rot_deg=1.0, max_offset_m=0.20, oversample=4,
                                  split_pert=False, pair_mode=False)
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2, collate_fn=collate_full)
    batch = next(iter(dl))
    true_uvd, dist_uvd, pad_mask = batch[1], batch[2], batch[3]
    s = args.sample; valid = (~pad_mask[s]).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=120)
    for ax, ck, lab in [(axes[0], args.ckpt_a, args.label_a), (axes[1], args.ckpt_b, args.label_b)]:
        m = load_model(ck, dev)
        per_pt = run(m, batch, dev)
        ds_, Sig, dg = gn_pose(per_pt, dist_uvd, pad_mask, batch, dev)
        # calibration over batch: mean |δ-δgt| vs predicted std sqrt(diag Σ_δ)
        e = (ds_ - dg)
        pred_std = torch.sqrt(torch.clamp(torch.diagonal(Sig, dim1=-2, dim2=-1), min=0))
        rot_err = e[:, :3].abs().mean().item(); t_err = e[:, 3:].abs().mean().item()
        rot_std = pred_std[:, :3].mean().item(); t_std = pred_std[:, 3:].mean().item()
        print(f'[{lab}] rot_err={rot_err:.3f}° (pred σ={rot_std:.3f}°)  '
              f't_err={t_err:.3f}m (pred σ={t_std:.3f}m)  '
              f'σ_px mean={math.exp(per_pt[s][valid][:,2].mean().item()):.2f}')
        draw(ax, true_uvd[s].numpy(), dist_uvd[s].numpy(), per_pt[s].numpy(), valid,
             args.k_show, f'{lab}: rot_err {rot_err:.3f}° t_err {t_err:.3f}m')
    fig.suptitle('× pert   ○ GT   + pred   ellipse=2σ info   |   left vs right = Σ re-distribution',
                 fontsize=12)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight'); print(f'saved {args.out}')


if __name__ == '__main__':
    main()
