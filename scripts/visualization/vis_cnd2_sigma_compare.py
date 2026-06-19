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
    """CANONICAL frame-safe GN (same as _ba_pose_loss): pts_cam_orig/duv_orig,
    float64. δ_pred = solve_pose(P0,−μ_orig,W); δ_gt = solve_pose(P0,−duv_orig,I);
    both parent-cam frame via the ONE module → no world/cam bug, no range/Z.
    Returns (δ_pred, Σ_δ=H⁻¹, δ_gt) on well-conditioned (PD) tiles only."""
    from scripts.ba.gn_pose import solve_pose
    pts0 = batch[8].to(dev).double(); duv_o = batch[9].to(dev).double()
    K = batch[10].to(dev).double(); cs_t = batch[11].to(dev)
    B, N, _ = dist_uvd.shape
    per_pt = per_pt.to(dev)
    Zc = pts0[..., 2]
    valid = (~pad_mask.to(dev)) & (Zc > 0.5)
    safe = torch.tensor([0., 0., 10.], device=dev, dtype=torch.float64)
    P0 = torch.where(valid.unsqueeze(-1), pts0, safe)
    s2o = (cs_t / 128.0).view(B, 1)
    mu_o = (per_pt[..., :2] * s2o.unsqueeze(-1)).double()
    sx = (per_pt[..., 2].exp().clamp(0.1, 50.0) * s2o).double()
    sy = (per_pt[..., 3].exp().clamp(0.1, 50.0) * s2o).double()
    rho = (per_pt[..., 4].tanh() * 0.95).double()
    W = make_info_from_sigma_rho(sx, sy, rho)
    Wi = make_info_from_sigma_rho(torch.ones(B, N, device=dev, dtype=torch.float64),
                                  torch.ones(B, N, device=dev, dtype=torch.float64),
                                  torch.zeros(B, N, device=dev, dtype=torch.float64))
    prior = torch.tensor([1/9., 1/9., 1/9., 1/0.09, 1/0.09, 1/0.09], device=dev, dtype=torch.float64)
    dpred, H = solve_pose(P0, -mu_o, W, K, valid=valid, n_iter=1, damping=1e-3, prior_diag=prior)
    dgt, _ = solve_pose(P0, -duv_o, Wi, K, valid=valid, n_iter=1, damping=1e-3, prior_diag=prior)
    Sigma = torch.linalg.inv(H)
    return dpred.cpu(), Sigma.cpu(), dgt.cpu()


def draw(ax, img, true_uvd, dist_uvd, per_pt, valid, k_show, title):
    # img: (3,S,S) tensor 0..1 → show the actual tile, overlay points in tile-px.
    im = img.permute(1, 2, 0).cpu().numpy()
    S = im.shape[0]
    ax.imshow(np.clip(im, 0, 1), extent=[0, S, S, 0])          # top-left origin
    idx = np.where(valid)[0]
    if len(idx) > k_show:
        idx = idx[np.linspace(0, len(idx) - 1, k_show).astype(int)]
    for i in idx:
        xp = dist_uvd[i, :2]; gt = true_uvd[i, :2]; mu = xp + per_pt[i, :2]
        ax.plot(xp[0], xp[1], 'x', c='red', ms=7, mew=2.0)
        ax.plot(gt[0], gt[1], 'o', mfc='none', mec='lime', ms=11, mew=2.0)
        ax.plot(mu[0], mu[1], '+', c='cyan', ms=9, mew=2.0)
        sx = math.exp(per_pt[i, 2]); sy = math.exp(per_pt[i, 3]); rho = math.tanh(per_pt[i, 4])
        cov = np.array([[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]])
        ev, evec = np.linalg.eigh(cov)
        ang = math.degrees(math.atan2(evec[1, 1], evec[0, 1]))
        ax.add_patch(Ellipse(mu, 2*math.sqrt(max(ev[1], 1e-6)),
                             2*math.sqrt(max(ev[0], 1e-6)), angle=ang,
                             fill=False, ec='cyan', alpha=0.85, lw=1.3))  # 1σ
    ax.set_xlim(0, S); ax.set_ylim(S, 0); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=10)


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
    imgs = batch[0]; true_uvd, dist_uvd, pad_mask = batch[1], batch[2], batch[3]
    img_s = imgs[args.sample].float() / 255.0                  # (3,S,S) for imshow
    s = args.sample; valid = (~pad_mask[s]).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=120)
    for ax, ck, lab in [(axes[0], args.ckpt_a, args.label_a), (axes[1], args.ckpt_b, args.label_b)]:
        m = load_model(ck, dev)
        per_pt = run(m, batch, dev)
        ds_, Sig, dg = gn_pose(per_pt, dist_uvd, pad_mask, batch, dev)
        # calibration over WELL-CONDITIONED tiles only (PD Σ_δ, finite): mean
        # |δ_pred-δ_gt| (actual pose err) vs predicted std sqrt(diag Σ_δ=H⁻¹).
        e = (ds_ - dg)
        diagS = torch.diagonal(Sig, dim1=-2, dim2=-1)
        okt = torch.isfinite(e).all(-1) & torch.isfinite(diagS).all(-1) & (diagS > 0).all(-1)
        e = e[okt]; pred_std = torch.sqrt(diagS[okt].clamp_min(0))
        rot_err = e[:, :3].abs().mean().item(); t_err = e[:, 3:].abs().mean().item()
        rot_std = pred_std[:, :3].mean().item(); t_std = pred_std[:, 3:].mean().item()
        # per-point σ (tile-local px) stats for THIS sample
        spx = torch.sqrt(per_pt[s][valid][:, 2].exp() * per_pt[s][valid][:, 3].exp())
        spx_med = float(spx.median())
        verdict = ('OVER-confident (IID)' if rot_std < 0.5 * rot_err
                   else 'calibrated' if rot_std < 2.0 * rot_err else 'under-confident')
        print(f'[{lab}] rot_err={rot_err:.3f}° predσ={rot_std:.3f}°  '
              f't_err={t_err:.3f}m predσ={t_std:.3f}m  σ_px med={spx_med:.2f}  → {verdict}')
        draw(ax, img_s, true_uvd[s].numpy(), dist_uvd[s].numpy(), per_pt[s].numpy(), valid,
             args.k_show,
             f'{lab}\nσ_px median={spx_med:.1f}px   pose σ_rot(pred)={rot_std:.3f}°')
    fig.suptitle('per-point 1σ ellipse on the tile (×pert ○GT +pred).  LEFT calib-'
                 'baseline: tight IID σ≈1px (over-counts → over-confident pose).  '
                 'RIGHT pose-supervised: σ INFLATED to the correlation-aware value.',
                 fontsize=10)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches='tight'); print(f'saved {args.out}')


if __name__ == '__main__':
    main()
