"""Model-free proof that range-as-Z unprojection floors the GN pose.

Uses PERFECT per-point correction μ = true_uv − dist_uv (no network), so the
ONLY error source is the unprojection geometry. Solves the 6-DoF GN pose with
(a) buggy: range used AS Z, (b) fixed: range→Z along the ray. Reports the
recovered pose error for both (buggy = the floor we saw; fixed ≈ 0) and
renders a per-point reprojection-residual overlay (× buggy, + fixed, ○ GT).
"""
import sys; sys.path.insert(0, '/home/hiro/git/e2e_calib')
import numpy as np, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
from scripts.ba.ba_torch import pinhole_jacobian, project_pinhole, gn_step, make_info_from_sigma_rho, _apply_extrinsic

CACHE = sys.argv[1] if len(sys.argv) > 1 else '/data'
OUT   = sys.argv[2] if len(sys.argv) > 2 else '/work/experiments/geom_proof.png'
dev = torch.device('cpu')

ds = PandaSetCalibDatasetFull(CACHE, split='val', center_band=0.5, u_band=0.0,
                              img_size=128, grid_n=16, min_crop_px=128, max_crop_px=512,
                              max_rot_deg=1.0, max_offset_m=0.20, oversample=4,
                              split_pert=False, pair_mode=False)
batch = next(iter(DataLoader(ds, batch_size=16, shuffle=True, num_workers=2, collate_fn=collate_full)))
true_uvd, dist_uvd, pad = batch[1], batch[2], batch[3]
K_orig, cs_t, u0_t, v0_t, pert = batch[10], batch[11], batch[13], batch[14], batch[7]
B, N, _ = dist_uvd.shape
valid = ~pad
S = 128.0; scale = S / cs_t
K = torch.zeros(B, 3, 3)
K[:, 0, 0] = K_orig[:, 0, 0] * scale; K[:, 1, 1] = K_orig[:, 1, 1] * scale
K[:, 0, 2] = (K_orig[:, 0, 2] - u0_t) * scale; K[:, 1, 2] = (K_orig[:, 1, 2] - v0_t) * scale
K[:, 2, 2] = 1.0
fx = K[..., 0, 0:1]; fy = K[..., 1, 1:2]; cx = K[..., 0, 2:3]; cy = K[..., 1, 2:3]
u, v = dist_uvd[..., 0], dist_uvd[..., 1]
r = (dist_uvd[..., 2] * 100.0)
safe_r = torch.where(valid, r, torch.full_like(r, 10.0)).clamp(min=0.5)
mu_perfect = true_uvd[..., :2] - dist_uvd[..., :2]      # GT correction, no network
uv_target = dist_uvd[..., :2] + mu_perfect             # == true_uv exactly
W = make_info_from_sigma_rho(torch.full((B, N), 2.0), torch.full((B, N), 2.0), torch.zeros(B, N))
dof = ('omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz')
prior = torch.tensor([1/9., 1/9., 1/9., 1/0.09, 1/0.09, 1/0.09])


def unproject(mode):
    if mode == 'buggy':       # range used AS Z
        Z = safe_r; X = (u - cx) * safe_r / fx; Y = (v - cy) * safe_r / fy
    else:                     # range -> Z along ray
        rx = (u - cx) / fx; ry = (v - cy) / fy
        Z = safe_r / torch.sqrt(rx*rx + ry*ry + 1.0); X = rx * Z; Y = ry * Z
    return torch.stack([X, Y, Z], -1)


def solve(pts_cam):
    omega = torch.zeros(B, 3); t = torch.zeros(B, 3)
    for _ in range(6):
        pts_t = _apply_extrinsic(pts_cam, omega, t)
        uv_proj = project_pinhole(pts_t, K)
        res = uv_target - uv_proj
        Xp, Yp, Zp = pts_t.unbind(-1)
        J = pinhole_jacobian(Xp, Yp, Zp, K, uv_proj.detach(), dof)
        d, _ = gn_step(J, W, res, valid=valid, damping=1e-3, prior_diag=prior)
        omega = omega + d[:, :3]; t = t + d[:, 3:]
    pts_t = _apply_extrinsic(pts_cam, omega, t)
    reproj = project_pinhole(pts_t, K)
    return omega, t, reproj


# GT pose target (axis-angle order = ypr.flip): the perturbation the GN should undo
omega_gt = pert[:, 3:6].flip(-1); t_gt = pert[:, :3]
fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), dpi=120)
for ax, mode in zip(axes, ['buggy', 'fixed']):
    om, t, reproj = solve(unproject(mode))
    rot_err = (om - omega_gt).abs().mean().item()
    t_err = (t - t_gt).abs().mean().item()
    # per-point reprojection residual after the recovered pose (should be ~0 if
    # geometry is exact, since μ is perfect)
    s = 0; vv = valid[s].numpy()
    gt = uv_target[s].numpy(); rp = reproj[s].detach().numpy()
    resid = np.linalg.norm((rp - gt)[vv], axis=-1)
    print(f'[{mode:5s}] PERFECT-μ pose: rot_err={rot_err:.3f}°  t_err={t_err:.3f}m  '
          f'reproj_resid mean={resid.mean():.2f}px max={resid.max():.2f}px')
    ax.plot(gt[vv, 0], gt[vv, 1], 'o', mfc='none', mec='green', ms=9, label='GT (=target)')
    ax.plot(rp[vv, 0], rp[vv, 1], '+', c='red', ms=9, mew=1.6,
            label='reproj @ recovered pose')
    for i in np.where(vv)[0]:
        ax.plot([gt[i, 0], rp[i, 0]], [gt[i, 1], rp[i, 1]], '-', c='red', alpha=0.4, lw=0.8)
    ax.set_title(f'{mode}: rot_err {rot_err:.3f}° t_err {t_err:.3f}m  '
                 f'resid {resid.mean():.2f}px', fontsize=12)
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.legend(fontsize=9)
fig.suptitle('PERFECT per-point μ → only geometry differs.  range-as-Z (buggy) '
             'cannot fit the pose (residual fans out) ; range→Z (fixed) ≈ exact', fontsize=12)
fig.tight_layout()
from pathlib import Path; Path(OUT).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches='tight'); print(f'saved {OUT}')
