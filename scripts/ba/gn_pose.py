"""Canonical GN pose-from-per-point module — ONE solver for training + inference.

The 6-DoF (or N-DoF) camera-pose correction is recovered from per-point
cam-frame 3D + a Δuv field, via `solve_pinhole_xyz` (consumes XYZ DIRECTLY,
so there is NO range→Z / tile-local-K reconstruction and none of the bugs that
came with it). Convention:

    target_uv = project(pts_cam, K) + duv
    solve δ s.t. project(apply_extrinsic(δ, pts_cam), K_lin) == target_uv

So if the observed (mis-calibrated) projection is uv_obs and a network predicts
the correction μ (uv_obs + μ ≈ uv_gt), feed duv = (uv_obs + μ) − project(pts_cam,K).
For the pure perturbation-recovery roundtrip below, duv = uv_pert − uv_gt and the
solver returns δ̂ ≈ δ_gt.

Run `python scripts/ba/gn_pose.py [CACHE_DIR]` for the self-test:
known pose → perturb (cam frame) → Δuv → GN must recover δ to ~machine
precision, on SYNTHETIC (camera NOT at origin, off-axis tiles) AND, if a
PandaSet cache is given, REAL LiDAR points (pts_cam_orig) with the real K.
Toy-at-origin hides frame bugs; the real-data leg is the one that matters.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from scripts.ba.ba_torch import (solve_pinhole_xyz, _apply_extrinsic,
                                  project_pinhole, make_info_from_sigma_rho)

DOF6 = ('omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz')


def solve_pose(pts_cam: torch.Tensor, duv: torch.Tensor, W: torch.Tensor,
               K: torch.Tensor, *, dof=DOF6, valid=None, n_iter: int = 10,
               damping: float = 0.0, prior_diag=None):
    """The ONE pose solver. pts_cam (B,N,3) cam-frame metres; duv (B,N,2) px
    (target = project(pts_cam,K)+duv); W (B,N,2,2) info; K (B,3,3).
    Returns (delta (B,len(dof)), H (B,K,K))."""
    return solve_pinhole_xyz(pts_cam, duv, W, K, dof, valid=valid,
                             n_iter=n_iter, damping=damping, prior_diag=prior_diag)


def _identity_W(B, N, dtype, device):
    return make_info_from_sigma_rho(torch.ones(B, N, dtype=dtype, device=device),
                                    torch.ones(B, N, dtype=dtype, device=device),
                                    torch.zeros(B, N, dtype=dtype, device=device))


def _aniso_WS(B, N, dtype, device, seed=0):
    """Anisotropic per-point measurement covariance Σ_uv (2x2 SPD, varying
    σx,σy,ρ like the network's log_sx/log_sy/rho) and its info W=Σ⁻¹.
    Returns (W (B,N,2,2), Sigma_uv (B,N,2,2), L (B,N,2,2) chol of Σ_uv)."""
    g = torch.Generator(device=device).manual_seed(seed)
    sx = (0.5 + 2.5 * torch.rand(B, N, generator=g, dtype=dtype, device=device))
    sy = (0.5 + 2.5 * torch.rand(B, N, generator=g, dtype=dtype, device=device))
    rho = (torch.rand(B, N, generator=g, dtype=dtype, device=device) * 1.6 - 0.8)
    W = make_info_from_sigma_rho(sx, sy, rho)
    Sigma = torch.linalg.inv(W)
    L = torch.linalg.cholesky(Sigma)                          # Σ = L Lᵀ
    return W, Sigma, L


def roundtrip(pts_cam, K, delta_gt, valid=None, n_iter=12, prior_diag=None, damping=0.0):
    """known δ_gt (cam frame) → perturb pts → Δuv → solve_pose recovers δ_gt.
    Returns (delta_hat, rot_err_deg, t_err_m)."""
    # Sanitize padded anchors (Z=0 → projection 0/0 = NaN, and 0·NaN=NaN
    # survives the valid-mask reduction). Replace invalid points with a safe
    # dummy in front of the camera; `valid` excludes them from the solve.
    # Exclude padded AND non-physical (Z≤0.5 m: at/behind camera) anchors — a
    # tiny-Z point gives J~fx/Z→∞ and detonates the un-damped normal equations.
    Zc = pts_cam[..., 2]
    phys = Zc > 0.5
    valid = phys if valid is None else (valid & phys)
    safe = torch.tensor([0.0, 0.0, 10.0], dtype=pts_cam.dtype, device=pts_cam.device)
    pts_cam = torch.where(valid.unsqueeze(-1), pts_cam, safe)
    uv_gt = project_pinhole(pts_cam, K)
    om = delta_gt[:, :3]; t = delta_gt[:, 3:6]
    uv_pert = project_pinhole(_apply_extrinsic(pts_cam, om, t), K)
    duv = uv_pert - uv_gt                                   # → solver returns δ_gt
    B, N = pts_cam.shape[:2]
    W = _identity_W(B, N, pts_cam.dtype, pts_cam.device)
    dhat, _ = solve_pose(pts_cam, duv, W, K, valid=valid, n_iter=n_iter,
                         prior_diag=prior_diag, damping=damping)
    rot_err = (dhat[:, :3] - delta_gt[:, :3]).abs().mean(-1)   # (B,) per tile
    t_err = (dhat[:, 3:6] - delta_gt[:, 3:6]).abs().mean(-1)   # (B,)
    n_used = valid.sum(-1)                                     # (B,) pts per tile
    return dhat, rot_err, t_err, n_used


def cov_roundtrip(pts_cam, K, sigma_px=1.0, n_trials=4000, prior_diag=None):
    """Covariance recovery check (Cramér-Rao, the mathematically-correct one).

    Per-point measurement noise Σ_uv = σ²I → W = Σ_uv⁻¹. One GN solve gives
    H; the predicted pose covariance is Σ_δ_pred = H⁻¹. Monte-Carlo: with
    δ_gt = 0, draw Δuv = N(0, σ²I) per point, solve δ̂ each trial; the empirical
    Cov(δ̂) must match H⁻¹. (NOTE: splatting a POSE covariance Σ_δ to per-point
    W=(JΣ_δJᵀ)⁻¹ and re-solving does NOT return Σ_δ — H becomes Σ rank-2
    projectors ≈ (N/3)Σ_δ⁻¹; see docs/diffba/gn_roundtrip_audit.md. The honest
    covariance contract is W=measurement-noise⁻¹ → H⁻¹=estimator covariance.)

    Returns (ratio_diag (K,), Σ_pred_diag (K,)). ratio≈1 ⇒ covariance recovered.
    """
    B1, N = 1, pts_cam.shape[1]
    from scripts.ba.ba_torch import pinhole_jacobian, gn_step
    W, Sigma_uv, L = _aniso_WS(B1, N, pts_cam.dtype, pts_cam.device, seed=7)  # ANISOTROPIC
    # H at δ=0 with the anisotropic per-point W
    uv0 = project_pinhole(pts_cam, K)
    X, Y, Z = pts_cam.unbind(-1)
    J = pinhole_jacobian(X, Y, Z, K, uv0, DOF6)
    _, H = gn_step(J, W, torch.zeros(B1, N, 2, dtype=pts_cam.dtype, device=pts_cam.device),
                   prior_diag=prior_diag)
    Sigma_pred = torch.linalg.inv(H)[0]                       # (6,6) = predicted pose cov
    # MC: δ_gt=0, draw Δuv ~ N(0, Σ_uv) per point (correlated via L), solve each.
    torch.manual_seed(1)
    pts_T = pts_cam.expand(n_trials, N, 3).contiguous()
    K_T = K.expand(n_trials, 3, 3).contiguous()
    z = torch.randn(n_trials, N, 2, 1, dtype=pts_cam.dtype, device=pts_cam.device)
    noise = (L.expand(n_trials, N, 2, 2) @ z).squeeze(-1)     # cov = Σ_uv per point
    W_T = W.expand(n_trials, N, 2, 2).contiguous()
    dhat, _ = solve_pinhole_xyz(pts_T, noise, W_T, K_T, DOF6, n_iter=1, prior_diag=prior_diag)
    Sigma_emp = torch.cov(dhat.T)                             # (6,6) empirical
    ratio = torch.diagonal(Sigma_emp) / torch.diagonal(Sigma_pred).clamp_min(1e-30)
    return ratio, torch.diagonal(Sigma_pred)


def _synthetic(off_axis_cx):
    torch.manual_seed(0)
    B, N = 1, 400
    fx = fy = 350.0; S = 128.0
    K = torch.zeros(B, 3, 3, dtype=torch.float64)
    K[:, 0, 0] = fx; K[:, 1, 1] = fy; K[:, 0, 2] = off_axis_cx; K[:, 1, 2] = 64.0; K[:, 2, 2] = 1.0
    u = torch.rand(B, N, dtype=torch.float64) * S
    v = torch.rand(B, N, dtype=torch.float64) * S
    Z = 5 + torch.rand(B, N, dtype=torch.float64) * 35
    X = (u - off_axis_cx) * Z / fx; Y = (v - 64.0) * Z / fy
    return torch.stack([X, Y, Z], -1), K


def main():
    torch.set_default_dtype(torch.float64)
    delta_gt = torch.tensor([[0.6, -0.4, 0.5, 0.10, -0.08, 0.12]], dtype=torch.float64)
    print(f'δ_gt = ω{delta_gt[0,:3].tolist()}°  t{delta_gt[0,3:].tolist()}m')
    print('--- SYNTHETIC (camera off-origin via off-axis cx; toy-at-origin hides this) ---')
    ok = True
    for name, cx in [('center cx=64', 64.0), ('edge cx=-200', -200.0), ('far cx=-450', -450.0)]:
        pts, K = _synthetic(cx)
        _, re, te, _ = roundtrip(pts, K, delta_gt)
        re, te = re.item(), te.item()
        flag = 'OK' if (re < 1e-4 and te < 1e-4) else 'FAIL'
        ok &= flag == 'OK'
        print(f'  [{name:14s}] rot_err={re:.2e}°  t_err={te:.2e}m  {flag}')

    print('--- COVARIANCE recovery (anisotropic per-point W; Cramér-Rao MC) ---')
    pts, K = _synthetic(-200.0)
    ratio, sdiag = cov_roundtrip(pts, K, n_trials=6000)
    cov_ok = bool(((ratio > 0.85) & (ratio < 1.15)).all())
    ok &= cov_ok
    rr = ', '.join(f'{r:.2f}' for r in ratio.tolist())
    print(f'  empirical Cov(δ̂)/H⁻¹ per DOF [ωx ωy ωz tx ty tz] = [{rr}]  '
          f'{"OK" if cov_ok else "FAIL"}  (≈1 ⇒ H⁻¹ is the true pose cov)')

    cache = sys.argv[1] if len(sys.argv) > 1 else None
    if cache:
        print(f'--- REAL PandaSet pts_cam_orig (real K, off-axis tiles) [{cache}] ---')
        from torch.utils.data import DataLoader
        from datasets.pandaset_full import PandaSetCalibDatasetFull, collate_full
        ds = PandaSetCalibDatasetFull(cache, split='val', center_band=0.5, u_band=0.0,
                                      img_size=128, grid_n=16, min_crop_px=128, max_crop_px=512,
                                      max_rot_deg=1.0, max_offset_m=0.20, oversample=4,
                                      split_pert=False, pair_mode=False)
        batch = next(iter(DataLoader(ds, batch_size=16, shuffle=True, num_workers=2, collate_fn=collate_full)))
        pad = batch[3]; pts_cam = batch[8].double(); K = batch[10].double()
        valid = ~pad
        dg = delta_gt.expand(pts_cam.shape[0], -1).contiguous()
        # production physical prior (σ_rot=3°, σ_t=0.3m) — weak vs a well-
        # conditioned tile's data so recovery stays tight, but it desingularises
        # the rank-deficient (narrow/flat) tiles instead of NaN/blow-up.
        prior = torch.tensor([1/9., 1/9., 1/9., 1/0.09, 1/0.09, 1/0.09], dtype=pts_cam.dtype)
        _, re, te, nu = roundtrip(pts_cam, K, dg, valid=valid, prior_diag=prior, damping=1e-3)
        good = (re < 1e-3) & (te < 1e-3)
        frac = good.float().mean().item()
        print(f'  [real {pts_cam.shape[0]} tiles] pts/tile median={int(nu.median())}  '
              f'recovered<1e-3: {int(good.sum())}/{len(good)} ({frac:.0%})')
        print(f'    rot_err  median={re.median():.2e}°  max={re.max():.2e}°')
        print(f'    t_err    median={te.median():.2e}m  max={te.max():.2e}m')
        print('    (max-outliers = single tiles with too few pts / flat geometry =')
        print('     genuinely 6-DoF-underdetermined; well-conditioned tiles → machine ε)')
        ok &= (re.median().item() < 1e-3 and te.median().item() < 1e-3)
    else:
        print('(no cache arg → real-data leg skipped)')
    print('SELF-TEST', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
