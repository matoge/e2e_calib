"""TOY BA experiment with CalibNet2 (cnd2) as the (mu, W) head.

Setup:
  - Synthetic 3D points in camera frame
  - Project to 2D with noise + biased subset (= mimics real failure mode)
  - cnd2 predicts per-point (mu, Sigma_inv) given image (uniform) + uvd
  - Differentiable BA solves for pose using cnd2 outputs
  - Loss = SE(3) pose error vs ground truth
  - Gradient flows: pose_loss → BA solver → cnd2

Multi-frame extension is trivial: stack M frames worth of (mu, W), solve M poses
jointly in one BA call, gradient propagates through entire batch.
"""
import sys, os
sys.path.insert(0, '/home/hiro/git/e2e_calib')

import argparse
import math
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from models.calibnet2 import CalibNet2


def build_bucket_uvd(uvd, IW, IH, grid_n=8, K_per_cell=8):
    """Bin points into (G*G, K_per_cell, 4) frustum buckets for cnd2 KV.
    uvd: (B, N, 4) [u, v, d, intensity] in image local coords.
    Returns: bucket_uvd (B, G*G, K_per_cell, 4), bucket_valid (B, G*G, K_per_cell).
    """
    B, N, _ = uvd.shape
    G = grid_n
    cell_S_u = IW / G
    cell_S_v = IH / G
    cu = (uvd[..., 0] / cell_S_u).long().clamp(0, G - 1)
    cv = (uvd[..., 1] / cell_S_v).long().clamp(0, G - 1)
    cell_id = cv * G + cu                       # (B, N)
    bucket_uvd = torch.zeros(B, G*G, K_per_cell, 4, device=uvd.device, dtype=uvd.dtype)
    bucket_valid = torch.zeros(B, G*G, K_per_cell, dtype=torch.bool, device=uvd.device)
    for b in range(B):
        # within-cell slot assignment
        cid = cell_id[b]
        # stable sort by cell, then assign first K
        order = torch.argsort(cid, stable=True)
        sorted_uvd = uvd[b, order]
        sorted_cid = cid[order]
        # counts per cell
        counts = torch.zeros(G*G, dtype=torch.long, device=uvd.device)
        counts.scatter_add_(0, sorted_cid, torch.ones_like(sorted_cid))
        starts = torch.zeros(G*G + 1, dtype=torch.long, device=uvd.device)
        starts[1:] = counts.cumsum(0)
        intra = torch.arange(N, device=uvd.device) - starts[sorted_cid]
        keep = intra < K_per_cell
        slots = intra[keep]
        cells = sorted_cid[keep]
        bucket_uvd[b, cells, slots] = sorted_uvd[keep]
        bucket_valid[b, cells, slots] = True
    return bucket_uvd, bucket_valid


# ─────────────────────────────────────────────────────────────────────
# SE(3) utilities (axis-angle + translation)
# ─────────────────────────────────────────────────────────────────────

def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """xi: (B, 6) [tx, ty, tz, rx, ry, rz] → T: (B, 4, 4)."""
    B = xi.shape[0]
    t = xi[:, :3]
    omega = xi[:, 3:]
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    k = omega / theta
    K = skew(k)
    I = torch.eye(3, device=xi.device, dtype=xi.dtype).expand(B, 3, 3)
    R = I + torch.sin(theta).unsqueeze(-1) * K \
        + (1 - torch.cos(theta).unsqueeze(-1)) * (K @ K)
    T = torch.zeros(B, 4, 4, device=xi.device, dtype=xi.dtype)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    T[:, 3, 3] = 1.0
    return T


def skew(v):
    """v: (B, 3) → K: (B, 3, 3)."""
    B = v.shape[0]
    K = torch.zeros(B, 3, 3, device=v.device, dtype=v.dtype)
    K[:, 0, 1] = -v[:, 2]; K[:, 0, 2] =  v[:, 1]
    K[:, 1, 0] =  v[:, 2]; K[:, 1, 2] = -v[:, 0]
    K[:, 2, 0] = -v[:, 1]; K[:, 2, 1] =  v[:, 0]
    return K


def se3_log(T: torch.Tensor) -> torch.Tensor:
    """T: (B, 4, 4) → xi: (B, 6). Small-angle stable."""
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = ((tr - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta).clamp(min=1e-9)
    R_diff = R - R.transpose(-1, -2)
    omega = torch.stack([
        R_diff[:, 2, 1], R_diff[:, 0, 2], R_diff[:, 1, 0]
    ], dim=-1) * (theta / (2.0 * sin_theta)).unsqueeze(-1)
    return torch.cat([t, omega], dim=-1)


def se3_distance(T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
    """Return per-batch scalar SE(3) distance."""
    delta = torch.linalg.inv(T1) @ T2
    xi = se3_log(delta)
    return xi.norm(dim=-1)


# ─────────────────────────────────────────────────────────────────────
# TOY data: synthetic 3D points + projected observations with bias
# ─────────────────────────────────────────────────────────────────────

def make_toy_batch(B, N, n_biased=20, noise_std=0.5, bias_uv=(3.0, 0.0),
                   depth_range=(10.0, 100.0), fx=500.0, fy=500.0,
                   IW=128, IH=128, device='cuda', max_pose_perturb=0.1,
                   render_image=True, dot_radius=1, dot_intensity=1.0):
    """Create one TOY batch.

    Returns dict with:
      pts_cam      (B, N, 3)  3D points in GT camera frame
      uvd          (B, N, 4)  observed [u, v, depth, intensity] (cnd2 input)
      uv_obs       (B, N, 2)  observed uv (= ground truth projection + noise + bias)
      uv_clean     (B, N, 2)  noiseless projection (for diagnostics)
      bias_mask    (B, N)     1 = this point is biased
      K            (B, 3, 3)  intrinsics
      T_gt         (B, 4, 4)  ground truth pose (= identity by construction)
      T_init       (B, 4, 4)  perturbed init pose (= what BA starts from)
      img          (B, 3, H, W) dummy gray image
    """
    # 3D points in front of camera (camera at origin, looking +z)
    z = torch.rand(B, N, device=device) * (depth_range[1] - depth_range[0]) + depth_range[0]
    # x, y based on FOV
    u_rand = torch.rand(B, N, device=device) * IW
    v_rand = torch.rand(B, N, device=device) * IH
    cx, cy = IW / 2, IH / 2
    x = (u_rand - cx) * z / fx
    y = (v_rand - cy) * z / fy
    pts_cam = torch.stack([x, y, z], dim=-1)  # (B, N, 3)

    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]],
                     device=device).expand(B, 3, 3).contiguous()

    # GT pose = identity (= cam frame is world frame)
    T_gt = torch.eye(4, device=device).expand(B, 4, 4).contiguous()

    # Project (= GT projection)
    uv_clean = (pts_cam[..., :2] / pts_cam[..., 2:3]) * torch.stack(
        [K[:, 0, 0:1], K[:, 1, 1:2]], dim=-1) + torch.stack(
        [K[:, 0, 2:3], K[:, 1, 2:3]], dim=-1)
    # ↑ shape: (B, N, 2)

    # Inject noise
    uv_obs = uv_clean + torch.randn_like(uv_clean) * noise_std

    # Inject bias on subset
    bias_mask = torch.zeros(B, N, device=device)
    perm = torch.stack([torch.randperm(N, device=device) for _ in range(B)])
    bias_idx = perm[:, :n_biased]  # (B, n_biased)
    for b in range(B):
        uv_obs[b, bias_idx[b], 0] += bias_uv[0]
        uv_obs[b, bias_idx[b], 1] += bias_uv[1]
        bias_mask[b, bias_idx[b]] = 1.0

    # cnd2 input: distorted_uvd = observed [u, v, depth_norm, intensity]
    depth_norm = pts_cam[..., 2] / depth_range[1]   # 0-1 normalized
    intensity = torch.ones(B, N, device=device) * 0.5
    uvd = torch.stack([uv_obs[..., 0], uv_obs[..., 1], depth_norm, intensity], dim=-1)

    # Perturbed initial pose
    xi_init = torch.randn(B, 6, device=device) * max_pose_perturb
    T_init = se3_exp(xi_init)

    # Render image with bright dots at GT projection (= "what camera should see")
    if render_image:
        img = torch.full((B, 3, IH, IW), 0.0, device=device)
        u_int = uv_clean[..., 0].round().long().clamp(0, IW - 1)
        v_int = uv_clean[..., 1].round().long().clamp(0, IH - 1)
        for db in range(B):
            for dn in range(N):
                u0, v0 = u_int[db, dn].item(), v_int[db, dn].item()
                u_lo = max(0, u0 - dot_radius); u_hi = min(IW, u0 + dot_radius + 1)
                v_lo = max(0, v0 - dot_radius); v_hi = min(IH, v0 + dot_radius + 1)
                img[db, :, v_lo:v_hi, u_lo:u_hi] = dot_intensity
    else:
        img = torch.full((B, 3, IH, IW), 0.5, device=device)

    return {
        'pts_cam': pts_cam, 'uvd': uvd, 'uv_obs': uv_obs, 'uv_clean': uv_clean,
        'bias_mask': bias_mask, 'K': K, 'T_gt': T_gt, 'T_init': T_init,
        'img': img,
    }


# ─────────────────────────────────────────────────────────────────────
# cnd2 output → mu, Sigma_inv
# ─────────────────────────────────────────────────────────────────────

def cnd2_out_to_mu_W(out, sigma_floor=0.1, sigma_ceil=50.0):
    """Convert cnd2 (..., 5) → (mu (2D), Sigma_inv (2x2)).

    out fields: [du, dv, log_sx, log_sy, rho_raw]
      σ_x = clamp(exp(log_sx), floor, ceil)
      σ_y = clamp(exp(log_sy), floor, ceil)
      ρ = tanh(rho_raw)
      Σ = [[σ_x², ρσ_xσ_y], [ρσ_xσ_y, σ_y²]]
      Σ⁻¹ closed-form
    """
    mu = out[..., :2]
    sx = torch.exp(out[..., 2]).clamp(sigma_floor, sigma_ceil)
    sy = torch.exp(out[..., 3]).clamp(sigma_floor, sigma_ceil)
    rho = torch.tanh(out[..., 4]) * 0.95   # avoid singular
    a = sx * sx
    b = rho * sx * sy
    c = sy * sy
    det = a * c - b * b
    # Sigma_inv = [[c, -b], [-b, a]] / det
    Sigma_inv = torch.stack([
        torch.stack([c, -b], dim=-1),
        torch.stack([-b, a], dim=-1),
    ], dim=-2) / det.unsqueeze(-1).unsqueeze(-1)
    return mu, Sigma_inv


# ─────────────────────────────────────────────────────────────────────
# Differentiable BA: LM + Cholesky
# ─────────────────────────────────────────────────────────────────────

def project(pts_cam_in_T, K):
    """T_world_from_cam @ pts_world inversely transformed → 2D.
    Here pts_cam_in_T already in transformed coords. project pinhole."""
    fx = K[:, 0, 0:1]; fy = K[:, 1, 1:2]
    cx = K[:, 0, 2:3]; cy = K[:, 1, 2:3]
    z = pts_cam_in_T[..., 2].clamp(min=0.1)
    u = fx * pts_cam_in_T[..., 0] / z + cx
    v = fy * pts_cam_in_T[..., 1] / z + cy
    return torch.stack([u, v], dim=-1)


def projection_jacobian(pts_cam, K):
    """Jacobian ∂uv / ∂T_se3 evaluated at the current point.
    For a 3D point [x,y,z] in current camera frame, after T applied:
      pts' = R @ pts + t
      uv = K @ [x'/z', y'/z']
    Standard pinhole se3 jacobian. Output (B, N, 2, 6).
    Order matches se3_exp: [tx, ty, tz, rx, ry, rz].
    """
    fx = K[:, 0, 0]; fy = K[:, 1, 1]
    x = pts_cam[..., 0]; y = pts_cam[..., 1]; z = pts_cam[..., 2].clamp(min=0.1)
    z2 = z * z
    # ∂uv / ∂(x',y',z') @ ∂(x',y',z') / ∂xi
    # du/dx' = fx/z, du/dy' = 0, du/dz' = -fx x'/z²
    # dv/dx' = 0, dv/dy' = fy/z, dv/dz' = -fy y'/z²
    # ∂(x',y',z')/∂t = I
    # ∂(x',y',z')/∂omega ≈ -skew(pts) (small angle approximation)
    fx_e = fx.unsqueeze(-1)  # (B, 1)
    fy_e = fy.unsqueeze(-1)
    z_inv = 1.0 / z
    z2_inv = 1.0 / z2
    du_dt = torch.stack([fx_e * z_inv, torch.zeros_like(z), -fx_e * x * z2_inv], dim=-1)
    dv_dt = torch.stack([torch.zeros_like(z), fy_e * z_inv, -fy_e * y * z2_inv], dim=-1)
    # rotation part: ∂(x', y', z') / ∂omega = -skew(x', y', z')
    # so du/domega = du_dxyz @ (-skew(pts))
    # build skew-applied directly:
    # -skew(pts) @ ω rotates pts: easier to compute as cross product
    # For each point at (x,y,z): -skew = [[0, z, -y], [-z, 0, x], [y, -x, 0]]
    sxr = torch.stack([
        torch.stack([torch.zeros_like(z), z, -y], dim=-1),
        torch.stack([-z, torch.zeros_like(z), x], dim=-1),
        torch.stack([y, -x, torch.zeros_like(z)], dim=-1),
    ], dim=-2)  # (B, N, 3, 3)
    # du/domega = du_dxyz @ sxr
    du_dxyz = torch.stack([fx_e * z_inv, torch.zeros_like(z), -fx_e * x * z2_inv], dim=-1)
    dv_dxyz = torch.stack([torch.zeros_like(z), fy_e * z_inv, -fy_e * y * z2_inv], dim=-1)
    du_domega = (du_dxyz.unsqueeze(-2) @ sxr).squeeze(-2)
    dv_domega = (dv_dxyz.unsqueeze(-2) @ sxr).squeeze(-2)
    J_u = torch.cat([du_dt, du_domega], dim=-1)  # (B, N, 6)
    J_v = torch.cat([dv_dt, dv_domega], dim=-1)
    J = torch.stack([J_u, J_v], dim=-2)           # (B, N, 2, 6)
    # Sign correction: both translation and rotation columns are negated
    # because (1) ∂pts_cam/∂δt = -I (translation),
    # (2) sxr above was -skew(p_c) but ∂pts_cam/∂δω = +skew(p_c) (rotation).
    return -J


def lm_ba_step(T, pts_world, uv_obs, mu, Sigma_inv, K, lam):
    """One LM step. Returns updated T and new loss."""
    # Apply current T to pts (= pts_world to current cam frame)
    # pts_cam = T⁻¹ @ pts_world. For TOY where T_gt=I, the search is
    # finding T such that T⁻¹ @ pts ≈ pts_gt.
    # Equivalently: pts_cam = R^T(pts - t).
    R = T[:, :3, :3]
    t = T[:, :3, 3:]
    pts_cam = (pts_world.transpose(-1, -2) - t)
    pts_cam = (R.transpose(-1, -2) @ pts_cam).transpose(-1, -2)
    # Project
    uv_proj = project(pts_cam, K)
    # Residual (after applying network correction mu)
    r = (uv_obs + mu) - uv_proj   # (B, N, 2)
    # Jacobian
    J = projection_jacobian(pts_cam, K)  # (B, N, 2, 6)
    # Normal equation: H δT = b
    # H_i = J_i^T @ Σ_inv_i @ J_i
    # b_i = J_i^T @ Σ_inv_i @ r_i
    JT_Sinv = J.transpose(-1, -2) @ Sigma_inv         # (B, N, 6, 2)
    H_per = JT_Sinv @ J                                # (B, N, 6, 6)
    b_per = (JT_Sinv @ r.unsqueeze(-1)).squeeze(-1)    # (B, N, 6)
    H = H_per.sum(dim=1)                               # (B, 6, 6)
    b = b_per.sum(dim=1)                               # (B, 6)
    # LM damping
    H = H + lam.unsqueeze(-1).unsqueeze(-1) * torch.eye(6, device=H.device).expand_as(H)
    # Cholesky solve
    try:
        L = torch.linalg.cholesky(H)
        delta_xi = torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
    except torch._C._LinAlgError:
        # Fallback: regularize more aggressively
        H_reg = H + 1e-3 * torch.eye(6, device=H.device).expand_as(H)
        L = torch.linalg.cholesky(H_reg)
        delta_xi = torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
    # Update T: right-multiply δT (= local-frame perturbation, matches Jacobian)
    dT = se3_exp(delta_xi)
    T_new = T @ dT
    # Compute new loss (= sum of Mahalanobis residuals)
    R_new = T_new[:, :3, :3]
    t_new = T_new[:, :3, 3:]
    pts_cam_new = ((pts_world.transpose(-1, -2) - t_new))
    pts_cam_new = (R_new.transpose(-1, -2) @ pts_cam_new).transpose(-1, -2)
    uv_proj_new = project(pts_cam_new, K)
    r_new = (uv_obs + mu) - uv_proj_new
    loss_new = ((r_new.unsqueeze(-2) @ Sigma_inv @ r_new.unsqueeze(-1))
                ).squeeze(-1).squeeze(-1).sum(dim=1)
    return T_new, loss_new


def diff_ba(T_init, pts_world, uv_obs, mu, Sigma_inv, K,
            n_iter=8, lam0=1e-3, lam_up=2.0, lam_down=0.5,
            lam_min=1e-6, lam_max=1e3):
    """Differentiable LM-BA solver."""
    B = T_init.shape[0]
    T = T_init
    lam = torch.full((B,), lam0, device=T.device, dtype=T.dtype)
    # Initial loss
    R0 = T[:, :3, :3]
    t0 = T[:, :3, 3:]
    pts_cam0 = ((pts_world.transpose(-1, -2) - t0))
    pts_cam0 = (R0.transpose(-1, -2) @ pts_cam0).transpose(-1, -2)
    uv_proj0 = project(pts_cam0, K)
    r0 = (uv_obs + mu) - uv_proj0
    prev_loss = ((r0.unsqueeze(-2) @ Sigma_inv @ r0.unsqueeze(-1))
                 ).squeeze(-1).squeeze(-1).sum(dim=1)
    for it in range(n_iter):
        T_new, loss_new = lm_ba_step(T, pts_world, uv_obs, mu, Sigma_inv, K, lam)
        # Adaptive damping (per-batch)
        better = loss_new < prev_loss
        # If better, accept + decrease lam; else reject + increase lam
        T = torch.where(better.unsqueeze(-1).unsqueeze(-1), T_new, T)
        prev_loss = torch.where(better, loss_new, prev_loss)
        lam = torch.where(better, lam * lam_down, lam * lam_up)
        lam = lam.clamp(lam_min, lam_max)
    return T


# ─────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)

    model = CalibNet2(
        d=128, img_size=128, in_channels=3,
        use_intensity=True, frustum_grid_n=args.grid_n,
        n_iter=2, n_heads=4,
        d_scalar=8, n_type1=40, use_info_head=True,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log_dir = Path(args.out)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / 'log.txt'
    log = open(log_path, 'a')

    for step in range(args.steps):
        batch = make_toy_batch(
            args.batch, args.n_points, n_biased=args.n_biased,
            noise_std=args.noise_std, bias_uv=(args.bias_u, args.bias_v),
            device=device,
        )
        # Build frustum buckets from uvd
        bucket_uvd, bucket_valid = build_bucket_uvd(
            batch['uvd'], IW=128, IH=128, grid_n=args.grid_n, K_per_cell=8)
        out = model(batch['img'], batch['uvd'],
                    bucket_uvd=bucket_uvd, bucket_valid=bucket_valid)
        # out is (B, N, 5) or maybe a tuple — adapt
        if isinstance(out, tuple):
            out = out[0]
        mu, Sigma_inv = cnd2_out_to_mu_W(out)

        T_solved = diff_ba(
            batch['T_init'], batch['pts_cam'], batch['uv_obs'], mu, Sigma_inv, batch['K'],
            n_iter=args.ba_iter,
        )
        loss = se3_distance(T_solved, batch['T_gt']).mean()

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                # bias σ vs clean σ diagnostics
                sx = torch.exp(out[..., 2])
                sy = torch.exp(out[..., 3])
                sig_mean = ((sx + sy) / 2).detach()
                bm = batch['bias_mask']
                sig_bias = (sig_mean * bm).sum() / bm.sum().clamp(min=1)
                sig_clean = (sig_mean * (1 - bm)).sum() / (1 - bm).sum().clamp(min=1)
            msg = (f'[step {step:6d}] loss={loss.item():.4f}  '
                   f'σ_bias={sig_bias.item():.3f}  σ_clean={sig_clean.item():.3f}  '
                   f'ratio={sig_bias.item()/sig_clean.item():.2f}')
            print(msg, flush=True)
            log.write(msg + '\n'); log.flush()
        if step % args.save_every == 0 and step > 0:
            torch.save(model.state_dict(), log_dir / f'ckpt_{step:06d}.pt')
    log.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--n-points', type=int, default=100)
    p.add_argument('--n-biased', type=int, default=20)
    p.add_argument('--noise-std', type=float, default=0.5)
    p.add_argument('--bias-u', type=float, default=3.0)
    p.add_argument('--bias-v', type=float, default=0.0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--steps', type=int, default=5000)
    p.add_argument('--ba-iter', type=int, default=8)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', default='/home/hiro/git/e2e_calib/scripts/calib_e2e_toy/runs/toy_001')
    p.add_argument('--grid-n', type=int, default=8, help='frustum grid size for cnd2')
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
