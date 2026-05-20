"""Torch / autograd-friendly closed-form BA solver.

GPU mirror of the numpy library:
    scripts/ba/ba_multicam_corr.py  (pinhole DOF_JAC + solve_dofs)
    scripts/ba/ba_kb_jac.py         (KB_DOF_JAC + solve_dofs_kb)

Designed to drop into a network's forward pass: gradients flow through
`torch.linalg.solve` (implicit-function theorem) so the per-point
information matrix W can be learned end-to-end via a pose loss.

Tensor shapes (B = batch, N = points per sample, K = len(dof_names)):
    uv      : (B, N, 2)    observation in pixel coords
    duv     : (B, N, 2)    Δuv from the network (uv_target = uv + duv)
    W       : (B, N, 2, 2) per-point info matrix (symmetric PSD)
    z       : (B, N)       cam-Z depth in metres
    valid   : (B, N) bool  per-point validity (zero contribution if False)
    K_int   : (B, 3, 3)    intrinsics
    dist    : (B, 4)       KB k1..k4

Returns:
    delta   : (B, K)       DoF correction (mean of the GN step)
    H       : (B, K, K)    information matrix at the final lin. point
                            (useful for multi-frame fusion: H_total = Σ_f H_f)
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
from torch import Tensor

_D2R = math.pi / 180.0


# ─── helpers ──────────────────────────────────────────────────────────

def make_info_from_sigma_rho(sigma_x: Tensor, sigma_y: Tensor,
                              rho: Tensor) -> Tensor:
    """Convert per-axis (σ_x, σ_y, ρ) ∈ (..., N) to W = Σ⁻¹ ∈ (..., N, 2, 2).

    For E2E learning, prefer to predict W directly via a Cholesky/lower-
    triangular parameterisation; this helper is for compatibility with the
    numpy-side `par[:, 2:5]` convention used in ba_multicam_corr.py."""
    det = sigma_x.pow(2) * sigma_y.pow(2) * (1.0 - rho.pow(2))
    W00 = sigma_y.pow(2) / det
    W11 = sigma_x.pow(2) / det
    W01 = -rho * sigma_x * sigma_y / det
    return torch.stack([
        torch.stack([W00, W01], dim=-1),
        torch.stack([W01, W11], dim=-1),
    ], dim=-2)


def _broadcast_intrinsics(K: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """K: (B, 3, 3) → (fx, fy, cx, cy) each shaped (B, 1) for broadcast
    against per-point (B, N) tensors."""
    fx = K[..., 0, 0].unsqueeze(-1)
    fy = K[..., 1, 1].unsqueeze(-1)
    cx = K[..., 0, 2].unsqueeze(-1)
    cy = K[..., 1, 2].unsqueeze(-1)
    return fx, fy, cx, cy


# ─── pinhole projection + Jacobian ────────────────────────────────────

def project_pinhole(P: Tensor, K: Tensor) -> Tensor:
    """P: (B, N, 3) cam-frame → uv: (B, N, 2). K: (B, 3, 3)."""
    fx, fy, cx, cy = _broadcast_intrinsics(K)
    X, Y, Z = P.unbind(-1)
    return torch.stack([fx * X / Z + cx, fy * Y / Z + cy], dim=-1)


def pinhole_jacobian(X: Tensor, Y: Tensor, Z: Tensor,
                      K: Tensor, uv: Tensor,
                      dof_names: Sequence[str]) -> Tensor:
    """Return J: (B, N, 2, K). Sign convention matches DOF_JAC in
    ba_multicam_corr.py — angles in DEGREES, translations in METERS,
    dfx/dfy as fractional (fx_new = fx · (1 + dfx))."""
    fx, fy, cx, cy = _broadcast_intrinsics(K)
    u, v = uv.unbind(-1)
    zero = torch.zeros_like(Z)
    one = torch.ones_like(Z)
    cols_u, cols_v = [], []
    for name in dof_names:
        if name == 'omega_x':
            ju = -(fx * X * Y) / (Z * Z) * _D2R
            jv = (-fy - (fy * Y * Y) / (Z * Z)) * _D2R
        elif name == 'omega_y':
            ju = (fx + (fx * X * X) / (Z * Z)) * _D2R
            jv = (fy * X * Y) / (Z * Z) * _D2R
        elif name == 'omega_z':
            ju = -fx * Y / Z * _D2R
            jv = fy * X / Z * _D2R
        elif name == 'tx':
            ju, jv = fx / Z * one, zero
        elif name == 'ty':
            ju, jv = zero, fy / Z * one
        elif name == 'tz':
            ju, jv = -fx * X / (Z * Z), -fy * Y / (Z * Z)
        elif name == 'dfx':
            ju, jv = (u - cx), zero
        elif name == 'dfy':
            ju, jv = zero, (v - cy)
        elif name == 'dcx':
            ju, jv = one, zero
        elif name == 'dcy':
            ju, jv = zero, one
        else:
            raise KeyError(f"unknown pinhole DoF '{name}'")
        cols_u.append(torch.broadcast_to(ju, Z.shape))
        cols_v.append(torch.broadcast_to(jv, Z.shape))
    Ju = torch.stack(cols_u, dim=-1)        # (B, N, K)
    Jv = torch.stack(cols_v, dim=-1)
    return torch.stack([Ju, Jv], dim=-2)    # (B, N, 2, K)


# ─── KB projection + Jacobian ─────────────────────────────────────────

def project_kb(P: Tensor, K: Tensor, dist: Tensor) -> Tensor:
    """P: (B, N, 3), K: (B, 3, 3), dist: (B, 4) → uv: (B, N, 2).

    Forward: r = √(X²+Y²); θ = atan2(r, Z); θ_d = θ·(1+k1θ²+...+k4θ⁸);
              u = fx·θ_d·X/r + cx; v = fy·θ_d·Y/r + cy.
    """
    fx, fy, cx, cy = _broadcast_intrinsics(K)
    X, Y, Z = P.unbind(-1)
    r = torch.sqrt(X * X + Y * Y + 1e-24)
    r_safe = torch.where(r > 1e-9, r, torch.ones_like(r))
    theta = torch.atan2(r, Z.clamp_min(1e-9))
    k1, k2, k3, k4 = dist[..., 0:1], dist[..., 1:2], dist[..., 2:3], dist[..., 3:4]
    t2 = theta * theta
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    poly = 1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8
    theta_d = theta * poly
    u = fx * theta_d * X / r_safe + cx
    v = fy * theta_d * Y / r_safe + cy
    return torch.stack([u, v], dim=-1)


def kb_jacobian(X: Tensor, Y: Tensor, Z: Tensor,
                 K: Tensor, dist: Tensor,
                 dof_names: Sequence[str]) -> Tensor:
    """Return J: (B, N, 2, K) for the Kannala-Brandt projection. Sign
    convention matches scripts/ba/ba_kb_jac.py:KB_DOF_JAC."""
    fx, fy, cx, cy = _broadcast_intrinsics(K)
    k1, k2, k3, k4 = dist[..., 0:1], dist[..., 1:2], dist[..., 2:3], dist[..., 3:4]
    r2 = X * X + Y * Y
    r = torch.sqrt(r2 + 1e-24)
    r_safe = torch.where(r > 1e-9, r, torch.ones_like(r))
    theta = torch.atan2(r, Z.clamp_min(1e-9))
    t2 = theta * theta
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    poly = 1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8
    theta_d = theta * poly
    dtd_dtheta = (1.0 + 3.0 * k1 * t2 + 5.0 * k2 * t4
                   + 7.0 * k3 * t6 + 9.0 * k4 * t8)
    r2pz2 = r2 + Z * Z + 1e-24
    inv_r = 1.0 / r_safe
    inv_r2 = inv_r * inv_r
    Xr = X * inv_r
    Yr = Y * inv_r

    def chain(dX, dY, dZ):
        dr = (X * dX + Y * dY) * inv_r
        dtheta = (Z * dr - r * dZ) / r2pz2
        dtd = dtd_dtheta * dtheta
        dXr = dX * inv_r - X * dr * inv_r2
        dYr = dY * inv_r - Y * dr * inv_r2
        du = fx * (dtd * Xr + theta_d * dXr)
        dv = fy * (dtd * Yr + theta_d * dYr)
        return du, dv

    zero = torch.zeros_like(Z)
    one = torch.ones_like(Z)
    cols_u, cols_v = [], []
    for name in dof_names:
        if name == 'omega_x':
            du, dv = chain(zero, -Z * _D2R, Y * _D2R)
        elif name == 'omega_y':
            du, dv = chain(Z * _D2R, zero, -X * _D2R)
        elif name == 'omega_z':
            du, dv = chain(-Y * _D2R, X * _D2R, zero)
        elif name == 'tx':
            du, dv = chain(one, zero, zero)
        elif name == 'ty':
            du, dv = chain(zero, one, zero)
        elif name == 'tz':
            du, dv = chain(zero, zero, one)
        elif name == 'dfx':
            du, dv = fx * theta_d * Xr, zero
        elif name == 'dfy':
            du, dv = zero, fy * theta_d * Yr
        elif name == 'dcx':
            du, dv = one, zero
        elif name == 'dcy':
            du, dv = zero, one
        else:
            raise KeyError(f"unknown KB DoF '{name}'")
        cols_u.append(torch.broadcast_to(du, Z.shape))
        cols_v.append(torch.broadcast_to(dv, Z.shape))
    Ju = torch.stack(cols_u, dim=-1)
    Jv = torch.stack(cols_v, dim=-1)
    return torch.stack([Ju, Jv], dim=-2)


# ─── Gauss-Newton step (model-agnostic) ───────────────────────────────

def gn_step(J: Tensor, W: Tensor, r: Tensor,
             valid: Tensor | None = None,
             damping: float = 0.0,
             ) -> Tuple[Tensor, Tensor]:
    """One Gauss-Newton step: δ = (JᵀWJ + λI)⁻¹ JᵀWr.

    J     : (B, N, 2, K)
    W     : (B, N, 2, 2)  (symmetric PSD; pre-zero invalid rows yourself
                            OR pass `valid` to do it here)
    r     : (B, N, 2)
    valid : (B, N) bool — False entries are zeroed before the reduction
    damping : Levenberg-Marquardt diagonal scale (added to H)

    Returns (delta, H) of shapes (B, K), (B, K, K).
    """
    if valid is not None:
        m = valid.to(J.dtype).unsqueeze(-1).unsqueeze(-1)   # (B, N, 1, 1)
        J = J * m
        r = r * m.squeeze(-1)
    # H[b,k,l] = Σ_n,i,j J[b,n,i,k] W[b,n,i,j] J[b,n,j,l]
    H = torch.einsum('bnik,bnij,bnjl->bkl', J, W, J)
    bvec = torch.einsum('bnik,bnij,bnj->bk', J, W, r)
    if damping > 0.0:
        K = H.shape[-1]
        eye = torch.eye(K, dtype=H.dtype, device=H.device).expand_as(H)
        H = H + damping * eye
    delta = torch.linalg.solve(H, bvec.unsqueeze(-1)).squeeze(-1)
    return delta, H


# ─── multi-step solvers (re-linearise at current δ) ────────────────────

def _apply_extrinsic(P: Tensor, omega_xyz_deg: Tensor, t_xyz: Tensor) -> Tensor:
    """P_new = R(ω) · P + t. ω in DEGREES, shape (B, 3); t in METERS, (B, 3).
    Uses Rodrigues with autograd-safe ops. P: (B, N, 3)."""
    theta = torch.linalg.vector_norm(omega_xyz_deg * _D2R, dim=-1, keepdim=True)
    # Avoid 0/0 when ω=0 — use small-angle limit (R = I + [ω]_x).
    safe = theta.clamp_min(1e-12)
    axis = (omega_xyz_deg * _D2R) / safe
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    ax, ay, az = axis.unbind(-1)
    zero = torch.zeros_like(ax)
    K_skew = torch.stack([
        torch.stack([zero, -az, ay], dim=-1),
        torch.stack([az, zero, -ax], dim=-1),
        torch.stack([-ay, ax, zero], dim=-1),
    ], dim=-2)                                                 # (B, 3, 3)
    eye3 = torch.eye(3, dtype=P.dtype, device=P.device).expand_as(K_skew)
    R = (eye3
         + sin_t.unsqueeze(-1) * K_skew
         + (1.0 - cos_t).unsqueeze(-1) * (K_skew @ K_skew))
    # Replace where theta was effectively 0 with identity-ish (above already
    # equals I when sin/cos is taken at 0 with K_skew small).
    P_rot = torch.einsum('bij,bnj->bni', R, P)
    return P_rot + t_xyz.unsqueeze(1)


def _K_with_delta(K: Tensor, dfx: Tensor, dfy: Tensor,
                   dcx: Tensor, dcy: Tensor) -> Tensor:
    """fx_new = fx · (1 + dfx); cx_new = cx + dcx; etc. K: (B, 3, 3),
    each delta: (B,)."""
    K_new = K.clone()
    K_new[..., 0, 0] = K[..., 0, 0] * (1.0 + dfx)
    K_new[..., 1, 1] = K[..., 1, 1] * (1.0 + dfy)
    K_new[..., 0, 2] = K[..., 0, 2] + dcx
    K_new[..., 1, 2] = K[..., 1, 2] + dcy
    return K_new


def _split_delta(delta: Tensor, dof_names: Sequence[str]) -> dict:
    """Slice (B, K) delta into named components, zero-fills missing keys."""
    out = {}
    B = delta.shape[0]
    z = torch.zeros(B, dtype=delta.dtype, device=delta.device)
    for nm in ('omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz',
               'dfx', 'dfy', 'dcx', 'dcy'):
        out[nm] = delta[..., dof_names.index(nm)] if nm in dof_names else z
    return out


def solve_pinhole(uv: Tensor, duv: Tensor, W: Tensor, z: Tensor,
                   K: Tensor, dof_names: Sequence[str],
                   *, valid: Tensor | None = None,
                   n_iter: int = 1, damping: float = 0.0
                   ) -> Tuple[Tensor, Tensor]:
    """Multi-step pinhole BA. Re-linearises at the current δ each step.

    Inputs/outputs described in the module docstring. `duv` follows the
    convention `uv_target = uv + duv` (network predicts the offset that
    should land the observation on truth). The estimate δ̂ is the
    correction to APPLY to (P_true, K) to land at the perturbed pose.
    """
    fx, fy, cx, cy = _broadcast_intrinsics(K)
    # Lift observation to truth-side cam-frame XYZ (pinhole back-projection).
    X0 = (uv[..., 0] - cx) * z / fx
    Y0 = (uv[..., 1] - cy) * z / fy
    Z0 = z
    P0 = torch.stack([X0, Y0, Z0], dim=-1)
    target_uv = uv + duv

    B, K_dim = X0.shape[0], len(dof_names)
    delta = torch.zeros(B, K_dim, dtype=uv.dtype, device=uv.device)
    H_last = None
    for _ in range(max(1, int(n_iter))):
        d = _split_delta(delta, dof_names)
        omega = torch.stack([d['omega_x'], d['omega_y'], d['omega_z']], dim=-1)
        t_v = torch.stack([d['tx'], d['ty'], d['tz']], dim=-1)
        P_lin = _apply_extrinsic(P0, omega, t_v)
        K_lin = _K_with_delta(K, d['dfx'], d['dfy'], d['dcx'], d['dcy'])
        uv_pred = project_pinhole(P_lin, K_lin)
        r = target_uv - uv_pred
        Xc, Yc, Zc = P_lin.unbind(-1)
        J = pinhole_jacobian(Xc, Yc, Zc, K_lin, uv_pred, dof_names)
        step, H_last = gn_step(J, W, r, valid=valid, damping=damping)
        delta = delta + step
    return delta, H_last


def solve_kb(uv: Tensor, duv: Tensor, W: Tensor, z: Tensor,
              K: Tensor, dist: Tensor, dof_names: Sequence[str],
              *, valid: Tensor | None = None,
              n_iter: int = 1, damping: float = 0.0
              ) -> Tuple[Tensor, Tensor]:
    """Multi-step KB BA. Same contract as `solve_pinhole` but with a
    fisheye projection model (`dist`: (B, 4) k1..k4)."""
    fx, fy, cx, cy = _broadcast_intrinsics(K)
    # Pinhole back-projection used as a depth anchor (good enough — the
    # perturbation acts on the projection, not the back-projection).
    X0 = (uv[..., 0] - cx) * z / fx
    Y0 = (uv[..., 1] - cy) * z / fy
    Z0 = z
    P0 = torch.stack([X0, Y0, Z0], dim=-1)
    target_uv = uv + duv

    B, K_dim = X0.shape[0], len(dof_names)
    delta = torch.zeros(B, K_dim, dtype=uv.dtype, device=uv.device)
    H_last = None
    for _ in range(max(1, int(n_iter))):
        d = _split_delta(delta, dof_names)
        omega = torch.stack([d['omega_x'], d['omega_y'], d['omega_z']], dim=-1)
        t_v = torch.stack([d['tx'], d['ty'], d['tz']], dim=-1)
        P_lin = _apply_extrinsic(P0, omega, t_v)
        K_lin = _K_with_delta(K, d['dfx'], d['dfy'], d['dcx'], d['dcy'])
        uv_pred = project_kb(P_lin, K_lin, dist)
        r = target_uv - uv_pred
        Xc, Yc, Zc = P_lin.unbind(-1)
        J = kb_jacobian(Xc, Yc, Zc, K_lin, dist, dof_names)
        step, H_last = gn_step(J, W, r, valid=valid, damping=damping)
        delta = delta + step
    return delta, H_last
