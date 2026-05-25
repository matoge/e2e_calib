"""Generate figures for docs/diffba/gn_roundtrip_pinhole_2dof.md.

Figures:
  fig1_convergence.png        — ‖δ_est − δ_gt‖∞ vs GN iteration (5 seeds)
  fig2_jacobian_field.png     — quiver: J_ωx, J_ωy at sampled points
  fig3_linear_quality.png     — true Δuv vs J·δ_gt across δ magnitudes
  fig4_covariance_match.png   — analytic σ vs MC σ scatter
  fig5_iteration_traces.png   — every iter's δ_est for one seed (path in 2D)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.ba.ba_torch import (
    project_pinhole, pinhole_jacobian, solve_pinhole,
    _apply_extrinsic, _K_with_delta,
)

torch.set_default_dtype(torch.float64)
OUT = REPO / 'docs' / 'assets' / 'diffba' / 'gn_roundtrip_pinhole_2dof'
OUT.mkdir(parents=True, exist_ok=True)

DOFS = ['omega_x', 'omega_y']


def make_cam_xyz(n, *, z_lo=2.0, z_hi=80.0, fov_half=0.6, seed=0):
    rng = np.random.RandomState(seed)
    z = np.exp(rng.uniform(np.log(z_lo), np.log(z_hi), n))
    xz = rng.uniform(-fov_half, fov_half, n)
    yz = rng.uniform(-fov_half, fov_half, n)
    return torch.tensor(np.stack([xz * z, yz * z, z], axis=-1), dtype=torch.float64)


def real_K(B=1):
    K = torch.zeros(B, 3, 3, dtype=torch.float64)
    K[:, 0, 0] = 900.0
    K[:, 1, 1] = 920.0
    K[:, 0, 2] = 512.0
    K[:, 1, 2] = 384.0
    K[:, 2, 2] = 1.0
    return K


def project_with_delta(P0, K, delta, dof_names):
    B = P0.shape[0]
    z = torch.zeros(B, dtype=P0.dtype)

    def g(name):
        return delta[..., dof_names.index(name)] if name in dof_names else z
    omega = torch.stack([g('omega_x'), g('omega_y'), g('omega_z')], dim=-1)
    t_v = torch.stack([g('tx'), g('ty'), g('tz')], dim=-1)
    K_pert = _K_with_delta(K, g('dfx'), g('dfy'), g('dcx'), g('dcy'))
    P_pert = _apply_extrinsic(P0, omega, t_v)
    return project_pinhole(P_pert, K_pert)


# ────────────────────────────────────────────────────────────────────
# fig1: convergence curves, 5 seeds × 6 iter
# ────────────────────────────────────────────────────────────────────

def fig1_convergence():
    print('[fig1] convergence curves')
    n_pts = 200
    K = real_K(1)
    valid = torch.ones(1, n_pts, dtype=torch.bool)
    W = torch.eye(2).expand(1, n_pts, 2, 2).contiguous()
    max_iter = 6

    fig, ax = plt.subplots(1, 1, figsize=(7, 4.6))
    for seed in range(5):
        P0 = make_cam_xyz(n_pts, seed=seed).unsqueeze(0)
        rng = np.random.RandomState(seed)
        delta_gt = torch.tensor(
            rng.uniform(-0.30, 0.30, len(DOFS)), dtype=torch.float64,
        ).unsqueeze(0)
        uv_truth = project_pinhole(P0, K)
        uv_pert = project_with_delta(P0, K, delta_gt, DOFS)
        duv = uv_pert - uv_truth
        z = P0[..., 2].clone()

        errs = []
        for ni in range(1, max_iter + 1):
            d_est, _ = solve_pinhole(uv_truth, duv, W, z, K, DOFS,
                                      valid=valid, n_iter=ni, damping=0.0)
            errs.append((d_est - delta_gt).abs().max().item())
        ax.semilogy(range(1, max_iter + 1), errs, '-o', ms=5,
                     label=f'seed {seed}  (‖δ_gt‖∞={delta_gt.abs().max().item():.3f}°)')

    ax.axhline(1e-12, color='gray', ls=':', label='float64 floor (1e-12)')
    ax.set_xlabel('GN iteration')
    ax.set_ylabel(r'$\|\hat\delta - \delta_{gt}\|_\infty$  [deg]')
    ax.set_title('Pinhole 2-DoF: noise-free pose roundtrip\n'
                 'iter 1: linearisation error  →  iter 2-3: machine precision')
    ax.grid(which='both', alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    p = OUT / 'fig1_convergence.png'
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {p}')


# ────────────────────────────────────────────────────────────────────
# fig2: Jacobian vector field — what each axis "looks like" in uv
# ────────────────────────────────────────────────────────────────────

def fig2_jacobian_field():
    print('[fig2] Jacobian flow field for ωx, ωy')
    # Place a grid of points across the image; pick Z so the points span
    # the whole frame, varying Z (3, 10, 50) so we see depth dependence
    K = real_K(1)
    fx, fy, cx, cy = 900.0, 920.0, 512.0, 384.0
    H, Wpx = 768, 1024
    pad = 80
    nu, nv = 9, 6
    us = np.linspace(pad, Wpx - pad, nu)
    vs = np.linspace(pad, H - pad, nv)
    UU, VV = np.meshgrid(us, vs, indexing='xy')

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    Zs = [3.0, 10.0, 50.0]
    for col, Z in enumerate(Zs):
        Xg = (UU - cx) * Z / fx
        Yg = (VV - cy) * Z / fy
        Zg = np.full_like(Xg, Z)
        Pg = torch.tensor(np.stack([Xg.ravel(), Yg.ravel(), Zg.ravel()], -1),
                          dtype=torch.float64).unsqueeze(0)
        uv = project_pinhole(Pg, K)
        Xc, Yc, Zc = Pg.unbind(-1)
        # pinhole_jacobian wants the broadcast-able tensors
        J = pinhole_jacobian(Xc, Yc, Zc, K, uv, DOFS)[0].numpy()  # (N, 2, 2)
        # Each column is the per-pixel response per 1° of motion
        for row, name in enumerate(DOFS):
            ax = axes[row, col]
            du = J[:, 0, row].reshape(nv, nu)
            dv = J[:, 1, row].reshape(nv, nu)
            mag = np.sqrt(du * du + dv * dv)
            ax.imshow(np.zeros((H, Wpx)), cmap='gray',
                       extent=[0, Wpx, H, 0], alpha=0.0)
            ax.quiver(UU, VV, du, dv, mag, cmap='viridis',
                       angles='xy', scale_units='xy',
                       scale=mag.max() / 70.0 if mag.max() > 0 else 1)
            ax.set_xlim(0, Wpx)
            ax.set_ylim(H, 0)
            ax.set_aspect('equal')
            if row == 0:
                ax.set_title(f'Z = {Z:.0f} m')
            if col == 0:
                ax.set_ylabel(f'∂uv / ∂{name}\n[px / deg]')
            ax.tick_params(labelbottom=(row == 1), labelleft=(col == 0))
            ax.grid(alpha=0.2)
    fig.suptitle('Pinhole 2-DoF Jacobian field — column = depth, '
                 'row = δ axis. Arrows: per-1° pixel motion.\n'
                 'Note: ωx (pitch) ≈ vertical flow + perspective '
                 'curvature; ωy (yaw) ≈ horizontal',
                 y=1.02)
    fig.tight_layout()
    p = OUT / 'fig2_jacobian_field.png'
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {p}')


# ────────────────────────────────────────────────────────────────────
# fig3: linearisation quality — when does the J·δ approximation break?
# ────────────────────────────────────────────────────────────────────

def fig3_linear_quality():
    print('[fig3] linearisation quality vs δ magnitude')
    n_pts = 200
    K = real_K(1)
    P0 = make_cam_xyz(n_pts, seed=0).unsqueeze(0)
    uv_truth = project_pinhole(P0, K)
    Xc, Yc, Zc = P0.unbind(-1)
    J = pinhole_jacobian(Xc, Yc, Zc, K, uv_truth, DOFS)[0]   # (N, 2, 2)

    deg_grid = np.geomspace(1e-4, 10.0, 30)
    rel_err = []
    abs_err = []
    for d_deg in deg_grid:
        delta = torch.tensor([[d_deg, 0.7 * d_deg]], dtype=torch.float64)
        uv_pert = project_with_delta(P0, K, delta, DOFS)
        duv_true = (uv_pert - uv_truth)[0]                   # (N, 2)
        duv_lin = torch.einsum('nij,j->ni', J, delta[0])     # J·δ
        e_abs = (duv_true - duv_lin).abs().max().item()
        e_rel = (e_abs / (duv_true.abs().max().item() + 1e-30))
        abs_err.append(e_abs)
        rel_err.append(e_rel)

    fig, ax1 = plt.subplots(1, 1, figsize=(7, 4.4))
    ax1.loglog(deg_grid, abs_err, '-o', color='tab:blue', label='abs err [px]')
    ax1.set_xlabel(r'$\|\delta\|$  [deg]')
    ax1.set_ylabel('max  | Δuv − J·δ |   [px]', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(which='both', alpha=0.3)
    ax2 = ax1.twinx()
    ax2.loglog(deg_grid, rel_err, '-s', color='tab:red',
                label='rel err  err / max|Δuv|')
    ax2.set_ylabel('rel error', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax1.axvline(0.30, color='tab:green', ls='--',
                 label=f'training pert (0.30°)')
    ax1.set_title('Pinhole 2-DoF: O(δ²) curvature\n'
                  'At δ=0.30° the linearisation misses ~10⁻⁵ of Δuv\n'
                  '— hence iter1 hits 1e-6 deg, iter2 hits floor')
    ax1.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    p = OUT / 'fig3_linear_quality.png'
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {p}')


# ────────────────────────────────────────────────────────────────────
# fig4: Cramér-Rao — analytic σ from H⁻¹ vs MC σ from 2000 trials
# ────────────────────────────────────────────────────────────────────

def fig4_covariance_match():
    print('[fig4] Cramér-Rao MC match')
    n_pts = 200
    n_trials = 2000
    sigma_pix = 1.0
    K = real_K(1)
    P0 = make_cam_xyz(n_pts, seed=11).unsqueeze(0)
    uv_truth = project_pinhole(P0, K)
    z = P0[..., 2].clone()
    valid = torch.ones(1, n_pts, dtype=torch.bool)
    W = torch.eye(2).expand(1, n_pts, 2, 2).contiguous() / sigma_pix ** 2

    Xc, Yc, Zc = P0.unbind(-1)
    J = pinhole_jacobian(Xc, Yc, Zc, K, uv_truth, DOFS)
    H = torch.einsum('bnik,bnij,bnjl->bkl', J, W, J)[0]
    Sigma_a = torch.linalg.inv(H)

    rng = np.random.RandomState(7)
    deltas = np.zeros((n_trials, len(DOFS)))
    for t in range(n_trials):
        noise = torch.tensor(rng.randn(1, n_pts, 2) * sigma_pix,
                              dtype=torch.float64)
        d_est, _ = solve_pinhole(uv_truth, noise, W, z, K, DOFS,
                                  valid=valid, n_iter=3, damping=0.0)
        deltas[t] = d_est[0].numpy()

    Sigma_mc = np.cov(deltas.T)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    # left: scatter of δ_est in 2D, with analytic and empirical 1-σ ellipses
    ax = axes[0]
    ax.scatter(deltas[:, 0], deltas[:, 1], s=4, alpha=0.3,
                color='tab:gray', label='2000 MC samples')
    # ellipse helper
    def ellipse(cov, color, label):
        ev, V = np.linalg.eigh(cov)
        ang = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        from matplotlib.patches import Ellipse
        e = Ellipse((0, 0), 2 * np.sqrt(ev[1]), 2 * np.sqrt(ev[0]),
                    angle=ang, fill=False, edgecolor=color, lw=2,
                    label=label)
        ax.add_patch(e)
    ellipse(Sigma_a.numpy(), 'tab:red', '1-σ analytic (H⁻¹)')
    ellipse(Sigma_mc, 'tab:blue', '1-σ empirical')
    ax.set_xlabel(r'$\delta_{\omega_x}$  [deg]')
    ax.set_ylabel(r'$\delta_{\omega_y}$  [deg]')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title('δ̂ scatter vs analytic Cramér-Rao bound\n'
                 'σ_pix = 1 px observation noise')

    # right: bar chart of σ ratios
    ax = axes[1]
    sa = np.sqrt(np.diag(Sigma_a.numpy()))
    sm = np.sqrt(np.diag(Sigma_mc))
    x = np.arange(len(DOFS))
    ax.bar(x - 0.18, sa * 1e3, width=0.35, color='tab:red',
            label=r'analytic $\sqrt{H^{-1}_{ii}}$')
    ax.bar(x + 0.18, sm * 1e3, width=0.35, color='tab:blue',
            label='MC stdev')
    for i in range(len(DOFS)):
        ax.text(i, max(sa[i], sm[i]) * 1e3 * 1.05,
                f'ratio={sm[i]/sa[i]:.3f}', ha='center', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(DOFS)
    ax.set_ylabel('σ  [10⁻³ deg]')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_title('Per-axis std-dev: analytic vs empirical')
    fig.tight_layout()
    p = OUT / 'fig4_covariance_match.png'
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {p}')


# ────────────────────────────────────────────────────────────────────
# fig5: per-iter δ trajectory in the (ωx, ωy) plane, 5 seeds
# ────────────────────────────────────────────────────────────────────

def fig5_iteration_traces():
    print('[fig5] iteration traces')
    n_pts = 200
    K = real_K(1)
    valid = torch.ones(1, n_pts, dtype=torch.bool)
    W = torch.eye(2).expand(1, n_pts, 2, 2).contiguous()
    max_iter = 5

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 6.0))
    for seed in range(5):
        P0 = make_cam_xyz(n_pts, seed=seed).unsqueeze(0)
        rng = np.random.RandomState(seed)
        delta_gt = torch.tensor(
            rng.uniform(-0.30, 0.30, len(DOFS)), dtype=torch.float64,
        ).unsqueeze(0)
        uv_truth = project_pinhole(P0, K)
        uv_pert = project_with_delta(P0, K, delta_gt, DOFS)
        duv = uv_pert - uv_truth
        z = P0[..., 2].clone()

        traj = [np.zeros(2)]
        for ni in range(1, max_iter + 1):
            d_est, _ = solve_pinhole(uv_truth, duv, W, z, K, DOFS,
                                      valid=valid, n_iter=ni, damping=0.0)
            traj.append(d_est[0].numpy())
        traj = np.array(traj)
        c = plt.cm.tab10(seed)
        ax.plot(traj[:, 0], traj[:, 1], '-o', color=c, ms=6,
                 label=f'seed {seed}')
        ax.scatter(delta_gt[0, 0], delta_gt[0, 1], color=c, marker='*',
                    s=180, edgecolors='black', linewidths=1, zorder=5)

    ax.set_xlabel(r'$\delta_{\omega_x}$  [deg]')
    ax.set_ylabel(r'$\delta_{\omega_y}$  [deg]')
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc='best')
    ax.set_title('GN trajectory  (○=iter, ★=δ_gt)\n'
                 '0 → iter1 covers 99.999% of the way; iter2+ vanish at this scale')
    fig.tight_layout()
    p = OUT / 'fig5_iteration_traces.png'
    fig.savefig(p, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {p}')


if __name__ == '__main__':
    fig1_convergence()
    fig2_jacobian_field()
    fig3_linear_quality()
    fig4_covariance_match()
    fig5_iteration_traces()
    print(f'\nAll figures → {OUT}')
