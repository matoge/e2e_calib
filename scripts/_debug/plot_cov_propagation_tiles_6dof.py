"""6-DoF closed-form roundtrip per tile combo: (ω_x, ω_y, ω_z, t_x, t_y, t_z).

Compared to 4-DoF, adding **roll (ω_z)** and **t_z** changes the
separability story:

  J_ωz : ∂u/∂ω_z = −fx · Y/Z ,   ∂v/∂ω_z = +fy · X/Z      (signed in X & Y)
  J_tz : ∂u/∂t_z = −fx · X/Z² ,  ∂v/∂t_z = −fy · Y/Z²     (signed in X & Y)

Both depend on the **sign of X**. A single-side tile (left OR right
only) loses access to the opposite sign, so:

  - ω_z vs t_z gets harder to separate (both produce u-flow ∝ Y, just
    with different Z-scaling)
  - left+right (X signs paired) breaks the symmetry — recovery should
    converge with much smaller N than left-only or right-only.

Output: docs/assets/diffba/cov_propagation_tiles_6dof/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.ba.ba_torch import (
    project_pinhole, pinhole_jacobian,
)

torch.set_default_dtype(torch.float64)
OUT = REPO / 'docs' / 'assets' / 'diffba' / 'cov_propagation_tiles_6dof'
OUT.mkdir(parents=True, exist_ok=True)

DOFS = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
K_DOF = len(DOFS)

FX, FY, CX, CY = 900.0, 920.0, 512.0, 384.0
H_PX, W_PX = 768, 1024
TILES = {
    'left':   (50,  362, 100, 668),
    'center': (362, 662, 100, 668),
    'right':  (662, 974, 100, 668),
}
TILE_COLORS = {'left': 'tab:blue', 'center': 'tab:orange', 'right': 'tab:green'}


def real_K(B=1):
    K = torch.zeros(B, 3, 3, dtype=torch.float64)
    K[:, 0, 0] = FX
    K[:, 1, 1] = FY
    K[:, 0, 2] = CX
    K[:, 1, 2] = CY
    K[:, 2, 2] = 1.0
    return K


# ───── 6-DoF Σ_δ ground truth ─────────────────────────────────────────
SIG_OMEGA_X = 0.10
SIG_OMEGA_Y = 0.20
SIG_OMEGA_Z = 0.15
SIG_TX      = 0.05
SIG_TY      = 0.04
SIG_TZ      = 0.06


def make_sigma_gt():
    stds = np.array([SIG_OMEGA_X, SIG_OMEGA_Y, SIG_OMEGA_Z,
                      SIG_TX, SIG_TY, SIG_TZ])
    rng = np.random.RandomState(7)
    A = rng.randn(K_DOF, K_DOF)
    Q, _ = np.linalg.qr(A)
    D = np.diag(stds ** 2)
    Sigma = Q @ D @ Q.T
    return torch.tensor(Sigma, dtype=torch.float64)


# ───── anchor sampling ────────────────────────────────────────────────

def _sample_one_tile(tile_uv, N, *, mode='full', seed=0):
    u_lo, u_hi, v_lo, v_hi = tile_uv
    rng = np.random.RandomState(seed)
    u = rng.uniform(u_lo, u_hi, N)
    v = rng.uniform(v_lo, v_hi, N)
    if mode == 'single_Z':
        z = np.full(N, 10.0)
    elif mode == 'two_Z':
        z = np.where(rng.rand(N) > 0.5, 6.0, 30.0)
    elif mode == 'full':
        z = np.exp(rng.uniform(np.log(2.0), np.log(50.0), N))
    else:
        raise ValueError(mode)
    X = (u - CX) * z / FX
    Y = (v - CY) * z / FY
    return np.stack([X, Y, z], -1)


def sample_anchors(tile_or_list, N, *, mode='full', seed=0):
    if isinstance(tile_or_list, tuple) and len(tile_or_list) == 4 and \
            all(isinstance(x, (int, float)) for x in tile_or_list):
        arr = _sample_one_tile(tile_or_list, N, mode=mode, seed=seed)
    else:
        tiles = list(tile_or_list)
        T = len(tiles)
        per = [N // T] * T
        per[-1] += N - sum(per)
        parts = [_sample_one_tile(tiles[i], per[i], mode=mode,
                                    seed=seed + i * 13)
                 for i in range(T)]
        arr = np.concatenate(parts, axis=0)
    return torch.tensor(arr, dtype=torch.float64).unsqueeze(0)


def kl_normal(S1, S2):
    S1, S2 = np.asarray(S1), np.asarray(S2)
    d = S1.shape[0]
    inv2 = np.linalg.inv(S2)
    tr = np.trace(inv2 @ S1)
    logdet = np.linalg.slogdet(S2)[1] - np.linalg.slogdet(S1)[1]
    return 0.5 * (tr - d + logdet)


def sym_kl(S1, S2):
    return 0.5 * (kl_normal(S1, S2) + kl_normal(S2, S1))


# ───── sym-vec helpers ────────────────────────────────────────────────

def _sym_basis(K):
    Es = []
    for i in range(K):
        for j in range(i, K):
            E = np.zeros((K, K))
            if i == j:
                E[i, i] = 1.0
            else:
                E[i, j] = E[j, i] = 1.0
            Es.append(E)
    return Es


def _sym_unvec(v, K):
    M = np.zeros((K, K))
    idx = 0
    for i in range(K):
        for j in range(i, K):
            M[i, j] = M[j, i] = v[idx]
            idx += 1
    return M


def closed_form_pool(P0, K, Sigma_gt, ridge=0.0):
    uv = project_pinhole(P0, K)
    Xc, Yc, Zc = P0.unbind(-1)
    J = pinhole_jacobian(Xc, Yc, Zc, K, uv, DOFS)[0].numpy()       # (N, 2, K_DOF)
    Sigma_uv = np.einsum('nij,jk,nlk->nil', J, Sigma_gt.numpy(), J)
    Es = _sym_basis(K_DOF)                                          # M = K(K+1)/2 = 21
    M = len(Es)
    N_anch = J.shape[0]
    A = np.zeros((N_anch * 3, M))
    b = np.zeros(N_anch * 3)
    for i in range(N_anch):
        Ji = J[i]
        Suv = Sigma_uv[i]
        b[i * 3 + 0] = Suv[0, 0]
        b[i * 3 + 1] = Suv[0, 1]
        b[i * 3 + 2] = Suv[1, 1]
        for a, E in enumerate(Es):
            JEJ = Ji @ E @ Ji.T
            A[i * 3 + 0, a] = JEJ[0, 0]
            A[i * 3 + 1, a] = JEJ[0, 1]
            A[i * 3 + 2, a] = JEJ[1, 1]
    if ridge > 0:
        A = np.vstack([A, np.sqrt(ridge) * np.eye(M)])
        b = np.concatenate([b, np.zeros(M)])
    s = np.linalg.svd(A, compute_uv=False)
    condA = s[0] / max(s[-1], 1e-300)
    rank = int((s > 1e-10 * s[0]).sum())
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    Sigma_est = _sym_unvec(c, K_DOF)
    return torch.tensor(Sigma_est, dtype=torch.float64), condA, rank, M


# ───── fig1: per-anchor Σ_uv ellipses (6-DoF) ────────────────────────

def fig1_uv_ellipses():
    print('[fig1] per-anchor Σ_uv ellipses (6-DoF Σ_δ → Σ_uv)')
    Sigma_gt = make_sigma_gt()
    K = real_K(1)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.0), sharey=True)
    for ax, (name, tile_uv) in zip(axes, TILES.items()):
        u_lo, u_hi, v_lo, v_hi = tile_uv
        col = TILE_COLORS[name]
        rng = np.random.RandomState(11 + hash(name) % 10)
        gu = np.linspace(u_lo + 25, u_hi - 25, 5)
        gv = np.linspace(v_lo + 25, v_hi - 25, 4)
        UU, VV = np.meshgrid(gu, gv, indexing='xy')
        n_vis = 14
        Z = np.exp(rng.uniform(np.log(4.0), np.log(30.0), UU.size))[:n_vis]
        u_anch = UU.ravel()[:n_vis]
        v_anch = VV.ravel()[:n_vis]
        X = (u_anch - CX) * Z / FX
        Y = (v_anch - CY) * Z / FY
        P0 = torch.tensor(np.stack([X, Y, Z], -1),
                          dtype=torch.float64).unsqueeze(0)
        uv0 = project_pinhole(P0, K)[0].numpy()
        Xc, Yc, Zc = P0.unbind(-1)
        J = pinhole_jacobian(Xc, Yc, Zc, K, project_pinhole(P0, K),
                              DOFS)[0].numpy()
        Sigma_uv = np.einsum('nij,jk,nlk->nil', J, Sigma_gt.numpy(), J)

        ax.add_patch(Rectangle((u_lo, v_lo), u_hi - u_lo, v_hi - v_lo,
                                fill=False, ec=col, lw=2.0, ls='--'))
        for i in range(n_vis):
            ev, V_eig = np.linalg.eigh(Sigma_uv[i])
            ev = np.clip(ev, 0, None)
            ang = np.degrees(np.arctan2(V_eig[1, 1], V_eig[0, 1]))
            w = 4 * np.sqrt(ev[1])
            h = 4 * np.sqrt(ev[0])
            ax.add_patch(Ellipse((uv0[i, 0], uv0[i, 1]), w, h, angle=ang,
                                  fill=False, edgecolor=col, lw=1.4, alpha=0.9))
            ax.plot(uv0[i, 0], uv0[i, 1], 'k.', ms=3.5)
            ax.text(uv0[i, 0], uv0[i, 1] - 8,
                    f'{Z[i]:.0f}m', fontsize=6.5, ha='center', alpha=0.7)
        ax.set_xlim(u_lo - 20, u_hi + 20)
        ax.set_ylim(v_hi + 20, v_lo - 20)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.set_xlabel('u [px]')
        ax.set_title(f'tile = {name}', color=col, fontweight='bold')
    axes[0].set_ylabel('v [px]')
    fig.suptitle('6-DoF Σ_δ → per-anchor Σ_uv = J_i Σ_δ J_iᵀ.', y=1.02)
    fig.tight_layout()
    p = OUT / 'fig1_uv_scatter.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {p}')


# ───── fig2: combo recovery vs N at single_Z ──────────────────────────

def fig2_combo():
    print('[fig2] combo recovery (6-DoF, single_Z)')
    Sigma_gt = make_sigma_gt()
    Sgt_np = Sigma_gt.numpy()
    Sgt_norm = np.linalg.norm(Sgt_np)
    K = real_K(1)
    Ns = [8, 10, 14, 20, 30, 50, 80, 150, 300, 1000]
    configs = [
        ('left only',         [TILES['left']],                 'tab:blue',   '-o'),
        ('center only',       [TILES['center']],               'tab:orange', '-o'),
        ('right only',        [TILES['right']],                'tab:green',  '-o'),
        ('left + right',      [TILES['left'], TILES['right']], 'tab:purple', '-s'),
        ('left + center + right', list(TILES.values()),        'tab:red',    '-^'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(20, 7.5))
    ax_kl, ax_cond = axes
    for label, tiles, color, marker in configs:
        fros, conds = [], []
        for N in Ns:
            P0 = sample_anchors(tiles, N, mode='single_Z', seed=0)
            Sigma_est, condA, rank, M = closed_form_pool(P0, K, Sigma_gt)
            fro = np.linalg.norm(Sigma_est.numpy() - Sgt_np) / Sgt_norm
            fros.append(max(fro, 1e-17))
            conds.append(condA)
        ax_kl.loglog(Ns, fros, marker, color=color, ms=6, lw=1.8,
                      label=label)
        ax_cond.loglog(Ns, conds, marker, color=color, ms=5, lw=1.6,
                        label=label)
    ax_kl.axhline(1e-15, color='gray', ls=':', label='float64 floor')
    ax_kl.set_xlabel('total N anchors')
    ax_kl.set_ylabel(r'$\|\hat\Sigma_\delta - \Sigma_\delta\|_F / \|\Sigma_\delta\|_F$')
    ax_kl.grid(which='both', alpha=0.3)
    ax_kl.legend(fontsize=9, loc='upper right')
    ax_kl.set_title('6-DoF Σ_δ recovery — single_Z config\n'
                     'Roll (ω_z) and t_z BOTH need X-sign symmetry')
    ax_cond.set_xlabel('total N anchors')
    ax_cond.set_ylabel('cond(A)')
    ax_cond.grid(which='both', alpha=0.3)
    ax_cond.legend(fontsize=9, loc='upper right')
    ax_cond.set_title('Sym-vec design conditioning (M = 21 unknowns)\n'
                       'left+right and all-three have order(s) better cond')
    fig.suptitle(
        'Adding roll → left-only / right-only is no longer enough; '
        'X-sign symmetry (left+right) becomes a hard requirement for'
        ' fast recovery.', y=1.02,
    )
    fig.tight_layout()
    p = OUT / 'fig2_combo.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {p}')


# ───── fig3: per-tile-combo bar at large N ───────────────────────────

def fig3_summary(N=3000):
    print(f'[fig3] per-DoF σ summary at N={N} (full-Z)')
    Sigma_gt = make_sigma_gt()
    K = real_K(1)
    configs = [
        ('left',          [TILES['left']]),
        ('center',        [TILES['center']]),
        ('right',         [TILES['right']]),
        ('left+right',    [TILES['left'], TILES['right']]),
        ('all 3',         list(TILES.values())),
    ]
    results = []
    for name, tiles in configs:
        P0 = sample_anchors(tiles, N, mode='full', seed=0)
        Sigma_est, _, _, _ = closed_form_pool(P0, K, Sigma_gt)
        results.append((name, Sigma_est.numpy()))

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    units = ['deg', 'deg', 'deg', 'm', 'm', 'm']
    for ax, idx, axname, unit in zip(axes.flat, range(K_DOF), DOFS, units):
        gt = np.sqrt(Sigma_gt[idx, idx].item())
        labels = [r[0] for r in results]
        x = np.arange(len(labels))
        vals = [np.sqrt(r[1][idx, idx]) for r in results]
        ax.axhline(gt, color='black', lw=2, label=f'input σ={gt:.4f}')
        ax.bar(x, vals, width=0.55, color='tab:blue')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(f'{axname}  [{unit}]')
        ax.grid(axis='y', alpha=0.3)
        for i in range(len(labels)):
            if abs(gt) > 1e-12:
                ax.text(x[i], vals[i] * 1.02,
                        f'{vals[i]/gt:.4f}', ha='center', fontsize=8)
        if idx == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f'6-DoF closed-form roundtrip at N={N} (full-Z): '
                 'σ vs input across configurations.', y=1.02)
    fig.tight_layout()
    p = OUT / 'fig3_summary.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {p}')


def main():
    fig1_uv_ellipses()
    fig2_combo()
    fig3_summary(N=3000)
    print()
    print(f'M = K(K+1)/2 = {K_DOF * (K_DOF + 1) // 2} unknowns;'
          ' ⌈M/3⌉ = {} anchors minimum'.format(
              (K_DOF * (K_DOF + 1) // 2 + 2) // 3))
    print(f'\nAll figures → {OUT}')


if __name__ == '__main__':
    main()
