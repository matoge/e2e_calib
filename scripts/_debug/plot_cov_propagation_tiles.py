"""Closed-form Σ_δ → Σ_uv → Σ_δ̂ roundtrip per tile.

No Monte Carlo. Σ_δ propagates through the per-anchor Jacobian as
    Σ_uv,i = J_i Σ_δ J_iᵀ
and is recovered exactly by either
  (A) per-anchor inversion :  Σ_δ = J_i⁻¹ Σ_uv,i J_i⁻ᵀ
  (B) N-point information pool :  Σ_δ = N · ( Σ_i J_iᵀ Σ_uv,i⁻¹ J_i )⁻¹

Both should hit float64 floor (~1e-16) regardless of N or tile location.

Figures:
  fig1 — per-anchor 2σ Σ_uv ellipses on the image plane, plus a zoomed
         single-anchor panel that shows the *deterministic* analytic ellipse
         and the principal-axis numbers.
  fig2 — sym-KL between recovered Σ_δ̂ and input Σ_δ as a function of N
         (1 → 1500), one curve per tile. Should be flat at ~1e-16.
  fig3 — per-tile bar chart of (σ_ωx, σ_ωy, ρ) — input vs recipe-A vs
         recipe-B at N=1500. All bars should be visually identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.ba.ba_torch import (
    project_pinhole, pinhole_jacobian,
)

torch.set_default_dtype(torch.float64)
OUT = REPO / 'docs' / 'assets' / 'diffba' / 'cov_propagation_tiles'
OUT.mkdir(parents=True, exist_ok=True)

DOFS = ['omega_x', 'omega_y']

# ───── Camera / image setup ───────────────────────────────────────────
FX, FY, CX, CY = 900.0, 920.0, 512.0, 384.0
H_PX, W_PX = 768, 1024


def real_K(B=1):
    K = torch.zeros(B, 3, 3, dtype=torch.float64)
    K[:, 0, 0] = FX
    K[:, 1, 1] = FY
    K[:, 0, 2] = CX
    K[:, 1, 2] = CY
    K[:, 2, 2] = 1.0
    return K


# ───── Tiles ──────────────────────────────────────────────────────────
TILES = {
    'left':   (50,  362, 100, 668),
    'center': (362, 662, 100, 668),
    'right':  (662, 974, 100, 668),
}
TILE_COLORS = {'left': 'tab:blue', 'center': 'tab:orange', 'right': 'tab:green'}


def sample_tile_anchors(tile_uv, *, n=400, z_lo=2.0, z_hi=50.0, seed=0):
    u_lo, u_hi, v_lo, v_hi = tile_uv
    rng = np.random.RandomState(seed)
    u = rng.uniform(u_lo, u_hi, n)
    v = rng.uniform(v_lo, v_hi, n)
    z = np.exp(rng.uniform(np.log(z_lo), np.log(z_hi), n))
    X = (u - CX) * z / FX
    Y = (v - CY) * z / FY
    P = torch.tensor(np.stack([X, Y, z], -1), dtype=torch.float64).unsqueeze(0)
    return P


# ───── Σ_δ ground truth ───────────────────────────────────────────────
SIGMA_WX = 0.10        # deg
SIGMA_WY = 0.20        # deg
RHO      = 0.5


def make_sigma_gt():
    sx, sy, rho = SIGMA_WX, SIGMA_WY, RHO
    return torch.tensor(
        [[sx * sx,        rho * sx * sy],
         [rho * sx * sy,  sy * sy      ]],
        dtype=torch.float64,
    )


def ellipse_from_cov(cov2x2, *, n_sigma=1.0):
    ev, V = np.linalg.eigh(cov2x2)
    ev = np.clip(ev, 0, None)
    angle = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
    w = 2 * n_sigma * np.sqrt(ev[1])
    h = 2 * n_sigma * np.sqrt(ev[0])
    return w, h, angle


def kl_normal(S1, S2):
    """KL( N(0, S1) || N(0, S2) ) for d×d SPD matrices."""
    S1, S2 = np.asarray(S1), np.asarray(S2)
    d = S1.shape[0]
    inv2 = np.linalg.inv(S2)
    tr = np.trace(inv2 @ S1)
    logdet = np.linalg.slogdet(S2)[1] - np.linalg.slogdet(S1)[1]
    return 0.5 * (tr - d + logdet)


def sym_kl(S1, S2):
    return 0.5 * (kl_normal(S1, S2) + kl_normal(S2, S1))


# ───── core closed-form roundtrip ─────────────────────────────────────

def closed_form_roundtrip(P0, K, Sigma_gt):
    """Returns (Sigma_uv_per_anchor, Sigma_est_A, Sigma_est_B,
                Sigma_est_each).

    A: average of per-anchor exact inversions
    B: N · (Σ_i Jᵀ Σ_uv,i⁻¹ J)⁻¹
    """
    uv = project_pinhole(P0, K)
    Xc, Yc, Zc = P0.unbind(-1)
    J = pinhole_jacobian(Xc, Yc, Zc, K, uv, DOFS)[0]              # (N, 2, 2)
    Sigma_uv = torch.einsum('nij,jk,nlk->nil', J, Sigma_gt, J)    # (N, 2, 2)
    # (A) per-anchor exact inversion
    J_inv = torch.linalg.inv(J)
    Sigma_each = torch.einsum('nij,njk,nlk->nil', J_inv, Sigma_uv, J_inv)
    Sigma_A = Sigma_each.mean(dim=0)
    # (B) information pool
    W = torch.linalg.inv(Sigma_uv)
    H = torch.einsum('nij,nik,nkl->jl', J, W, J)
    N = P0.shape[1]
    Sigma_B = N * torch.linalg.inv(H)
    return Sigma_uv, Sigma_A, Sigma_B, Sigma_each, J


# ───── fig1: per-anchor Σ_uv on image (deterministic ellipses) ──────

def fig1_uv_ellipses():
    print('[fig1] per-anchor Σ_uv ellipses (deterministic)')
    Sigma_gt = make_sigma_gt()
    K = real_K(1)

    fig, axes = plt.subplots(2, 3, figsize=(17, 11.0),
                              gridspec_kw={'height_ratios': [1.4, 1.0]})

    for col_i, (name, tile_uv) in enumerate(TILES.items()):
        ax = axes[0, col_i]
        u_lo, u_hi, v_lo, v_hi = tile_uv
        col = TILE_COLORS[name]

        # sparse 5x4 grid inside the tile, anchor depth varies
        nu, nv = 5, 4
        gu = np.linspace(u_lo + 25, u_hi - 25, nu)
        gv = np.linspace(v_lo + 25, v_hi - 25, nv)
        UU, VV = np.meshgrid(gu, gv, indexing='xy')
        rng = np.random.RandomState(11 + hash(name) % 10)
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
            w, h, ang = ellipse_from_cov(Sigma_uv[i], n_sigma=2.0)
            ax.add_patch(Ellipse((uv0[i, 0], uv0[i, 1]), w, h, angle=ang,
                                  fill=False, edgecolor=col, lw=1.4, alpha=0.9))
            ax.plot(uv0[i, 0], uv0[i, 1], 'k.', ms=3.5)
            ax.text(uv0[i, 0], uv0[i, 1] - 6,
                    f'{Z[i]:.0f}m', fontsize=6.5, ha='center',
                    color='black', alpha=0.7)
        ax.set_xlim(u_lo - 20, u_hi + 20)
        ax.set_ylim(v_hi + 20, v_lo - 20)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.set_xlabel('u [px]')
        ax.set_title(f'tile = {name}', color=col, fontweight='bold')
        if col_i == 0:
            ax.set_ylabel('v [px]')

        # zoom on central anchor
        i_z = n_vis // 2
        ax_z = axes[1, col_i]
        w, h, ang = ellipse_from_cov(Sigma_uv[i_z], n_sigma=2.0)
        ax_z.add_patch(Ellipse((uv0[i_z, 0], uv0[i_z, 1]), w, h, angle=ang,
                                fill=False, edgecolor=col, lw=2.2,
                                label=r'2σ analytic $J_i \Sigma_\delta J_i^\top$'))
        # also draw 1σ ellipse
        w1, h1, ang1 = ellipse_from_cov(Sigma_uv[i_z], n_sigma=1.0)
        ax_z.add_patch(Ellipse((uv0[i_z, 0], uv0[i_z, 1]), w1, h1, angle=ang1,
                                fill=False, edgecolor=col, lw=1.0, ls=':',
                                label=r'1σ analytic'))
        ax_z.plot(uv0[i_z, 0], uv0[i_z, 1], 'k.', ms=6)
        ev, V_eig = np.linalg.eigh(Sigma_uv[i_z])
        # axis half-window
        ax_size = 2.5 * np.sqrt(ev[1]) + 1.0
        ax_z.set_xlim(uv0[i_z, 0] - ax_size, uv0[i_z, 0] + ax_size)
        ax_z.set_ylim(uv0[i_z, 1] + ax_size, uv0[i_z, 1] - ax_size)
        ax_z.set_aspect('equal')
        ax_z.grid(alpha=0.3)
        ax_z.set_xlabel('u [px]')
        ax_z.set_title(
            f'zoom: anchor (u={uv0[i_z, 0]:.0f}, v={uv0[i_z, 1]:.0f}), '
            f'Z={Z[i_z]:.0f} m\n'
            rf'Σ_uv eigvals: $\sqrt{{\lambda}}$=({np.sqrt(max(ev[1],0)):.2f}, '
            rf'{np.sqrt(max(ev[0],0)):.2f}) px',
            color=col, fontsize=10,
        )
        if col_i == 0:
            ax_z.set_ylabel('v [px]')
            ax_z.legend(loc='upper right', fontsize=8)

    fig.suptitle(
        rf'Forward $\Sigma_\delta \to \Sigma_{{uv}}$ via per-anchor Jacobian. '
        rf'Deterministic — no MC.   '
        rf'$\sigma_{{\omega_x}}={SIGMA_WX}°$, $\sigma_{{\omega_y}}={SIGMA_WY}°$, '
        rf'$\rho={RHO}$.',
        y=1.02,
    )
    fig.tight_layout()
    p = OUT / 'fig1_uv_scatter.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {p}')


# ───── fig2: KL vs N for closed-form roundtrip ───────────────────────

def fig2_kl_vs_N():
    print('[fig2] sym-KL roundtrip vs N (closed-form)')
    Sigma_gt = make_sigma_gt()
    K = real_K(1)
    Ns = [1, 2, 3, 5, 10, 30, 100, 300, 1000, 1500]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5.0))
    table = []
    for name, tile_uv in TILES.items():
        kls_A, kls_B = [], []
        for N in Ns:
            P0 = sample_tile_anchors(tile_uv, n=N,
                                       seed=hash(name) % 100)
            _, SA, SB, _, _ = closed_form_roundtrip(P0, K, Sigma_gt)
            kls_A.append(max(sym_kl(SA.numpy(), Sigma_gt.numpy()), 1e-20))
            kls_B.append(max(sym_kl(SB.numpy(), Sigma_gt.numpy()), 1e-20))
        col = TILE_COLORS[name]
        ax.semilogy(Ns, kls_A, '-o', color=col, ms=6, lw=2,
                     label=f'{name}  (A: per-anchor J⁻¹)')
        ax.semilogy(Ns, kls_B, '--s', color=col, ms=5, lw=1.5,
                     alpha=0.7, label=f'{name}  (B: N·H⁻¹)')
        table.append((name, kls_A, kls_B))
    ax.axhline(1e-16, color='gray', ls=':', label='float64 floor (1e-16)')
    ax.set_xscale('log')
    ax.set_xlabel('N anchors')
    ax.set_ylabel('sym-KL ( N(0, Σ̂_δ)  ‖  N(0, Σ_δ) )')
    ax.set_ylim(1e-20, 1e-12)
    ax.grid(which='both', alpha=0.3)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.set_title(
        r'Closed-form roundtrip $\Sigma_\delta \to \Sigma_{uv} \to \hat\Sigma_\delta$' '\n'
        'sym-KL = 0 to float64 floor for every N ≥ 1, every tile.'
    )
    fig.tight_layout()
    p = OUT / 'fig2_delta_recovery.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {p}')
    return table


# ───── fig3: per-tile bar (input / A / B at N=1500) ──────────────────

def fig3_summary(N=1500):
    print('[fig3] per-tile σ / ρ bar summary (closed-form, N=1500)')
    Sigma_gt = make_sigma_gt()
    K = real_K(1)

    def stats(cov):
        sx = np.sqrt(cov[0, 0])
        sy = np.sqrt(cov[1, 1])
        rho = cov[0, 1] / (sx * sy + 1e-30)
        return sx, sy, rho

    rows = ['σ_ωx [deg]', 'σ_ωy [deg]', 'ρ (off-diag)']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    results = {}
    for name, tile_uv in TILES.items():
        P0 = sample_tile_anchors(tile_uv, n=N, seed=hash(name) % 100)
        _, SA, SB, _, _ = closed_form_roundtrip(P0, K, Sigma_gt)
        results[name] = {'gt': Sigma_gt.numpy(),
                          'A': SA.numpy(), 'B': SB.numpy()}
    for ax, row_idx, row_name in zip(axes, range(3), rows):
        labels = list(TILES.keys())
        x = np.arange(len(labels))
        gt = [stats(results[k]['gt'])[row_idx] for k in labels]
        a  = [stats(results[k]['A'])[row_idx]  for k in labels]
        b  = [stats(results[k]['B'])[row_idx]  for k in labels]
        ax.bar(x - 0.27, gt, width=0.25, color='black',     label='input Σ_δ')
        ax.bar(x       , a,  width=0.25, color='tab:red',   label='(A) J⁻¹ recipe')
        ax.bar(x + 0.27, b,  width=0.25, color='tab:blue',  label='(B) N·H⁻¹ recipe')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(row_name)
        ax.grid(axis='y', alpha=0.3)
        if row_idx == 0:
            ax.legend(loc='best', fontsize=9)
        for i in range(len(labels)):
            if abs(gt[i]) > 1e-12:
                rat_a = a[i] / gt[i]
                rat_b = b[i] / gt[i]
                ax.text(x[i],         a[i] * 1.02, f'{rat_a:.6f}',
                         ha='center', fontsize=7)
                ax.text(x[i] + 0.27, b[i] * 1.02, f'{rat_b:.6f}',
                         ha='center', fontsize=7)
    fig.suptitle(
        f'Closed-form roundtrip at N={N}: input vs recipe-A vs recipe-B. '
        'All bars are identical to ~1e-16.',
        y=1.04,
    )
    fig.tight_layout()
    p = OUT / 'fig3_summary.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {p}')
    return results


def main():
    fig1_uv_ellipses()
    table = fig2_kl_vs_N()
    res = fig3_summary(N=1500)
    print('\nClosed-form roundtrip summary (N=1500):')
    print('-' * 78)
    for name, r in res.items():
        Sgt, SA, SB = r['gt'], r['A'], r['B']
        print(f'  [{name:6s}]  '
              f'sym-KL(A)={sym_kl(SA, Sgt):.2e}   '
              f'sym-KL(B)={sym_kl(SB, Sgt):.2e}   '
              f'fro_rel(A)={np.linalg.norm(SA-Sgt)/np.linalg.norm(Sgt):.2e}   '
              f'fro_rel(B)={np.linalg.norm(SB-Sgt)/np.linalg.norm(Sgt):.2e}')
    print('-' * 78)
    print(f'\nAll figures → {OUT}')


if __name__ == '__main__':
    main()
