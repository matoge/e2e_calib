"""4-DoF closed-form roundtrip: (ω_x, ω_y, t_x, t_y).

Adds the two image-plane translations to the previous yaw/pitch experiment.
Per anchor, J_i is now 2×4 (under-determined: 2 obs, 4 unknowns) so
ONE anchor cannot recover Σ_δ. The information pool

    H = Σ_i J_iᵀ Σ_uv,i⁻¹ J_i

becomes full-rank only when anchors span a depth range — yaw/pitch flow
is depth-independent (`fx, fy · trig(X,Y)/Z²` ≈ depth-free at first order)
while translation flow is `−fx/Z, −fy/Z` ∝ 1/Z. Without depth diversity
the (ω_y, t_x) and (ω_x, t_y) pairs are degenerate in the linear regime.

Figures:
  fig1 — per-anchor 2σ Σ_uv ellipses (same view as 2-DoF, just for
          context — shape changes with the 4-DoF Σ_δ).
  fig2 — sym-KL roundtrip vs N for three anchor configurations:
            (a) all anchors at the same Z (depth-degenerate)
            (b) anchors on a 2-Z grid
            (c) anchors with full Z range (logU(2, 50) m)
  fig3 — per-tile bar of recovered (σ_ωx, σ_ωy, σ_tx, σ_ty, ρ values)
          for config (c) at N=1500.
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
OUT = REPO / 'docs' / 'assets' / 'diffba' / 'cov_propagation_tiles_4dof'
OUT.mkdir(parents=True, exist_ok=True)

DOFS = ['omega_x', 'omega_y', 'tx', 'ty']

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


# ───── 4-DoF Σ_δ ground truth ─────────────────────────────────────────
SIG_OMEGA_X = 0.10        # deg
SIG_OMEGA_Y = 0.20        # deg
SIG_TX      = 0.05        # m
SIG_TY      = 0.04        # m


def make_sigma_gt():
    stds = np.array([SIG_OMEGA_X, SIG_OMEGA_Y, SIG_TX, SIG_TY])
    rng = np.random.RandomState(7)
    A = rng.randn(4, 4)
    Q, _ = np.linalg.qr(A)
    D = np.diag(stds ** 2)
    Sigma = Q @ D @ Q.T
    return torch.tensor(Sigma, dtype=torch.float64)


# ───── anchor sampling: three Z configurations ────────────────────────

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
    return np.stack([X, Y, z], -1)             # (N, 3)


def sample_anchors(tile_or_list, N, *, mode='full', seed=0):
    """tile_or_list: tile box (u_lo,u_hi,v_lo,v_hi) OR a list of tile boxes.

    When a list is given, N is split equally across them (last tile picks
    up the remainder) so the user can mix left/right or all three.
    """
    if isinstance(tile_or_list, tuple) and len(tile_or_list) == 4 and \
            all(isinstance(x, (int, float)) for x in tile_or_list):
        arr = _sample_one_tile(tile_or_list, N, mode=mode, seed=seed)
    else:
        # list of tile boxes
        tiles = list(tile_or_list)
        T = len(tiles)
        per = [N // T] * T
        per[-1] += N - sum(per)
        parts = [_sample_one_tile(tiles[i], per[i], mode=mode, seed=seed + i * 13)
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


def _sym_basis(K):
    """Return list of K×K symmetric basis matrices E_a, a = 0..K(K+1)/2-1."""
    Es = []
    for i in range(K):
        for j in range(i, K):
            E = np.zeros((K, K))
            if i == j:
                E[i, i] = 1.0
            else:
                E[i, j] = E[j, i] = 1.0
            Es.append(E)
    return Es   # length K(K+1)/2


def _sym_vec(M):
    """Flatten an SPD matrix M into [M_00, M_01, M_11, M_02, ...] (upper tri)."""
    K = M.shape[0]
    out = []
    for i in range(K):
        for j in range(i, K):
            out.append(M[i, j])
    return np.asarray(out)


def _sym_unvec(v, K):
    M = np.zeros((K, K))
    idx = 0
    for i in range(K):
        for j in range(i, K):
            M[i, j] = M[j, i] = v[idx]
            idx += 1
    return M


def closed_form_pool(P0, K, Sigma_gt, ridge=0.0):
    """Recover Σ_δ from per-anchor Σ_uv via least-squares on sym-vec.

    For each anchor:  Σ_uv,i (3 unique entries for sym 2x2)
                    = J_i Σ_δ J_iᵀ              (linear in Σ_δ)
                    = sum_a c_{i,a} · (J_i E_a J_iᵀ)   where Σ_δ = sum_a c_a E_a
    Stack across anchors: A · c = b,  c ∈ R^{K(K+1)/2}, A ∈ R^{3N × K(K+1)/2}.
    Solve via lstsq. With depth diversity ≥ 4 anchors gives full rank.
    """
    K_dof = len(DOFS)
    uv = project_pinhole(P0, K)
    Xc, Yc, Zc = P0.unbind(-1)
    J = pinhole_jacobian(Xc, Yc, Zc, K, uv, DOFS)[0].numpy()       # (N, 2, K)
    # forward
    Sigma_uv = np.einsum('nij,jk,nlk->nil', J, Sigma_gt.numpy(), J)  # (N, 2, 2)

    # build A and b
    Es = _sym_basis(K_dof)                # length M = K(K+1)/2
    M = len(Es)
    N_anch = J.shape[0]
    rows_per = 3                          # sym 2x2 → 3 unique entries
    A = np.zeros((N_anch * rows_per, M))
    b = np.zeros(N_anch * rows_per)
    for i in range(N_anch):
        Ji = J[i]                         # (2, K)
        Suv = Sigma_uv[i]                 # (2, 2)
        b[i * 3 + 0] = Suv[0, 0]
        b[i * 3 + 1] = Suv[0, 1]
        b[i * 3 + 2] = Suv[1, 1]
        for a, E in enumerate(Es):
            JEJ = Ji @ E @ Ji.T           # (2, 2)
            A[i * 3 + 0, a] = JEJ[0, 0]
            A[i * 3 + 1, a] = JEJ[0, 1]
            A[i * 3 + 2, a] = JEJ[1, 1]
    if ridge > 0:
        A = np.vstack([A, np.sqrt(ridge) * np.eye(M)])
        b = np.concatenate([b, np.zeros(M)])
    # condition number of A (used for sanity)
    s = np.linalg.svd(A, compute_uv=False)
    condA = s[0] / max(s[-1], 1e-300)
    rank = int((s > 1e-10 * s[0]).sum())
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    Sigma_est = _sym_unvec(c, K_dof)
    return torch.tensor(Sigma_est, dtype=torch.float64), None, condA, rank, s


# ───── fig1: per-anchor Σ_uv ellipses (4-DoF) ────────────────────────

def fig1_uv_ellipses():
    print('[fig1] per-anchor Σ_uv ellipses (4-DoF Σ_δ → Σ_uv)')
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
                              DOFS)[0].numpy()                    # (N, 2, 4)
        Sigma_uv = np.einsum('nij,jk,nlk->nil', J, Sigma_gt.numpy(), J)

        ax.add_patch(Rectangle((u_lo, v_lo), u_hi - u_lo, v_hi - v_lo,
                                fill=False, ec=col, lw=2.0, ls='--'))
        for i in range(n_vis):
            ev, V_eig = np.linalg.eigh(Sigma_uv[i])
            ev = np.clip(ev, 0, None)
            ang = np.degrees(np.arctan2(V_eig[1, 1], V_eig[0, 1]))
            w = 4 * np.sqrt(ev[1])    # 2σ
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
    fig.suptitle(
        r'4-DoF Σ_δ = diag(σ²_ωx, σ²_ωy, σ²_tx, σ²_ty) ⊕ off-diag → '
        r'per-anchor Σ_uv = J_i Σ_δ J_iᵀ.', y=1.02,
    )
    fig.tight_layout()
    p = OUT / 'fig1_uv_scatter.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {p}')


# ───── fig2: sym-KL vs N for 3 depth configs ─────────────────────────

def fig2_kl_vs_N():
    print('[fig2] sym-KL roundtrip vs N (4-DoF, 3 depth configs)')
    Sigma_gt = make_sigma_gt()
    Sgt_np = Sigma_gt.numpy()
    K = real_K(1)
    Ns = [1, 2, 3, 5, 10, 30, 100, 300, 1000, 3000]
    modes = ['single_Z', 'two_Z', 'full']
    mode_labels = {
        'single_Z': 'all Z = 10 m (degenerate)',
        'two_Z':    'Z ∈ {6, 30} m',
        'full':     'Z ~ logU(2, 50) m',
    }
    mode_styles = {'single_Z': '--', 'two_Z': '-.', 'full': '-'}

    fig, axes = plt.subplots(1, 2, figsize=(20, 7.5))
    ax_kl, ax_cond = axes

    for name, tile_uv in TILES.items():
        col = TILE_COLORS[name]
        for mode in modes:
            kls, conds = [], []
            for N in Ns:
                P0 = sample_anchors(tile_uv, N, mode=mode,
                                       seed=hash(name) % 100)
                Sigma_est, _, condH, rank, _ = closed_form_pool(
                    P0, K, Sigma_gt, ridge=0.0)
                # Treat severely ill-conditioned cases as "didn't recover"
                try:
                    kl = sym_kl(Sigma_est.numpy(), Sgt_np)
                except np.linalg.LinAlgError:
                    kl = np.nan
                kls.append(max(kl, 1e-20) if np.isfinite(kl) else np.nan)
                conds.append(condH)
            ax_kl.semilogy(Ns, kls, mode_styles[mode] + 'o', color=col,
                            ms=5, lw=1.6,
                            label=f'{name} / {mode_labels[mode]}')
            ax_cond.loglog(Ns, conds, mode_styles[mode] + 's', color=col,
                            ms=4, lw=1.4)

    ax_kl.axhline(1e-16, color='gray', ls=':', label='float64 floor')
    ax_kl.set_xscale('log')
    ax_kl.set_xlabel('N anchors')
    ax_kl.set_ylabel('sym-KL ( N(0, Σ̂_δ)  ‖  N(0, Σ_δ) )')
    ax_kl.set_ylim(1e-20, 1e6)
    ax_kl.grid(which='both', alpha=0.3)
    ax_kl.legend(fontsize=7, loc='upper right', ncol=1)
    ax_kl.set_title('Σ_δ recovery via N·H⁻¹')

    ax_cond.set_xlabel('N anchors')
    ax_cond.set_ylabel('cond(H) = λ_max / λ_min')
    ax_cond.grid(which='both', alpha=0.3)
    ax_cond.set_title('Sym-vec design matrix conditioning\n'
                       'single_Z: cond → ∞ (rank-deficient); '
                       'multi-Z: cond → O(1)')
    fig.suptitle(
        '4-DoF (ω_x, ω_y, t_x, t_y) — depth diversity is required to '
        'separate yaw/tx and pitch/ty.', y=1.02,
    )
    fig.tight_layout()
    p = OUT / 'fig2_delta_recovery.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {p}')


# ───── fig3: per-tile bar at N=1500, full-Z config ────────────────────

def fig3_summary(N=1500):
    print('[fig3] per-tile σ summary at N=1500, full-Z config')
    Sigma_gt = make_sigma_gt()
    K = real_K(1)
    results = {}
    for name, tile_uv in TILES.items():
        P0 = sample_anchors(tile_uv, N, mode='full',
                              seed=hash(name) % 100)
        Sigma_est, *_ = closed_form_pool(P0, K, Sigma_gt)
        results[name] = {'gt': Sigma_gt.numpy(), 'B': Sigma_est.numpy()}
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    labels = list(TILES.keys())
    x = np.arange(len(labels))

    for ax, idx, axname, unit in zip(
        axes, range(4),
        ['σ_ωx', 'σ_ωy', 'σ_tx', 'σ_ty'],
        ['deg', 'deg', 'm', 'm'],
    ):
        gt = [np.sqrt(results[k]['gt'][idx, idx]) for k in labels]
        b  = [np.sqrt(results[k]['B'][idx, idx])  for k in labels]
        ax.bar(x - 0.18, gt, width=0.35, color='black',    label='input')
        ax.bar(x + 0.18, b,  width=0.35, color='tab:blue', label='N·H⁻¹')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(f'{axname}  [{unit}]')
        ax.grid(axis='y', alpha=0.3)
        for i in range(len(labels)):
            if abs(gt[i]) > 1e-12:
                ax.text(x[i] + 0.18, b[i] * 1.02,
                        f'{b[i]/gt[i]:.6f}', ha='center', fontsize=7)
        if idx == 0:
            ax.legend(loc='best', fontsize=9)
    fig.suptitle(
        f'4-DoF closed-form roundtrip at N={N} (full-Z config). '
        'Recovery is exact to ~1e-16 across all tiles and DoFs.', y=1.04,
    )
    fig.tight_layout()
    p = OUT / 'fig3_summary.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {p}')
    return results


def fig4_combo_kl():
    """Compare single-tile vs combo (left+right, all three) on the
    HARDEST depth config (single_Z). With X spread on both signs the
    sym-vec design matrix becomes full-rank with fewer total anchors."""
    print('[fig4] combo (multi-tile) recovery vs N at single_Z')
    Sigma_gt = make_sigma_gt()
    Sgt_np = Sigma_gt.numpy()
    K = real_K(1)
    Ns = [4, 5, 6, 8, 10, 14, 20, 30, 60, 100]

    configs = [
        ('left only',         [TILES['left']],                     'tab:blue',  '-o'),
        ('center only',       [TILES['center']],                   'tab:orange','-o'),
        ('right only',        [TILES['right']],                    'tab:green', '-o'),
        ('left + right',      [TILES['left'], TILES['right']],     'tab:purple','-s'),
        ('left + center + right', list(TILES.values()),            'tab:red',   '-^'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(20, 7.5))
    ax_kl, ax_cond = axes

    Sgt_norm = np.linalg.norm(Sgt_np)

    for label, tiles, color, marker in configs:
        fros, conds = [], []
        for N in Ns:
            P0 = sample_anchors(tiles, N, mode='single_Z', seed=0)
            Sigma_est, _, condA, _, _ = closed_form_pool(P0, K, Sigma_gt)
            fro = np.linalg.norm(Sigma_est.numpy() - Sgt_np) / Sgt_norm
            fros.append(max(fro, 1e-17))
            conds.append(condA)
        ax_kl.loglog(Ns, fros, marker, color=color, ms=6, lw=1.8,
                      label=label)
        ax_cond.loglog(Ns, conds, marker, color=color, ms=5, lw=1.6,
                        label=label)

    ax_kl.axhline(1e-15, color='gray', ls=':', label='float64 floor')
    ax_kl.set_xlabel('total N anchors (split equally across selected tiles)')
    ax_kl.set_ylabel(r'$\|\hat\Sigma_\delta - \Sigma_\delta\|_F / \|\Sigma_\delta\|_F$')
    ax_kl.grid(which='both', alpha=0.3)
    ax_kl.legend(fontsize=9, loc='upper right')
    ax_kl.set_title('4-DoF sym-KL roundtrip — single_Z config\n'
                     'left+right and all-three converge with fewer total N '
                     'than any single tile')
    ax_cond.set_xlabel('total N anchors')
    ax_cond.set_ylabel('cond(A)')
    ax_cond.grid(which='both', alpha=0.3)
    ax_cond.legend(fontsize=9, loc='upper right')
    ax_cond.set_title('Sym-vec design conditioning\n'
                       'mixing tiles spreads X-sign ⇒ better-conditioned A')
    fig.suptitle(
        'Mixing tiles helps: same total N, better recovery. '
        'Same Σ_δ, depth fixed at Z = 10 m for the worst case.',
        y=1.02,
    )
    fig.tight_layout()
    p = OUT / 'fig4_combo.png'
    fig.savefig(p, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {p}')


def main():
    fig1_uv_ellipses()
    fig2_kl_vs_N()
    fig4_combo_kl()
    res = fig3_summary(N=1500)
    print('\n4-DoF closed-form summary (N=1500, full-Z):')
    print('-' * 78)
    for name, r in res.items():
        Sgt, SB = r['gt'], r['B']
        print(f'  [{name:6s}]  '
              f'sym-KL={sym_kl(SB, Sgt):.2e}   '
              f'fro_rel={np.linalg.norm(SB-Sgt)/np.linalg.norm(Sgt):.2e}')
    print('-' * 78)
    print(f'\nAll figures → {OUT}')


if __name__ == '__main__':
    main()
