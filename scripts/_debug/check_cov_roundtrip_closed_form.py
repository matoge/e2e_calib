"""Closed-form Σ_δ → Σ_uv → Σ_δ̂ roundtrip — should hit float64 floor.

No Monte Carlo, no sampling. Σ_δ is propagated through J as
    Σ_uv,i = J_i Σ_δ J_iᵀ
then inverted back analytically.

We check both reconstruction recipes:

  (A) Per-anchor exact inversion:
        Σ_δ_est_i = J_i⁻¹ Σ_uv,i J_i⁻ᵀ            (= Σ_δ exactly, ∀ i)
        Σ_δ_est = mean_i Σ_δ_est_i

  (B) Information pool + normalisation:
        H = Σ_i J_iᵀ Σ_uv,i⁻¹ J_i               (= N · Σ_δ⁻¹)
        Σ_δ_est = N · H⁻¹                        (= Σ_δ exactly)

Both should give Σ_δ_est == Σ_δ_gt to machine precision (~1e-16).
Metric: symmetric KL between N(0, Σ_δ_est) and N(0, Σ_δ_gt) — also expected
to vanish at float64 floor.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts._debug.plot_cov_propagation_tiles import (
    TILES, sample_tile_anchors, real_K, make_sigma_gt, DOFS,
)
from scripts.ba.ba_torch import project_pinhole, pinhole_jacobian

torch.set_default_dtype(torch.float64)


def kl_normal(S1, S2):
    """KL( N(0, S1) || N(0, S2) ) for 2x2 SPD matrices."""
    S1, S2 = np.asarray(S1), np.asarray(S2)
    d = S1.shape[0]
    inv2 = np.linalg.inv(S2)
    tr = np.trace(inv2 @ S1)
    logdet = np.linalg.slogdet(S2)[1] - np.linalg.slogdet(S1)[1]
    return 0.5 * (tr - d + logdet)


def sym_kl(S1, S2):
    return 0.5 * (kl_normal(S1, S2) + kl_normal(S2, S1))


def run_tile(tile_name, tile_uv, *, n_anchors=1500, seed=0):
    Sigma_gt = make_sigma_gt()
    K = real_K(1)
    P0 = sample_tile_anchors(tile_uv, n=n_anchors, seed=seed)
    uv = project_pinhole(P0, K)
    Xc, Yc, Zc = P0.unbind(-1)
    J = pinhole_jacobian(Xc, Yc, Zc, K, uv, DOFS)[0]            # (N, 2, 2)

    # Forward: per-anchor Σ_uv,i = J_i Σ_gt J_iᵀ
    Sigma_uv = torch.einsum('nij,jk,nlk->nil', J, Sigma_gt, J)  # (N, 2, 2)

    # ── (A) per-anchor exact inversion ───────────────────────────────
    J_inv = torch.linalg.inv(J)                                  # (N, 2, 2)
    Sigma_est_each = torch.einsum('nij,njk,nlk->nil',
                                    J_inv, Sigma_uv, J_inv)
    # all should equal Sigma_gt exactly
    Sigma_est_A = Sigma_est_each.mean(dim=0).numpy()
    err_per_anchor_max = (Sigma_est_each - Sigma_gt).abs().max().item()

    # ── (B) information pool + N-normalisation ───────────────────────
    W = torch.linalg.inv(Sigma_uv)                               # (N, 2, 2)
    H = torch.einsum('nij,nik,nkl->jl', J, W, J)                 # (2, 2)
    Sigma_est_B = (n_anchors * torch.linalg.inv(H)).numpy()

    Sgt = Sigma_gt.numpy()

    # frobenius and KL metrics
    fro_A = np.linalg.norm(Sigma_est_A - Sgt) / np.linalg.norm(Sgt)
    fro_B = np.linalg.norm(Sigma_est_B - Sgt) / np.linalg.norm(Sgt)
    kl_A = sym_kl(Sigma_est_A, Sgt)
    kl_B = sym_kl(Sigma_est_B, Sgt)

    print(f'[tile={tile_name}]  N={n_anchors}')
    print(f'  Σ_gt =\n{Sgt}')
    print(f'  (A) per-anchor max |Σ_est_i − Σ_gt| = {err_per_anchor_max:.2e}')
    print(f'      mean Σ_est_A =\n{Sigma_est_A}')
    print(f'      ‖ΔΣ‖_F / ‖Σ‖_F = {fro_A:.2e}    sym-KL = {kl_A:.2e}')
    print(f'  (B) Σ_est_B = N · H⁻¹ =\n{Sigma_est_B}')
    print(f'      ‖ΔΣ‖_F / ‖Σ‖_F = {fro_B:.2e}    sym-KL = {kl_B:.2e}')
    print()


if __name__ == '__main__':
    for name, t in TILES.items():
        run_tile(name, t, n_anchors=1500, seed=hash(name) % 100)
    # also sanity at very small N to confirm both are deterministic
    for n in (1, 2, 10, 100):
        print(f'### sanity: N = {n} ###')
        run_tile('center', TILES['center'], n_anchors=n, seed=42)
