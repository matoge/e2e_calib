"""Per-trial residual ‖δ̂ - δ_gt‖∞ for the tile roundtrip.

If GN converged, each trial's residual should be at machine precision
(linear case) or at the O(δ²) curvature floor (non-linear case).
The 3.6% in fig3 is purely sample-covariance noise of T=1500 trials.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts._debug.plot_cov_propagation_tiles import (
    TILES, sample_tile_anchors, real_K, make_sigma_gt,
    forward_nonlinear, cholesky_np, DOFS,
)
from scripts.ba.ba_torch import project_pinhole, solve_pinhole

torch.set_default_dtype(torch.float64)


def run(tile_name, tile_uv, *, n_anchors=200, n_trials=1500, n_iter=3, seed=0):
    Sigma_gt = make_sigma_gt()
    K = real_K(1)
    L = cholesky_np(Sigma_gt.numpy())
    P0 = sample_tile_anchors(tile_uv, n=n_anchors, seed=seed)
    uv_truth = project_pinhole(P0, K)
    z = P0[..., 2].clone()
    valid = torch.ones(1, n_anchors, dtype=torch.bool)
    W = torch.eye(2, dtype=torch.float64).expand(1, n_anchors, 2, 2).contiguous()

    rng = np.random.RandomState(seed + 100)
    deltas_gt = (L @ rng.randn(2, n_trials)).T              # (T, 2)
    deltas_gt_t = torch.tensor(deltas_gt, dtype=torch.float64)
    uv_pert = forward_nonlinear(P0, K, deltas_gt_t)
    duv = uv_pert - uv_truth.expand_as(uv_pert)

    # Solve all trials for several n_iter values, see how the residual decays.
    W_T = W.expand(n_trials, n_anchors, 2, 2).contiguous()
    valid_T = valid.expand(n_trials, n_anchors).contiguous()
    z_T = z.expand(n_trials, n_anchors).contiguous()
    K_T = K.expand(n_trials, 3, 3).contiguous()
    uv_truth_T = uv_truth.expand(n_trials, n_anchors, 2).contiguous()

    print(f'[tile={tile_name}]  T={n_trials}  ‖δ_gt‖∞ max={np.max(np.abs(deltas_gt)):.4f} deg')
    for k in (1, 2, 3, 4, 6, 10):
        d_est, _ = solve_pinhole(uv_truth_T, duv, W_T, z_T, K_T, DOFS,
                                  valid=valid_T, n_iter=k, damping=0.0)
        residual = (d_est.numpy() - deltas_gt)            # (T, 2) [deg]
        per_trial_inf = np.max(np.abs(residual), axis=1)
        print(f'   n_iter={k:2d} :  '
              f'residual ‖∞  median={np.median(per_trial_inf):.2e} deg  '
              f'p95={np.percentile(per_trial_inf, 95):.2e}  '
              f'max={np.max(per_trial_inf):.2e}')

    # Sampling-noise expectation on Cov(δ̂):
    #   Var(s²_diag) ≈ 2 σ⁴ / (T-1)   ⇒   relstd(σ_diag) ≈ √(2/(T-1)) / 2 ≈ √(1/(2T))
    # Hmm let me recompute: Var(s²) = 2σ⁴/(T-1), so std(s²) = σ² √(2/(T-1)),
    # relative = √(2/(T-1)). σ = √(s²) ⇒ rel.std(σ) ≈ ½ rel.std(s²) = ½ √(2/(T-1))
    rel_pred = 0.5 * np.sqrt(2.0 / (n_trials - 1))
    print(f'   ⇒ predicted rel.std on σ from sample-cov noise: ±{rel_pred * 100:.2f}%')


if __name__ == '__main__':
    for name, t in TILES.items():
        run(name, t, n_trials=1500, seed=hash(name) % 100)
        print()
