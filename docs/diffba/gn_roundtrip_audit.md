# GN Solver Roundtrip Audit

**Date:** 2026-05-21
**Goal:** prove the closed-form GN solver in `scripts/ba/ba_torch.py`
behaves with the right *signs* and *scales* in 6/10-DoF, before letting
the network's W flow through it.

## Why this matters

The pose loss flows back to `InfoHead2x2` through

```
Σ_δ = H⁻¹                    Cramér-Rao bound (Σ_pose)
H   = Σᵢ Jᵢᵀ Wᵢ Jᵢ          GN information matrix
```

A scale bug in `J` propagates through this with a **fourth power**:
`Σ_uv = J Σ_δ Jᵀ` (squared), then inverted into `W`, then sandwiched
again into `H`, then inverted. So a 3× error in `J` lands as ~80×
in the recovered Σ_δ. **That** is what was driving `⟨log det W⟩ → −19`
in Phase 1.c — the network was forced to absorb a broken solver scale
by collapsing W globally.

A standalone roundtrip test, with the network unplugged, is the only
honest way to certify the solver is unit-clean.

## Test script

`scripts/_debug/test_gn_roundtrip.py`. Geometry-only (no dataset, no
CalibNetDepth). Float64. Realistic intrinsics: `fx≈900, fy≈920,
cx=512, cy=384`, KB dist `[-0.05, 0.01, -0.002, 0.0005]`.

### (A) Pose roundtrip

For each (model ∈ {pinhole, KB}) × (DoF ∈ {2, 6, 10}):

1. sample `δ_gt`
2. project `P0` under truth (`uv_truth`) and under perturbed pose
   (`uv_pert`). `Δuv = uv_pert − uv_truth`
3. call `solve_*(uv_truth, Δuv, W=I, …, n_iter=k)` and check
   `‖δ_est − δ_gt‖∞`

### (B) Monte-Carlo Cramér-Rao check

`δ_gt = 0`, add `N(0, σ²I)` pixel noise to `Δuv`, run GN, collect
`Cov(δ̂)` over 2000 trials, compare against analytic `H⁻¹`.

## Results

```
==============================================================================
  (A) POSE ROUNDTRIP — # iterations to ‖δ_est − δ_gt‖∞ < 1e-10
==============================================================================
  [pinhole 2dof ]  n_iter ∈ 2-3   final_err 3.4e-15 .. 8.7e-12
  [pinhole 6dof ]  n_iter = 4     final_err 3.7e-13 .. 2.1e-11
  [pinhole 10dof]  n_iter = 4     final_err 4.5e-12 .. 3.1e-11
  [kb      2dof ]  STUCK at 6.8e-2  (20 iter, no progress past iter 1)
  [kb      6dof ]  STUCK at 3.5e-1
  [kb      10dof]  STUCK at 22.0
==============================================================================
  (B) MC COVARIANCE — empirical Cov(δ̂) vs H⁻¹ (Cramér-Rao)
==============================================================================
  [pinhole 2dof ]  ratio (mc/analytic) = 1.011, 1.012
  [pinhole 6dof ]  ratio = 0.978–1.010 across all 6 axes
  [pinhole 10dof]  ratio = 0.976–1.021 across all 10 axes
  [kb      2dof ]  ratio = 1.005, 1.002
  [kb      6dof ]  ratio = 0.81–1.01 (tz off by 19%)
  [kb      10dof]  ratio = 0.87–1.27 (dcx/dcy noticeably off)
```

## Findings

### ✅ Pinhole solver is unit-clean

- **2-DoF** converges in 2 GN iterations to `1e-12`.
- **6-DoF** in 4 iterations.
- **10-DoF** in 4 iterations.
- Empirical `Cov(δ̂)` matches `H⁻¹` to ≤2.5% on every axis.

This means signs and scales in `pinhole_jacobian` are consistent with
`project_pinhole`, and the linear-noise Cramér-Rao bound is correct.
*Pinhole side has no scale bug.*

### ❌ `solve_kb` is broken

The KB solver does NOT converge. The first iteration fires, then every
subsequent iteration sits at the same residual (`6.8e-2` for 2-DoF).
Inspection:

```python
# scripts/ba/ba_torch.py:340-345
def solve_kb(...):
    fx, fy, cx, cy = _broadcast_intrinsics(K)
    # Pinhole back-projection used as a depth anchor (good enough — the
    # perturbation acts on the projection, not the back-projection).
    X0 = (uv[..., 0] - cx) * z / fx
    Y0 = (uv[..., 1] - cy) * z / fy
    Z0 = z
```

The "depth anchor" comment is wrong: with KB, `uv ↔ (X, Y, Z)` is
non-pinhole, so seeding `P0` from a *pinhole* back-projection puts the
linearisation point on the wrong manifold. The Jacobian is computed at
this bogus point, and GN cannot escape it.

**This is the bug we need to fix before anything else.** The training
script `overfit_2dof_ba_multiframe_unlock_ddp.py` calls `solve_pinhole`,
not `solve_kb`, so PandaSet (front camera, KB lens) was being run
through pinhole geometry — *another* scale bug by 1–2% across the
image, which compounds with the vfp/Z mismatch.

### ❌ Analytic covariance roundtrip — the test was wrong

A first attempt computed:

```
Σ_uv,i = Jᵢ Σ_δ Jᵢᵀ            (per-point 2×2)
W_i    = Σ_uv,i⁻¹
H      = Σᵢ Jᵢᵀ W_i Jᵢ
Σ_δ_est = H⁻¹
```

and asked `Σ_δ_est == Σ_δ_gt`. **This identity does not hold in general.**

Reason: each per-point term `Jᵢᵀ (Jᵢ Σ Jᵢᵀ)⁻¹ Jᵢ` is a rank-2
projection (`= Σ⁻¹/² Pᵢ Σ⁻¹/²` where `Pᵢ` projects onto the row-space
of `M_i = Jᵢ Σ¹/²`). The sum of rank-2 projections is **not** the
identity, even when N→∞ — it converges to whatever average projection
the geometry induces.

The mathematically clean roundtrip is the **noise-free MC**:

1. sample `δ_gt ~ N(0, Σ_δ_gt)`
2. propagate `Δuv = J δ_gt` (no observation noise)
3. solve `δ_est = (Jᵀ J)⁻¹ Jᵀ Δuv`  ⇒  `δ_est = δ_gt` (linear)
4. ⇒ `Cov(δ_est) = Cov(δ_gt) = Σ_δ_gt` to machine precision

In the **non-linear** regime (KB or large δ), this only holds
asymptotically as `Σ_δ_gt → 0`. With finite Σ_δ, residual non-linearity
shows up as a small bias.

## Next steps

1. **Fix `solve_kb`**: seed `P0` either from the true cam-frame XYZ
   (passed in by caller, since the dataset already has it) or from the
   KB back-projection. Re-run (A) — KB pose roundtrip should match
   pinhole within a couple of iterations.
2. **Plumb cam-frame XYZ through the dataset**: per yesterday's design,
   `apply_perturbation_explicit` needs to return `K_full`, `dist`,
   `cp`, `R_gt` so `_build_K_batch` and `solve_kb` use original-camera
   units, not vfp/local.
3. **Switch training script to `solve_kb`** with proper `dist` plumbing.
4. **Replace test (B0)** with the noise-free MC roundtrip.
5. Once all three pass at machine precision (or near-machine for KB),
   re-kick the 2-DoF unfreeze run.
