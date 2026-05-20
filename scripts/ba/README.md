# `scripts/ba/` — Closed-form Bundle Adjustment library

Per-tile Δuv (network output) → 6/10-DoF camera pose & intrinsic correction `δ`,
in **one matrix solve**. Both numpy (CPU, reference) and torch (GPU, autograd)
implementations live here and are kept byte-equivalent at 1-step.

## TL;DR

```python
# torch (GPU, autograd, in-network)
from scripts.ba.ba_torch import solve_pinhole, solve_kb, make_info_from_sigma_rho

# Inputs (B = batch frame, N = points / tiles per frame, K = len(dof))
#   uv    : (B, N, 2)    image-plane observation (pixels)
#   duv   : (B, N, 2)    network's predicted Δuv  (uv_target = uv + duv)
#   W     : (B, N, 2, 2) per-point info matrix    (W = Σ⁻¹, symmetric PSD)
#   z     : (B, N)       cam-Z depth (metres)
#   K_int : (B, 3, 3)    intrinsics
#   dist  : (B, 4)       Kannala-Brandt k1..k4   (KB only)
#   dof   : list[str]    e.g. ['omega_x','omega_y','omega_z','tx','ty','tz']

delta, H = solve_pinhole(uv, duv, W, z, K_int, dof, n_iter=3, damping=0.0)
# delta : (B, K)         GN correction in DOF order (deg, m, fractional, px)
# H     : (B, K, K)      info matrix at final lin. point — sum across frames
#                          for multi-frame fusion, or use H⁻¹ as δ-cov
```

```python
# numpy (CPU, reference, no autograd)
from scripts.ba.ba_multicam_corr import solve_dofs           # pinhole
from scripts.ba.ba_kb_jac          import solve_dofs_kb       # KB

#   uv  : (N, 2)   par : (N, 5) = [Δu, Δv, σ_x, σ_y, ρ]
#   z   : (N,)     K   : (3, 3)   dist : (4,)   dof : list[str]
delta = solve_dofs   (uv, par, z, K,        dof_names=dof, damping=0.0,
                       huber_k=None, n_iter=1)
delta = solve_dofs_kb(uv, par, z, K, dist,  dof_names=dof, damping=0.0,
                       huber_k=2.5, n_iter=10)
```

## Files

| file                       | role                                                          |
|----------------------------|---------------------------------------------------------------|
| `ba_torch.py`              | **GPU + autograd** (pinhole + KB). E2E-learning library.      |
| `ba_multicam_corr.py`      | numpy reference (pinhole). 1-step closed-form.                |
| `ba_kb_jac.py`             | numpy reference (Kannala-Brandt). Multi-step Huber-IRLS.      |
| (others: `ba_*.py`)        | older / standalone tools — not part of the library API.       |

`ba_torch.py` is a **strict mirror** of the numpy versions: same DoF names, same
sign convention, same internal Jacobians. Test `T1`/`T2` enforce 1-step parity
to ≈1e-14.

## DoF names (shared by numpy & torch)

| name      | meaning                            | unit       |
|-----------|------------------------------------|------------|
| `omega_x` | rotation about cam-X (axis-angle)  | **deg**    |
| `omega_y` | rotation about cam-Y                | deg        |
| `omega_z` | rotation about cam-Z                | deg        |
| `tx, ty, tz` | translation in cam frame         | metres     |
| `dfx`     | fractional fx update: `fx_new = fx · (1 + dfx)` | unitless |
| `dfy`     | fractional fy update                 | unitless |
| `dcx`     | additive cx update: `cx_new = cx + dcx` | px     |
| `dcy`     | additive cy update                   | px       |

`δ` is what to ADD to the current `(P_lin, K_lin)` to land at the target pose.
Sign: `solve(*)` returns `δ̂` such that applying R(δ̂_ω)·P + δ̂_t and the
intrinsic deltas reproduces `uv_target = uv + duv` to first order.

## Math (one-paragraph)

For each point we have a 2D residual `r = uv_target − uv_pred(δ)` and an
analytic Jacobian `J = ∂uv_pred/∂δ ∈ R^{2×K}`. With per-point info matrix
`W ∈ R^{2×2}` (= Σ⁻¹), the Gauss-Newton normal equations are

```
H · δ = b           H = Σₙ Jₙᵀ Wₙ Jₙ      (K × K)
                    b = Σₙ Jₙᵀ Wₙ rₙ      (K,)
δ = solve(H + λI, b)
```

Then re-linearise at `δ_cum + δ` and repeat (`n_iter` times). Cost per step
is **O(B·N·K²)** matmuls + **O(B·K³)** solve; with K≤10 the K³ is free.

## E2E learning contract (torch only)

`ba_torch` is autograd-friendly: gradients flow through `torch.linalg.solve`
via the implicit-function theorem.

- Predict `L: (B, N, 2, 2)` lower-triangular → `W = L Lᵀ` is PSD by construction.
- Pass `(uv, duv, W, z, K, dof, n_iter, damping)` → get `δ̂`.
- Pose loss (e.g. `‖δ̂ − δ_true‖²`) backprops to BOTH `duv` (network's mean
  prediction) AND `L` (network's heteroscedastic uncertainty head).

This is what makes the "BA-as-network-layer" formulation possible.

## Numerical guarantees (verified by tests)

| level                                  | result                          |
|----------------------------------------|---------------------------------|
| pinhole 2/6-DoF tile, clean            | `< 1e-3` deg / `< 1e-3` m       |
| pinhole 10-DoF full FOV, clean         | ω: `< 1e-2`° · dcx,y: `< 5e-2` px |
| KB 6-DoF tile, clean                   | `< 1e-3` deg / m                |
| KB 10-DoF full FOV, clean              | ω: `< 1e-2`° · dcx,y: `< 5e-2` px |
| pinhole 6-DoF, σ_uv = 1 px, 3-step GN  | `emp_std ≈ CRLB · σ_uv`         |
| **GN convergence (clean, 6-DoF)**      | n=1: 4e-4 → n=2: 7e-7 → n=3: **1e-9** deg |
| **np ↔ torch parity (1-step)**         | `≤ 5e-15` (pinhole), `≤ 5e-14` (KB) |
| **autograd ∂loss/∂(duv, L)**           | finite, non-zero                |

`make_info_from_sigma_rho(σ_x, σ_y, ρ) → W` is provided for compatibility
with the numpy-side `par[:, 2:5]` convention. For learning, construct W from
a Cholesky head directly.

## Tile-scale degeneracy ⚠

For 10-DoF (intrinsics included) within ONE tile, `ω_y` and `dcx` are
**numerically degenerate** because `(u − cx)` ≈ const inside a small patch,
making their Jacobian columns collinear. Either

- include points spanning the full FOV (multi-tile fusion), or
- restrict to 6-DoF (extrinsic only) per-frame and lift to 10-DoF after
  multi-frame `H` accumulation.

L3 / L5 tests use full-FOV synthetic points to exercise the 10-DoF case; the
real per-tile inference path stays at 6-DoF.

## Tests

```bash
# numpy toy tests (L1-L5 clean, L6 noise robustness)
python tests/ba/test_ba_pipeline_perfect_input.py

# torch parity / autograd / GPU
python tests/ba/test_ba_torch_parity.py
```

`T5` GPU is **skipped** on hardware that doesn't match the installed torch
wheel's CC list (e.g. local DGX V100 = CC 7.0 with a wheel built only for
CC ≥ 7.5). On a supported device, `cpu` and `cuda` results agree to
`< 1e-10`.

## Where this fits in the project

```
input image (B, 3, H, W) ─ network ─► duv (B, N, 2),  L (B, N, 2, 2)
                                              │              │
                                              ▼              ▼
   uv (tile centres), z, K, dof  ──► solve_pinhole / solve_kb
                                              │
                                              ▼
                                       δ̂ (B, K)  +  H (B, K, K)
                                              │
                                       pose loss / multi-frame fusion
                                              │
                                              ▼
                                          autograd
```

Per-tile inference path (`scripts/serving/caaas_app.py`) calls the **numpy**
solver after collecting all tile Δuv into `par`. The training path that runs
δ̂ inside the network's forward graph uses **torch** (`ba_torch.py`).
