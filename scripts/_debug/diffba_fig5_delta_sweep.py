"""Why does step 1 already saturate Fig 5? — linearisation error vs noise.

The 6-DoF GN linearisation error scales as O(δ²) at step 1.  At δ ≈ 0.1° this
is ~1e-4°·fx ≈ tiny px, which is dwarfed by σ=0.3 px sensor noise.  So step 1
already lands within noise floor and step 2/3 cannot improve.

Crank δ_true up 10× (≈1° rotations) → linearisation error grows ~100× → now
step 2 visibly helps.  This script prints both regimes.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.ba.ba_multicam_corr import solve_dofs

K = np.array([[800.0, 0.0, 320.0],
              [0.0, 800.0, 240.0],
              [0.0, 0.0,   1.0]], dtype=np.float64)


def _pinhole(P, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.stack([fx * P[:, 0] / P[:, 2] + cx,
                     fy * P[:, 1] / P[:, 2] + cy], axis=1)


def _R(rotvec_deg):
    return Rotation.from_rotvec(np.deg2rad(rotvec_deg)).as_matrix()


def _build_par(duv):
    par = np.zeros((len(duv), 5)); par[:, 0:2] = duv
    par[:, 2] = par[:, 3] = 1.0; par[:, 4] = 0.0
    return par


rng = np.random.RandomState(42)
Z = rng.uniform(5, 30, 300)
X = (0.25 + rng.uniform(-0.05, 0.05, 300)) * Z
Y = (0.10 + rng.uniform(-0.05, 0.05, 300)) * Z
P_true = np.stack([X, Y, Z], axis=1)
dof = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
sigma = 0.3
np.random.seed(5)
noise_full = np.random.RandomState(5).normal(0, sigma, (300, 2))


def run(delta_true, sigma_uv, label):
    P_pert = P_true @ _R(delta_true[:3]).T + delta_true[3:6]
    uv_true = _pinhole(P_true, K)
    uv_pert_clean = _pinhole(P_pert, K)
    uv_obs = uv_pert_clean + noise_full * (sigma_uv / sigma)

    print(f'\n  === {label} ===')
    print(f'  δ_true (deg/m) = {delta_true}')
    print(f'  σ_uv injected  = {sigma_uv} px   (noise floor RMS √2σ = {np.sqrt(2)*sigma_uv:.3f} px)')
    print()
    print('  n_iter   res_RMS [px]   |δ̂ − δ_true|_max')
    delta_cum = np.zeros(6)
    for n in range(1, 6):
        P_lin = P_true @ _R(delta_cum[:3]).T + delta_cum[3:6]
        uv_lin = _pinhole(P_lin, K)
        par = _build_par(uv_obs - uv_lin)
        step = solve_dofs(uv_lin, par, P_lin[:, 2], K, dof_names=dof,
                            damping=0.0, huber_k=None, n_iter=1)
        delta_cum = delta_cum + step
        P_corr = P_true @ _R(delta_cum[:3]).T + delta_cum[3:6]
        uv_corr = _pinhole(P_corr, K)
        res = uv_obs - uv_corr
        rms = np.sqrt((res ** 2).sum(axis=1).mean())
        derr = np.abs(np.abs(delta_cum) - np.abs(delta_true)).max()
        print(f'  {n:>5d}   {rms:>10.6f}   {derr:.3e}')


# Original Fig 5 setup
run(np.array([0.10, 0.15, 0.05, 0.02, -0.03, 0.04]),
    0.3, 'A: original Fig 5  (small δ, σ=0.3 px)')

# Same δ, much smaller noise → linearisation error visible
run(np.array([0.10, 0.15, 0.05, 0.02, -0.03, 0.04]),
    0.001, 'B: small δ, σ=0.001 px  (sub-pixel: lin. error visible)')

# δ × 10 → linearisation error swamps noise floor → step 2/3 visibly helps
run(np.array([1.0, 1.5, 0.5, 0.2, -0.3, 0.4]),
    0.3, 'C: large δ (≈1°), σ=0.3 px  (lin. error matters)')

# δ × 10, no noise → quadratic GN convergence is exposed
run(np.array([1.0, 1.5, 0.5, 0.2, -0.3, 0.4]),
    0.0, 'D: large δ, σ=0  (clean — pure GN quadratic convergence)')
