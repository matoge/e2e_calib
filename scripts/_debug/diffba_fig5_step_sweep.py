"""Sweep n_iter ∈ {1,2,3,5} on the Fig 5 setup; print residual RMS.

Expectation: residual RMS plateaus at √2·σ_uv = √2·0.3 ≈ 0.424 px after
step 1, because step 1 already removed the structured 6-DoF component;
all that remains is iid noise which cannot be reduced by any GN step.
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
delta_true = np.array([0.10, 0.15, 0.05, 0.02, -0.03, 0.04])
P_pert = P_true @ _R(delta_true[:3]).T + delta_true[3:6]
uv_true = _pinhole(P_true, K)
uv_pert_clean = _pinhole(P_pert, K)
sigma = 0.3
uv_obs = uv_pert_clean + np.random.RandomState(5).normal(0, sigma, uv_pert_clean.shape)

dof = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']

print(f'  σ_uv injected = {sigma} px  →  noise floor RMS = √2·σ = {np.sqrt(2)*sigma:.4f} px')
print(f'  300 points, σ_per_axis is iid → finite-sample E[RMS] ≈ {np.sqrt(2)*sigma * np.sqrt(1 - 1/(2*300)):.4f}')
print()
print('  n_iter   res_RMS [px]   |δ̂ − δ_true|_max')
print('  ------   ------------   ----------------')

# Walk from truth side toward perturbation, n_iter Gauss-Newton steps.
delta_cum = np.zeros(6)
for n in range(1, 6):
    P_lin = P_true @ _R(delta_cum[:3]).T + delta_cum[3:6]
    uv_lin = _pinhole(P_lin, K)
    par = _build_par(uv_obs - uv_lin)
    step = solve_dofs(uv_lin, par, P_lin[:, 2], K, dof_names=dof,
                       damping=0.0, huber_k=None, n_iter=1)
    delta_cum = delta_cum + step
    # Residual at current cumulative δ
    P_corr = P_true @ _R(delta_cum[:3]).T + delta_cum[3:6]
    uv_corr = _pinhole(P_corr, K)
    res = uv_obs - uv_corr
    rms = np.sqrt((res ** 2).sum(axis=1).mean())
    derr = np.abs(np.abs(delta_cum) - np.abs(delta_true)).max()
    print(f'  {n:>5d}   {rms:>10.6f}   {derr:.3e}')
