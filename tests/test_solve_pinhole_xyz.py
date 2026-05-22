"""Guard: uv-free pinhole solver consumes (P0_cam_orig, duv_orig, K_orig).

The legacy `solve_pinhole(uv, duv, W, z, K, ...)` back-projects uv → P0
inside the solver. That path mixes units when (uv, K) are in local-128px
but z is in metres — exactly the (cs/S)² scale bug we hit on the unlock
DDP run. The new `solve_pinhole_xyz(P0, duv, W, K, ...)` takes the cam-
frame XYZ directly so the solver runs purely in original-camera SE3
units.

This file pins:

  1. With identity W, the solver recovers a synthetic δ applied to known
     P0 (orig metric XYZ + orig px duv + parent K) at high precision.
  2. The information matrix H = JᵀWJ scales linearly with W (i.e. no
     hidden cs/S factor leaking through).
  3. Output δ has the same shape contract as the legacy solver
     (B, K_dof) and H is (B, K_dof, K_dof) symmetric PSD.
  4. The new solver agrees with the legacy `solve_pinhole` when fed
     consistent inputs (single-shot, identity W, well-conditioned K).
     This is a back-compat sanity check, not a unit-equivalence claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.ba.ba_torch import (  # noqa: E402
    project_pinhole, solve_pinhole, _apply_extrinsic,
)


# Try to import the new solver — test will skip if the function is not
# yet implemented (so the file can be checked into the repo before the
# implementation lands; the test stays a guard once it is).
try:
    from scripts.ba.ba_torch import solve_pinhole_xyz  # type: ignore
    _HAS_XYZ = True
except ImportError:
    _HAS_XYZ = False


def _real_K(B, fx=1972.0, fy=1972.0, cx=960.0, cy=540.0, dtype=torch.float64):
    K = torch.zeros(B, 3, 3, dtype=dtype)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy
    K[:, 2, 2] = 1.0
    return K


def _sample_P0(B, N, *, seed=0, dtype=torch.float64):
    rng = np.random.RandomState(seed)
    Z = np.exp(rng.uniform(np.log(3.0), np.log(40.0), (B, N)))
    u = rng.uniform(60, 1860, (B, N))
    v = rng.uniform(60, 1020, (B, N))
    fx, fy, cx, cy = 1972.0, 1972.0, 960.0, 540.0
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return torch.tensor(np.stack([X, Y, Z], axis=-1), dtype=dtype)


def _apply_2dof_perturbation(P0, omega_x_deg, omega_y_deg):
    """Use the same Rodrigues helper the solver uses, so the test
    perturbation lives on the SE(3) manifold the solver linearises
    against (no small-angle approximation)."""
    B = P0.shape[0]
    omega = torch.zeros(B, 3, dtype=P0.dtype, device=P0.device)
    omega[:, 0] = float(omega_x_deg)
    omega[:, 1] = float(omega_y_deg)
    t = torch.zeros(B, 3, dtype=P0.dtype, device=P0.device)
    return _apply_extrinsic(P0, omega, t)


@pytest.mark.skipif(not _HAS_XYZ, reason='solve_pinhole_xyz not yet implemented')
def test_recover_2dof_identity_W():
    """Apply small ω_x, ω_y to P0_true → get P0_perturbed → uv difference
    is duv_orig. Solving with identity W should recover (ω_x, ω_y) to
    < 1e-6 deg in float64."""
    B, N = 2, 200
    K = _real_K(B)
    P0 = _sample_P0(B, N, seed=1)
    omega_x_true = [0.30, -0.15]
    omega_y_true = [-0.20, 0.40]
    P0_pert = torch.cat([
        _apply_2dof_perturbation(P0[k:k+1], omega_x_true[k], omega_y_true[k])
        for k in range(B)
    ], dim=0)
    uv_true = project_pinhole(P0, K)
    uv_pert = project_pinhole(P0_pert, K)
    duv_orig = uv_pert - uv_true                                       # (B, N, 2)
    W = torch.eye(2, dtype=P0.dtype).expand(B, N, 2, 2).contiguous()

    delta, H = solve_pinhole_xyz(P0, duv_orig, W, K,
                                  dof_names=('omega_x', 'omega_y'),
                                  n_iter=4)
    assert delta.shape == (B, 2)
    assert H.shape == (B, 2, 2)
    for k in range(B):
        assert abs(delta[k, 0].item() - omega_x_true[k]) < 1e-5, \
            f'sample {k}: ω_x recovered {delta[k, 0].item()}, want {omega_x_true[k]}'
        assert abs(delta[k, 1].item() - omega_y_true[k]) < 1e-5, \
            f'sample {k}: ω_y recovered {delta[k, 1].item()}, want {omega_y_true[k]}'


@pytest.mark.skipif(not _HAS_XYZ, reason='solve_pinhole_xyz not yet implemented')
def test_H_scales_with_W():
    """H = JᵀWJ is linear in W. Doubling W must double H exactly."""
    B, N = 1, 80
    K = _real_K(B)
    P0 = _sample_P0(B, N, seed=2)
    duv = torch.zeros(B, N, 2, dtype=P0.dtype)
    W1 = torch.eye(2, dtype=P0.dtype).expand(B, N, 2, 2).contiguous()
    W2 = 2.0 * W1
    _, H1 = solve_pinhole_xyz(P0, duv, W1, K, dof_names=('omega_x', 'omega_y'))
    _, H2 = solve_pinhole_xyz(P0, duv, W2, K, dof_names=('omega_x', 'omega_y'))
    assert torch.allclose(H2, 2.0 * H1, atol=1e-9)


@pytest.mark.skipif(not _HAS_XYZ, reason='solve_pinhole_xyz not yet implemented')
def test_H_symmetric_psd():
    B, N = 3, 60
    K = _real_K(B)
    P0 = _sample_P0(B, N, seed=3)
    duv = torch.zeros(B, N, 2, dtype=P0.dtype)
    W = torch.eye(2, dtype=P0.dtype).expand(B, N, 2, 2).contiguous()
    _, H = solve_pinhole_xyz(P0, duv, W, K,
                              dof_names=('omega_x', 'omega_y', 'tx', 'ty'))
    assert H.shape == (B, 4, 4)
    # Symmetric
    assert torch.allclose(H, H.transpose(-1, -2), atol=1e-9)
    # PSD (eigvals >= 0 up to numeric)
    eigs = torch.linalg.eigvalsh(H)
    assert (eigs > -1e-9).all(), f'H is not PSD: min eig = {eigs.min().item()}'


@pytest.mark.skipif(not _HAS_XYZ, reason='solve_pinhole_xyz not yet implemented')
def test_agrees_with_legacy_solve_pinhole():
    """Feeding (uv = project(P0), duv = computed Δuv) to the legacy uv-
    based solver must give the same δ as the new XYZ-based solver, since
    the legacy path back-projects uv → P0 with the same K."""
    B, N = 2, 100
    K = _real_K(B)
    P0 = _sample_P0(B, N, seed=4)
    omega_x_true = [0.25, -0.10]
    omega_y_true = [-0.18, 0.30]
    P0_pert = torch.cat([
        _apply_2dof_perturbation(P0[k:k+1], omega_x_true[k], omega_y_true[k])
        for k in range(B)
    ], dim=0)
    uv_true = project_pinhole(P0, K)
    uv_pert = project_pinhole(P0_pert, K)
    duv = uv_pert - uv_true
    z = P0[..., 2].clone()
    W = torch.eye(2, dtype=P0.dtype).expand(B, N, 2, 2).contiguous()

    delta_legacy, _ = solve_pinhole(uv_true, duv, W, z, K,
                                     dof_names=('omega_x', 'omega_y'))
    delta_xyz,    _ = solve_pinhole_xyz(P0, duv, W, K,
                                         dof_names=('omega_x', 'omega_y'))
    assert torch.allclose(delta_legacy, delta_xyz, atol=1e-7), \
        f'legacy {delta_legacy} vs xyz {delta_xyz}'
