"""Parity tests for the torch BA library vs the numpy reference.

  T1  pinhole 6-DoF  : numpy `solve_dofs` ≈ torch `solve_pinhole(n_iter=1)`
  T2  KB 6-DoF       : numpy `solve_dofs_kb` ≈ torch `solve_kb`
  T3  multi-step    : torch `solve_pinhole(n_iter=3)` recovers δ_true to
                        machine precision (Gauss-Newton convergence)
  T4  autograd      : ∂δ̂/∂W is non-zero and finite (gradient flows)
  T5  GPU           : run on CUDA if available, compare to CPU result
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scipy.spatial.transform import Rotation
from scripts.ba.ba_multicam_corr import solve_dofs
from scripts.ba.ba_kb_jac import solve_dofs_kb, project_kb as project_kb_np
from scripts.ba.ba_torch import (
    solve_pinhole, solve_kb,
    project_pinhole, project_kb,
    pinhole_jacobian, kb_jacobian,
    make_info_from_sigma_rho,
)


# ─── helpers ───────────────────────────────────────────────────────────

def _make_points_cpu(n: int = 200, seed: int = 0,
                      cx_frac: float = 0.25, cy_frac: float = 0.10,
                      half: float = 0.05,
                      z_min: float = 5.0, z_max: float = 30.0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    Z = rng.uniform(z_min, z_max, n)
    X = (cx_frac + rng.uniform(-half, half, n)) * Z
    Y = (cy_frac + rng.uniform(-half, half, n)) * Z
    return np.stack([X, Y, Z], axis=1)


def _proj_pinhole_np(P, K):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.stack([fx * P[:, 0] / P[:, 2] + cx,
                     fy * P[:, 1] / P[:, 2] + cy], axis=1)


def _R(rotvec_deg):
    return Rotation.from_rotvec(np.deg2rad(rotvec_deg)).as_matrix()


# ─── T1: pinhole 6-DoF parity ─────────────────────────────────────────

def test_T1_pinhole_6dof_parity():
    K_np = np.array([[800.0, 0.0, 320.0],
                     [0.0, 800.0, 240.0],
                     [0.0, 0.0,   1.0]], dtype=np.float64)
    P_true = _make_points_cpu(n=300, seed=7)
    dof = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
    delta_true = np.array([0.10, 0.15, 0.05, 0.02, -0.03, 0.04])

    R_pert = _R(delta_true[:3])
    P_pert = P_true @ R_pert.T + delta_true[3:6]
    uv_true = _proj_pinhole_np(P_true, K_np)
    uv_pert = _proj_pinhole_np(P_pert, K_np)
    z_pert = P_pert[:, 2]
    duv = uv_true - uv_pert

    par = np.zeros((len(duv), 5)); par[:, 0:2] = duv
    par[:, 2] = par[:, 3] = 1.0; par[:, 4] = 0.0
    delta_np = solve_dofs(uv_pert, par, z_pert, K_np, dof_names=dof,
                           damping=0.0, huber_k=None, n_iter=1)

    # torch (B=1, n_iter=1, no re-linearisation → matches numpy 1-step)
    t = torch.float64
    uv_t   = torch.from_numpy(uv_pert)[None].to(t)
    duv_t  = torch.from_numpy(duv)[None].to(t)
    z_t    = torch.from_numpy(z_pert)[None].to(t)
    K_t    = torch.from_numpy(K_np)[None].to(t)
    sx = torch.ones(1, len(duv), dtype=t)
    sy = torch.ones(1, len(duv), dtype=t)
    rho = torch.zeros(1, len(duv), dtype=t)
    W = make_info_from_sigma_rho(sx, sy, rho)
    delta_pt, _ = solve_pinhole(uv_t, duv_t, W, z_t, K_t, dof,
                                 n_iter=1, damping=0.0)
    delta_pt_np = delta_pt[0].cpu().numpy()

    diff = np.abs(delta_np - delta_pt_np).max()
    print(f'\n  T1 pinhole 6-DoF parity (1-step):')
    print(f'    np:    {delta_np}')
    print(f'    torch: {delta_pt_np}')
    print(f'    max |np - torch| = {diff:.2e}')
    assert diff < 1e-9, f'parity broken: {diff}'


# ─── T2: KB 6-DoF parity ──────────────────────────────────────────────

def test_T2_kb_6dof_parity():
    K_np = np.array([[1200.0, 0.0, 1024.0],
                     [0.0, 1200.0, 768.0],
                     [0.0, 0.0,    1.0]], dtype=np.float64)
    dist_np = np.array([-0.05, 0.01, -0.002, 0.0005], dtype=np.float64)
    P_true = _make_points_cpu(n=400, seed=23)
    dof = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
    delta_true = np.array([0.10, 0.15, 0.05, 0.02, -0.03, 0.04])

    R_pert = _R(delta_true[:3])
    P_pert = P_true @ R_pert.T + delta_true[3:6]
    uv_true = project_kb_np(P_true, K_np, dist_np)
    uv_pert = project_kb_np(P_pert, K_np, dist_np)
    z_pert = P_pert[:, 2]
    duv = uv_true - uv_pert

    par = np.zeros((len(duv), 5)); par[:, 0:2] = duv
    par[:, 2] = par[:, 3] = 1.0; par[:, 4] = 0.0
    # numpy reference: 1 iter, no Huber, linearise at the OBSERVED pose
    # (= post-perturbation), so it doesn't re-project P_pert through KB.
    # Match that exactly: use solve_dofs_kb with x0=0, n_iter=1.
    delta_np = solve_dofs_kb(uv_pert, par, z_pert, K_np, dist_np,
                              dof_names=dof, damping=0.0,
                              huber_k=None, n_iter=1)

    t = torch.float64
    uv_t   = torch.from_numpy(uv_pert)[None].to(t)
    duv_t  = torch.from_numpy(duv)[None].to(t)
    z_t    = torch.from_numpy(z_pert)[None].to(t)
    K_t    = torch.from_numpy(K_np)[None].to(t)
    dist_t = torch.from_numpy(dist_np)[None].to(t)
    sx = torch.ones(1, len(duv), dtype=t)
    sy = torch.ones(1, len(duv), dtype=t)
    rho = torch.zeros(1, len(duv), dtype=t)
    W = make_info_from_sigma_rho(sx, sy, rho)
    delta_pt, _ = solve_kb(uv_t, duv_t, W, z_t, K_t, dist_t, dof,
                            n_iter=1, damping=0.0)
    delta_pt_np = delta_pt[0].cpu().numpy()

    diff = np.abs(delta_np - delta_pt_np).max()
    print(f'\n  T2 KB 6-DoF parity (1-step):')
    print(f'    np:    {delta_np}')
    print(f'    torch: {delta_pt_np}')
    print(f'    max |np - torch| = {diff:.2e}')
    assert diff < 1e-7, f'parity broken: {diff}'


# ─── T3: torch multi-step convergence ─────────────────────────────────

def test_T3_torch_multistep_machine_precision():
    K_np = np.array([[800.0, 0.0, 320.0],
                     [0.0, 800.0, 240.0],
                     [0.0, 0.0,   1.0]], dtype=np.float64)
    P_true = _make_points_cpu(n=300, seed=7)
    dof = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
    delta_true = np.array([0.10, 0.15, 0.05, 0.02, -0.03, 0.04])

    R_pert = _R(delta_true[:3])
    P_pert = P_true @ R_pert.T + delta_true[3:6]
    uv_obs = _proj_pinhole_np(P_pert, K_np)         # noiseless
    z_true = P_true[:, 2]                            # truth-side depth
    uv_true = _proj_pinhole_np(P_true, K_np)
    duv = uv_obs - uv_true                            # network "perfect" Δuv

    t = torch.float64
    uv_t   = torch.from_numpy(uv_true)[None].to(t)
    duv_t  = torch.from_numpy(duv)[None].to(t)
    z_t    = torch.from_numpy(z_true)[None].to(t)
    K_t    = torch.from_numpy(K_np)[None].to(t)
    sx = torch.ones(1, len(duv), dtype=t)
    sy = torch.ones(1, len(duv), dtype=t)
    rho = torch.zeros(1, len(duv), dtype=t)
    W = make_info_from_sigma_rho(sx, sy, rho)

    print(f'\n  T3 torch multi-step convergence (clean):')
    for n_iter in (1, 2, 3):
        delta_pt, _ = solve_pinhole(uv_t, duv_t, W, z_t, K_t, dof,
                                     n_iter=n_iter, damping=0.0)
        d = delta_pt[0].cpu().numpy()
        err = np.abs(d - delta_true).max()
        print(f'    n_iter={n_iter}  max |δ̂ - δ_true| = {err:.2e}')
        if n_iter == 3:
            assert err < 1e-8, f'3-step did not converge: {err}'


# ─── T4: autograd through BA ──────────────────────────────────────────

def test_T4_autograd_through_BA():
    """Verify ∂loss/∂(per-pt input) is non-zero and finite. Setup: feed a
    NOISY duv (so δ̂ is off from δ_true by O(σ_uv)), then differentiate
    the pose loss w.r.t. both `duv` (= network's mean prediction) and `L`
    (= Cholesky factor of the per-pt info matrix W). Both gradients must
    be finite & non-zero — this is what makes E2E pose learning possible."""
    K_np = np.array([[800.0, 0.0, 320.0],
                     [0.0, 800.0, 240.0],
                     [0.0, 0.0,   1.0]], dtype=np.float64)
    P_true = _make_points_cpu(n=200, seed=7)
    dof = ['omega_x', 'omega_y']
    delta_true = np.array([0.10, 0.15])

    R_pert = _R([delta_true[0], delta_true[1], 0.0])
    P_pert = P_true @ R_pert.T
    uv_true = _proj_pinhole_np(P_true, K_np)
    uv_obs_clean = _proj_pinhole_np(P_pert, K_np)
    rng = np.random.RandomState(0)
    uv_obs = uv_obs_clean + rng.normal(0.0, 1.0, uv_obs_clean.shape)
    duv = uv_obs - uv_true
    z_true = P_true[:, 2]

    t = torch.float64
    uv_t   = torch.from_numpy(uv_true)[None].to(t)
    duv_t  = torch.from_numpy(duv)[None].to(t).requires_grad_(True)
    z_t    = torch.from_numpy(z_true)[None].to(t)
    K_t    = torch.from_numpy(K_np)[None].to(t)

    # Random per-pt L (heteroscedastic info matrix), learnable.
    rng2 = np.random.RandomState(1)
    L_init = np.tile(np.eye(2), (1, len(duv), 1, 1))
    L_init[..., 0, 0] = 0.5 + rng2.uniform(0, 1, len(duv))[None]
    L_init[..., 1, 1] = 0.5 + rng2.uniform(0, 1, len(duv))[None]
    L_init[..., 1, 0] = rng2.uniform(-0.2, 0.2, len(duv))[None]
    L = torch.tensor(L_init, dtype=t, requires_grad=True)
    W = L @ L.transpose(-1, -2)

    delta_pt, _ = solve_pinhole(uv_t, duv_t, W, z_t, K_t, dof,
                                 n_iter=2, damping=0.0)
    pose_loss = ((delta_pt - torch.tensor(delta_true, dtype=t)) ** 2).sum()
    pose_loss.backward()

    print(f'\n  T4 autograd through BA (with σ=1px noise on duv):')
    print(f'    pose_loss = {pose_loss.item():.6e}')
    print(f'    |∂loss/∂duv| min/max = {duv_t.grad.abs().min().item():.3e}'
          f' / {duv_t.grad.abs().max().item():.3e}')
    print(f'    |∂loss/∂L|  min/max = {L.grad.abs().min().item():.3e}'
          f' / {L.grad.abs().max().item():.3e}')
    assert duv_t.grad is not None and torch.isfinite(duv_t.grad).all()
    assert L.grad is not None and torch.isfinite(L.grad).all()
    assert duv_t.grad.abs().sum().item() > 1e-6
    assert L.grad.abs().sum().item() > 1e-6


# ─── T5: GPU run + CPU/GPU agreement ──────────────────────────────────

def test_T5_gpu_run():
    if not torch.cuda.is_available():
        print('\n  T5 GPU: SKIPPED (no CUDA)')
        return
    # The installed wheel may not have kernels for this device's CC
    # (e.g. local DGX V100 = CC 7.0 with a wheel built only for CC ≥ 7.5).
    try:
        torch.zeros(1, device='cuda').add_(1.0)
    except Exception as e:
        print(f'\n  T5 GPU: SKIPPED (CUDA kernel mismatch: {type(e).__name__})')
        return
    K_np = np.array([[800.0, 0.0, 320.0],
                     [0.0, 800.0, 240.0],
                     [0.0, 0.0,   1.0]], dtype=np.float64)
    P_true = _make_points_cpu(n=300, seed=7)
    dof = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
    delta_true = np.array([0.10, 0.15, 0.05, 0.02, -0.03, 0.04])
    R_pert = _R(delta_true[:3])
    P_pert = P_true @ R_pert.T + delta_true[3:6]
    uv_true = _proj_pinhole_np(P_true, K_np)
    uv_obs = _proj_pinhole_np(P_pert, K_np)
    duv = uv_obs - uv_true

    t = torch.float64
    def _pack(device):
        uv_t   = torch.from_numpy(uv_true)[None].to(t).to(device)
        duv_t  = torch.from_numpy(duv)[None].to(t).to(device)
        z_t    = torch.from_numpy(P_true[:, 2])[None].to(t).to(device)
        K_t    = torch.from_numpy(K_np)[None].to(t).to(device)
        sx = torch.ones(1, len(duv), dtype=t, device=device)
        sy = torch.ones(1, len(duv), dtype=t, device=device)
        rho = torch.zeros(1, len(duv), dtype=t, device=device)
        W = make_info_from_sigma_rho(sx, sy, rho)
        return uv_t, duv_t, W, z_t, K_t

    cpu = _pack('cpu')
    gpu = _pack('cuda')
    delta_cpu, _ = solve_pinhole(*cpu, dof, n_iter=3, damping=0.0)
    delta_gpu, _ = solve_pinhole(*gpu, dof, n_iter=3, damping=0.0)
    diff = (delta_cpu.cpu() - delta_gpu.cpu()).abs().max().item()
    print(f'\n  T5 CPU vs GPU agreement (3-step):')
    print(f'    cpu: {delta_cpu[0].cpu().numpy()}')
    print(f'    gpu: {delta_gpu[0].cpu().numpy()}')
    print(f'    max |cpu - gpu| = {diff:.2e}')
    assert diff < 1e-10


if __name__ == '__main__':
    test_T1_pinhole_6dof_parity()
    test_T2_kb_6dof_parity()
    test_T3_torch_multistep_machine_precision()
    test_T4_autograd_through_BA()
    test_T5_gpu_run()
    print('\nAll torch parity / autograd / GPU tests PASS')
