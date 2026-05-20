"""Pure-synthetic toy unit tests for the closed-form BA solver.

NO cache. NO dataset. NO build_window grid. NO network.

Levels:
  L1  pinhole, 2-DoF  (omega_x, omega_y)
  L2  pinhole, 6-DoF  (omega_xyz, txyz)
  L3  pinhole, 10-DoF (omega_xyz, txyz, dfx, dfy, dcx, dcy)
  L4  KB,      6-DoF  (omega_xyz, txyz)
  L5  KB,      10-DoF (omega_xyz, txyz, dfx, dfy, dcx, dcy)

Recipe (same for every level):
  1. Random 3-D points in cam-front cone.
  2. Apply known δ_true to (a) extrinsic via Rotation/translation, and
     (b) intrinsic via fx_new = fx·(1+dfx), cx_new = cx + dcx, etc.
  3. Project both true and perturbed via project_pinhole / project_kb.
  4. Build par = [Δu = uv_true - uv_pert, Δv = ..., σ=1, σ=1, ρ=0].
  5. Call solver linearised AT the perturbed pose and assert δ̂ ≈ -δ_true.
     (Sign flips because we apply the perturbation true→pert; the solver
      returns the correction pert→true, which is the negative.)

Run:
    pytest tests/ba/test_ba_pipeline_perfect_input.py -v
    or:    python tests/ba/test_ba_pipeline_perfect_input.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

try:
    import pytest
except ImportError:                    # local dev env may be pytest-less
    class _M:
        def skipif(self, *a, **k):
            def _wrap(fn): return fn
            return _wrap
    class _Pytest:
        mark = _M()
        @staticmethod
        def skip(msg): raise AssertionError(f'SKIP: {msg}')
    pytest = _Pytest()

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scipy.spatial.transform import Rotation
from scripts.ba.ba_multicam_corr import solve_dofs
from scripts.ba.ba_kb_jac import (
    kb_jacobian, project_kb, KB_DOF_JAC,
)


# ─── shared helpers ────────────────────────────────────────────────────

def _make_points(n: int = 100, *, seed: int = 0,
                  z_min: float = 3.0, z_max: float = 40.0,
                  xy_frac: float = 0.5) -> np.ndarray:
    """Random 3-D points in front of the cam. Returns (n, 3)."""
    rng = np.random.RandomState(seed)
    Z = rng.uniform(z_min, z_max, n)
    X = rng.uniform(-1.0, 1.0, n) * Z * xy_frac
    Y = rng.uniform(-0.5, 0.5, n) * Z * xy_frac
    return np.stack([X, Y, Z], axis=1)


def _make_tile_points(n: int = 200, *, seed: int = 0,
                       z_min: float = 5.0, z_max: float = 30.0,
                       cx_frac: float = 0.25, cy_frac: float = 0.10,
                       half_x_frac: float = 0.05,
                       half_y_frac: float = 0.05) -> np.ndarray:
    """Tile-scale points: a small angular patch (≈ 256/4096 of FOV) off
    the principal axis. xy/Z ratios stay close together so the projection
    is locally near-linear, which is the regime our 1-step closed-form
    BA was designed for."""
    rng = np.random.RandomState(seed)
    Z = rng.uniform(z_min, z_max, n)
    X = (cx_frac + rng.uniform(-half_x_frac, half_x_frac, n)) * Z
    Y = (cy_frac + rng.uniform(-half_y_frac, half_y_frac, n)) * Z
    return np.stack([X, Y, Z], axis=1)


def _per_axis_tol(dof_names, *, tight: bool = False):
    """Tolerance per DoF unit, sized for δ ≈ 0.1 unit perturbations.
    Coupling residual scales as O(δ²); user spec is "0.01° even at 10-DoF",
    so we hold ω at 1e-2 deg and translations at 1e-3 m. Set tight=True
    for the 2/6-DoF levels where the linearisation is essentially exact."""
    tol = []
    for nm in dof_names:
        if nm.startswith('omega'):
            tol.append(1e-3 if tight else 1e-2)
        elif nm in ('tx', 'ty', 'tz'):
            tol.append(1e-3)
        elif nm in ('dfx', 'dfy'):
            tol.append(2e-4)
        elif nm in ('dcx', 'dcy'):
            tol.append(5e-2)
        else:
            tol.append(1e-3)
    return np.asarray(tol)


def _project_pinhole(P: np.ndarray, K: np.ndarray) -> np.ndarray:
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.stack([fx * X / Z + cx, fy * Y / Z + cy], axis=1)


def _apply_extrinsic_delta(P: np.ndarray, ox: float, oy: float, oz: float,
                            tx: float, ty: float, tz: float) -> np.ndarray:
    """ω in DEGREES, t in METERS. P_new = R·P + t."""
    R = Rotation.from_rotvec(np.deg2rad([ox, oy, oz])).as_matrix()
    t = np.array([tx, ty, tz])
    return P @ R.T + t


def _K_with_delta(K: np.ndarray, dfx: float, dfy: float,
                   dcx: float, dcy: float) -> np.ndarray:
    K2 = K.copy()
    K2[0, 0] *= (1.0 + dfx)
    K2[1, 1] *= (1.0 + dfy)
    K2[0, 2] += dcx
    K2[1, 2] += dcy
    return K2


def _solve_kb_one_step(uv_pert: np.ndarray, par: np.ndarray, z_pert: np.ndarray,
                        P_pert: np.ndarray, K: np.ndarray, dist: np.ndarray,
                        dof_names: list) -> np.ndarray:
    """One Gauss-Newton step using KB analytic Jacobian, linearised at the
    EXACT perturbed cam-frame XYZ (no pinhole back-projection round-trip).
    This isolates KB_DOF_JAC from any depth/back-projection bookkeeping."""
    Xc, Yc, Zc = P_pert[:, 0], P_pert[:, 1], P_pert[:, 2]
    Ju, Jv = kb_jacobian(Xc, Yc, Zc, K, dist, dof_names)
    r_u = par[:, 0]
    r_v = par[:, 1]
    su, sv, rho = par[:, 2], par[:, 3], par[:, 4]
    det = su * su * sv * sv * (1 - rho * rho)
    Wuu = (sv * sv) / det
    Wvv = (su * su) / det
    Wuv = -(rho * su * sv) / det
    n = len(dof_names)
    H = np.zeros((n, n)); b = np.zeros(n)
    for i in range(n):
        for j in range(n):
            H[i, j] = ((Ju[:, i] * Wuu * Ju[:, j]).sum()
                       + (Jv[:, i] * Wvv * Jv[:, j]).sum()
                       + (Ju[:, i] * Wuv * Jv[:, j]).sum()
                       + (Jv[:, i] * Wuv * Ju[:, j]).sum())
        b[i] = ((Ju[:, i] * Wuu * r_u).sum()
                + (Jv[:, i] * Wvv * r_v).sum()
                + (Ju[:, i] * Wuv * r_v).sum()
                + (Jv[:, i] * Wuv * r_u).sum())
    return np.linalg.solve(H, b)


def _build_par(duv: np.ndarray) -> np.ndarray:
    par = np.zeros((len(duv), 5), dtype=np.float64)
    par[:, 0] = duv[:, 0]
    par[:, 1] = duv[:, 1]
    par[:, 2] = 1.0
    par[:, 3] = 1.0
    par[:, 4] = 0.0
    return par


def _print_report(level: str, dof_names, true_vec, est_vec, tol_vec):
    err = np.abs(np.abs(est_vec) - np.abs(true_vec))
    print(f'\n  {level}')
    for nm, t_, e_, er, tl in zip(dof_names, true_vec, est_vec, err, tol_vec):
        unit = 'deg' if nm.startswith('omega') else 'm' if nm in ('tx','ty','tz') else \
               'frac' if nm.startswith('df') else 'px'
        ok = 'OK' if er < tl else 'FAIL'
        print(f'    {nm:>8s}  true={t_:+.6e} {unit}  est={e_:+.6e}  '
              f'|err|={er:.2e}  tol={tl:.0e}  [{ok}]')


# ─── L1: pinhole 2-DoF ─────────────────────────────────────────────────

def test_L1_pinhole_2dof():
    K = np.array([[800.0, 0.0, 320.0],
                  [0.0, 800.0, 240.0],
                  [0.0, 0.0,   1.0]], dtype=np.float64)
    P_true = _make_tile_points(n=200, seed=42)
    dof = ['omega_x', 'omega_y']
    delta_true = np.array([0.20, 0.30])     # deg

    P_pert = _apply_extrinsic_delta(P_true, delta_true[0], delta_true[1],
                                     0.0, 0.0, 0.0, 0.0)
    uv_true = _project_pinhole(P_true, K)
    uv_pert = _project_pinhole(P_pert, K)
    z_pert  = P_pert[:, 2]
    par = _build_par(uv_true - uv_pert)

    delta_hat = solve_dofs(uv_pert, par, z_pert, K, dof_names=dof,
                            damping=0.0, huber_k=None, n_iter=1)

    tol = _per_axis_tol(dof, tight=True)
    _print_report('L1 pinhole 2-DoF', dof, delta_true, delta_hat, tol)
    err = np.abs(np.abs(delta_hat) - np.abs(delta_true))
    assert (err < tol).all(), f'δ̂ = {delta_hat}, true = {delta_true}'


# ─── L2: pinhole 6-DoF ─────────────────────────────────────────────────

def test_L2_pinhole_6dof():
    K = np.array([[800.0, 0.0, 320.0],
                  [0.0, 800.0, 240.0],
                  [0.0, 0.0,   1.0]], dtype=np.float64)
    P_true = _make_tile_points(n=300, seed=7)
    dof = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
    delta_true = np.array([0.10, 0.15, 0.05, 0.02, -0.03, 0.04])

    P_pert = _apply_extrinsic_delta(P_true, *delta_true)
    uv_true = _project_pinhole(P_true, K)
    uv_pert = _project_pinhole(P_pert, K)
    z_pert  = P_pert[:, 2]
    par = _build_par(uv_true - uv_pert)

    delta_hat = solve_dofs(uv_pert, par, z_pert, K, dof_names=dof,
                            damping=0.0, huber_k=None, n_iter=1)

    tol = _per_axis_tol(dof, tight=True)
    _print_report('L2 pinhole 6-DoF', dof, delta_true, delta_hat, tol)
    err = np.abs(np.abs(delta_hat) - np.abs(delta_true))
    assert (err < tol).all(), f'δ̂ = {delta_hat}, true = {delta_true}'


# ─── L3: pinhole 10-DoF (incl. dfx, dfy, dcx, dcy) ─────────────────────

def test_L3_pinhole_10dof():
    """10-DoF needs FULL FOV: per-tile, ω_y vs dcx/dfx is degenerate
    (Jacobian columns become collinear since (u-cx) ≈ const inside one
    tile). Spread points across the whole image to break the symmetry."""
    K = np.array([[800.0, 0.0, 320.0],
                  [0.0, 800.0, 240.0],
                  [0.0, 0.0,   1.0]], dtype=np.float64)
    P_true = _make_points(n=2000, seed=11,
                           z_min=4.0, z_max=30.0, xy_frac=0.35)
    dof = ['omega_x', 'omega_y', 'omega_z',
           'tx', 'ty', 'tz',
           'dfx', 'dfy', 'dcx', 'dcy']
    delta_true = np.array([
        0.10, 0.15, 0.05,
        0.02, -0.03, 0.04,
        +0.01, -0.005,
        +1.5, -0.8,
    ])

    P_pert = _apply_extrinsic_delta(P_true, *delta_true[:6])
    K_pert = _K_with_delta(K, delta_true[6], delta_true[7],
                                delta_true[8], delta_true[9])
    uv_true = _project_pinhole(P_true, K)
    uv_pert = _project_pinhole(P_pert, K_pert)
    z_pert  = P_pert[:, 2]
    par = _build_par(uv_true - uv_pert)

    delta_hat = solve_dofs(uv_pert, par, z_pert, K_pert, dof_names=dof,
                            damping=0.0, huber_k=None, n_iter=1)

    tol = _per_axis_tol(dof)
    _print_report('L3 pinhole 10-DoF', dof, delta_true, delta_hat, tol)
    err = np.abs(np.abs(delta_hat) - np.abs(delta_true))
    assert (err < tol).all(), (
        f'\n  dof={dof}\n  δ̂ = {delta_hat}\n  true = {delta_true}\n  err = {err}')


# ─── L4: KB 6-DoF ──────────────────────────────────────────────────────

def test_L4_kb_6dof():
    K = np.array([[1200.0, 0.0, 1024.0],
                  [0.0, 1200.0, 768.0],
                  [0.0, 0.0,    1.0]], dtype=np.float64)
    dist = np.array([-0.05, 0.01, -0.002, 0.0005], dtype=np.float64)
    P_true = _make_tile_points(n=400, seed=23,
                                cx_frac=0.30, cy_frac=0.10,
                                half_x_frac=0.05, half_y_frac=0.05)
    dof = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
    delta_true = np.array([0.10, 0.15, 0.05, 0.02, -0.03, 0.04])

    P_pert = _apply_extrinsic_delta(P_true, *delta_true)
    uv_true = project_kb(P_true, K, dist)
    uv_pert = project_kb(P_pert, K, dist)
    par = _build_par(uv_true - uv_pert)

    delta_hat = _solve_kb_one_step(uv_pert, par, P_pert[:, 2],
                                    P_pert, K, dist, dof)

    tol = _per_axis_tol(dof, tight=True)
    _print_report('L4 KB 6-DoF', dof, delta_true, delta_hat, tol)
    err = np.abs(np.abs(delta_hat) - np.abs(delta_true))
    assert (err < tol).all(), f'δ̂ = {delta_hat}, true = {delta_true}'


# ─── L5: KB 10-DoF ─────────────────────────────────────────────────────

def test_L5_kb_10dof():
    """10-DoF KB also needs full-FOV; intra-tile ω_y vs dcx is degenerate
    just like the pinhole case."""
    K = np.array([[1200.0, 0.0, 1024.0],
                  [0.0, 1200.0, 768.0],
                  [0.0, 0.0,    1.0]], dtype=np.float64)
    dist = np.array([-0.05, 0.01, -0.002, 0.0005], dtype=np.float64)
    P_true = _make_points(n=2000, seed=31,
                           z_min=4.0, z_max=30.0, xy_frac=0.7)
    dof = ['omega_x', 'omega_y', 'omega_z',
           'tx', 'ty', 'tz',
           'dfx', 'dfy', 'dcx', 'dcy']
    delta_true = np.array([
        0.10, 0.15, 0.05,
        0.02, -0.03, 0.04,
        +0.01, -0.005,
        +1.5, -0.8,
    ])

    P_pert = _apply_extrinsic_delta(P_true, *delta_true[:6])
    K_pert = _K_with_delta(K, delta_true[6], delta_true[7],
                                delta_true[8], delta_true[9])
    uv_true = project_kb(P_true, K, dist)
    uv_pert = project_kb(P_pert, K_pert, dist)
    par = _build_par(uv_true - uv_pert)

    delta_hat = _solve_kb_one_step(uv_pert, par, P_pert[:, 2],
                                    P_pert, K_pert, dist, dof)

    tol = _per_axis_tol(dof)
    _print_report('L5 KB 10-DoF', dof, delta_true, delta_hat, tol)
    err = np.abs(np.abs(delta_hat) - np.abs(delta_true))
    assert (err < tol).all(), (
        f'\n  dof={dof}\n  δ̂ = {delta_hat}\n  true = {delta_true}\n  err = {err}')


# ─── L6: noise robustness ─────────────────────────────────────────────
# Add iid Gaussian noise σ_uv (px) to uv_pert, sweep σ ∈ {0.5, 1, 2, 5} px,
# 50 trials each, compare empirical std(δ̂) to Cramér-Rao lower bound
# (σ̂_δ = sqrt(diag(H⁻¹))·σ_uv with H built at σ=1).

def _noise_trial(K, P_true, dof, delta_true, sigma_px, *, seed,
                  n_steps: int = 1):
    """Match the real pipeline.

    World points P_true; "truth" projection is project(P_true, K).
    The mis-calibrated camera moves the same world points to a perturbed
    image: uv_pert_clean = project(P_pert, K_pert). Sensor / network noise
    acts on this: uv_obs = uv_pert_clean + N(0, σ_uv²). The solver starts
    at δ_cum = 0 (= truth side, P_true / K) and walks toward +δ_true so
    its prediction matches uv_obs.

    Each GN step k:
      1. P_lin = apply δ_cum extrinsic to P_true; K_lin = K with δ_cum intrinsic
      2. r = uv_obs − project(P_lin, K_lin)
      3. solve δ_step, δ_cum += δ_step.

    Clean σ=0 → δ_cum → +δ_true to machine precision in 2-3 steps.
    """
    rng = np.random.RandomState(seed)
    # Observation = projection through the MIS-calibrated camera + noise.
    P_pert = _apply_extrinsic_delta(P_true, *delta_true[:6])
    K_pert = _K_with_delta(K,
                            delta_true[dof.index('dfx')] if 'dfx' in dof else 0.0,
                            delta_true[dof.index('dfy')] if 'dfy' in dof else 0.0,
                            delta_true[dof.index('dcx')] if 'dcx' in dof else 0.0,
                            delta_true[dof.index('dcy')] if 'dcy' in dof else 0.0)
    uv_pert_clean = _project_pinhole(P_pert, K_pert)
    noise = rng.normal(0.0, sigma_px, uv_pert_clean.shape)
    uv_obs = uv_pert_clean + noise

    has = lambda nm: nm in dof
    idx = {nm: dof.index(nm) for nm in dof}

    delta_cum = np.zeros(len(dof))
    for _ in range(n_steps):
        get = lambda nm: float(delta_cum[idx[nm]]) if has(nm) else 0.0
        # Linearisation point starts at TRUTH (P_true, K) and walks to
        # the mis-calib (P_pert, K_pert).
        P_lin = _apply_extrinsic_delta(P_true,
                                        get('omega_x'), get('omega_y'),
                                        get('omega_z'),
                                        get('tx'), get('ty'), get('tz'))
        K_lin = _K_with_delta(K, get('dfx'), get('dfy'),
                                  get('dcx'), get('dcy'))
        uv_lin = _project_pinhole(P_lin, K_lin)
        par = _build_par(uv_obs - uv_lin)
        z_lin = P_lin[:, 2]
        delta_step = solve_dofs(uv_lin, par, z_lin, K_lin, dof_names=dof,
                                 damping=0.0, huber_k=None, n_iter=1)
        delta_cum = delta_cum + delta_step
    return delta_cum


def test_L6_pinhole_6dof_noise_robustness():
    """With σ_uv noise on the observation, compare 1-step vs 2-step GN.

    Variance: CRLB-bounded; 2-step DOES NOT reduce it (information-bound).
    Bias:   shrinks dramatically with 2-step because re-linearising at the
              post-step-1 pose removes the deterministic O(δ²) coupling
              residual that 1-step inherits from linearising at the pert
              pose.
    """
    K = np.array([[800.0, 0.0, 320.0],
                  [0.0, 800.0, 240.0],
                  [0.0, 0.0,   1.0]], dtype=np.float64)
    P_true = _make_tile_points(n=300, seed=7)
    dof = ['omega_x', 'omega_y', 'omega_z', 'tx', 'ty', 'tz']
    delta_true = np.array([0.10, 0.15, 0.05, 0.02, -0.03, 0.04])

    print('\n  L6 pinhole 6-DoF + UV noise (300 tile points, 50 trials/σ)')
    for sigma_px in [0.0, 0.5, 1.0, 2.0, 5.0]:
        for n_steps in (1, 2, 3):
            ests = []
            for s in range(50):
                ests.append(_noise_trial(K, P_true, dof, delta_true,
                                          sigma_px, seed=1000 + s,
                                          n_steps=n_steps))
            ests = np.asarray(ests)
            bias  = np.mean(ests, axis=0) - delta_true
            emp_std = np.std(ests, axis=0)
            cov = solve_dofs._last_cov
            crlb = np.sqrt(np.diag(cov)) * sigma_px

            print(f'    σ_uv = {sigma_px:.1f} px  (n_steps={n_steps})')
            for i, nm in enumerate(dof):
                unit = 'deg' if nm.startswith('omega') else 'm'
                print(f'      {nm:>8s}  bias={bias[i]:+.2e}  '
                      f'emp_std={emp_std[i]:.2e}  CRLB={crlb[i]:.2e} {unit}')

            # 2-step should still be CRLB-bounded in variance.
            if sigma_px > 0.0:
                ratio = emp_std / np.maximum(crlb, 1e-12)
                assert (ratio < 1.5).all() and (ratio > 0.7).all(), (
                    f'σ={sigma_px} n_steps={n_steps}: emp_std/CRLB = {ratio}')


if __name__ == '__main__':
    test_L1_pinhole_2dof()
    test_L2_pinhole_6dof()
    test_L3_pinhole_10dof()
    test_L4_kb_6dof()
    test_L5_kb_10dof()
    test_L6_pinhole_6dof_noise_robustness()
    print('\nAll levels (L1..L6) PASS')
