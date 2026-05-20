"""Kannala-Brandt analytic Jacobian for closed-form 6-DoF BA.

Pinhole `DOF_JAC` (in ba_multicam_corr.py) under-estimates δ̂ for fisheye
images because the image-plane Δuv response near the edge is bigger than
the pinhole `(fx + fx X²/Z²)·dω` model predicts. This module supplies the
analytic ∂uv/∂δ for the KB 4-coeff fisheye projection so a single 6×6
normal equation can solve the same problem with image-edge bias gone.

Forward (must match scripts/util/projection.project_kannala):
    r       = sqrt(X² + Y²)
    θ       = atan2(r, Z)
    θ_d     = θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸)
    u       = fx · θ_d · X / r + cx
    v       = fy · θ_d · Y / r + cy

Convention: angles in DEGREES (`_D2R` applied internally), translations in
METERS, fx/cx/fy/cy in PIXELS. δ has the SAME sign convention as
ba_multicam_corr.DOF_JAC (positive δ moves the projection in the +Δuv
direction reported by the model).

Public API:
    kb_jacobian(X, Y, Z, K, dist, dof_names) → (Ju (N,k), Jv (N,k))
    solve_dofs_kb(uv, par, z, K, dist, dof_names,
                   damping=1e-3, huber_k=2.5, n_iter=10, x0=None)
"""
from __future__ import annotations

import numpy as np

_D2R = np.pi / 180.0


def _per_pt(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
             K: np.ndarray, dist: np.ndarray):
    """Common KB intermediates evaluated at the linearisation point."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    k1, k2, k3, k4 = float(dist[0]), float(dist[1]), float(dist[2]), float(dist[3])
    r2 = X * X + Y * Y
    r = np.sqrt(np.maximum(r2, 1e-24))
    r_safe = np.where(r > 1e-9, r, 1.0)
    theta = np.arctan2(r, np.maximum(Z, 1e-9))
    t2 = theta * theta
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    poly = 1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8
    theta_d = theta * poly
    # ∂θ_d / ∂θ : derivative w.r.t. theta (chain rule first piece)
    dtd_dtheta = 1.0 + 3.0 * k1 * t2 + 5.0 * k2 * t4 + 7.0 * k3 * t6 + 9.0 * k4 * t8
    return dict(fx=fx, fy=fy, cx=cx, cy=cy,
                X=X, Y=Y, Z=Z, r=r, r_safe=r_safe,
                theta=theta, theta_d=theta_d, dtd_dtheta=dtd_dtheta,
                r2_plus_z2=r2 + Z * Z + 1e-24)


def _duv_from_dxyz(p: dict, dX: np.ndarray, dY: np.ndarray, dZ: np.ndarray):
    """Push (∂X, ∂Y, ∂Z) through the KB chain to (∂u, ∂v).

    Eqs:
        ∂r/∂δ      = (X·dX + Y·dY) / r
        ∂θ/∂δ      = (Z·∂r/∂δ - r·dZ) / (r² + Z²)
        ∂θ_d/∂δ    = (∂θ_d/∂θ) · ∂θ/∂δ
        ∂(X/r)/∂δ  = dX / r - X · ∂r/∂δ / r²
        ∂u/∂δ      = fx · [∂θ_d/∂δ · (X/r) + θ_d · ∂(X/r)/∂δ]
    Same for v with Y.
    """
    X, Y, Z, r, r_safe = p['X'], p['Y'], p['Z'], p['r'], p['r_safe']
    theta_d = p['theta_d']
    dr = (X * dX + Y * dY) / r_safe
    dtheta = (Z * dr - r * dZ) / p['r2_plus_z2']
    dtd = p['dtd_dtheta'] * dtheta
    dXr = dX / r_safe - X * dr / (r_safe * r_safe)
    dYr = dY / r_safe - Y * dr / (r_safe * r_safe)
    du = p['fx'] * (dtd * (X / r_safe) + theta_d * dXr)
    dv = p['fy'] * (dtd * (Y / r_safe) + theta_d * dYr)
    return du, dv


# ── Per-DoF Jacobian: each returns (du, dv) of shape (N,) given p = _per_pt() ──

def _jac_omega_x(p):
    # Rotation about cam-X by dθ rad: (X, Y, Z) → (X, Y - Z·dθ, Z + Y·dθ)
    # ⇒ ∂(X, Y, Z) / ∂(ω_x in DEG) = (0, -Z, Y) · _D2R
    z = np.zeros_like(p['X'])
    return _duv_from_dxyz(p, z, -p['Z'] * _D2R, p['Y'] * _D2R)


def _jac_omega_y(p):
    # ∂(X, Y, Z) / ∂(ω_y in DEG) = (Z, 0, -X) · _D2R
    z = np.zeros_like(p['X'])
    return _duv_from_dxyz(p, p['Z'] * _D2R, z, -p['X'] * _D2R)


def _jac_omega_z(p):
    # ∂(X, Y, Z) / ∂(ω_z in DEG) = (-Y, X, 0) · _D2R
    z = np.zeros_like(p['X'])
    return _duv_from_dxyz(p, -p['Y'] * _D2R, p['X'] * _D2R, z)


def _jac_tx(p):
    o = np.ones_like(p['X']); z = np.zeros_like(p['X'])
    return _duv_from_dxyz(p, o, z, z)


def _jac_ty(p):
    o = np.ones_like(p['X']); z = np.zeros_like(p['X'])
    return _duv_from_dxyz(p, z, o, z)


def _jac_tz(p):
    o = np.ones_like(p['X']); z = np.zeros_like(p['X'])
    return _duv_from_dxyz(p, z, z, o)


def _jac_dfx(p):
    # fx_new = fx · (1 + dfx_pct). u = fx · θ_d · X/r + cx
    # ⇒ ∂u/∂dfx = fx · θ_d · X / r = (u_kb - cx). v independent.
    z = np.zeros_like(p['X'])
    du = p['fx'] * p['theta_d'] * p['X'] / p['r_safe']
    return du, z


def _jac_dfy(p):
    z = np.zeros_like(p['X'])
    dv = p['fy'] * p['theta_d'] * p['Y'] / p['r_safe']
    return z, dv


def _jac_dcx(p):
    o = np.ones_like(p['X']); z = np.zeros_like(p['X'])
    return o, z


def _jac_dcy(p):
    o = np.ones_like(p['X']); z = np.zeros_like(p['X'])
    return z, o


KB_DOF_JAC = {
    'omega_x': _jac_omega_x,
    'omega_y': _jac_omega_y,
    'omega_z': _jac_omega_z,
    'tx':      _jac_tx,
    'ty':      _jac_ty,
    'tz':      _jac_tz,
    'dfx':     _jac_dfx,
    'dfy':     _jac_dfy,
    'dcx':     _jac_dcx,
    'dcy':     _jac_dcy,
}


def kb_jacobian(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                 K: np.ndarray, dist: np.ndarray, dof_names: list):
    """Stack per-DoF Jacobian columns. Returns (Ju (N,k), Jv (N,k))."""
    p = _per_pt(X, Y, Z, K, dist)
    Jus, Jvs = [], []
    for name in dof_names:
        if name not in KB_DOF_JAC:
            raise KeyError(f"unknown KB DoF '{name}' — valid: {sorted(KB_DOF_JAC)}")
        ju, jv = KB_DOF_JAC[name](p)
        Jus.append(np.broadcast_to(ju, Z.shape))
        Jvs.append(np.broadcast_to(jv, Z.shape))
    return np.column_stack(Jus), np.column_stack(Jvs)


# ── KB projection forward (mirrors scripts.util.projection.project_kannala
# but using the same chain we differentiated above, so re-linearisation
# at a non-zero δ stays self-consistent). ──

def project_kb(pts_cam: np.ndarray, K: np.ndarray, dist: np.ndarray):
    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    r = np.sqrt(X * X + Y * Y)
    theta = np.arctan2(r, np.maximum(Z, 1e-9))
    k1, k2, k3, k4 = float(dist[0]), float(dist[1]), float(dist[2]), float(dist[3])
    t2 = theta * theta
    theta_d = theta * (1.0 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4)
    r_safe = np.where(r > 1e-9, r, 1.0)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = fx * (theta_d * X / r_safe) + cx
    v = fy * (theta_d * Y / r_safe) + cy
    return np.stack([u, v], axis=-1)


def solve_dofs_kb(uv: np.ndarray, par: np.ndarray, z: np.ndarray,
                   K: np.ndarray, dist: np.ndarray, dof_names: list,
                   damping: float | np.ndarray = 1e-3,
                   huber_k: float | None = 2.5,
                   n_iter: int = 10, x0: np.ndarray | None = None):
    """Closed-form GN with KB analytic Jacobian. Multiple iterations re-
    linearise at the current δ, so unlike `solve_dofs` (pinhole, line-
    arises only at δ=0) it can converge from a warm-start far from the
    truth.

    Inputs match `solve_dofs`:
        uv  : (N, 2) observation in PARENT-image px (the pose under
              which the model was queried)
        par : (N, 5) [Δu, Δv, σx, σy, ρ]
        z   : (N,)   cam-Z depth in metres
        K   : (3, 3) intrinsics
        dist: (4,)   Kannala k1..k4
        dof_names : list of DoF names in declaration order
        x0 : (k,) initial δ. If None, zeros (i.e. linearise at observation pose).
    Returns δ (k,) in DOF declaration order.
    """
    n_dof = len(dof_names)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Lift observed uv to cam-frame XYZ through the pinhole-from-uv
    # back-projection (same as solve_dofs uses); good enough as a depth
    # anchor — the perturbation affects projection, not back-projection.
    X0 = (uv[:, 0] - cx) * z / fx
    Y0 = (uv[:, 1] - cy) * z / fy
    Z0 = z

    target_uv = uv + par[:, :2]                # uv_target = obs + model Δuv
    su, sv, rho = par[:, 2], par[:, 3], par[:, 4]
    det = su * su * sv * sv * (1 - rho * rho)
    Wuu0 = (sv * sv) / det
    Wvv0 = (su * su) / det
    Wuv0 = -(rho * su * sv) / det

    delta = np.zeros(n_dof) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
    weights = np.ones_like(z)

    H = np.zeros((n_dof, n_dof))
    b = np.zeros(n_dof)
    for it in range(max(1, int(n_iter))):
        # Re-project pts3 with current δ (rotvec + translation in cam-frame).
        from scipy.spatial.transform import Rotation
        R = Rotation.from_rotvec(np.deg2rad(np.array([
            delta[dof_names.index('omega_x')] if 'omega_x' in dof_names else 0.0,
            delta[dof_names.index('omega_y')] if 'omega_y' in dof_names else 0.0,
            delta[dof_names.index('omega_z')] if 'omega_z' in dof_names else 0.0,
        ]))).as_matrix()
        t = np.array([
            delta[dof_names.index('tx')] if 'tx' in dof_names else 0.0,
            delta[dof_names.index('ty')] if 'ty' in dof_names else 0.0,
            delta[dof_names.index('tz')] if 'tz' in dof_names else 0.0,
        ])
        pts3 = np.stack([X0, Y0, Z0], axis=1) @ R.T + t
        Xc, Yc, Zc = pts3[:, 0], pts3[:, 1], pts3[:, 2]
        uv_pred = project_kb(pts3, K, dist)
        r_u = target_uv[:, 0] - uv_pred[:, 0]
        r_v = target_uv[:, 1] - uv_pred[:, 1]
        # Jacobian at current linearisation point.
        Ju, Jv = kb_jacobian(Xc, Yc, Zc, K, dist, dof_names)

        Wuu = Wuu0 * weights
        Wvv = Wvv0 * weights
        Wuv = Wuv0 * weights
        H[:] = 0.0
        b[:] = 0.0
        for i in range(n_dof):
            for j in range(n_dof):
                H[i, j] = ((Ju[:, i] * Wuu * Ju[:, j]).sum()
                           + (Jv[:, i] * Wvv * Jv[:, j]).sum()
                           + (Ju[:, i] * Wuv * Jv[:, j]).sum()
                           + (Jv[:, i] * Wuv * Ju[:, j]).sum())
            b[i] = ((Ju[:, i] * Wuu * r_u).sum()
                    + (Jv[:, i] * Wvv * r_v).sum()
                    + (Ju[:, i] * Wuv * r_v).sum()
                    + (Jv[:, i] * Wuv * r_u).sum())
        damp_arr = np.asarray(damping, dtype=np.float64)
        if damp_arr.ndim == 0:
            H_reg = H + float(damp_arr) * np.eye(n_dof)
        else:
            H_reg = H + np.diag(damp_arr)
        step = np.linalg.solve(H_reg, b)
        delta = delta + step
        # Huber re-weighting based on Mahalanobis distance of the
        # post-step residual.
        if huber_k is None:
            break
        ru = r_u - Ju @ step
        rv = r_v - Jv @ step
        d2 = (ru * Wuu0 * ru) + (rv * Wvv0 * rv) + 2.0 * (ru * Wuv0 * rv)
        d = np.sqrt(np.maximum(d2, 1e-12))
        weights = np.where(d <= huber_k, 1.0, huber_k / d)
        if np.linalg.norm(step) < 1e-5:
            break
    damp_arr = np.asarray(damping, dtype=np.float64)
    H_final = H + (float(damp_arr) * np.eye(n_dof) if damp_arr.ndim == 0
                   else np.diag(damp_arr))
    solve_dofs_kb._last_cov = np.linalg.inv(H_final)
    solve_dofs_kb._last_H = H
    solve_dofs_kb._last_b = b
    solve_dofs_kb._last_weights = weights
    return delta
