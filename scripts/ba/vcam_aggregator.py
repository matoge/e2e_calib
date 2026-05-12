"""VCAM-frame BA aggregator — combine per-tile (μ, Σ) outputs from the CLS
pose-head (trained with pose_frame='vcam') into a single orig-camera 6-DoF
δ via information-matrix sum. No learning, purely linear algebra.

Each tile produces (μ_v, log_σ_v) where μ_v ∈ R^5 (tx_v, ty_v, tz_v, yaw_v,
pitch_v) is the perturbation expressed in the tile's VCAM frame whose
optical axis is the ray through tile center. The orig→VCAM rotation R_i
(known from tile geometry, no learning) maps orig δ_orig (6-DoF: t_orig
+ yaw + pitch + roll) onto the tile's VCAM 5-DoF observation:

    μ_i = J_i @ δ_orig + ε_i,  ε_i ~ N(0, Σ_i)

    J_i = [[ R_i, 0   ],     # 3×3 translation block (orig→VCAM)
           [ 0,   R_i_rot ]]  # 2×3 rotation block (drop roll_v row)

H δ_orig = b solves for the MLE:
    H = Σ_i J_i^T Σ_i^{-1} J_i,  b = Σ_i J_i^T Σ_i^{-1} μ_i
    δ_orig = H^{-1} b,  Cov(δ_orig) = H^{-1}

Usage:
    from scripts.ba.vcam_aggregator import aggregate_vcam_to_orig
    delta_orig, cov_orig = aggregate_vcam_to_orig(
        mus=[mu_1, mu_2, ...],         # (N, 5) per-tile VCAM mean
        log_sigmas=[ls_1, ls_2, ...],  # (N, 5) per-tile log_sigma (diag)
        tile_centers_uv=[(u_c1, v_c1), ...],
        K_orig=K)                     # (3, 3) orig camera intrinsics
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def _R_orig_to_vcam(uc: float, vc: float, K: np.ndarray) -> np.ndarray:
    """Rotation orig → VCAM such that r_i (ray through tile center, in orig
    cam frame) is mapped to [0,0,1]^T (VCAM optical axis)."""
    ray = np.array([(uc - K[0,2]) / K[0,0],
                     (vc - K[1,2]) / K[1,1],
                     1.0], dtype=np.float64)
    r_i = ray / (np.linalg.norm(ray) + 1e-12)
    z_ax = np.array([0., 0., 1.])
    axis = np.cross(r_i, z_ax)
    an = np.linalg.norm(axis)
    if an < 1e-9:
        return np.eye(3) if r_i[2] > 0 else -np.eye(3)
    axis = axis / an
    angle = float(np.arccos(np.clip(r_i @ z_ax, -1.0, 1.0)))
    return Rotation.from_rotvec(axis * angle).as_matrix()


def aggregate_vcam_to_orig(*, mus, log_sigmas, tile_centers_uv,
                            K_orig: np.ndarray, damping: float = 1e-6):
    """Aggregate N per-tile VCAM observations into orig 6-DoF δ + cov.

    Args:
      mus:               (N, 5) per-tile (tx_v, ty_v, tz_v, yaw_v, pitch_v).
                         All in meters (translation) / degrees (rotation).
      log_sigmas:        (N, 5) per-tile log_sigma (diagonal Gaussian).
      tile_centers_uv:   (N, 2) (u_center, v_center) of each tile in orig px.
      K_orig:            (3, 3) orig camera intrinsic matrix.
      damping:           Levenberg term on H diag (default 1e-6).

    Returns:
      delta_orig:        (6,) [tx, ty, tz, yaw, pitch, roll] in orig frame
                         (translation in meters, ypr in degrees).
      cov_orig:          (6, 6) posterior covariance.
    """
    mus = np.asarray(mus, dtype=np.float64)              # (N, 5)
    log_sigmas = np.asarray(log_sigmas, dtype=np.float64)
    centers = np.asarray(tile_centers_uv, dtype=np.float64)
    N = len(mus)
    assert mus.shape == (N, 5) and log_sigmas.shape == (N, 5) and centers.shape == (N, 2)

    H = np.zeros((6, 6))
    b = np.zeros(6)
    for i in range(N):
        R_o_v = _R_orig_to_vcam(centers[i, 0], centers[i, 1], K_orig)
        # Jacobian J_i: (5, 6)
        #   t_v[0:3] = R_o_v @ t_o[0:3]   → rows 0-2, cols 0-2 = R_o_v
        #   ypr_v[0:2] = (R_o_v applied to angular vel) but yaw/pitch only
        #     For small angles: ω_v = R_o_v @ ω_o; we keep ω_v_z (yaw) and ω_v_y (pitch).
        #     Convention: ypr_o = [yaw, pitch, roll] where rotation = R_zyx(yaw, pitch, roll).
        #     Angular velocity components: ω_o = [ω_x, ω_y, ω_z] = [roll, pitch, yaw] order.
        #     yaw_v = ω_v_z; pitch_v = ω_v_y.
        #     ω_v = R_o_v @ ω_o → ω_v[2] = R_o_v[2] @ ω_o; ω_v[1] = R_o_v[1] @ ω_o.
        #     ω_o order is (roll, pitch, yaw) for axes (x, y, z) — but our δ_orig
        #     is laid out as (yaw, pitch, roll) per our convention. So:
        #       δ_orig[3] = yaw_o = ω_z_o
        #       δ_orig[4] = pitch_o = ω_y_o
        #       δ_orig[5] = roll_o = ω_x_o
        #     → ω_o = [δ[5], δ[4], δ[3]] (x, y, z components)
        #     yaw_v = ω_v_z = R_o_v[2,:] @ [δ[5], δ[4], δ[3]]
        #     pitch_v = ω_v_y = R_o_v[1,:] @ [δ[5], δ[4], δ[3]]
        J = np.zeros((5, 6))
        J[0:3, 0:3] = R_o_v
        # Map δ_orig[3..5] = (yaw, pitch, roll) to ω_o = (roll, pitch, yaw)
        # then apply R_o_v rows 2 (yaw_v) and 1 (pitch_v)
        # Equivalent matrix: J[3:5, 3:6] = R_o_v[[2, 1], :][:, [2, 1, 0]]
        rot_perm = R_o_v[[2, 1], :][:, [2, 1, 0]]   # (2, 3) maps (yaw_o, pitch_o, roll_o) → (yaw_v, pitch_v)
        J[3:5, 3:6] = rot_perm

        sig = np.exp(log_sigmas[i])
        var = sig * sig
        Sinv = np.diag(1.0 / np.maximum(var, 1e-12))  # (5, 5)
        H += J.T @ Sinv @ J
        b += J.T @ Sinv @ mus[i]

    H_reg = H + damping * np.eye(6)
    delta_orig = np.linalg.solve(H_reg, b)
    cov_orig = np.linalg.inv(H_reg)
    return delta_orig, cov_orig


if __name__ == '__main__':
    # quick self-test
    np.random.seed(0)
    K = np.array([[1900.0, 0, 960.0], [0, 1900.0, 540.0], [0, 0, 1]])
    delta_true = np.array([0.30, -0.20, 0.10, 0.80, -0.50, 0.30])
    N = 30
    centers = np.random.rand(N, 2) * np.array([1920, 1080])
    mus = np.zeros((N, 5))
    log_sigmas = np.full((N, 5), np.log(0.1))
    for i in range(N):
        R_o_v = _R_orig_to_vcam(centers[i, 0], centers[i, 1], K)
        J = np.zeros((5, 6))
        J[0:3, 0:3] = R_o_v
        J[3:5, 3:6] = R_o_v[[2, 1], :][:, [2, 1, 0]]
        mus[i] = J @ delta_true + np.random.randn(5) * 0.02
    delta_est, cov_est = aggregate_vcam_to_orig(
        mus=mus, log_sigmas=log_sigmas, tile_centers_uv=centers, K_orig=K)
    print(f'true: {delta_true.round(3)}')
    print(f'est : {delta_est.round(3)}')
    print(f'err : {(delta_true - delta_est).round(4)}')
    print(f'σ   : {np.sqrt(np.diag(cov_est)).round(4)}')
