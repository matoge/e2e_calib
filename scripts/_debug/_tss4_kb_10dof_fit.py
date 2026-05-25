"""10-DoF KB Gauss-Newton on full-frame npz: (omega_xyz, t_xyz, k1..k4).

Per-cell hard mean of (X, Y, Z) (metres, original-camera frame) gives the
3D point each cell represents; per-cell info-weighted Δuv (parent-px)
gives the residual; per-cell info matrix W aggregates all sub-crop
samples that fell into that cell. We linearise KB at the current
estimate and step δ ∈ R^10 until convergence.

Usage:
  python scripts/_debug/_tss4_kb_10dof_fit.py \\
      --npz scripts/_debug/_outputs/v2/tss4_full_frame_<ckpt>_cs256.npz \\
      --n-thresh 200 --v-frac 0.72 --v-min-frac 0.362 --n-iter 8
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


_D2R = np.pi / 180.0


def rodrigues(omega_deg: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(omega_deg) * _D2R
    if th < 1e-12:
        return np.eye(3)
    axis = (omega_deg * _D2R) / th
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def kb_project(P: np.ndarray, fx: float, fy: float, cx: float, cy: float,
               k: np.ndarray) -> np.ndarray:
    """KB forward: P=(M,3) cam frame → (M,2) px."""
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    r = np.sqrt(X * X + Y * Y + 1e-24)
    theta = np.arctan2(r, np.maximum(Z, 1e-9))
    t2 = theta**2; t4 = t2**2; t6 = t4*t2; t8 = t4**2
    poly = 1 + k[0]*t2 + k[1]*t4 + k[2]*t6 + k[3]*t8
    theta_d = theta * poly
    Xr = X / np.maximum(r, 1e-9)
    Yr = Y / np.maximum(r, 1e-9)
    u = fx * theta_d * Xr + cx
    v = fy * theta_d * Yr + cy
    return np.stack([u, v], axis=-1)


def kb_jacobian_10(P: np.ndarray, fx: float, fy: float,
                   k: np.ndarray) -> np.ndarray:
    """Per-point 2×10 KB Jacobian: cols = [omega_x..z (deg), tx..tz (m), k1..k4]."""
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    r2 = X*X + Y*Y
    r = np.sqrt(r2 + 1e-24)
    r_safe = np.where(r > 1e-9, r, 1.0)
    theta = np.arctan2(r, np.maximum(Z, 1e-9))
    t2 = theta**2; t4 = t2**2; t6 = t4*t2; t8 = t4**2
    poly = 1 + k[0]*t2 + k[1]*t4 + k[2]*t6 + k[3]*t8
    theta_d = theta * poly
    dtd_dtheta = (1 + 3*k[0]*t2 + 5*k[1]*t4 + 7*k[2]*t6 + 9*k[3]*t8)
    r2pz2 = r2 + Z*Z + 1e-24
    inv_r = 1.0 / r_safe
    inv_r2 = inv_r * inv_r
    Xr = X * inv_r
    Yr = Y * inv_r

    def chain(dX, dY, dZ):
        dr = (X * dX + Y * dY) * inv_r
        dtheta = (Z * dr - r * dZ) / r2pz2
        dtd = dtd_dtheta * dtheta
        dXr = dX * inv_r - X * dr * inv_r2
        dYr = dY * inv_r - Y * dr * inv_r2
        du = fx * (dtd * Xr + theta_d * dXr)
        dv = fy * (dtd * Yr + theta_d * dYr)
        return du, dv

    M = X.size
    zero = np.zeros(M)
    one = np.ones(M)
    cols_u = np.zeros((M, 10))
    cols_v = np.zeros((M, 10))
    # Note: ω in degrees → ∂P/∂ω = D2R · (skew rotation of P)
    # ∂(R(0)·P)/∂ω_x = D2R · [0, -Z, Y]; ∂/∂ω_y = D2R · [Z, 0, -X]; ∂/∂ω_z = D2R · [-Y, X, 0]
    cols_u[:, 0], cols_v[:, 0] = chain(zero, -Z * _D2R, Y * _D2R)
    cols_u[:, 1], cols_v[:, 1] = chain(Z * _D2R, zero, -X * _D2R)
    cols_u[:, 2], cols_v[:, 2] = chain(-Y * _D2R, X * _D2R, zero)
    # translation
    cols_u[:, 3], cols_v[:, 3] = chain(one, zero, zero)
    cols_u[:, 4], cols_v[:, 4] = chain(zero, one, zero)
    cols_u[:, 5], cols_v[:, 5] = chain(zero, zero, one)
    # k1..k4 (radial)
    cphi = Xr; sphi = Yr
    powers = np.stack([theta*t2, theta*t4, theta*t6, theta*t8], axis=-1)  # (M,4)
    cols_u[:, 6:] = (fx * cphi)[:, None] * powers
    cols_v[:, 6:] = (fy * sphi)[:, None] * powers
    return np.stack([cols_u, cols_v], axis=1)  # (M, 2, 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=Path, required=True)
    ap.add_argument('--n-thresh', type=int, default=200)
    ap.add_argument('--v-frac', type=float, default=0.72)
    ap.add_argument('--v-min-frac', type=float, default=0.362)
    ap.add_argument('--n-iter', type=int, default=8)
    ap.add_argument('--prior-omega-deg', type=float, default=2.0,
                    help='Gaussian prior σ on ω axes (deg)')
    ap.add_argument('--prior-t-m', type=float, default=0.5,
                    help='Gaussian prior σ on t axes (m)')
    ap.add_argument('--prior-k', type=float, default=1.0,
                    help='Gaussian prior σ on k_i (unitless), 1.0 = effectively off')
    ap.add_argument('--cache', type=Path,
                    default=Path('/mnt/fsx/tmp/hfunaya/e2e_calib_cache/tss4_v3_tiled'))
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    sum_n = z['sum_n']
    sum_W = z['sum_W']; sum_Wd = z['sum_Wd']
    sum_X = z['sum_X']; sum_Y = z['sum_Y']; sum_Z = z['sum_Z']
    cell_px = int(z['cell_px']); IW = int(z['IW']); IH = int(z['IH'])
    nH, nW = sum_n.shape

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from datasets.pandaset_full import PandaSetCalibDatasetFull
    ds = PandaSetCalibDatasetFull(
        cache_dir=args.cache, split='train',
        img_size=128, min_crop_px=128, max_crop_px=512,
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=16, center_band=0.0, preload=False)
    inst = ds._load_inst(0)
    K = inst['K_full'].numpy()
    dist0 = inst['distortion'].numpy().astype(np.float64)
    fx, fy, cx, cy = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])
    print(f'[10dof] K: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}')
    print(f'[10dof] dist_init k1..k4 = {dist0}')

    # mask
    cells_u = (np.arange(nW) + 0.5) * cell_px
    cells_v = (np.arange(nH) + 0.5) * cell_px
    UU, VV = np.meshgrid(cells_u, cells_v)
    v_max = args.v_frac * IH
    v_min = args.v_min_frac * IH
    band = (VV < v_max) & (VV > v_min)
    ok = (sum_n >= args.n_thresh) & band
    print(f'[10dof] active cells (n>={args.n_thresh}, {v_min:.0f}<v<{v_max:.0f}): '
          f'{int(ok.sum())}/{nW*nH}')

    cv_idx, cu_idx = np.where(ok)
    M = cv_idx.size
    n_c = sum_n[cv_idx, cu_idx].astype(np.float64)
    X_c = sum_X[cv_idx, cu_idx] / np.maximum(n_c, 1.0)  # hard mean of pts in metres
    Y_c = sum_Y[cv_idx, cu_idx] / np.maximum(n_c, 1.0)
    Z_c = sum_Z[cv_idx, cu_idx] / np.maximum(n_c, 1.0)
    P_c = np.stack([X_c, Y_c, Z_c], axis=-1)
    W_c = sum_W[cv_idx, cu_idx]
    Wd_c = sum_Wd[cv_idx, cu_idx]
    # info-weighted Δuv per cell
    d_meas = np.zeros((M, 2))
    keep = np.ones(M, dtype=bool)
    for i in range(M):
        try:
            d_meas[i] = np.linalg.solve(W_c[i], Wd_c[i])
        except np.linalg.LinAlgError:
            keep[i] = False
    cv_idx = cv_idx[keep]; cu_idx = cu_idx[keep]
    P_c = P_c[keep]; W_c = W_c[keep]; d_meas = d_meas[keep]
    M = int(keep.sum())
    print(f'[10dof] kept {M} cells with valid info-mean')

    # uv_obs at cell center is the model's "where the point lies"; uv_target = uv_obs + d_meas
    # We linearise around the current (R, t, k) and step. r = uv_target - kb_project(R·P+t; k).
    # Initial: R=I, t=0, k=k_init. Then uv_obs ≈ kb_project(P_c, k_init), so
    # initial residual = (cell_center + d_meas) - kb_project(P_c, k_init).
    # But cell centers were derived from observations (true_uv after model forward),
    # not exactly kb_project(P_c, k_init). To stay consistent with the per-cell
    # KB-only fit's setup, we treat d_meas itself as the residual the model wants
    # absorbed by Δ(R,t,k). That means uv_target = kb_project(P_c, R, t, k) + d_meas.
    # Linearised: r = d_meas - J · δ at δ=0 each iter? No — we update (R,t,k) by δ
    # absolutely each iteration, so the ACCUMULATED state is what defines the
    # current KB projection, and r = d_meas_residual_left.
    #
    # Implement as: maintain δ_acc (R, t, k). Each iter, transform P_c by current
    # (R, t), compute predicted uv (KB with current k), measure remaining residual
    # (target_uv - uv_pred), where target_uv = kb_project(P_c_orig, k_init) + d_meas
    # (i.e., what the network said the projection should be).
    target_uv = kb_project(P_c, fx, fy, cx, cy, dist0) + d_meas

    # GN state
    omega = np.zeros(3)   # deg
    t = np.zeros(3)       # m
    k = dist0.copy()
    prior_diag = np.zeros(10)
    prior_diag[0:3] = 1.0 / (args.prior_omega_deg ** 2)
    prior_diag[3:6] = 1.0 / (args.prior_t_m ** 2)
    prior_diag[6:10] = 1.0 / (args.prior_k ** 2)

    print(f'\n[10dof] iterating GN:')
    print(f'  iter   |δ|     ωxyz(°)             txyz(m)              k1..k4')
    print(f'  init                                                       k={k}')
    for it in range(args.n_iter):
        R = rodrigues(omega)
        P_lin = (R @ P_c.T).T + t  # (M, 3)
        uv_pred = kb_project(P_lin, fx, fy, cx, cy, k)
        r = target_uv - uv_pred                  # (M, 2)
        J = kb_jacobian_10(P_lin, fx, fy, k)     # (M, 2, 10)
        # GN with info matrices W_c
        # H = Σ J^T W J ; b = Σ J^T W r
        WJ = np.einsum('mij,mjk->mik', W_c, J)   # (M, 2, 10)
        H = np.einsum('mij,mik->jk', J, WJ)      # (10, 10)
        b = np.einsum('mij,mi->j', WJ, r)        # (10,)
        cost = float(np.einsum('mi,mij,mj->', r, W_c, r))
        wrms = float(np.sqrt(cost / max(1, np.einsum('mij->', W_c))))
        H = H + np.diag(prior_diag)
        try:
            delta = np.linalg.solve(H + 1e-9 * np.eye(10), b)
        except np.linalg.LinAlgError:
            delta = np.zeros(10)
        omega = omega + delta[0:3]
        t     = t + delta[3:6]
        k     = k + delta[6:10]
        print(f'  {it:3d}  {np.linalg.norm(delta):.2e}  '
              f'[{omega[0]:+.3f},{omega[1]:+.3f},{omega[2]:+.3f}]  '
              f'[{t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}]  '
              f'[{k[0]:+.3e},{k[1]:+.3e},{k[2]:+.3e},{k[3]:+.3e}]  '
              f'cost={cost:.3e} wrms={wrms:.3f}px')

    # final RMS at converged state
    R = rodrigues(omega)
    P_lin = (R @ P_c.T).T + t
    uv_pred = kb_project(P_lin, fx, fy, cx, cy, k)
    r = target_uv - uv_pred
    cost = float(np.einsum('mi,mij,mj->', r, W_c, r))
    wrms = float(np.sqrt(cost / max(1, np.einsum('mij->', W_c))))
    print(f'\n[10dof] final  ω={omega}  t={t}  k={k}')
    print(f'[10dof]  Δk = {k - dist0}')
    print(f'[10dof] final weighted-RMS = {wrms:.3f} px')

    out_json = args.npz.with_name(args.npz.stem +
        f'_10dof_n{args.n_thresh}_v{int(v_min)}-{int(v_max)}.json')
    out_json.write_text(json.dumps({
        'n_thresh': args.n_thresh, 'v_min': float(v_min), 'v_max': float(v_max),
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
        'dist_init': dist0.tolist(),
        'omega_deg': omega.tolist(),
        't_m':       t.tolist(),
        'dist_fit':  k.tolist(),
        'delta_k':   (k - dist0).tolist(),
        'final_wrms_px': wrms,
        'n_cells_used': int(M),
        'prior_omega_deg': args.prior_omega_deg,
        'prior_t_m': args.prior_t_m,
        'prior_k': args.prior_k,
    }, indent=2))
    print(f'[10dof] wrote {out_json}')


if __name__ == '__main__':
    main()
