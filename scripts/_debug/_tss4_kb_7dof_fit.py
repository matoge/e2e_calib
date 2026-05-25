"""7-DoF KB GN on full-frame npz: (ω_x, ω_y, ω_z, k1..k4).

Rotation + KB radial only — translation is omitted (would require XYZ;
also degenerates with k_i in 1-frame periphery fits). Geometry:

  per-cell (u_c, v_c) → unit-ray ν via KB-inv (φ, θ at k_init)
  apply ω : ν' = R(ω) ν
  re-project: r' = √(ν'_x²+ν'_y²), θ' = atan2(r', ν'_z)
  KB(θ', k) → uv_pred
  residual = (u_c + d_meas) - uv_pred

Per-cell info matrix W as weighting.
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


def kb_theta_from_uv(u, v, fx, fy, cx, cy, k, n_newton=8):
    du = (u - cx); dv = (v - cy)
    r_pix = np.sqrt(du**2 + dv**2)
    f_mean = 0.5 * (fx + fy)
    theta_d = r_pix / f_mean
    theta = theta_d.copy()
    for _ in range(n_newton):
        t2 = theta**2; t4 = t2**2; t6 = t4*t2; t8 = t4**2
        f = theta * (1 + k[0]*t2 + k[1]*t4 + k[2]*t6 + k[3]*t8) - theta_d
        df = (1 + 3*k[0]*t2 + 5*k[1]*t4 + 7*k[2]*t6 + 9*k[3]*t8)
        theta = theta - f / np.maximum(df, 1e-9)
    return theta


def project_unit_ray(nu, fx, fy, cx, cy, k):
    """nu (M,3) unit cam ray → (M,2) px via KB."""
    X, Y, Z = nu[:, 0], nu[:, 1], nu[:, 2]
    r = np.sqrt(X*X + Y*Y + 1e-24)
    theta = np.arctan2(r, np.maximum(Z, 1e-9))
    t2 = theta**2; t4 = t2**2; t6 = t4*t2; t8 = t4**2
    poly = 1 + k[0]*t2 + k[1]*t4 + k[2]*t6 + k[3]*t8
    theta_d = theta * poly
    Xr = X / np.maximum(r, 1e-9); Yr = Y / np.maximum(r, 1e-9)
    u = fx * theta_d * Xr + cx
    v = fy * theta_d * Yr + cy
    return np.stack([u, v], axis=-1)


def kb_jacobian_11(nu, fx, fy, k):
    """(M, 2, 11) cols: [ωx, ωy, ωz (deg), k1..k4, dfx, dfy, dcx, dcy].
    dfx/dfy multiplicative (fx_new = fx·(1+dfx)); dcx/dcy additive (px)."""
    X, Y, Z = nu[:, 0], nu[:, 1], nu[:, 2]
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
    Xr = X * inv_r; Yr = Y * inv_r

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
    cols_u = np.zeros((M, 11))
    cols_v = np.zeros((M, 11))
    cols_u[:, 0], cols_v[:, 0] = chain(zero, -Z * _D2R, Y * _D2R)
    cols_u[:, 1], cols_v[:, 1] = chain(Z * _D2R, zero, -X * _D2R)
    cols_u[:, 2], cols_v[:, 2] = chain(-Y * _D2R, X * _D2R, zero)
    powers = np.stack([theta*t2, theta*t4, theta*t6, theta*t8], axis=-1)
    cols_u[:, 3:7] = (fx * Xr)[:, None] * powers
    cols_v[:, 3:7] = (fy * Yr)[:, None] * powers
    # dfx (multiplicative): u contributed by fx · θ_d · X/r ; ∂u/∂dfx = fx · θ_d · X/r
    cols_u[:, 7]  = fx * theta_d * Xr;  cols_v[:, 7]  = zero
    cols_u[:, 8]  = zero;               cols_v[:, 8]  = fy * theta_d * Yr
    # dcx, dcy: additive in px
    cols_u[:, 9]  = one;  cols_v[:, 9]  = zero
    cols_u[:, 10] = zero; cols_v[:, 10] = one
    return np.stack([cols_u, cols_v], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=Path, required=True)
    ap.add_argument('--n-thresh', type=int, default=200)
    ap.add_argument('--v-frac', type=float, default=0.72)
    ap.add_argument('--v-min-frac', type=float, default=0.362)
    ap.add_argument('--n-iter', type=int, default=10)
    ap.add_argument('--prior-omega-deg', type=float, default=2.0)
    ap.add_argument('--prior-k', type=float, default=1.0)
    ap.add_argument('--prior-dfxy', type=float, default=0.05,
                    help='σ on relative focal change (5% default)')
    ap.add_argument('--prior-dcxy-px', type=float, default=50.0,
                    help='σ on principal-point shift (px)')
    ap.add_argument('--cache', type=Path,
                    default=Path('/mnt/fsx/tmp/hfunaya/e2e_calib_cache/tss4_v3_tiled'))
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    sum_n = z['sum_n']; sum_W = z['sum_W']; sum_Wd = z['sum_Wd']
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
    print(f'[7dof] K: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}')
    print(f'[7dof] dist_init = {dist0}')

    cells_u = (np.arange(nW) + 0.5) * cell_px
    cells_v = (np.arange(nH) + 0.5) * cell_px
    UU, VV = np.meshgrid(cells_u, cells_v)
    v_max = args.v_frac * IH
    v_min = args.v_min_frac * IH
    band = (VV < v_max) & (VV > v_min)
    ok = (sum_n >= args.n_thresh) & band
    print(f'[7dof] active cells: {int(ok.sum())}/{nW*nH}')

    cv_idx, cu_idx = np.where(ok)
    M = cv_idx.size
    u_c = UU[cv_idx, cu_idx]; v_c = VV[cv_idx, cu_idx]
    W_c = sum_W[cv_idx, cu_idx]
    Wd_c = sum_Wd[cv_idx, cu_idx]
    d_meas = np.zeros((M, 2))
    keep = np.ones(M, dtype=bool)
    for i in range(M):
        try:
            d_meas[i] = np.linalg.solve(W_c[i], Wd_c[i])
        except np.linalg.LinAlgError:
            keep[i] = False
    u_c = u_c[keep]; v_c = v_c[keep]
    W_c = W_c[keep]; d_meas = d_meas[keep]
    M = int(keep.sum())
    print(f'[7dof] kept {M} cells')

    # cell-center → unit ray via KB-inv at k_init
    theta0 = kb_theta_from_uv(u_c, v_c, fx, fy, cx, cy, dist0, n_newton=10)
    phi0 = np.arctan2(v_c - cy, u_c - cx)
    nu_orig = np.stack([
        np.sin(theta0) * np.cos(phi0),
        np.sin(theta0) * np.sin(phi0),
        np.cos(theta0),
    ], axis=-1)  # (M, 3) unit cam ray at k_init geometry

    # target: cell center + measured Δuv (where the network says it should land)
    target_uv = np.stack([u_c, v_c], axis=-1) + d_meas

    omega = np.zeros(3)
    k = dist0.copy()
    dfx = 0.0; dfy = 0.0; dcx = 0.0; dcy = 0.0
    prior_diag = np.zeros(11)
    prior_diag[0:3]  = 1.0 / (args.prior_omega_deg ** 2)
    prior_diag[3:7]  = 1.0 / (args.prior_k ** 2)
    prior_diag[7:9]  = 1.0 / (args.prior_dfxy ** 2)
    prior_diag[9:11] = 1.0 / (args.prior_dcxy_px ** 2)

    print(f'\n[11dof] iterating GN (ω + k1..k4 + dfx,dfy,dcx,dcy):')
    for it in range(args.n_iter):
        R = rodrigues(omega)
        nu_lin = nu_orig @ R.T
        fx_e = fx * (1.0 + dfx); fy_e = fy * (1.0 + dfy)
        cx_e = cx + dcx;         cy_e = cy + dcy
        uv_pred = project_unit_ray(nu_lin, fx_e, fy_e, cx_e, cy_e, k)
        r = target_uv - uv_pred
        # Jacobian uses CURRENT effective intrinsics so partials are correct
        J = kb_jacobian_11(nu_lin, fx_e, fy_e, k)
        WJ = np.einsum('mij,mjk->mik', W_c, J)
        H = np.einsum('mij,mik->jk', J, WJ)
        b = np.einsum('mij,mi->j', WJ, r)
        cost = float(np.einsum('mi,mij,mj->', r, W_c, r))
        wrms = float(np.sqrt(cost / max(1, np.einsum('mij->', W_c))))
        H = H + np.diag(prior_diag)
        try:
            delta = np.linalg.solve(H + 1e-9 * np.eye(11), b)
        except np.linalg.LinAlgError:
            delta = np.zeros(11)
        omega = omega + delta[0:3]
        k     = k + delta[3:7]
        # dfx/dfy are absolute increments to the multiplicative offset
        dfx   = dfx + delta[7]
        dfy   = dfy + delta[8]
        dcx   = dcx + delta[9]
        dcy   = dcy + delta[10]
        print(f'  it={it:2d}  |δ|={np.linalg.norm(delta):.2e}  '
              f'ω=[{omega[0]:+.3f},{omega[1]:+.3f},{omega[2]:+.3f}]°  '
              f'k=[{k[0]:+.3f},{k[1]:+.3f},{k[2]:+.3f},{k[3]:+.3f}]  '
              f'fx*={1+dfx:.4f} fy*={1+dfy:.4f} cx+={dcx:+.1f} cy+={dcy:+.1f}  '
              f'wrms={wrms:.3f}px')

    # final
    R = rodrigues(omega)
    nu_lin = nu_orig @ R.T
    fx_e = fx * (1.0 + dfx); fy_e = fy * (1.0 + dfy)
    cx_e = cx + dcx;         cy_e = cy + dcy
    uv_pred = project_unit_ray(nu_lin, fx_e, fy_e, cx_e, cy_e, k)
    r = target_uv - uv_pred
    cost = float(np.einsum('mi,mij,mj->', r, W_c, r))
    wrms = float(np.sqrt(cost / max(1, np.einsum('mij->', W_c))))
    print(f'\n[11dof] final ω={omega}°  k={k}')
    print(f'[11dof] fx_new={fx_e:.3f}  fy_new={fy_e:.3f}  '
          f'cx_new={cx_e:.3f}  cy_new={cy_e:.3f}')
    print(f'[11dof] Δk = {k - dist0}')
    print(f'[11dof] final weighted-RMS = {wrms:.3f} px')

    out_json = args.npz.with_name(args.npz.stem +
        f'_11dof_n{args.n_thresh}_v{int(v_min)}-{int(v_max)}.json')
    out_json.write_text(json.dumps({
        'fx_init': fx, 'fy_init': fy, 'cx_init': cx, 'cy_init': cy,
        'fx_fit': fx_e, 'fy_fit': fy_e, 'cx_fit': cx_e, 'cy_fit': cy_e,
        'dfx': dfx, 'dfy': dfy, 'dcx': dcx, 'dcy': dcy,
        'dist_init': dist0.tolist(),
        'omega_deg': omega.tolist(),
        'dist_fit': k.tolist(),
        'delta_k': (k - dist0).tolist(),
        'final_wrms_px': wrms,
        'n_cells_used': int(M),
    }, indent=2))
    print(f'[11dof] wrote {out_json}')


if __name__ == '__main__':
    main()
