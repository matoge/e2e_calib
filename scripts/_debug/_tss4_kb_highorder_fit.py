"""High-order KB GN fit.  Variable radial order  K ∈ {4, 5, 6, 8}.

  θ_d = θ · (1 + Σ_{i=1..K} k_i · θ^{2i})

Other DOF kept identical to the 11-DoF baseline:
   ω_xyz, dfx, dfy, dcx, dcy, [k_1..k_K]   (cxy lockable)

Designed to test "is the right-edge 30 px residual just higher-order
radial that θ⁸ couldn't reach?" TSS4 θ_max ≈ 0.7 rad → θ¹⁰ ≈ 0.028,
θ¹² ≈ 0.014 — small but nonzero, may absorb a chunk if it's truly
radial.

Usage:
   python scripts/_debug/_tss4_kb_highorder_fit.py \
       --npz <npz> --order 6 [--lock-cxy]
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


def kb_theta_from_uv(u, v, fx, fy, cx, cy, k4_init, n_newton=10):
    du = (u - cx); dv = (v - cy)
    r_pix = np.sqrt(du**2 + dv**2)
    f_mean = 0.5 * (fx + fy)
    theta_d = r_pix / f_mean
    theta = theta_d.copy()
    for _ in range(n_newton):
        t2 = theta**2; t4 = t2**2; t6 = t4*t2; t8 = t4**2
        f = theta * (1 + k4_init[0]*t2 + k4_init[1]*t4
                     + k4_init[2]*t6 + k4_init[3]*t8) - theta_d
        df = (1 + 3*k4_init[0]*t2 + 5*k4_init[1]*t4
              + 7*k4_init[2]*t6 + 9*k4_init[3]*t8)
        theta = theta - f / np.maximum(df, 1e-9)
    return theta


def project_unit_ray_K(nu, fx, fy, cx, cy, k):
    X, Y, Z = nu[:, 0], nu[:, 1], nu[:, 2]
    r = np.sqrt(X*X + Y*Y + 1e-24)
    theta = np.arctan2(r, np.maximum(Z, 1e-9))
    poly = np.ones_like(theta)
    t2 = theta * theta
    tp = t2.copy()
    for ki in k:
        poly = poly + ki * tp
        tp = tp * t2
    theta_d = theta * poly
    Xr = X / np.maximum(r, 1e-9); Yr = Y / np.maximum(r, 1e-9)
    u = fx * theta_d * Xr + cx
    v = fy * theta_d * Yr + cy
    return np.stack([u, v], axis=-1)


def jacobian_K(nu, fx, fy, k):
    K_order = len(k)
    P = K_order + 7
    X, Y, Z = nu[:, 0], nu[:, 1], nu[:, 2]
    r2 = X*X + Y*Y
    r = np.sqrt(r2 + 1e-24)
    r_safe = np.where(r > 1e-9, r, 1.0)
    theta = np.arctan2(r, np.maximum(Z, 1e-9))
    t2 = theta * theta
    poly = np.ones_like(theta)
    dpoly = np.zeros_like(theta)
    tp = t2.copy()
    for i, ki in enumerate(k):
        poly = poly + ki * tp
        dpoly = dpoly + (2*(i+1)) * ki * (theta ** (2*(i+1) - 1))
        tp = tp * t2
    theta_d = theta * poly
    dtd_dtheta = poly + theta * dpoly
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
    zero = np.zeros(M); one = np.ones(M)
    cols_u = np.zeros((M, P)); cols_v = np.zeros((M, P))
    cols_u[:, 0], cols_v[:, 0] = chain(zero, -Z * _D2R, Y * _D2R)
    cols_u[:, 1], cols_v[:, 1] = chain(Z * _D2R, zero, -X * _D2R)
    cols_u[:, 2], cols_v[:, 2] = chain(-Y * _D2R, X * _D2R, zero)
    powers = np.stack([theta ** (2*(i+1) + 1) for i in range(K_order)], axis=-1)
    cols_u[:, 3:3+K_order] = (fx * Xr)[:, None] * powers
    cols_v[:, 3:3+K_order] = (fy * Yr)[:, None] * powers
    cols_u[:, 3+K_order]   = fx * theta_d * Xr
    cols_v[:, 3+K_order]   = zero
    cols_u[:, 4+K_order]   = zero
    cols_v[:, 4+K_order]   = fy * theta_d * Yr
    cols_u[:, 5+K_order]   = one;  cols_v[:, 5+K_order] = zero
    cols_u[:, 6+K_order]   = zero; cols_v[:, 6+K_order] = one
    return np.stack([cols_u, cols_v], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=Path, required=True)
    ap.add_argument('--order', type=int, default=6,
                    help='KB radial order K (k_1..k_K). 4=baseline, 5/6/8 high.')
    ap.add_argument('--n-thresh', type=int, default=200)
    ap.add_argument('--v-frac', type=float, default=0.72)
    ap.add_argument('--v-min-frac', type=float, default=0.362)
    ap.add_argument('--n-iter', type=int, default=15)
    ap.add_argument('--prior-omega-deg', type=float, default=2.0)
    ap.add_argument('--prior-k', type=float, default=1.0)
    ap.add_argument('--prior-dfxy', type=float, default=0.05)
    ap.add_argument('--prior-dcxy-px', type=float, default=50.0)
    ap.add_argument('--lock-cxy', action='store_true')
    ap.add_argument('--cache', type=Path,
                    default=Path('/mnt/fsx/tmp/hfunaya/e2e_calib_cache/tss4_v3_tiled'))
    args = ap.parse_args()
    K = int(args.order)
    assert K >= 4

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
    K_intr = inst['K_full'].numpy()
    dist0_4 = inst['distortion'].numpy().astype(np.float64)
    fx, fy, cx, cy = float(K_intr[0,0]), float(K_intr[1,1]), float(K_intr[0,2]), float(K_intr[1,2])
    print(f'[KB{K}] K: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}')
    print(f'[KB{K}] dist_init (4) = {dist0_4}')

    cells_u = (np.arange(nW) + 0.5) * cell_px
    cells_v = (np.arange(nH) + 0.5) * cell_px
    UU, VV = np.meshgrid(cells_u, cells_v)
    v_max = args.v_frac * IH; v_min = args.v_min_frac * IH
    band = (VV < v_max) & (VV > v_min)
    ok = (sum_n >= args.n_thresh) & band
    cv_idx, cu_idx = np.where(ok)
    M = cv_idx.size
    u_c = UU[cv_idx, cu_idx]; v_c = VV[cv_idx, cu_idx]
    W_c = sum_W[cv_idx, cu_idx]; Wd_c = sum_Wd[cv_idx, cu_idx]
    d_meas = np.zeros((M, 2)); keep = np.ones(M, dtype=bool)
    for i in range(M):
        try: d_meas[i] = np.linalg.solve(W_c[i], Wd_c[i])
        except np.linalg.LinAlgError: keep[i] = False
    u_c=u_c[keep]; v_c=v_c[keep]; W_c=W_c[keep]; d_meas=d_meas[keep]
    M = int(keep.sum())
    print(f'[KB{K}] active cells: {M}')

    theta0 = kb_theta_from_uv(u_c, v_c, fx, fy, cx, cy, dist0_4, n_newton=10)
    phi0 = np.arctan2(v_c - cy, u_c - cx)
    nu_orig = np.stack([np.sin(theta0)*np.cos(phi0),
                        np.sin(theta0)*np.sin(phi0),
                        np.cos(theta0)], axis=-1)
    target_uv = np.stack([u_c, v_c], axis=-1) + d_meas

    omega = np.zeros(3)
    k = np.zeros(K, dtype=np.float64)
    k[:4] = dist0_4
    dfx=0.0; dfy=0.0; dcx=0.0; dcy=0.0
    P = K + 7
    prior_diag = np.zeros(P)
    prior_diag[0:3]            = 1.0 / (args.prior_omega_deg ** 2)
    prior_diag[3:3+K]          = 1.0 / (args.prior_k ** 2)
    prior_diag[3+K:5+K]        = 1.0 / (args.prior_dfxy ** 2)
    prior_diag[5+K:7+K]        = 1.0 / (args.prior_dcxy_px ** 2)
    if args.lock_cxy:
        prior_diag[5+K:7+K] = 1.0 / (1e-6 ** 2)
        print(f'[KB{K}] cxy LOCKED')

    print(f'\n[KB{K}] iterating GN (P={P}):')
    for it in range(args.n_iter):
        R = rodrigues(omega); nu_lin = nu_orig @ R.T
        fx_e = fx * (1.0 + dfx); fy_e = fy * (1.0 + dfy)
        cx_e = cx + dcx;         cy_e = cy + dcy
        uv_pred = project_unit_ray_K(nu_lin, fx_e, fy_e, cx_e, cy_e, k)
        r = target_uv - uv_pred
        J = jacobian_K(nu_lin, fx_e, fy_e, k)
        WJ = np.einsum('mij,mjk->mik', W_c, J)
        H = np.einsum('mij,mik->jk', J, WJ)
        b = np.einsum('mij,mi->j', WJ, r)
        cost = float(np.einsum('mi,mij,mj->', r, W_c, r))
        wrms = float(np.sqrt(cost / max(1, np.einsum('mij->', W_c))))
        H = H + np.diag(prior_diag)
        try:
            delta = np.linalg.solve(H + 1e-9 * np.eye(P), b)
        except np.linalg.LinAlgError:
            delta = np.zeros(P)
        omega = omega + delta[0:3]
        k     = k + delta[3:3+K]
        dfx   = dfx + delta[3+K]
        dfy   = dfy + delta[4+K]
        dcx   = dcx + delta[5+K]
        dcy   = dcy + delta[6+K]
        ks_str = ','.join(f'{kk:+.3f}' for kk in k)
        print(f'  it={it:2d}  |δ|={np.linalg.norm(delta):.2e}  '
              f'ω=[{omega[0]:+.3f},{omega[1]:+.3f},{omega[2]:+.3f}]°  '
              f'fxy*=[{1+dfx:.4f},{1+dfy:.4f}] cxy+=[{dcx:+.1f},{dcy:+.1f}]  '
              f'k=[{ks_str}]  wrms={wrms:.3f}')

    R = rodrigues(omega); nu_lin = nu_orig @ R.T
    fx_e = fx * (1.0 + dfx); fy_e = fy * (1.0 + dfy)
    cx_e = cx + dcx;         cy_e = cy + dcy
    uv_pred = project_unit_ray_K(nu_lin, fx_e, fy_e, cx_e, cy_e, k)
    r = target_uv - uv_pred
    cost = float(np.einsum('mi,mij,mj->', r, W_c, r))
    wrms = float(np.sqrt(cost / max(1, np.einsum('mij->', W_c))))
    print(f'\n[KB{K}] final wrms = {wrms:.3f} px   ω={omega}°  k={k}')

    out_json = args.npz.with_name(args.npz.stem +
        f'_KB{K}_n{args.n_thresh}_v{int(v_min)}-{int(v_max)}'
        + ('_lockcxy' if args.lock_cxy else '') + '.json')
    out_json.write_text(json.dumps({
        'fx_init': fx, 'fy_init': fy, 'cx_init': cx, 'cy_init': cy,
        'fx_fit': fx_e, 'fy_fit': fy_e, 'cx_fit': cx_e, 'cy_fit': cy_e,
        'omega_deg': omega.tolist(),
        'dist_init_4': dist0_4.tolist(),
        'dist_fit':    k.tolist(),
        'order_K':     K,
        'lock_cxy':    bool(args.lock_cxy),
        'final_wrms_px': wrms,
        'n_cells': int(M),
    }, indent=2))
    print(f'[KB{K}] wrote {out_json}')


if __name__ == '__main__':
    main()
