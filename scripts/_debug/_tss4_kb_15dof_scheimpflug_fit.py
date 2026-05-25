"""13-DoF KB+tangential GN fit on full-frame npz.

Params: (ω_x, ω_y, ω_z, k1..k4, dfx, dfy, dcx, dcy, p1, p2)

Forward model (per cell):
  unit ray  ν₀ ← KB-inv at (K_init, k_init) on cell center
  ν       = R(ω) ν₀
  θ       = atan2(√(X²+Y²), Z),  Xr = X/r, Yr = Y/r
  θ_d     = θ·(1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸)
  X'      = θ_d · Xr,    Y' = θ_d · Yr        (post-radial normalized px)
  r'²     = X'² + Y'²  ( = θ_d² )
  Δu_tan  = 2 p1 X' Y' + p2 (r'² + 2 X'²)     (Brown–Conrady)
  Δv_tan  = p1 (r'² + 2 Y'²) + 2 p2 X' Y'
  u       = fx·(1+dfx)·(X' + Δu_tan) + (cx + dcx)
  v       = fy·(1+dfy)·(Y' + Δv_tan) + (cy + dcy)

Tangential models a tilted lens-vs-sensor: it produces ASYMMETRIC
left-right and top-bottom residual signatures that pure KB radial
cannot. Critical for the rightmost cells where 11-DoF still left
dv ≈ -0.5..-1 px aggregate (and 30+ px individual cells).

Per-cell info matrix W weighting; Gaussian priors on each block.
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


def kb_theta_from_uv(u, v, fx, fy, cx, cy, k, n_newton=10):
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


def project_unit_ray_tan(nu, fx, fy, cx, cy, k, p):
    """Forward with KB radial + Brown–Conrady tangential."""
    X, Y, Z = nu[:, 0], nu[:, 1], nu[:, 2]
    r = np.sqrt(X*X + Y*Y + 1e-24)
    theta = np.arctan2(r, np.maximum(Z, 1e-9))
    t2 = theta**2; t4 = t2**2; t6 = t4*t2; t8 = t4**2
    poly = 1 + k[0]*t2 + k[1]*t4 + k[2]*t6 + k[3]*t8
    theta_d = theta * poly
    Xr = X / np.maximum(r, 1e-9); Yr = Y / np.maximum(r, 1e-9)
    Xp = theta_d * Xr; Yp = theta_d * Yr
    r2p = Xp*Xp + Yp*Yp
    du_t = 2*p[0]*Xp*Yp + p[1]*(r2p + 2*Xp*Xp)
    dv_t = p[0]*(r2p + 2*Yp*Yp) + 2*p[1]*Xp*Yp
    u = fx * (Xp + du_t) + cx
    v = fy * (Yp + dv_t) + cy
    return np.stack([u, v], axis=-1)


def kb_jacobian_13(nu, fx, fy, k, p):
    """(M, 2, 13) cols: [ωx, ωy, ωz, k1..k4, dfx, dfy, dcx, dcy, p1, p2].

    NOTE: ω, k, dfxy chain uses RADIAL-ONLY partials (small-p approx).
    p1, p2, dcxy are exact. This is fine for GN convergence — the
    second-order tangential cross-term is ≪ residual scale at small p.
    """
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
    Xp = theta_d * Xr; Yp = theta_d * Yr
    r2p = Xp*Xp + Yp*Yp

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
    cols_u = np.zeros((M, 13))
    cols_v = np.zeros((M, 13))
    cols_u[:, 0], cols_v[:, 0] = chain(zero, -Z * _D2R, Y * _D2R)
    cols_u[:, 1], cols_v[:, 1] = chain(Z * _D2R, zero, -X * _D2R)
    cols_u[:, 2], cols_v[:, 2] = chain(-Y * _D2R, X * _D2R, zero)
    powers = np.stack([theta*t2, theta*t4, theta*t6, theta*t8], axis=-1)
    cols_u[:, 3:7] = (fx * Xr)[:, None] * powers
    cols_v[:, 3:7] = (fy * Yr)[:, None] * powers
    cols_u[:, 7]  = fx * theta_d * Xr;  cols_v[:, 7]  = zero
    cols_u[:, 8]  = zero;               cols_v[:, 8]  = fy * theta_d * Yr
    cols_u[:, 9]  = one;  cols_v[:, 9]  = zero
    cols_u[:, 10] = zero; cols_v[:, 10] = one
    # tangential p1, p2 — closed form on Xp, Yp
    cols_u[:, 11] = fx * 2 * Xp * Yp
    cols_v[:, 11] = fy * (r2p + 2*Yp*Yp)
    cols_u[:, 12] = fx * (r2p + 2*Xp*Xp)
    cols_v[:, 12] = fy * 2 * Xp * Yp
    return np.stack([cols_u, cols_v], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=Path, required=True)
    ap.add_argument('--n-thresh', type=int, default=200)
    ap.add_argument('--v-frac', type=float, default=0.72)
    ap.add_argument('--v-min-frac', type=float, default=0.362)
    ap.add_argument('--n-iter', type=int, default=12)
    ap.add_argument('--prior-omega-deg', type=float, default=2.0)
    ap.add_argument('--prior-k', type=float, default=1.0)
    ap.add_argument('--prior-dfxy', type=float, default=0.05)
    ap.add_argument('--prior-dcxy-px', type=float, default=50.0)
    ap.add_argument('--prior-p', type=float, default=0.01,
                    help='σ on Brown–Conrady p1, p2 (unitless)')
    ap.add_argument('--lock-cxy', action='store_true',
                    help='freeze dcx=dcy=0 (cx,cy are degenerate w/ yaw,pitch)')
    ap.add_argument('--init-from-json', type=Path, default=None,
                    help='start GN from a previous FIT json (uses *_fit values '
                         'as the new baseline calib; the previous omega is '
                         'pre-applied to nu_baseline; this iteration solves a '
                         'small delta on top).')
    ap.add_argument('--out-suffix', type=str, default='',
                    help='extra suffix added to output json filename '
                         '(e.g. _iter2)')
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
    # K_recalib / dist_recalib are the calib that npz residuals were produced
    # under — keep these for the KB-inv step (cell uv → unit ray ν₀).
    fx_rc = float(K[0, 0]); fy_rc = float(K[1, 1])
    cx_rc = float(K[0, 2]); cy_rc = float(K[1, 2])
    dist_rc = dist0.copy()
    fx, fy, cx, cy = fx_rc, fy_rc, cx_rc, cy_rc
    prev_omega = np.zeros(3)
    prev_p = np.zeros(2)
    if args.init_from_json is not None:
        prev = json.loads(args.init_from_json.read_text())
        fx = float(prev['fx_fit']); fy = float(prev['fy_fit'])
        cx = float(prev['cx_fit']); cy = float(prev['cy_fit'])
        dist0 = np.asarray(prev['dist_fit'], dtype=np.float64)
        if dist0.size != 4:
            raise SystemExit(f'init-from-json: dist_fit must be 4 KB coefs, '
                             f'got {dist0.size}')
        prev_omega = np.asarray(prev['omega_deg'], dtype=np.float64)
        if 'tangential_p' in prev:
            prev_p = np.asarray(prev['tangential_p'], dtype=np.float64)
        print(f'[13dof] init-from-json: {args.init_from_json.name}')
        print(f'[13dof]   prev ω={prev_omega}°  prev p={prev_p}')
        print(f'[13dof]   prev wrms={prev.get("final_wrms_px","?"):.4f}px')
    print(f'[13dof] K_baseline: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}')
    print(f'[13dof] dist_baseline = {dist0}')
    print(f'[13dof] K_recalib (for KB-inv): fx={fx_rc:.2f} fy={fy_rc:.2f} '
          f'cx={cx_rc:.2f} cy={cy_rc:.2f}')

    cells_u = (np.arange(nW) + 0.5) * cell_px
    cells_v = (np.arange(nH) + 0.5) * cell_px
    UU, VV = np.meshgrid(cells_u, cells_v)
    v_max = args.v_frac * IH
    v_min = args.v_min_frac * IH
    band = (VV < v_max) & (VV > v_min)
    ok = (sum_n >= args.n_thresh) & band
    print(f'[13dof] active cells: {int(ok.sum())}/{nW*nH}')

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
    print(f'[13dof] kept {M} cells')

    # KB-inv on the BASELINE calib (= the calib the npz residuals were
    # produced under). For iter1 that's K_recalib/dist_recalib; for iter2+
    # it's K_iter_prev/dist_iter_prev (set by --init-from-json above).
    # Doing KB-inv with the matching K/D yields nu in the correct camera
    # frame, so no R_prev pre-rotation is needed; R_total = R at the end.
    theta0 = kb_theta_from_uv(u_c, v_c, fx, fy, cx, cy, dist0, n_newton=10)
    phi0 = np.arctan2(v_c - cy, u_c - cx)
    nu_baseline = np.stack([
        np.sin(theta0) * np.cos(phi0),
        np.sin(theta0) * np.sin(phi0),
        np.cos(theta0),
    ], axis=-1)
    target_uv = np.stack([u_c, v_c], axis=-1) + d_meas

    omega = np.zeros(3)
    k = dist0.copy()
    dfx = 0.0; dfy = 0.0; dcx = 0.0; dcy = 0.0
    p = prev_p.copy()
    prior_diag = np.zeros(13)
    prior_diag[0:3]   = 1.0 / (args.prior_omega_deg ** 2)
    prior_diag[3:7]   = 1.0 / (args.prior_k ** 2)
    prior_diag[7:9]   = 1.0 / (args.prior_dfxy ** 2)
    prior_diag[9:11]  = 1.0 / (args.prior_dcxy_px ** 2)
    prior_diag[11:13] = 1.0 / (args.prior_p ** 2)
    if args.lock_cxy:
        # huge prior pin → solver returns ~0 for cols 9, 10
        prior_diag[9:11] = 1.0 / (1e-6 ** 2)
        print('[13dof] cxy LOCKED (dcx=dcy≡0)')

    print(f'\n[13dof] iterating GN (ω + k1..k4 + dfx,dfy,dcx,dcy + p1,p2):')
    for it in range(args.n_iter):
        R = rodrigues(omega)
        nu_lin = nu_baseline @ R.T
        fx_e = fx * (1.0 + dfx); fy_e = fy * (1.0 + dfy)
        cx_e = cx + dcx;         cy_e = cy + dcy
        uv_pred = project_unit_ray_tan(nu_lin, fx_e, fy_e, cx_e, cy_e, k, p)
        r = target_uv - uv_pred
        J = kb_jacobian_13(nu_lin, fx_e, fy_e, k, p)
        WJ = np.einsum('mij,mjk->mik', W_c, J)
        H = np.einsum('mij,mik->jk', J, WJ)
        b = np.einsum('mij,mi->j', WJ, r)
        cost = float(np.einsum('mi,mij,mj->', r, W_c, r))
        wrms = float(np.sqrt(cost / max(1, np.einsum('mij->', W_c))))
        H = H + np.diag(prior_diag)
        try:
            delta = np.linalg.solve(H + 1e-9 * np.eye(13), b)
        except np.linalg.LinAlgError:
            delta = np.zeros(13)
        omega = omega + delta[0:3]
        k     = k + delta[3:7]
        dfx   = dfx + delta[7]
        dfy   = dfy + delta[8]
        dcx   = dcx + delta[9]
        dcy   = dcy + delta[10]
        p     = p + delta[11:13]
        print(f'  it={it:2d}  |δ|={np.linalg.norm(delta):.2e}  '
              f'ω=[{omega[0]:+.3f},{omega[1]:+.3f},{omega[2]:+.3f}]°  '
              f'fxy*=[{1+dfx:.4f},{1+dfy:.4f}] cxy+=[{dcx:+.1f},{dcy:+.1f}]  '
              f'p=[{p[0]:+.4f},{p[1]:+.4f}]  '
              f'wrms={wrms:.3f}px')

    R = rodrigues(omega)
    nu_lin = nu_baseline @ R.T
    fx_e = fx * (1.0 + dfx); fy_e = fy * (1.0 + dfy)
    cx_e = cx + dcx;         cy_e = cy + dcy
    uv_pred = project_unit_ray_tan(nu_lin, fx_e, fy_e, cx_e, cy_e, k, p)
    r = target_uv - uv_pred
    cost = float(np.einsum('mi,mij,mj->', r, W_c, r))
    wrms = float(np.sqrt(cost / max(1, np.einsum('mij->', W_c))))

    # Compose total rotation: nu_baseline is already in the iter_prev cam
    # frame (KB-inv used K_prev/D_prev), so the new ω directly sits ON TOP
    # of the previous one in additive sense — but axes don't add naively,
    # so compose via R_total = R_new @ R_prev where R_prev = R(prev_omega).
    R_prev = rodrigues(prev_omega)
    R_total = R @ R_prev
    th_total = np.arccos(np.clip(0.5 * (np.trace(R_total) - 1.0), -1.0, 1.0))
    if th_total < 1e-12:
        omega_total = np.zeros(3)
    else:
        ax = np.array([R_total[2, 1] - R_total[1, 2],
                       R_total[0, 2] - R_total[2, 0],
                       R_total[1, 0] - R_total[0, 1]]) / (2.0 * np.sin(th_total))
        omega_total = np.degrees(ax * th_total)

    print(f'\n[13dof] final Δω={omega}°  Δk={k}  p={p}  (this iter)')
    print(f'[13dof] composed total ω={omega_total}°')
    print(f'[13dof] fx_new={fx_e:.3f} fy_new={fy_e:.3f} '
          f'cx_new={cx_e:.3f} cy_new={cy_e:.3f}')
    print(f'[13dof] final weighted-RMS = {wrms:.3f} px')

    out_json = args.npz.with_name(args.npz.stem +
        f'_13dof_n{args.n_thresh}_v{int(v_min)}-{int(v_max)}'
        f'{args.out_suffix}.json')
    out_json.write_text(json.dumps({
        'fx_init': fx_rc, 'fy_init': fy_rc, 'cx_init': cx_rc, 'cy_init': cy_rc,
        'fx_fit': fx_e, 'fy_fit': fy_e, 'cx_fit': cx_e, 'cy_fit': cy_e,
        'dfx': dfx, 'dfy': dfy, 'dcx': dcx, 'dcy': dcy,
        'dist_init': dist_rc.tolist(),
        'omega_deg': omega_total.tolist(),
        'dist_fit': k.tolist(),
        'tangential_p': p.tolist(),
        'final_wrms_px': wrms,
        'n_cells_used': int(M),
        'iter_delta_omega_deg': omega.tolist(),
        'baseline_omega_deg': prev_omega.tolist(),
        'baseline_dist': dist0.tolist(),
        'baseline_K': [fx, fy, cx, cy],
        'init_from_json': (str(args.init_from_json)
                           if args.init_from_json is not None else None),
    }, indent=2))
    print(f'[13dof] wrote {out_json}')


if __name__ == '__main__':
    main()
