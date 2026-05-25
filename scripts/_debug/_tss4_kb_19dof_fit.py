"""Generalised KB-N + tangential GN fit on full-frame npz.

Params: (ω_x, ω_y, ω_z, k1..kN, dfx, dfy, dcx, dcy, p1, p2)  → P = 9 + N

Forward model (per cell), N-th order odd-power radial polynomial:
  unit ray  ν₀ ← KB-inv at (K_baseline, k_baseline) on cell center
  ν       = R(ω) ν₀
  θ       = atan2(√(X²+Y²), Z)
  poly    = 1 + Σ_{i=1..N} k_i · θ^(2i)
  θ_d     = θ · poly
  X'      = θ_d · Xr,    Y' = θ_d · Yr
  r'²     = X'² + Y'²
  Δu_tan  = 2 p1 X'Y' + p2 (r'² + 2 X'²)
  Δv_tan  = p1 (r'² + 2 Y'²) + 2 p2 X'Y'
  u       = fx·(1+dfx)·(X' + Δu_tan) + (cx + dcx)
  v       = fy·(1+dfy)·(Y' + Δv_tan) + (cy + dcy)

N=4 reduces to the original KB4 + tangential 13-DoF fit.
N=10 (default for this script) extends to k1..k10 (θ²..θ²⁰), enough to
absorb non-linear edge distortion that pure KB4 leaves at the right edge.

When --init-from-json points at a KB4 fit, k5..kN are initialised to 0.
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


def kb_theta_from_uv(u, v, fx, fy, cx, cy, k, n_newton=12):
    """Invert θ_d = θ·(1 + Σ k_i θ^(2i)) for arbitrary-order k."""
    du = (u - cx); dv = (v - cy)
    r_pix = np.sqrt(du**2 + dv**2)
    f_mean = 0.5 * (fx + fy)
    theta_d = r_pix / f_mean
    theta = theta_d.copy()
    N = len(k)
    for _ in range(n_newton):
        t2 = theta * theta
        # poly = 1 + k1 θ² + k2 θ⁴ + ...; dpoly/dθ via per-i (2i)·k_i·θ^(2i-1)
        poly = np.ones_like(theta)
        dpoly_dth = np.zeros_like(theta)
        tp = t2.copy()  # tp = θ^(2i) at i=1
        for i in range(N):
            poly = poly + k[i] * tp
            # d/dθ [k_i · θ^(2(i+1))] = 2(i+1) · k_i · θ^(2(i+1)-1)
            dpoly_dth = dpoly_dth + (2 * (i + 1)) * k[i] * (tp / np.maximum(theta, 1e-12))
            tp = tp * t2
        f = theta * poly - theta_d
        df = poly + theta * dpoly_dth
        theta = theta - f / np.maximum(df, 1e-9)
    return theta


def _poly_and_dpoly(theta, k):
    """Return poly = 1 + Σ k_i θ^(2i),  dtd_dtheta = poly + θ·dpoly/dθ
       = 1 + Σ (2i+1) k_i θ^(2i)."""
    t2 = theta * theta
    poly = np.ones_like(theta)
    dtd_dtheta = np.ones_like(theta)
    tp = t2.copy()  # θ^(2(i+1)) at i=0
    powers = []     # θ · θ^(2(i+1)) = θ^(2i+1)? -- careful with index
    # We want columns col_i = θ^(2i+1) for i=1..N (used in dθ_d/dk_i).
    # k_i multiplies θ^(2i+1) in θ_d.
    for i in range(len(k)):
        poly = poly + k[i] * tp
        dtd_dtheta = dtd_dtheta + (2 * (i + 1) + 1) * k[i] * tp
        # Wait: dtd/dθ of [θ · k_i · θ^(2(i+1))] = (2(i+1)+1) · k_i · θ^(2(i+1))
        # so coefficient is (2(i+1)+1) — but in KB4 original code it was
        # (1 + 3k1 t² + 5k2 t⁴ + 7k3 t⁶ + 9k4 t⁸) → coeffs 3,5,7,9 = 2i+1 with i=1..4.
        # Above I wrote (2(i+1)+1) for i=0..N-1 → 3,5,7,9,11,...  ✓
        powers.append(theta * tp)  # θ · θ^(2(i+1)) = θ^(2i+3)?  Let's recheck.
        tp = tp * t2
    # Hmm — we want dθ_d/dk_j  =  θ · θ^(2j)  =  θ^(2j+1)  for j=1..N.
    # Above `powers[i]` (i=0) = θ · θ² = θ³ = θ^(2·1+1)  ✓ for k_1
    # `powers[i]` (i=1) = θ · θ⁴ = θ⁵ = θ^(2·2+1)  ✓ for k_2
    # So powers stack matches Jacobian column 3+i for k_{i+1}.
    return poly, dtd_dtheta, np.stack(powers, axis=-1)


def project_unit_ray_tan(nu, fx, fy, cx, cy, k, p):
    X, Y, Z = nu[:, 0], nu[:, 1], nu[:, 2]
    r = np.sqrt(X*X + Y*Y + 1e-24)
    theta = np.arctan2(r, np.maximum(Z, 1e-9))
    poly, _, _ = _poly_and_dpoly(theta, k)
    theta_d = theta * poly
    Xr = X / np.maximum(r, 1e-9); Yr = Y / np.maximum(r, 1e-9)
    Xp = theta_d * Xr; Yp = theta_d * Yr
    r2p = Xp*Xp + Yp*Yp
    du_t = 2*p[0]*Xp*Yp + p[1]*(r2p + 2*Xp*Xp)
    dv_t = p[0]*(r2p + 2*Yp*Yp) + 2*p[1]*Xp*Yp
    u = fx * (Xp + du_t) + cx
    v = fy * (Yp + dv_t) + cy
    return np.stack([u, v], axis=-1)


def kb_jacobian(nu, fx, fy, k, p):
    """(M, 2, P) cols: [ωx, ωy, ωz, k1..kN, dfx, dfy, dcx, dcy, p1, p2]
       where P = 9 + N.

    NOTE: ω, k, dfxy chain uses RADIAL-ONLY partials (small-p approx).
    p1, p2, dcxy are exact.
    """
    N = len(k)
    P = 9 + N
    X, Y, Z = nu[:, 0], nu[:, 1], nu[:, 2]
    r2 = X*X + Y*Y
    r = np.sqrt(r2 + 1e-24)
    r_safe = np.where(r > 1e-9, r, 1.0)
    theta = np.arctan2(r, np.maximum(Z, 1e-9))
    poly, dtd_dtheta, powers = _poly_and_dpoly(theta, k)
    theta_d = theta * poly
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
    cols_u = np.zeros((M, P))
    cols_v = np.zeros((M, P))
    # ω
    cols_u[:, 0], cols_v[:, 0] = chain(zero, -Z * _D2R, Y * _D2R)
    cols_u[:, 1], cols_v[:, 1] = chain(Z * _D2R, zero, -X * _D2R)
    cols_u[:, 2], cols_v[:, 2] = chain(-Y * _D2R, X * _D2R, zero)
    # k1..kN  (powers[:, i] = θ^(2(i+1)+1))
    cols_u[:, 3:3+N] = (fx * Xr)[:, None] * powers
    cols_v[:, 3:3+N] = (fy * Yr)[:, None] * powers
    # dfx, dfy
    cols_u[:, 3+N]   = fx * theta_d * Xr;  cols_v[:, 3+N]   = zero
    cols_u[:, 3+N+1] = zero;               cols_v[:, 3+N+1] = fy * theta_d * Yr
    # dcx, dcy
    cols_u[:, 3+N+2] = one;  cols_v[:, 3+N+2] = zero
    cols_u[:, 3+N+3] = zero; cols_v[:, 3+N+3] = one
    # p1, p2
    cols_u[:, 3+N+4] = fx * 2 * Xp * Yp
    cols_v[:, 3+N+4] = fy * (r2p + 2*Yp*Yp)
    cols_u[:, 3+N+5] = fx * (r2p + 2*Xp*Xp)
    cols_v[:, 3+N+5] = fy * 2 * Xp * Yp
    return np.stack([cols_u, cols_v], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=Path, required=True)
    ap.add_argument('--n-k', type=int, default=10,
                    help='order of radial polynomial (number of k coefs).'
                         ' n=4 → KB4; n=10 → k1..k10 (θ²..θ²⁰)')
    ap.add_argument('--n-thresh', type=int, default=200)
    ap.add_argument('--v-frac', type=float, default=0.72)
    ap.add_argument('--v-min-frac', type=float, default=0.362)
    ap.add_argument('--n-iter', type=int, default=12)
    ap.add_argument('--prior-omega-deg', type=float, default=2.0)
    ap.add_argument('--prior-k-low', type=float, default=1.0,
                    help='σ on k1..k4 (existing KB4 coefs)')
    ap.add_argument('--prior-k-high', type=float, default=0.5,
                    help='σ on k5..kN (extension coefs, kept tight to start)')
    ap.add_argument('--prior-dfxy', type=float, default=0.05)
    ap.add_argument('--prior-dcxy-px', type=float, default=50.0)
    ap.add_argument('--prior-p', type=float, default=0.01)
    ap.add_argument('--lock-cxy', action='store_true')
    ap.add_argument('--lock-roll', action='store_true',
                    help='pin ω_z=0 via huge prior (physical: vehicle roll≈0)')
    ap.add_argument('--init-from-json', type=Path, default=None,
                    help='start GN from a previous FIT json. dist_fit may be '
                         'shorter than n-k; trailing coefs init to 0.')
    ap.add_argument('--out-suffix', type=str, default='')
    ap.add_argument('--cache', type=Path,
                    default=Path('/mnt/fsx/tmp/hfunaya/e2e_calib_cache/tss4_v3_tiled'))
    args = ap.parse_args()

    N = int(args.n_k)
    P = 9 + N
    tag = f'kb{N}'
    print(f'[{tag}] P={P}  (3 + {N} k + 2 + 2 + 2)')

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
    dist_inst = inst['distortion'].numpy().astype(np.float64)
    fx_rc = float(K[0, 0]); fy_rc = float(K[1, 1])
    cx_rc = float(K[0, 2]); cy_rc = float(K[1, 2])
    dist_rc = dist_inst.copy()  # only used for output bookkeeping

    # baseline = inst by default (n=4); if init_from_json, override
    fx, fy, cx, cy = fx_rc, fy_rc, cx_rc, cy_rc
    k_baseline = np.zeros(N)
    k_baseline[:min(N, len(dist_inst))] = dist_inst[:min(N, len(dist_inst))]
    prev_omega = np.zeros(3)
    prev_p = np.zeros(2)

    if args.init_from_json is not None:
        prev = json.loads(args.init_from_json.read_text())
        fx = float(prev['fx_fit']); fy = float(prev['fy_fit'])
        cx = float(prev['cx_fit']); cy = float(prev['cy_fit'])
        prev_dist = np.asarray(prev['dist_fit'], dtype=np.float64)
        k_baseline = np.zeros(N)
        n_copy = min(N, len(prev_dist))
        k_baseline[:n_copy] = prev_dist[:n_copy]
        prev_omega = np.asarray(prev['omega_deg'], dtype=np.float64)
        if 'tangential_p' in prev:
            prev_p = np.asarray(prev['tangential_p'], dtype=np.float64)
        print(f'[{tag}] init-from-json: {args.init_from_json.name}')
        print(f'[{tag}]   prev ω={prev_omega}°  prev p={prev_p}')
        print(f'[{tag}]   prev k(len={len(prev_dist)}) → padded to N={N}: '
              f'{k_baseline}')
        try:
            print(f'[{tag}]   prev wrms={prev["final_wrms_px"]:.4f}px')
        except Exception:
            pass

    print(f'[{tag}] K_baseline: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}')
    print(f'[{tag}] k_baseline = {k_baseline}')

    cells_u = (np.arange(nW) + 0.5) * cell_px
    cells_v = (np.arange(nH) + 0.5) * cell_px
    UU, VV = np.meshgrid(cells_u, cells_v)
    v_max = args.v_frac * IH
    v_min = args.v_min_frac * IH
    band = (VV < v_max) & (VV > v_min)
    ok = (sum_n >= args.n_thresh) & band
    print(f'[{tag}] active cells: {int(ok.sum())}/{nW*nH}')

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
    print(f'[{tag}] kept {M} cells')

    # KB-inv with the BASELINE (= prev FIT for iter2+)
    theta0 = kb_theta_from_uv(u_c, v_c, fx, fy, cx, cy, k_baseline, n_newton=12)
    phi0 = np.arctan2(v_c - cy, u_c - cx)
    nu_baseline = np.stack([
        np.sin(theta0) * np.cos(phi0),
        np.sin(theta0) * np.sin(phi0),
        np.cos(theta0),
    ], axis=-1)
    target_uv = np.stack([u_c, v_c], axis=-1) + d_meas

    omega = np.zeros(3)
    k = k_baseline.copy()
    dfx = 0.0; dfy = 0.0; dcx = 0.0; dcy = 0.0
    p = prev_p.copy()
    prior_diag = np.zeros(P)
    prior_diag[0:3]   = 1.0 / (args.prior_omega_deg ** 2)
    # k1..k4 (low-order) get the looser prior; k5..kN get tighter prior
    n_low = min(4, N)
    prior_diag[3:3+n_low]   = 1.0 / (args.prior_k_low ** 2)
    if N > 4:
        prior_diag[3+n_low:3+N] = 1.0 / (args.prior_k_high ** 2)
    prior_diag[3+N]         = 1.0 / (args.prior_dfxy ** 2)
    prior_diag[3+N+1]       = 1.0 / (args.prior_dfxy ** 2)
    prior_diag[3+N+2]       = 1.0 / (args.prior_dcxy_px ** 2)
    prior_diag[3+N+3]       = 1.0 / (args.prior_dcxy_px ** 2)
    prior_diag[3+N+4]       = 1.0 / (args.prior_p ** 2)
    prior_diag[3+N+5]       = 1.0 / (args.prior_p ** 2)
    if args.lock_cxy:
        prior_diag[3+N+2] = 1.0 / (1e-6 ** 2)
        prior_diag[3+N+3] = 1.0 / (1e-6 ** 2)
        print(f'[{tag}] cxy LOCKED (dcx=dcy≡0)')
    if args.lock_roll:
        prior_diag[2] = 1.0 / (1e-6 ** 2)
        print(f'[{tag}] roll LOCKED (ω_z≡0)')

    print(f'\n[{tag}] iterating GN (ω + k1..k{N} + dfx,dfy,dcx,dcy + p1,p2):')
    for it in range(args.n_iter):
        R = rodrigues(omega)
        nu_lin = nu_baseline @ R.T
        fx_e = fx * (1.0 + dfx); fy_e = fy * (1.0 + dfy)
        cx_e = cx + dcx;         cy_e = cy + dcy
        uv_pred = project_unit_ray_tan(nu_lin, fx_e, fy_e, cx_e, cy_e, k, p)
        r = target_uv - uv_pred
        J = kb_jacobian(nu_lin, fx_e, fy_e, k, p)
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
        k     = k + delta[3:3+N]
        dfx   = dfx + delta[3+N]
        dfy   = dfy + delta[3+N+1]
        dcx   = dcx + delta[3+N+2]
        dcy   = dcy + delta[3+N+3]
        p     = p + delta[3+N+4:3+N+6]
        k_str = ','.join(f'{ki:+.3f}' for ki in k)
        print(f'  it={it:2d}  |δ|={np.linalg.norm(delta):.2e}  '
              f'ω=[{omega[0]:+.3f},{omega[1]:+.3f},{omega[2]:+.3f}]°  '
              f'fxy*=[{1+dfx:.4f},{1+dfy:.4f}] cxy+=[{dcx:+.1f},{dcy:+.1f}]  '
              f'p=[{p[0]:+.4f},{p[1]:+.4f}]  '
              f'k=[{k_str}]  '
              f'wrms={wrms:.3f}px')

    R = rodrigues(omega)
    nu_lin = nu_baseline @ R.T
    fx_e = fx * (1.0 + dfx); fy_e = fy * (1.0 + dfy)
    cx_e = cx + dcx;         cy_e = cy + dcy
    uv_pred = project_unit_ray_tan(nu_lin, fx_e, fy_e, cx_e, cy_e, k, p)
    r = target_uv - uv_pred
    cost = float(np.einsum('mi,mij,mj->', r, W_c, r))
    wrms = float(np.sqrt(cost / max(1, np.einsum('mij->', W_c))))

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

    print(f'\n[{tag}] final Δω={omega}°  Δk={k}  p={p}  (this iter)')
    print(f'[{tag}] composed total ω={omega_total}°')
    print(f'[{tag}] fx_new={fx_e:.3f} fy_new={fy_e:.3f} '
          f'cx_new={cx_e:.3f} cy_new={cy_e:.3f}')
    print(f'[{tag}] final weighted-RMS = {wrms:.3f} px')

    out_json = args.npz.with_name(args.npz.stem +
        f'_{tag}_n{args.n_thresh}_v{int(v_min)}-{int(v_max)}'
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
        'baseline_dist': k_baseline.tolist(),
        'baseline_K': [fx, fy, cx, cy],
        'n_k': N,
        'init_from_json': (str(args.init_from_json)
                           if args.init_from_json is not None else None),
    }, indent=2))
    print(f'[{tag}] wrote {out_json}')


if __name__ == '__main__':
    main()
