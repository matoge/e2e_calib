"""KB-N + tangential + Δt(rear→cam) GN fit on full-frame npz.

Params: (ω_x, ω_y, ω_z, k1..kN, dfx, dfy, dcx, dcy, p1, p2, Δtx, Δty, Δtz)
       → P = 12 + N

Compared to _tss4_kb_19dof_fit.py (P = 9 + N), this adds 3 translation
parameters that perturb the rear→cam translation seen by every cell.
The 3D point in the camera frame is

    pts_cam_new(i) = R(ω) · ν0(i) · Z_i + Δt

where ν0(i) is the unit ray from KB-inv at the cell center on the BASELINE
(prev FIT) calib, and Z_i is the per-cell average depth (m) carried in the
npz under sum_X/Y/Z (must be present — i.e. iter1_xyz npz, not the older
iter1 npz that lacks them).

After Δt is applied we re-project with the FULL forward (KB-N radial + p1,p2
tangential).  The Jacobian columns for Δt come from
   ∂uv/∂Δt = J_proj · (∂nu_lin/∂Δt)
with ∂nu_lin/∂Δt = -nu_lin/Z_world ?  No — easier to keep the projector
expressed in pts_cam metric (X,Y,Z) and let dX,dY,dZ chain through the same
`chain()` helper used for ω.

Usage (typical):
  /home/hfunaya/.pyenv/versions/3.10.4/bin/python \
      scripts/_debug/_tss4_kb_22dof_fit.py \
        --npz scripts/_debug/_outputs/<...>_xyz.npz \
        --n-k 10 \
        --prior-k-low 1.0 --prior-k-high 100.0 \
        --prior-dt-m 0.10 \
        --lock-cxy --lock-roll \
        --out-suffix _iter1_kb10_dt_lockroll
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
    du = (u - cx); dv = (v - cy)
    r_pix = np.sqrt(du**2 + dv**2)
    f_mean = 0.5 * (fx + fy)
    theta_d = r_pix / f_mean
    theta = theta_d.copy()
    N = len(k)
    for _ in range(n_newton):
        t2 = theta * theta
        poly = np.ones_like(theta)
        dpoly_dth = np.zeros_like(theta)
        tp = t2.copy()
        for i in range(N):
            poly = poly + k[i] * tp
            dpoly_dth = dpoly_dth + (2 * (i + 1)) * k[i] * (tp / np.maximum(theta, 1e-12))
            tp = tp * t2
        f = theta * poly - theta_d
        df = poly + theta * dpoly_dth
        theta = theta - f / np.maximum(df, 1e-9)
    return theta


def _poly_and_dpoly(theta, k):
    t2 = theta * theta
    poly = np.ones_like(theta)
    dtd_dtheta = np.ones_like(theta)
    tp = t2.copy()
    powers = []
    for i in range(len(k)):
        poly = poly + k[i] * tp
        dtd_dtheta = dtd_dtheta + (2 * (i + 1) + 1) * k[i] * tp
        powers.append(theta * tp)
        tp = tp * t2
    return poly, dtd_dtheta, np.stack(powers, axis=-1)


def project_pts_tan(pts, fx, fy, cx, cy, k, p):
    """pts (M, 3) in metres → uv (M, 2). Full KB-N + Brown tangential."""
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(X * X + Y * Y + 1e-24)
    theta = np.arctan2(r, np.maximum(Z, 1e-9))
    poly, _, _ = _poly_and_dpoly(theta, k)
    theta_d = theta * poly
    Xr = X / np.maximum(r, 1e-9); Yr = Y / np.maximum(r, 1e-9)
    Xp = theta_d * Xr; Yp = theta_d * Yr
    r2p = Xp * Xp + Yp * Yp
    du_t = 2 * p[0] * Xp * Yp + p[1] * (r2p + 2 * Xp * Xp)
    dv_t = p[0] * (r2p + 2 * Yp * Yp) + 2 * p[1] * Xp * Yp
    u = fx * (Xp + du_t) + cx
    v = fy * (Yp + dv_t) + cy
    return np.stack([u, v], axis=-1)


def kb_jacobian(pts, fx, fy, k, p):
    """(M, 2, P) cols: [ωx, ωy, ωz, k1..kN, dfx, dfy, dcx, dcy, p1, p2,
                       Δtx, Δty, Δtz].   P = 12 + N.

    pts: (M, 3) metres in current cam frame (after R(ω) and +Δt have been
         applied — i.e. pts == pts_lin in the GN loop).

    NOTE: ω, k, dfxy chains use radial-only partials (small-p approximation,
    matching the 19dof script).  p1,p2,dcxy are exact.  Δt uses the same
    radial-only chain via dX/dY/dZ.
    """
    N = len(k)
    P = 12 + N
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    r2 = X * X + Y * Y
    r = np.sqrt(r2 + 1e-24)
    r_safe = np.where(r > 1e-9, r, 1.0)
    theta = np.arctan2(r, np.maximum(Z, 1e-9))
    poly, dtd_dtheta, powers = _poly_and_dpoly(theta, k)
    theta_d = theta * poly
    r2pz2 = r2 + Z * Z + 1e-24
    inv_r = 1.0 / r_safe
    inv_r2 = inv_r * inv_r
    Xr = X * inv_r; Yr = Y * inv_r
    Xp = theta_d * Xr; Yp = theta_d * Yr
    r2p = Xp * Xp + Yp * Yp

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
    # ω (acts on the unit-ray pre-Δt; we feed pts directly here, so use the
    # same skew-symmetric form on (X,Y,Z) — gives the rotational derivative
    # of pts_lin at the current operating point, ignoring Δt's coupling
    # which is small for |Δt| ≪ Z).
    cols_u[:, 0], cols_v[:, 0] = chain(zero, -Z * _D2R, Y * _D2R)
    cols_u[:, 1], cols_v[:, 1] = chain(Z * _D2R, zero, -X * _D2R)
    cols_u[:, 2], cols_v[:, 2] = chain(-Y * _D2R, X * _D2R, zero)
    # k1..kN
    cols_u[:, 3:3 + N] = (fx * Xr)[:, None] * powers
    cols_v[:, 3:3 + N] = (fy * Yr)[:, None] * powers
    # dfx, dfy
    cols_u[:, 3 + N]     = fx * theta_d * Xr;  cols_v[:, 3 + N]     = zero
    cols_u[:, 3 + N + 1] = zero;               cols_v[:, 3 + N + 1] = fy * theta_d * Yr
    # dcx, dcy
    cols_u[:, 3 + N + 2] = one;  cols_v[:, 3 + N + 2] = zero
    cols_u[:, 3 + N + 3] = zero; cols_v[:, 3 + N + 3] = one
    # p1, p2
    cols_u[:, 3 + N + 4] = fx * 2 * Xp * Yp
    cols_v[:, 3 + N + 4] = fy * (r2p + 2 * Yp * Yp)
    cols_u[:, 3 + N + 5] = fx * (r2p + 2 * Xp * Xp)
    cols_v[:, 3 + N + 5] = fy * 2 * Xp * Yp
    # Δtx, Δty, Δtz: ∂pts/∂Δt = I, chain straight through
    cols_u[:, 3 + N + 6], cols_v[:, 3 + N + 6] = chain(one, zero, zero)
    cols_u[:, 3 + N + 7], cols_v[:, 3 + N + 7] = chain(zero, one, zero)
    cols_u[:, 3 + N + 8], cols_v[:, 3 + N + 8] = chain(zero, zero, one)
    return np.stack([cols_u, cols_v], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=Path, required=True,
                    help='must contain sum_X, sum_Y, sum_Z (xyz npz)')
    ap.add_argument('--n-k', type=int, default=10)
    ap.add_argument('--n-thresh', type=int, default=200)
    ap.add_argument('--v-frac', type=float, default=0.72)
    ap.add_argument('--v-min-frac', type=float, default=0.362)
    ap.add_argument('--n-iter', type=int, default=12)
    ap.add_argument('--prior-omega-deg', type=float, default=2.0)
    ap.add_argument('--prior-k-low', type=float, default=1.0)
    ap.add_argument('--prior-k-high', type=float, default=0.5)
    ap.add_argument('--prior-dfxy', type=float, default=0.05)
    ap.add_argument('--prior-dcxy-px', type=float, default=50.0)
    ap.add_argument('--prior-p', type=float, default=0.01)
    ap.add_argument('--prior-dt-m', type=float, default=0.10,
                    help='σ on Δtx,Δty,Δtz in metres (default 10cm)')
    ap.add_argument('--lock-cxy', action='store_true')
    ap.add_argument('--lock-roll', action='store_true')
    ap.add_argument('--lock-dt', action='store_true',
                    help='pin Δt=0 (sanity: should reproduce 19dof results)')
    ap.add_argument('--init-from-json', type=Path, default=None)
    ap.add_argument('--edge-boost', type=float, default=1.0,
                    help='multiply per-cell info weight W by this factor for '
                         'cells in left/right edge bands (cu < edge_w or '
                         'cu >= nW-edge_w). 1.0 = off.')
    ap.add_argument('--edge-band-frac', type=float, default=0.125,
                    help='fraction of nW counted as edge on each side '
                         '(default 0.125 = nW/8 each side, matches viz)')
    ap.add_argument('--use-pts-bar', action='store_true',
                    help='use per-cell (Xbar, Ybar, Zbar) directly instead of '
                         'ν0(K_baseline)·Zbar. Reason: cell-center KB-inv diverges '
                         'nonlinearly from real per-point centroids on fisheye '
                         'edges, causing solver-converges-but-apply-roundtrip '
                         'still leaks residual on the right band.')
    ap.add_argument('--out-suffix', type=str, default='')
    ap.add_argument('--cache', type=Path,
        default=Path('/mnt/fsx/tmp/hfunaya/e2e_calib_cache/tss4_v3_tiled'))
    args = ap.parse_args()

    N = int(args.n_k)
    P = 12 + N
    tag = f'kb{N}dt'
    print(f'[{tag}] P={P}  (3 + {N} k + 2 + 2 + 2 + 3)')

    z = np.load(args.npz, allow_pickle=False)
    if 'sum_X' not in z.files or 'sum_Z' not in z.files:
        raise SystemExit(f'npz missing sum_X/Y/Z: {args.npz}\n'
                         f'  re-run _tss4_full_frame_stats.py to regenerate.')
    sum_n = z['sum_n']; sum_W = z['sum_W']; sum_Wd = z['sum_Wd']
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
    dist_inst = inst['distortion'].numpy().astype(np.float64)
    fx_rc = float(K[0, 0]); fy_rc = float(K[1, 1])
    cx_rc = float(K[0, 2]); cy_rc = float(K[1, 2])
    dist_rc = dist_inst.copy()

    fx, fy, cx, cy = fx_rc, fy_rc, cx_rc, cy_rc
    k_baseline = np.zeros(N)
    k_baseline[:min(N, len(dist_inst))] = dist_inst[:min(N, len(dist_inst))]
    prev_omega = np.zeros(3)
    prev_p = np.zeros(2)
    prev_dt = np.zeros(3)

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
        if 'delta_t_m' in prev:
            prev_dt = np.asarray(prev['delta_t_m'], dtype=np.float64)
        print(f'[{tag}] init-from-json: {args.init_from_json.name}')
        print(f'[{tag}]   prev ω={prev_omega.round(4).tolist()}°  '
              f'prev Δt={(prev_dt*1000).round(1).tolist()}mm  '
              f'prev p={prev_p.round(5).tolist()}')

    print(f'[{tag}] K_baseline: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}')
    print(f'[{tag}] k_baseline = {k_baseline}')

    cells_u = (np.arange(nW) + 0.5) * cell_px
    cells_v = (np.arange(nH) + 0.5) * cell_px
    UU, VV = np.meshgrid(cells_u, cells_v)
    band = (VV < args.v_frac * IH) & (VV > args.v_min_frac * IH)
    ok = (sum_n >= args.n_thresh) & band
    print(f'[{tag}] active cells: {int(ok.sum())}/{nW*nH}')

    cv_idx, cu_idx = np.where(ok)
    M = cv_idx.size
    u_c = UU[cv_idx, cu_idx]; v_c = VV[cv_idx, cu_idx]
    W_c = sum_W[cv_idx, cu_idx].copy()
    Wd_c = sum_Wd[cv_idx, cu_idx].copy()
    n_pts = sum_n[cv_idx, cu_idx].astype(np.float64)
    if args.edge_boost != 1.0:
        edge_w = max(1, int(round(nW * args.edge_band_frac)))
        edge_mask = (cu_idx < edge_w) | (cu_idx >= nW - edge_w)
        W_c[edge_mask]  = W_c[edge_mask]  * args.edge_boost
        Wd_c[edge_mask] = Wd_c[edge_mask] * args.edge_boost
        n_edge = int(edge_mask.sum())
        print(f'[{tag}] edge-boost ×{args.edge_boost:g} applied to '
              f'{n_edge} edge cells (edge_w={edge_w} per side)')
    # per-cell average XYZ from sums
    Xbar = sum_X[cv_idx, cu_idx] / np.maximum(n_pts, 1.0)
    Ybar = sum_Y[cv_idx, cu_idx] / np.maximum(n_pts, 1.0)
    Zbar = sum_Z[cv_idx, cu_idx] / np.maximum(n_pts, 1.0)
    d_meas = np.zeros((M, 2))
    keep = np.ones(M, dtype=bool)
    for i in range(M):
        try:
            d_meas[i] = np.linalg.solve(W_c[i], Wd_c[i])
        except np.linalg.LinAlgError:
            keep[i] = False
    u_c = u_c[keep]; v_c = v_c[keep]
    W_c = W_c[keep]; d_meas = d_meas[keep]
    Xbar = Xbar[keep]; Ybar = Ybar[keep]; Zbar = Zbar[keep]
    M = int(keep.sum())
    print(f'[{tag}] kept {M} cells   ⟨Z⟩={Zbar.mean():.2f}m  '
          f'Z-range=[{Zbar.min():.1f}, {Zbar.max():.1f}]m')

    if args.use_pts_bar:
        pts_baseline = np.stack([Xbar, Ybar, Zbar], axis=-1)
        uv_at_baseline = project_pts_tan(pts_baseline, fx, fy, cx, cy,
                                         k_baseline, prev_p)
        target_uv = uv_at_baseline + d_meas
        print(f'[{tag}] --use-pts-bar: pts_baseline=(Xbar,Ybar,Zbar), '
              f'target_uv = proj(pts_bar; K_baseline) + d_meas '
              f'(consistent with per-point apply path)')
    else:
        # KB-inv with BASELINE → unit ray nu0
        theta0 = kb_theta_from_uv(u_c, v_c, fx, fy, cx, cy, k_baseline, n_newton=12)
        phi0 = np.arctan2(v_c - cy, u_c - cx)
        nu0 = np.stack([
            np.sin(theta0) * np.cos(phi0),
            np.sin(theta0) * np.sin(phi0),
            np.cos(theta0),
        ], axis=-1)
        pts_baseline = nu0 * Zbar[:, None]
        target_uv = np.stack([u_c, v_c], axis=-1) + d_meas

    omega = np.zeros(3)
    k = k_baseline.copy()
    dfx = 0.0; dfy = 0.0; dcx = 0.0; dcy = 0.0
    p = prev_p.copy()
    dt = np.zeros(3)
    prior_diag = np.zeros(P)
    prior_diag[0:3]   = 1.0 / (args.prior_omega_deg ** 2)
    n_low = min(4, N)
    prior_diag[3:3 + n_low]   = 1.0 / (args.prior_k_low ** 2)
    if N > 4:
        prior_diag[3 + n_low:3 + N] = 1.0 / (args.prior_k_high ** 2)
    prior_diag[3 + N]         = 1.0 / (args.prior_dfxy ** 2)
    prior_diag[3 + N + 1]     = 1.0 / (args.prior_dfxy ** 2)
    prior_diag[3 + N + 2]     = 1.0 / (args.prior_dcxy_px ** 2)
    prior_diag[3 + N + 3]     = 1.0 / (args.prior_dcxy_px ** 2)
    prior_diag[3 + N + 4]     = 1.0 / (args.prior_p ** 2)
    prior_diag[3 + N + 5]     = 1.0 / (args.prior_p ** 2)
    prior_diag[3 + N + 6:3 + N + 9] = 1.0 / (args.prior_dt_m ** 2)
    if args.lock_cxy:
        prior_diag[3 + N + 2] = 1.0 / (1e-6 ** 2)
        prior_diag[3 + N + 3] = 1.0 / (1e-6 ** 2)
        print(f'[{tag}] cxy LOCKED')
    if args.lock_roll:
        prior_diag[2] = 1.0 / (1e-6 ** 2)
        print(f'[{tag}] roll LOCKED (ω_z≡0)')
    if args.lock_dt:
        prior_diag[3 + N + 6:3 + N + 9] = 1.0 / (1e-6 ** 2)
        print(f'[{tag}] Δt LOCKED (sanity, should match 19dof)')
    print(f'[{tag}] prior_dt_m={args.prior_dt_m:.4f}m  '
          f'(λ_dt = {1.0/(args.prior_dt_m**2):.2f})')

    print(f'\n[{tag}] iterating GN (ω + k1..k{N} + dfxy + dcxy + p1p2 + Δt):')
    for it in range(args.n_iter):
        R = rodrigues(omega)
        pts_lin = pts_baseline @ R.T + dt[None, :]
        fx_e = fx * (1.0 + dfx); fy_e = fy * (1.0 + dfy)
        cx_e = cx + dcx;         cy_e = cy + dcy
        uv_pred = project_pts_tan(pts_lin, fx_e, fy_e, cx_e, cy_e, k, p)
        r = target_uv - uv_pred
        J = kb_jacobian(pts_lin, fx_e, fy_e, k, p)
        WJ = np.einsum('mij,mjk->mik', W_c, J)
        H = np.einsum('mij,mik->jk', J, WJ)
        b = np.einsum('mij,mi->j', WJ, r)
        cost = float(np.einsum('mi,mij,mj->', r, W_c, r))
        wt_total = float(np.einsum('mij->', W_c))
        wrms = float(np.sqrt(cost / max(1, wt_total)))
        H = H + np.diag(prior_diag)
        try:
            delta = np.linalg.solve(H + 1e-9 * np.eye(P), b)
        except np.linalg.LinAlgError:
            delta = np.zeros(P)
        omega = omega + delta[0:3]
        k     = k + delta[3:3 + N]
        dfx   = dfx + delta[3 + N]
        dfy   = dfy + delta[3 + N + 1]
        dcx   = dcx + delta[3 + N + 2]
        dcy   = dcy + delta[3 + N + 3]
        p     = p + delta[3 + N + 4:3 + N + 6]
        dt    = dt + delta[3 + N + 6:3 + N + 9]
        k_str = ','.join(f'{ki:+.3f}' for ki in k)
        print(f'  it={it:2d}  |δ|={np.linalg.norm(delta):.2e}  '
              f'ω=[{omega[0]:+.3f},{omega[1]:+.3f},{omega[2]:+.3f}]°  '
              f'fxy*=[{1+dfx:.4f},{1+dfy:.4f}]  '
              f'p=[{p[0]:+.4f},{p[1]:+.4f}]  '
              f'Δt=[{dt[0]*1000:+.0f},{dt[1]*1000:+.0f},{dt[2]*1000:+.0f}]mm  '
              f'wrms={wrms:.3f}px')

    R = rodrigues(omega)
    pts_lin = pts_baseline @ R.T + dt[None, :]
    fx_e = fx * (1.0 + dfx); fy_e = fy * (1.0 + dfy)
    cx_e = cx + dcx;         cy_e = cy + dcy
    uv_pred = project_pts_tan(pts_lin, fx_e, fy_e, cx_e, cy_e, k, p)
    r = target_uv - uv_pred
    cost = float(np.einsum('mi,mij,mj->', r, W_c, r))
    wt_total = float(np.einsum('mij->', W_c))
    wrms = float(np.sqrt(cost / max(1, wt_total)))

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

    # Composed total Δt: applying R_prev then Δt_prev, then R, then Δt of this
    # iter, equals first applying R_total then Δt_total where
    #   Δt_total = R · Δt_prev + Δt_iter
    dt_total = R @ prev_dt + dt

    print(f'\n[{tag}] final Δω={omega}°  Δt={dt*1000}mm  (this iter)')
    print(f'[{tag}] composed total ω={omega_total}°  Δt={(dt_total*1000)}mm')
    print(f'[{tag}] fx_new={fx_e:.3f} fy_new={fy_e:.3f} '
          f'cx_new={cx_e:.3f} cy_new={cy_e:.3f}')
    print(f'[{tag}] final weighted-RMS = {wrms:.3f} px')

    out_json = args.npz.with_name(args.npz.stem +
        f'_kb{N}dt_n{args.n_thresh}_v{int(args.v_min_frac*IH)}-{int(args.v_frac*IH)}'
        f'{args.out_suffix}.json')
    out_json.write_text(json.dumps({
        'fx_init': fx_rc, 'fy_init': fy_rc, 'cx_init': cx_rc, 'cy_init': cy_rc,
        'fx_fit': fx_e, 'fy_fit': fy_e, 'cx_fit': cx_e, 'cy_fit': cy_e,
        'dfx': dfx, 'dfy': dfy, 'dcx': dcx, 'dcy': dcy,
        'dist_init': dist_rc.tolist(),
        'omega_deg': omega_total.tolist(),
        'dist_fit': k.tolist(),
        'tangential_p': p.tolist(),
        'delta_t_m': dt_total.tolist(),
        'final_wrms_px': wrms,
        'n_cells_used': int(M),
        'iter_delta_omega_deg': omega.tolist(),
        'iter_delta_t_m': dt.tolist(),
        'baseline_omega_deg': prev_omega.tolist(),
        'baseline_delta_t_m': prev_dt.tolist(),
        'baseline_dist': k_baseline.tolist(),
        'baseline_K': [fx, fy, cx, cy],
        'n_k': N,
        'init_from_json': (str(args.init_from_json)
                           if args.init_from_json is not None else None),
    }, indent=2))
    print(f'[{tag}] wrote {out_json}')


if __name__ == '__main__':
    main()
