"""Visualise observed vs predicted Δuv with the converged 11-DoF params.

For each active cell:
  observed = (Σ W)⁻¹ Σ W·d                       (info-weighted residual)
  predicted = uv_pred(R(ω) ν₀, k_fit, fxy_fit, cxy_fit) - cell_center
                where ν₀ = unit ray at cell center under (K_init, k_init)

Output: 1×3 figure
  (a) observed quiver
  (b) predicted quiver  (same color/scale)
  (c) residual quiver = obs - pred  (what's still left)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

_D2R = np.pi / 180.0


def rodrigues(omega_deg):
    th = np.linalg.norm(omega_deg) * _D2R
    if th < 1e-12: return np.eye(3)
    axis = (omega_deg * _D2R) / th
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)


def _poly_kbN(theta, k):
    t2 = theta * theta
    poly = np.ones_like(theta)
    tp = t2.copy()
    for ki in k:
        poly = poly + ki * tp
        tp = tp * t2
    return poly


def _dpoly_kbN_dtheta(theta, k):
    """∂(θ·poly(θ))/∂θ for KB-N polynomial."""
    t2 = theta * theta
    out = np.ones_like(theta)
    tp = t2.copy()
    for i, ki in enumerate(k):
        out = out + (2 * (i + 1) + 1) * ki * tp
        tp = tp * t2
    return out


def kb_theta_from_uv(u, v, fx, fy, cx, cy, k, n_newton=12):
    du = u - cx; dv = v - cy
    r_pix = np.sqrt(du**2 + dv**2)
    f_mean = 0.5 * (fx + fy)
    theta_d = r_pix / f_mean
    theta = theta_d.copy()
    for _ in range(n_newton):
        f = theta * _poly_kbN(theta, k) - theta_d
        df = _dpoly_kbN_dtheta(theta, k)
        theta = theta - f / np.maximum(df, 1e-9)
    return theta


def project_pts(pts, fx, fy, cx, cy, k, p=None):
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    r = np.sqrt(X * X + Y * Y + 1e-24)
    theta = np.arctan2(r, np.maximum(Z, 1e-9))
    poly = _poly_kbN(theta, k)
    theta_d = theta * poly
    Xr = X / np.maximum(r, 1e-9); Yr = Y / np.maximum(r, 1e-9)
    Xp = theta_d * Xr; Yp = theta_d * Yr
    if p is not None:
        r2p = Xp * Xp + Yp * Yp
        du_t = 2 * p[0] * Xp * Yp + p[1] * (r2p + 2 * Xp * Xp)
        dv_t = p[0] * (r2p + 2 * Yp * Yp) + 2 * p[1] * Xp * Yp
        Xp = Xp + du_t; Yp = Yp + dv_t
    u = fx * Xp + cx
    v = fy * Yp + cy
    return np.stack([u, v], axis=-1)


def project_unit_ray(nu, fx, fy, cx, cy, k, p=None):
    return project_pts(nu, fx, fy, cx, cy, k, p=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=Path, required=True)
    ap.add_argument('--json', type=Path, required=True)
    ap.add_argument('--n-thresh', type=int, default=200)
    ap.add_argument('--v-frac', type=float, default=0.72)
    ap.add_argument('--v-min-frac', type=float, default=0.362)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--u-range', type=str, default=None,
                    help='restrict cells to u_min,u_max in parent-px '
                         '(e.g. "3328,3840" for t19)')
    ap.add_argument('--v-range', type=str, default=None,
                    help='restrict cells to v_min,v_max in parent-px '
                         '(e.g. "677,1189" for t19)')
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    sum_n=z['sum_n']; sum_W=z['sum_W']; sum_Wd=z['sum_Wd']
    cell_px=int(z['cell_px']); IW=int(z['IW']); IH=int(z['IH'])
    nH,nW = sum_n.shape
    has_xyz = ('sum_X' in z.files) and ('sum_Z' in z.files)
    if has_xyz:
        sum_X = z['sum_X']; sum_Y = z['sum_Y']; sum_Z = z['sum_Z']

    j = json.loads(args.json.read_text())
    fx0 = j['fx_init']; fy0 = j['fy_init']; cx0 = j['cx_init']; cy0 = j['cy_init']
    fx_fit = j['fx_fit']; fy_fit = j['fy_fit']; cx_fit = j['cx_fit']; cy_fit = j['cy_fit']
    k0 = np.array(j['dist_init']); k_fit = np.array(j['dist_fit'])
    omega = np.array(j['omega_deg'])
    p_fit = np.array(j['tangential_p']) if 'tangential_p' in j else None
    dt_fit = np.array(j['delta_t_m']) if 'delta_t_m' in j else None
    # Use baseline_K (= pre-fit K seen by GN), not init_K, when present.
    if 'baseline_K' in j:
        bK = j['baseline_K']
        fx_b, fy_b, cx_b, cy_b = float(bK[0]), float(bK[1]), float(bK[2]), float(bK[3])
        k_b = np.array(j.get('baseline_dist', j['dist_init']))
        omega_apply = np.array(j.get('iter_delta_omega_deg', omega))
        # When chained, the npz already had prev Δt applied during forward.
        # Predicted uv must use ONLY this-iter Δt against the npz frame.
        if 'iter_delta_t_m' in j:
            dt_fit = np.array(j['iter_delta_t_m'])
    else:
        fx_b, fy_b, cx_b, cy_b = fx0, fy0, cx0, cy0
        k_b = k0
        omega_apply = omega

    cells_u = (np.arange(nW)+0.5)*cell_px
    cells_v = (np.arange(nH)+0.5)*cell_px
    UU,VV = np.meshgrid(cells_u, cells_v)
    v_max = args.v_frac*IH; v_min = args.v_min_frac*IH
    band = (VV<v_max)&(VV>v_min)
    ok = (sum_n>=args.n_thresh)&band
    if args.u_range is not None:
        umin, umax = [float(x) for x in args.u_range.split(',')]
        ok = ok & (UU >= umin) & (UU < umax)
    if args.v_range is not None:
        vmin2, vmax2 = [float(x) for x in args.v_range.split(',')]
        ok = ok & (VV >= vmin2) & (VV < vmax2)
    print(f'[viz] active cells: {int(ok.sum())} '
          f'(u_range={args.u_range}, v_range={args.v_range})')
    cv_idx, cu_idx = np.where(ok)
    M = cv_idx.size
    u_c = UU[cv_idx, cu_idx]; v_c = VV[cv_idx, cu_idx]

    # observed info-weighted Δuv
    W_c = sum_W[cv_idx, cu_idx]; Wd_c = sum_Wd[cv_idx, cu_idx]
    obs = np.zeros((M,2))
    keep = np.ones(M, dtype=bool)
    for i in range(M):
        try: obs[i] = np.linalg.solve(W_c[i], Wd_c[i])
        except np.linalg.LinAlgError: keep[i] = False
    cv_idx=cv_idx[keep]; cu_idx=cu_idx[keep]
    u_c=u_c[keep]; v_c=v_c[keep]; obs=obs[keep]; W_c=W_c[keep]
    M = int(keep.sum())

    # predicted: cell center → unit ray (BASELINE geometry) → scale by per-cell
    # mean depth Z (from sum_X/Y/Z if available) → R(Δω)·pts + Δt → KB-N + tang.
    theta0 = kb_theta_from_uv(u_c, v_c, fx_b, fy_b, cx_b, cy_b, k_b, n_newton=12)
    phi0 = np.arctan2(v_c - cy_b, u_c - cx_b)
    nu = np.stack([np.sin(theta0) * np.cos(phi0),
                   np.sin(theta0) * np.sin(phi0),
                   np.cos(theta0)], axis=-1)
    R = rodrigues(omega_apply)
    if has_xyz and dt_fit is not None:
        n_pts = sum_n[cv_idx, cu_idx].astype(np.float64)
        Zbar = sum_Z[cv_idx, cu_idx] / np.maximum(n_pts, 1.0)
        pts_baseline = nu * Zbar[:, None]
        pts_lin = pts_baseline @ R.T + dt_fit[None, :]
        uv_pred = project_pts(pts_lin, fx_fit, fy_fit, cx_fit, cy_fit,
                              k_fit, p=p_fit)
        print(f'[viz] using xyz npz + Δt; ⟨Z⟩={Zbar.mean():.2f}m')
    else:
        nu_rot = nu @ R.T
        uv_pred = project_unit_ray(nu_rot, fx_fit, fy_fit, cx_fit, cy_fit,
                                   k_fit, p=p_fit)
        if dt_fit is not None:
            print(f'[viz] WARN: fit has Δt but npz lacks sum_X/Y/Z → ignoring Δt')
    pred = uv_pred - np.stack([u_c, v_c], axis=-1)

    resid = obs - pred

    # info-weighted norms for stats
    def wrms(d):
        cost = float(np.einsum('mi,mij,mj->', d, W_c, d))
        wt   = float(np.einsum('mij->', W_c))
        return np.sqrt(cost/max(1,wt))
    print(f'[viz] wrms obs   = {wrms(obs):.3f} px')
    print(f'[viz] wrms pred  = {wrms(pred):.3f} px')
    print(f'[viz] wrms resid = {wrms(resid):.3f} px')

    # decompose by column band
    edge_w = max(1, nW//8)
    bands = [
        ('left',  cu_idx < edge_w),
        ('mid',   (cu_idx>=edge_w) & (cu_idx<nW-edge_w)),
        ('right', cu_idx >= nW-edge_w),
    ]
    print(f'\n[viz] info-weighted column-band means (obs / pred / resid):')
    print(f'  band     obs (du, dv)        pred (du, dv)       resid (du, dv)')
    for name, sel in bands:
        if sel.sum()==0: continue
        Wb = W_c[sel].sum(axis=0)
        def info_mean(d):
            Wd = np.einsum('mij,mj->i', W_c[sel], d[sel])
            try: return np.linalg.solve(Wb, Wd)
            except np.linalg.LinAlgError: return np.array([np.nan, np.nan])
        ob = info_mean(obs); pb = info_mean(pred); rb = info_mean(resid)
        print(f'  {name:6s}  ({ob[0]:+6.2f},{ob[1]:+6.2f})    '
              f'({pb[0]:+6.2f},{pb[1]:+6.2f})    '
              f'({rb[0]:+6.2f},{rb[1]:+6.2f})')

    # plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    aspect = IH/IW
    fig_w = 21.0
    fig_h = (fig_w/3)*aspect + 1.2
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), dpi=130)

    panels = [('(a) observed Δuv (info-weighted)', obs, 'turbo'),
              ('(b) predicted Δuv from 11-DoF fit', pred, 'turbo'),
              ('(c) residual = obs − pred',  resid, 'turbo')]
    norms = [np.linalg.norm(obs, axis=-1),
             np.linalg.norm(pred, axis=-1),
             np.linalg.norm(resid, axis=-1)]
    cmax_op = float(np.percentile(np.concatenate(norms[:2]), 95))
    cmax_r  = float(np.percentile(norms[2], 95))
    panel_cmax = [cmax_op, cmax_op, cmax_r]
    panel_scale = [0.6, 0.6, 0.6 * (cmax_r / max(cmax_op, 1e-6))]

    for ax, (title, d, cmap), n, cm, sc_arrow in zip(
            axes, panels, norms, panel_cmax, panel_scale):
        ax.set_facecolor('#0d0d0d')
        ax.set_xlim(0, IW); ax.set_ylim(IH, 0); ax.set_aspect('equal')
        sc = ax.quiver(u_c, v_c, d[:,0], d[:,1], n,
                       cmap=cmap, clim=(0, cm),
                       angles='xy', scale_units='xy', scale=sc_arrow,
                       width=0.0015, headwidth=4, headlength=5)
        cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label('|Δuv| px', fontsize=8)
        ax.axhline(v_max, color='#ff4040', lw=0.6, ls='--', alpha=0.6)
        ax.axhline(v_min, color='#ff4040', lw=0.6, ls='--', alpha=0.6)
        ax.set_title(title, fontsize=9)

    p_str = f'  p={p_fit.round(4).tolist()}' if p_fit is not None else ''
    dt_str = (f'  Δt=[{dt_fit[0]*1000:+.0f},{dt_fit[1]*1000:+.0f},'
              f'{dt_fit[2]*1000:+.0f}]mm' if dt_fit is not None else '')
    iter_tag = f'KB{len(k_fit)}-DoF fit (P={3+len(k_fit)+(2 if "fx_fit" in j else 0)+(2 if "tangential_p" in j else 0)+(3 if dt_fit is not None else 0)+2})'
    fig.suptitle(f'{iter_tag}  ω={omega.round(3).tolist()}°{dt_str}  '
                 f'fx*={fx_fit/fx0:.4f} fy*={fy_fit/fy0:.4f}  '
                 f'cx{cx_fit-cx0:+.1f}px cy{cy_fit-cy0:+.1f}px  '
                 f'k={k_fit.round(3).tolist()}{p_str}  '
                 f'wrms obs/pred/resid = {wrms(obs):.2f}/{wrms(pred):.2f}/{wrms(resid):.2f} px',
                 fontsize=10)
    fig.tight_layout()
    out_path = args.out or args.npz.with_name(args.npz.stem +
        f'_11dof_viz_n{args.n_thresh}_v{int(v_min)}-{int(v_max)}.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[viz] wrote {out_path}')


if __name__ == '__main__':
    main()
