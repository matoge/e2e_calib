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


def kb_theta_from_uv(u, v, fx, fy, cx, cy, k, n_newton=10):
    du = u - cx; dv = v - cy
    r_pix = np.sqrt(du**2 + dv**2)
    f_mean = 0.5*(fx+fy)
    theta_d = r_pix / f_mean
    theta = theta_d.copy()
    for _ in range(n_newton):
        t2=theta**2; t4=t2**2; t6=t4*t2; t8=t4**2
        f = theta*(1+k[0]*t2+k[1]*t4+k[2]*t6+k[3]*t8) - theta_d
        df = 1+3*k[0]*t2+5*k[1]*t4+7*k[2]*t6+9*k[3]*t8
        theta = theta - f/np.maximum(df, 1e-9)
    return theta


def project_unit_ray(nu, fx, fy, cx, cy, k):
    X,Y,Z = nu[:,0], nu[:,1], nu[:,2]
    r = np.sqrt(X*X+Y*Y+1e-24)
    theta = np.arctan2(r, np.maximum(Z, 1e-9))
    t2=theta**2; t4=t2**2; t6=t4*t2; t8=t4**2
    poly = 1+k[0]*t2+k[1]*t4+k[2]*t6+k[3]*t8
    theta_d = theta*poly
    Xr = X/np.maximum(r,1e-9); Yr = Y/np.maximum(r,1e-9)
    u = fx*theta_d*Xr + cx
    v = fy*theta_d*Yr + cy
    return np.stack([u,v], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=Path, required=True)
    ap.add_argument('--json', type=Path, required=True)
    ap.add_argument('--n-thresh', type=int, default=200)
    ap.add_argument('--v-frac', type=float, default=0.72)
    ap.add_argument('--v-min-frac', type=float, default=0.362)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    sum_n=z['sum_n']; sum_W=z['sum_W']; sum_Wd=z['sum_Wd']
    cell_px=int(z['cell_px']); IW=int(z['IW']); IH=int(z['IH'])
    nH,nW = sum_n.shape

    j = json.loads(args.json.read_text())
    fx0 = j['fx_init']; fy0 = j['fy_init']; cx0 = j['cx_init']; cy0 = j['cy_init']
    fx_fit = j['fx_fit']; fy_fit = j['fy_fit']; cx_fit = j['cx_fit']; cy_fit = j['cy_fit']
    k0 = np.array(j['dist_init']); k_fit = np.array(j['dist_fit'])
    omega = np.array(j['omega_deg'])

    cells_u = (np.arange(nW)+0.5)*cell_px
    cells_v = (np.arange(nH)+0.5)*cell_px
    UU,VV = np.meshgrid(cells_u, cells_v)
    v_max = args.v_frac*IH; v_min = args.v_min_frac*IH
    band = (VV<v_max)&(VV>v_min)
    ok = (sum_n>=args.n_thresh)&band
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

    # predicted: cell center → unit ray (init geometry) → R · ν → KB(k_fit, fxy_fit, cxy_fit)
    theta0 = kb_theta_from_uv(u_c, v_c, fx0, fy0, cx0, cy0, k0, n_newton=10)
    phi0 = np.arctan2(v_c-cy0, u_c-cx0)
    nu = np.stack([np.sin(theta0)*np.cos(phi0),
                   np.sin(theta0)*np.sin(phi0),
                   np.cos(theta0)], axis=-1)
    R = rodrigues(omega)
    nu_rot = nu @ R.T
    uv_pred = project_unit_ray(nu_rot, fx_fit, fy_fit, cx_fit, cy_fit, k_fit)
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
    cmax = float(np.percentile(np.concatenate(norms), 95))

    for ax, (title, d, cmap), n in zip(axes, panels, norms):
        ax.set_facecolor('#0d0d0d')
        ax.set_xlim(0, IW); ax.set_ylim(IH, 0); ax.set_aspect('equal')
        sc = ax.quiver(u_c, v_c, d[:,0], d[:,1], n,
                       cmap=cmap, clim=(0, cmax),
                       angles='xy', scale_units='xy', scale=0.6,
                       width=0.0015, headwidth=4, headlength=5)
        cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label('|Δuv| px', fontsize=8)
        ax.axhline(v_max, color='#ff4040', lw=0.6, ls='--', alpha=0.6)
        ax.axhline(v_min, color='#ff4040', lw=0.6, ls='--', alpha=0.6)
        ax.set_title(title, fontsize=9)

    fig.suptitle(f'11-DoF KB fit  ω={omega.round(3).tolist()}°  '
                 f'fx*={fx_fit/fx0:.4f} fy*={fy_fit/fy0:.4f}  '
                 f'cx{cx_fit-cx0:+.1f}px cy{cy_fit-cy0:+.1f}px  '
                 f'k={k_fit.round(3).tolist()}  '
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
