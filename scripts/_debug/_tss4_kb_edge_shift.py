"""Predict edge-band (du, dv) shift induced by Δk = k_fit - k_init.

Given the fitted KB params, compare what the KB delta predicts at the
left/right edges with what the model's info-weighted residuals say.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def kb_theta_from_uv(u, v, fx, fy, cx, cy, k, n_newton=6):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=Path, required=True)
    ap.add_argument('--json', type=Path, required=True,
                    help='kb_fit json (has dist_init, dist_fit, fx/fy/cx/cy)')
    ap.add_argument('--n-thresh', type=int, default=200)
    ap.add_argument('--v-frac', type=float, default=0.72)
    ap.add_argument('--v-min-frac', type=float, default=0.362)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    sum_n = z['sum_n']
    sum_W = z['sum_W']; sum_Wd = z['sum_Wd']
    cell_px = int(z['cell_px']); IW = int(z['IW']); IH = int(z['IH'])
    nH, nW = sum_n.shape

    j = json.loads(args.json.read_text())
    fx, fy, cx, cy = j['fx'], j['fy'], j['cx'], j['cy']
    k_init = np.array(j['dist_init'], dtype=np.float64)
    k_fit  = np.array(j['dist_fit'],  dtype=np.float64)
    dk = k_fit - k_init
    print(f'[edge] k_init = {k_init}')
    print(f'[edge] k_fit  = {k_fit}')
    print(f'[edge] Δk     = {dk}')

    cells_u = (np.arange(nW) + 0.5) * cell_px
    cells_v = (np.arange(nH) + 0.5) * cell_px
    UU, VV = np.meshgrid(cells_u, cells_v)
    v_max = args.v_frac * IH
    v_min = args.v_min_frac * IH
    band = (VV < v_max) & (VV > v_min)
    ok = (sum_n >= args.n_thresh) & band

    edge_w = max(1, nW // 8)
    bands = [
        (f'left   (cu=0..{edge_w-1})', list(range(0, edge_w))),
        (f'mid    (cu={edge_w}..{nW-edge_w-1})', list(range(edge_w, nW-edge_w))),
        (f'right  (cu={nW-edge_w}..{nW-1})', list(range(nW-edge_w, nW))),
    ]

    print(f'\n[edge] Predicted vs observed shift per column band')
    print(f'  band                         pred du   pred dv | obs du    obs dv')
    for label, cus in bands:
        sel = ok[:, cus]
        if sel.sum() == 0:
            continue
        cv_idx, cu_off = np.where(sel)
        cu_idx = np.array([cus[c] for c in cu_off])
        u_c = UU[cv_idx, cu_idx]; v_c = VV[cv_idx, cu_idx]
        W_c = sum_W[cv_idx, cu_idx]
        Wd_c = sum_Wd[cv_idx, cu_idx]
        # observed info-weighted band mean
        Wb = W_c.sum(axis=0); Wdb = Wd_c.sum(axis=0)
        try:
            ob = np.linalg.solve(Wb, Wdb)
        except np.linalg.LinAlgError:
            ob = np.array([np.nan, np.nan])
        # predicted shift at each cell using k_init geometry, then J·Δk
        theta = kb_theta_from_uv(u_c, v_c, fx, fy, cx, cy, k_init, n_newton=6)
        t2 = theta**2; t4 = t2**2; t6 = t4*t2; t8 = t4**2
        du0 = u_c - cx; dv0 = v_c - cy
        r0 = np.sqrt(du0**2 + dv0**2 + 1e-12)
        cphi = du0 / r0; sphi = dv0 / r0
        powers = np.stack([theta*t2, theta*t4, theta*t6, theta*t8], axis=-1)
        Ju = (fx * cphi)[:, None] * powers
        Jv = (fy * sphi)[:, None] * powers
        pred_du_per = Ju @ dk
        pred_dv_per = Jv @ dk
        # info-weighted aggregate of predicted
        # use same W to combine (so units match the obs band mean)
        d_pred = np.stack([pred_du_per, pred_dv_per], axis=-1)
        Wd_pred_b = np.einsum('mij,mj->i', W_c, d_pred)
        try:
            pb = np.linalg.solve(Wb, Wd_pred_b)
        except np.linalg.LinAlgError:
            pb = np.array([np.nan, np.nan])
        print(f'  {label:30s}  {pb[0]:+7.2f}  {pb[1]:+7.2f}  | '
              f'{ob[0]:+7.2f}  {ob[1]:+7.2f}')


if __name__ == '__main__':
    main()
