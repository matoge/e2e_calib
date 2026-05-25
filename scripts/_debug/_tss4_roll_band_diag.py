"""ROLL signal localisation diagnostic.

Question: ω_z = -0.20° という KB10 from-scratch の fit は、中央セル（W-dominant）
から来ているのか、それとも端から来ているのか？

手順:
  1) iter1 npz から (sum_n, sum_W, sum_Wd) → 各セルの d_meas, W
  2) lockroll fit (ω_z=0) と direct fit (ω_z=-0.20) を読む
  3) 各セルで両方の forward を計算 → r_lock, r_dir
  4) u 軸で 5-band に分割し、band ごとに：
     - cell 数 / Σ trace(W)（info weight 寄与）
     - info-weighted ⟨du⟩, ⟨dv⟩ (lockroll の残差 ＝ ω_z を許せば消える信号)
     - cost = Σ rᵀWr  と Δcost = cost_lock - cost_dir（負＝direct のほうが良い）

中央バンドが Δcost 寄与の大半を占めれば「中央起源」、端バンドなら「端起源」。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


_D2R = np.pi / 180.0


def rodrigues(omega_deg):
    th = float(np.linalg.norm(omega_deg)) * _D2R
    if th < 1e-12:
        return np.eye(3)
    axis = (np.asarray(omega_deg, dtype=np.float64) * _D2R) / th
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def kb_theta_from_uv(u, v, fx, fy, cx, cy, k, n_newton=12):
    du = u - cx; dv = v - cy
    r_pix = np.sqrt(du * du + dv * dv)
    theta_d = r_pix / (0.5 * (fx + fy))
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


def project(nu, fx, fy, cx, cy, k, p=None):
    X, Y, Z = nu[:, 0], nu[:, 1], nu[:, 2]
    r = np.sqrt(X * X + Y * Y + 1e-24)
    theta = np.arctan2(r, np.maximum(Z, 1e-9))
    t2 = theta * theta
    poly = np.ones_like(theta)
    tp = t2.copy()
    for ki in k:
        poly = poly + ki * tp
        tp = tp * t2
    theta_d = theta * poly
    Xr = X / np.maximum(r, 1e-9); Yr = Y / np.maximum(r, 1e-9)
    Xp = theta_d * Xr; Yp = theta_d * Yr
    if p is not None:
        r2p = Xp * Xp + Yp * Yp
        du_t = 2 * p[0] * Xp * Yp + p[1] * (r2p + 2 * Xp * Xp)
        dv_t = p[0] * (r2p + 2 * Yp * Yp) + 2 * p[1] * Xp * Yp
        Xp = Xp + du_t
        Yp = Yp + dv_t
    return np.stack([fx * Xp + cx, fy * Yp + cy], axis=-1)


def load_fit(jp):
    j = json.loads(Path(jp).read_text())
    return {
        'fx': float(j['fx_fit']), 'fy': float(j['fy_fit']),
        'cx': float(j['cx_fit']), 'cy': float(j['cy_fit']),
        'k':  np.asarray(j['dist_fit'], dtype=np.float64),
        'p':  np.asarray(j.get('tangential_p', [0.0, 0.0]), dtype=np.float64),
        'omega': np.asarray(j['omega_deg'], dtype=np.float64),
        'baseline_omega': np.asarray(j['baseline_omega_deg'], dtype=np.float64),
        'baseline_K': np.asarray(j['baseline_K'], dtype=np.float64),
        'baseline_dist': np.asarray(j['baseline_dist'], dtype=np.float64),
        'final_wrms_px': float(j['final_wrms_px']),
        'name': Path(jp).stem,
    }


def cell_residuals(npz, fit, n_thresh=200, v_min_frac=0.362, v_frac=0.72):
    """Returns u, v, W, r (target_uv - pred_uv) for active cells."""
    z = np.load(npz, allow_pickle=False)
    sum_n = z['sum_n']; sum_W = z['sum_W']; sum_Wd = z['sum_Wd']
    cell_px = int(z['cell_px']); IW = int(z['IW']); IH = int(z['IH'])
    nH, nW = sum_n.shape
    cells_u = (np.arange(nW) + 0.5) * cell_px
    cells_v = (np.arange(nH) + 0.5) * cell_px
    UU, VV = np.meshgrid(cells_u, cells_v)
    band = (VV < v_frac * IH) & (VV > v_min_frac * IH)
    ok = (sum_n >= n_thresh) & band
    cv_idx, cu_idx = np.where(ok)
    u_c = UU[cv_idx, cu_idx]; v_c = VV[cv_idx, cu_idx]
    W_c = sum_W[cv_idx, cu_idx]
    Wd_c = sum_Wd[cv_idx, cu_idx]
    M = u_c.size
    d_meas = np.zeros((M, 2))
    keep = np.ones(M, dtype=bool)
    for i in range(M):
        try:
            d_meas[i] = np.linalg.solve(W_c[i], Wd_c[i])
        except np.linalg.LinAlgError:
            keep[i] = False
    u_c = u_c[keep]; v_c = v_c[keep]
    W_c = W_c[keep]; d_meas = d_meas[keep]

    # baseline (KB-N inv) → unit ray
    bk = fit['baseline_K']
    bdist = fit['baseline_dist']
    theta0 = kb_theta_from_uv(u_c, v_c, bk[0], bk[1], bk[2], bk[3], bdist)
    phi0 = np.arctan2(v_c - bk[3], u_c - bk[2])
    nu_baseline = np.stack([
        np.sin(theta0) * np.cos(phi0),
        np.sin(theta0) * np.sin(phi0),
        np.cos(theta0),
    ], axis=-1)
    target_uv = np.stack([u_c, v_c], axis=-1) + d_meas

    # Δω relative to baseline (the iter's ω stored in 'iter_delta_omega_deg' is
    # what we want; but 'omega' in our load_fit is the COMPOSED total. We need
    # to extract Δω = R_total · R_baselineᵀ then logmap.)
    R_total = rodrigues(fit['omega'])
    R_base = rodrigues(fit['baseline_omega'])
    R_delta = R_total @ R_base.T
    nu_lin = nu_baseline @ R_delta.T

    pred = project(nu_lin, fit['fx'], fit['fy'], fit['cx'], fit['cy'],
                   fit['k'], fit['p'])
    r = target_uv - pred
    return u_c, v_c, W_c, r, IW, IH


def per_band_summary(u, v, W, r, IW, n_bands=5, label=''):
    edges = np.linspace(0, IW, n_bands + 1)
    rows = []
    cost_total = 0.0
    wt_total = 0.0
    for i in range(n_bands):
        m = (u >= edges[i]) & (u < edges[i + 1])
        if not m.any():
            rows.append((i, edges[i], edges[i+1], 0, 0.0, 0.0, 0.0, 0.0, 0.0))
            continue
        Wm = W[m]; rm = r[m]
        # info weights
        trW = np.einsum('mij->m', Wm)  # tr(W)
        wt_band = float(trW.sum())
        # info-weighted ⟨du⟩, ⟨dv⟩
        du_mean = float((trW * rm[:, 0]).sum() / max(wt_band, 1e-12))
        dv_mean = float((trW * rm[:, 1]).sum() / max(wt_band, 1e-12))
        # cost = Σ rᵀWr
        cost = float(np.einsum('mi,mij,mj->', rm, Wm, rm))
        wrms = float(np.sqrt(cost / max(wt_band, 1e-12)))
        rows.append((i, edges[i], edges[i+1], int(m.sum()),
                     wt_band, du_mean, dv_mean, cost, wrms))
        cost_total += cost
        wt_total += wt_band
    print(f'\n=== {label} ===')
    print(f'{"band":>4} {"u-range":>14} {"M":>4} {"Σtr(W)":>14} '
          f'{"⟨du⟩":>9} {"⟨dv⟩":>9} {"cost":>14} {"wrms":>7} {"%cost":>7} {"%wt":>7}')
    for (i, u0, u1, M, wt, du_, dv_, cost, wrms) in rows:
        pc = 100.0 * cost / max(cost_total, 1e-12)
        pw = 100.0 * wt / max(wt_total, 1e-12)
        print(f'{i:>4} {u0:>6.0f}–{u1:<7.0f} {M:>4d} {wt:>14.4g} '
              f'{du_:>+9.3f} {dv_:>+9.3f} {cost:>14.4g} {wrms:>7.3f} '
              f'{pc:>6.1f}% {pw:>6.1f}%')
    print(f'TOTAL  cost={cost_total:.4g}  Σtr(W)={wt_total:.4g}  '
          f'wrms={np.sqrt(cost_total/max(wt_total,1e-12)):.3f}')
    return rows, cost_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=Path,
        default=Path('scripts/_debug/_outputs/tss4_full_frame_km_wv_wm_n4_img128_ml_dense_pe_dgx2_200ep_v2_cs256.npz'))
    ap.add_argument('--fit-lock', type=Path,
        default=Path('scripts/_debug/_outputs/tss4_full_frame_km_wv_wm_n4_img128_ml_dense_pe_dgx2_200ep_v2_cs256_kb10_n200_v600-1194_kb10_iter1_lockroll.json'))
    ap.add_argument('--fit-dir', type=Path,
        default=Path('scripts/_debug/_outputs/tss4_full_frame_km_wv_wm_n4_img128_ml_dense_pe_dgx2_200ep_v2_cs256_kb10_n200_v600-1194_kb10_iter1_direct.json'))
    ap.add_argument('--n-bands', type=int, default=5)
    args = ap.parse_args()

    fit_lock = load_fit(args.fit_lock)
    fit_dir = load_fit(args.fit_dir)
    print(f'[diag] LOCK fit: ω={fit_lock["omega"]}°  wrms={fit_lock["final_wrms_px"]:.3f}')
    print(f'[diag]  DIR fit: ω={fit_dir["omega"]}°  wrms={fit_dir["final_wrms_px"]:.3f}')

    u, v, W, r_lock, IW, IH = cell_residuals(args.npz, fit_lock)
    _, _, _, r_dir, _, _ = cell_residuals(args.npz, fit_dir)
    print(f'[diag] active cells: {u.size}  IW={IW} IH={IH}')

    rows_lock, cost_lock = per_band_summary(u, v, W, r_lock, IW,
        n_bands=args.n_bands, label='LOCKROLL fit residuals (per u-band)')
    rows_dir, cost_dir = per_band_summary(u, v, W, r_dir, IW,
        n_bands=args.n_bands, label='DIRECT (ω_z=-0.20°) fit residuals (per u-band)')

    print('\n=== Δcost per band (lock − dir) — どこで roll が稼いでいるか ===')
    print(f'{"band":>4} {"u-range":>14} {"Δcost":>14} {"%share":>8} '
          f'{"wrms_lock":>10} {"wrms_dir":>10}')
    deltas = []
    for L, D in zip(rows_lock, rows_dir):
        i, u0, u1, M, wt, _, _, c_l, w_l = L
        _, _, _, _, _, _, _, c_d, w_d = D
        deltas.append((i, u0, u1, c_l - c_d, w_l, w_d))
    total_delta = sum(d[3] for d in deltas)
    for (i, u0, u1, dc, w_l, w_d) in deltas:
        share = 100.0 * dc / max(total_delta, 1e-12)
        print(f'{i:>4} {u0:>6.0f}–{u1:<7.0f} {dc:>+14.4g} {share:>+7.1f}% '
              f'{w_l:>10.3f} {w_d:>10.3f}')
    print(f'TOTAL Δcost = {total_delta:.4g}  (>0 means roll helps)')

    # Same again, but binned by radial distance from optical center
    cx0 = fit_lock['cx']; cy0 = fit_lock['cy']
    r_pix = np.sqrt((u - cx0) ** 2 + (v - cy0) ** 2)
    r_max = float(r_pix.max())
    n_rb = args.n_bands
    edges = np.linspace(0, r_max + 1, n_rb + 1)
    print('\n=== Δcost per RADIAL band (concentric rings) ===')
    print(f'{"ring":>4} {"r-range(px)":>16} {"M":>4} {"Δcost":>14} {"%share":>8}')
    rb_deltas = []
    for i in range(n_rb):
        m = (r_pix >= edges[i]) & (r_pix < edges[i + 1])
        if not m.any():
            rb_deltas.append((i, edges[i], edges[i+1], 0, 0.0))
            continue
        c_l = float(np.einsum('mi,mij,mj->', r_lock[m], W[m], r_lock[m]))
        c_d = float(np.einsum('mi,mij,mj->', r_dir[m], W[m], r_dir[m]))
        rb_deltas.append((i, edges[i], edges[i+1], int(m.sum()), c_l - c_d))
    tot_rb = sum(d[4] for d in rb_deltas)
    for (i, r0, r1, M, dc) in rb_deltas:
        share = 100.0 * dc / max(tot_rb, 1e-12)
        print(f'{i:>4} {r0:>7.0f}–{r1:<7.0f} {M:>4d} {dc:>+14.4g} {share:>+7.1f}%')
    print(f'TOTAL Δcost = {tot_rb:.4g}')


if __name__ == '__main__':
    main()
