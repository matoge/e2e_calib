"""KB k1..k4 fit on full-frame npz.

Per-cell info-weighted mean duv (Σ W)⁻¹ Σ W·d を residual として、各 cell
中心 (u_c, v_c) と既知 K/dist_init から KB の k1..k4 だけを Gauss-Newton
で解く。3D 不要 (φ = atan2(v-cy, u-cx) と KB-inv で θ を逆算)。

Usage:
  python scripts/_debug/_tss4_kb_fit.py \
      --npz scripts/_debug/_outputs/tss4_full_frame_<ckpt>_cs256.npz \
      --n-thresh 200 --v-frac 0.72 --n-iter 6
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def kb_theta_from_uv(u: np.ndarray, v: np.ndarray,
                     fx: float, fy: float, cx: float, cy: float,
                     k: np.ndarray, *, n_newton: int = 6) -> np.ndarray:
    """Invert KB radially: given measured (u,v), find θ such that
    θ_d(θ) = θ·(1+k1θ²+...+k4θ⁸) = r_pix / fmean (treating fx≈fy)."""
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
    ap.add_argument('--n-thresh', type=int, default=200)
    ap.add_argument('--v-frac', type=float, default=0.72)
    ap.add_argument('--v-min-frac', type=float, default=0.24)
    ap.add_argument('--n-iter', type=int, default=6)
    ap.add_argument('--cache', type=Path,
                    default=Path('/mnt/fsx/tmp/hfunaya/e2e_calib_cache/tss4_v3_tiled'))
    ap.add_argument('--ckpt-run', default=None,
                    help='not used for fit, only for getting K/dist_init')
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    sum_n  = z['sum_n']
    sum_W  = z['sum_W']; sum_Wd = z['sum_Wd']
    cell_px = int(z['cell_px']); IW = int(z['IW']); IH = int(z['IH'])
    nH, nW = sum_n.shape

    # K / dist_init from any inst (they're identical across TSS4 frames).
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
    print(f'[kb] K: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}')
    print(f'[kb] dist_init k1..k4 = {dist0}')

    # Build per-cell info-weighted mean and posterior cov.
    cells_u = (np.arange(nW) + 0.5) * cell_px
    cells_v = (np.arange(nH) + 0.5) * cell_px
    UU, VV = np.meshgrid(cells_u, cells_v)
    v_max = args.v_frac * IH
    v_min = args.v_min_frac * IH
    band = (VV < v_max) & (VV > v_min)
    ok = (sum_n >= args.n_thresh) & band
    print(f'[kb] active cells (n>={args.n_thresh}, {v_min:.0f}<v<{v_max:.0f}): '
          f'{int(ok.sum())}/{nW*nH}')

    # Per-cell info-mean and posterior W (= sum_W stays in info form, big = sure).
    cells = np.where(ok)
    cv_idx, cu_idx = cells
    M = cv_idx.size
    u_c = UU[cv_idx, cu_idx]
    v_c = VV[cv_idx, cu_idx]
    W_c = sum_W[cv_idx, cu_idx]      # (M, 2, 2)
    Wd_c = sum_Wd[cv_idx, cu_idx]    # (M, 2)
    # info mean
    d_meas = np.zeros((M, 2), dtype=np.float64)
    keep = np.ones(M, dtype=bool)
    for i in range(M):
        try:
            d_meas[i] = np.linalg.solve(W_c[i], Wd_c[i])
        except np.linalg.LinAlgError:
            keep[i] = False
    M0 = M
    M = int(keep.sum())
    cv_idx = cv_idx[keep]; cu_idx = cu_idx[keep]
    u_c = u_c[keep]; v_c = v_c[keep]
    W_c = W_c[keep]; Wd_c = Wd_c[keep]; d_meas = d_meas[keep]
    print(f'[kb] kept {M}/{M0} cells after info-mean solve')

    # geometry: φ from current cell center (= where the model says the point lies)
    du0 = u_c - cx; dv0 = v_c - cy
    r0 = np.sqrt(du0**2 + dv0**2 + 1e-12)
    cphi = du0 / r0
    sphi = dv0 / r0

    # Initial KB params: start from dist_init and update Δk = (k - dist_init).
    k = dist0.copy()
    print(f'\n[kb] iterating GN (using info matrix W):')
    print(f'  iter   |Δk|    k1          k2          k3          k4   '
          f'    cost    weighted-RMS(px)')
    for it in range(args.n_iter):
        theta = kb_theta_from_uv(u_c, v_c, fx, fy, cx, cy, k, n_newton=6)
        t2 = theta**2; t4 = t2**2; t6 = t4*t2; t8 = t4**2
        # KB column derivatives (∂u/∂k_i, ∂v/∂k_i) at the current θ.
        # u_pred = fx · θ·(1+k1t2+...+k4t8) · cos φ + cx
        # ∂u/∂k_i = fx · θ^(2i+1) · cos φ ; ∂v/∂k_i = fy · θ^(2i+1) · sin φ
        powers = np.stack([theta*t2, theta*t4, theta*t6, theta*t8], axis=-1)  # (M,4)
        Ju = (fx * cphi)[:, None] * powers   # (M,4)
        Jv = (fy * sphi)[:, None] * powers   # (M,4)
        J = np.stack([Ju, Jv], axis=1)       # (M, 2, 4)

        # Gauss-Newton with per-cell info matrix W_c:
        # H = Σ Jᵀ W J, b = Σ Jᵀ W r ; r = d_meas (treat as residual we want
        # the model to absorb by adjusting k from current k).
        # H (4,4), b (4,)
        WJ = np.einsum('mij,mjk->mik', W_c, J)               # (M,2,4)
        H = np.einsum('mij,mik->jk', J, WJ)                  # (4,4)
        b = np.einsum('mij,mi->j', WJ, d_meas)               # (4,)
        # weighted cost = Σ rᵀ W r
        cost = float(np.einsum('mi,mij,mj->', d_meas, W_c, d_meas))
        wrms = float(np.sqrt(cost / max(1, np.einsum('mij->', W_c))))
        try:
            dk = np.linalg.solve(H + 1e-9 * np.eye(4), b)
        except np.linalg.LinAlgError:
            dk = np.zeros(4)
        k = k + dk
        # update residual after step (apply step to predict new induced duv)
        # d_meas_new = d_meas - J · dk
        d_meas = d_meas - np.einsum('mij,j->mi', J, dk)
        print(f'  {it:3d}  {np.linalg.norm(dk):8.2e}  '
              f'{k[0]:+.4e}  {k[1]:+.4e}  {k[2]:+.4e}  {k[3]:+.4e}  '
              f'{cost:.3e}  {wrms:.3f}')

    print(f'\n[kb] final k1..k4 = {k}')
    print(f'[kb] dist_init     = {dist0}')
    print(f'[kb] Δk            = {k - dist0}')

    # Save
    out_json = args.npz.with_name(args.npz.stem +
        f'_kb_n{args.n_thresh}_v{int(v_min)}-{int(v_max)}.json')
    out_json.write_text(json.dumps({
        'n_thresh': args.n_thresh, 'v_max': float(v_max),
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
        'dist_init': dist0.tolist(),
        'dist_fit':  k.tolist(),
        'delta_k':   (k - dist0).tolist(),
        'n_cells_used': int(M),
    }, indent=2))
    print(f'[kb] wrote {out_json}')


if __name__ == '__main__':
    main()
