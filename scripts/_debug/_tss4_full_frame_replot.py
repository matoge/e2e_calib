"""npz を読んで mask 条件 (sum_n threshold + v_max) を変えて再描画する。

Usage:
  python scripts/_debug/_tss4_full_frame_replot.py \
      --npz scripts/_debug/_outputs/tss4_full_frame_<ckpt>_cs256.npz \
      --n-thresh 200 --v-frac 0.72
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', type=Path, required=True)
    ap.add_argument('--n-thresh', type=int, default=200,
                    help='min sum_n per cell')
    ap.add_argument('--v-frac', type=float, default=0.72,
                    help='only cells with v_center < v_frac * IH (drop dashboard)')
    ap.add_argument('--v-min-frac', type=float, default=0.24,
                    help='only cells with v_center > v_min_frac * IH (drop sky/top)')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    sum_n  = z['sum_n']
    sum_du = z['sum_du']; sum_dv = z['sum_dv']
    sum_du2 = z['sum_du2']; sum_dv2 = z['sum_dv2']
    sum_W  = z['sum_W'];  sum_Wd = z['sum_Wd']
    cell_px = int(z['cell_px']); IW = int(z['IW']); IH = int(z['IH'])
    nH, nW = sum_n.shape
    print(f'[replot] grid={nW}x{nH} cell={cell_px} IW={IW} IH={IH}')
    print(f'[replot] total points: {int(sum_n.sum()):,}')

    # quick sum_n vs v histogram
    n_per_v = sum_n.sum(axis=1)
    print(f'[replot] sum_n per v-row (cv: count):')
    for cv in range(nH):
        bar = '#' * int(40 * n_per_v[cv] / max(1, n_per_v.max()))
        v_center = (cv + 0.5) * cell_px
        print(f'  cv={cv:3d}  v={v_center:6.0f}  n={int(n_per_v[cv]):8d}  {bar}')

    n_safe = np.where(sum_n > 0, sum_n, 1).astype(np.float64)
    mean_du = sum_du / n_safe
    mean_dv = sum_dv / n_safe
    mean_norm = np.sqrt(mean_du**2 + mean_dv**2)
    var_du = np.maximum(sum_du2 / n_safe - mean_du**2, 0.0)
    var_dv = np.maximum(sum_dv2 / n_safe - mean_dv**2, 0.0)
    std_combined = np.sqrt(var_du + var_dv)
    cons = mean_norm / np.where(std_combined > 1e-3, std_combined, 1e-3)

    info_du = np.zeros_like(mean_du)
    info_dv = np.zeros_like(mean_dv)
    post_uu = np.zeros_like(mean_du)
    post_vv = np.zeros_like(mean_du)
    for cv in range(nH):
        for cu in range(nW):
            if sum_n[cv, cu] < 1:
                continue
            W = sum_W[cv, cu]
            try:
                C = np.linalg.inv(W)
            except np.linalg.LinAlgError:
                continue
            d = C @ sum_Wd[cv, cu]
            info_du[cv, cu] = d[0]; info_dv[cv, cu] = d[1]
            post_uu[cv, cu] = C[0, 0]; post_vv[cv, cu] = C[1, 1]
    info_norm = np.sqrt(info_du**2 + info_dv**2)
    post_std = np.sqrt(np.maximum(post_uu + post_vv, 0.0))
    info_cons = info_norm / np.where(post_std > 1e-3, post_std, 1e-3)

    # cell mask: enough points AND above dashboard
    v_centers = (np.arange(nH) + 0.5) * cell_px
    v_max = args.v_frac * IH
    v_min = args.v_min_frac * IH
    band = ((v_centers < v_max) & (v_centers > v_min))[:, None]   # (nH, 1)
    ok = (sum_n >= args.n_thresh) & band
    print(f'[replot] mask: n>={args.n_thresh} AND {v_min:.0f}<v<{v_max:.0f}  '
          f'→ active cells: {int(ok.sum())}/{nW*nH}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb

    aspect = IH / IW
    fig_w = 21.0
    fig_h = (fig_w / 3) * aspect + 1.5
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), dpi=130)

    cells_u = (np.arange(nW) + 0.5) * cell_px
    cells_v = (np.arange(nH) + 0.5) * cell_px
    UU, VV = np.meshgrid(cells_u, cells_v)

    # ── direction-encoded color: hue = atan2(dv, du), saturation = |mean|
    #    (so longer arrows are vivid, near-zero arrows are gray)
    def dir_rgba(du_arr, dv_arr, conf_arr):
        ang = np.arctan2(dv_arr, du_arr)            # -π..π
        H = (ang / (2 * np.pi)) % 1.0
        # saturation grows with confidence (clip outliers via 95th pctile)
        c95 = np.percentile(conf_arr[np.isfinite(conf_arr) & (conf_arr > 0)],
                            95) if np.any(conf_arr > 0) else 1.0
        S = np.clip(conf_arr / max(c95, 1e-6), 0.0, 1.0)
        V = np.ones_like(H)
        rgb = hsv_to_rgb(np.stack([H, S, V], axis=-1))
        return rgb

    # (a) hard quiver — color = direction, intensity = |mean|/std
    ax = axes[0]
    ax.set_facecolor('#0d0d0d')
    ax.set_xlim(0, IW); ax.set_ylim(IH, 0)
    ax.set_aspect('equal')
    if ok.any():
        rgb = dir_rgba(mean_du[ok], mean_dv[ok], cons[ok])
        ax.quiver(UU[ok], VV[ok], mean_du[ok], mean_dv[ok],
                  color=rgb,
                  angles='xy', scale_units='xy',
                  scale=0.6, width=0.0015, headwidth=4, headlength=5)
    ax.axhline(v_max, color='#ff4040', lw=0.6, ls='--', alpha=0.6)
    ax.axhline(v_min, color='#ff4040', lw=0.6, ls='--', alpha=0.6)
    ax.set_title(f'(a) hard-bin  (n>={args.n_thresh}, {v_min:.0f}<v<{v_max:.0f})',
                 fontsize=9)

    # (b) info quiver — color = direction, intensity = |mean|/√tr(Σ_post)
    ax = axes[1]
    ax.set_facecolor('#0d0d0d')
    ax.set_xlim(0, IW); ax.set_ylim(IH, 0)
    ax.set_aspect('equal')
    if ok.any():
        rgb = dir_rgba(info_du[ok], info_dv[ok], info_cons[ok])
        ax.quiver(UU[ok], VV[ok], info_du[ok], info_dv[ok],
                  color=rgb,
                  angles='xy', scale_units='xy',
                  scale=0.6, width=0.0015, headwidth=4, headlength=5)
    ax.axhline(v_max, color='#ff4040', lw=0.6, ls='--', alpha=0.6)
    ax.set_title('(b) info-weighted  (hue=direction, sat=confidence)', fontsize=9)

    # (c) direction wheel as inline legend over heatmap of |info-mean|
    ax = axes[2]
    masked = np.ma.masked_where(~ok, info_norm)
    im = ax.imshow(masked, origin='upper', cmap='magma',
                    extent=(0, IW, IH, 0))
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label('|info-mean| (px)', fontsize=8)
    ax.axhline(v_max, color='#ff4040', lw=0.6, ls='--', alpha=0.6)
    ax.set_title('(c) info |mean|', fontsize=9)

    fig.suptitle(f'TSS4 full-frame  hue=direction (→red ↓green ←cyan ↑purple)  '
                 f'sat=conf  n>={args.n_thresh}  v<{v_max:.0f}', fontsize=10)
    fig.tight_layout()

    # Direction-wheel legend in lower-right inset.
    wheel_ax = fig.add_axes([0.84, 0.02, 0.13, 0.13], polar=False)
    th = np.linspace(0, 2 * np.pi, 256)
    rr = np.linspace(0, 1, 64)
    TH, RR = np.meshgrid(th, rr)
    H = (TH / (2 * np.pi)) % 1.0
    S = RR
    V = np.ones_like(H)
    wheel_rgb = hsv_to_rgb(np.stack([H, S, V], axis=-1))
    XX = RR * np.cos(TH); YY = RR * np.sin(TH)
    wheel_ax.scatter(XX, YY, c=wheel_rgb.reshape(-1, 3), s=4, marker='s',
                     linewidths=0)
    wheel_ax.set_xlim(-1.1, 1.1); wheel_ax.set_ylim(-1.1, 1.1)
    wheel_ax.set_aspect('equal'); wheel_ax.set_xticks([]); wheel_ax.set_yticks([])
    wheel_ax.set_title('direction', fontsize=7)
    # cardinal labels (image coords: +u right, +v down)
    wheel_ax.annotate('→ right', (1.0, 0), fontsize=6, va='center')
    wheel_ax.annotate('↓ down',  (0, 1.0), fontsize=6, ha='center')
    wheel_ax.annotate('← left',  (-1.0, 0), fontsize=6, va='center', ha='right')
    wheel_ax.annotate('↑ up',    (0, -1.0), fontsize=6, ha='center', va='bottom')

    out_path = args.out or args.npz.with_name(
        args.npz.stem +
        f'_n{args.n_thresh}_v{int(v_min)}-{int(v_max)}.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[replot] wrote {out_path}')

    # column band summary using ONLY masked cells
    edge_w = max(1, nW // 8)
    print(f'\n[replot] column-band weighted means (masked):')
    for label, cus in [(f'left   (cu=0..{edge_w-1})', list(range(0, edge_w))),
                        (f'mid    (cu={edge_w}..{nW-edge_w-1})',
                            list(range(edge_w, nW - edge_w))),
                        (f'right  (cu={nW-edge_w}..{nW-1})',
                            list(range(nW - edge_w, nW)))]:
        sel = ok[:, cus]
        nm = sum_n[:, cus] * sel
        if nm.sum() == 0:
            continue
        wmu = (sum_du[:, cus] * sel).sum() / nm.sum()
        wmv = (sum_dv[:, cus] * sel).sum() / nm.sum()
        Wb  = (sum_W[:, cus]  * sel[..., None, None]).sum(axis=(0, 1))
        Wdb = (sum_Wd[:, cus] * sel[..., None]).sum(axis=(0, 1))
        try:
            ib = np.linalg.solve(Wb, Wdb)
            iwmu, iwmv = float(ib[0]), float(ib[1])
        except np.linalg.LinAlgError:
            iwmu = iwmv = float('nan')
        print(f'  {label}  hard du={wmu:+.3f} dv={wmv:+.3f}   '
              f'info du={iwmu:+.3f} dv={iwmv:+.3f}   '
              f'(n_pts={int(nm.sum())})')


if __name__ == '__main__':
    main()
