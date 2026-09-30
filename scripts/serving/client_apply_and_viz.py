"""Independent client: POST to auto_calib_server, then render the exact
tile-based overlay style used by apply_frame0.py --viz.

    python scripts/serving/client_apply_and_viz.py \\
        --seq <...> --api http://localhost:8501 \\
        --stride 5 --max-frames 10 --viz-frame-idx 0 --out out.png

The joint 6-DoF pool comes from POST /calibrate/sequence. The per-tile HAT/μ
points drawn on top come from POST /calibrate/frame with return_per_point=1
(the same forward the sequence run does internally, exposed for viz).
"""
from __future__ import annotations
import argparse, json, sys, requests
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse

REPO = Path(__file__).resolve().parents[2]

CS = 512
S  = 256


def _tile_grid(iw, ih, cs=CS):
    nx = max(1, int(np.ceil(iw / cs)))
    ny = max(1, int(np.ceil(ih / cs)))
    sx = max(0, iw - cs) / max(nx - 1, 1)
    sy = max(0, ih - cs) / max(ny - 1, 1)
    return [(int(round(sx * i)), int(round(sy * j)))
            for j in range(ny) for i in range(nx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True, type=Path)
    ap.add_argument('--api', default='http://127.0.0.1:8501')
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--max-frames', type=int, default=10)
    ap.add_argument('--viz-frame-idx', type=int, default=0)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()

    # ── 1) joint pool over the whole sequence
    print(f'[client] POST {args.api}/calibrate/sequence ...', flush=True)
    seq_resp = requests.post(f'{args.api}/calibrate/sequence',
                              json={'seq_dir': str(args.seq),
                                    'stride': args.stride,
                                    'max_frames': args.max_frames,
                                    'use_hood_mask': True,
                                    'use_gate3': True},
                              timeout=120).json()
    d  = seq_resp['deltas']; sd = seq_resp['sd']
    print(f'[client]   F={seq_resp["n_frames_used"]} k={seq_resp["k_dispersion"]} '
          f'elapsed={seq_resp["elapsed_s"]}s   '
          f'pitch {np.degrees(d["rot_y"]):+.4f}°±{np.degrees(sd["rot_y"]):.4f}°',
          flush=True)

    # ── 2) per-tile HAT/μ for the frame we want to visualize
    print(f'[client] POST {args.api}/calibrate/frame  frame_idx={args.viz_frame_idx}',
          flush=True)
    frm_resp = requests.post(f'{args.api}/calibrate/frame',
                              json={'seq_dir': str(args.seq),
                                    'frame_idx': args.viz_frame_idx,
                                    'use_hood_mask': True,
                                    'return_per_point': True,
                                    'return_hood_polygon': True},
                              timeout=120).json()
    if frm_resp.get('status') != 'success':
        raise RuntimeError(f'frame request failed: {frm_resp}')
    IW, IH = frm_resp['image_size']
    fid = frm_resp['fid']

    # ── 3) load the image from disk (client-side, no need to ship bytes)
    img = cv2.cvtColor(cv2.imread(str(args.seq / 'tss4_fcm' / f'{fid}.jpg')),
                        cv2.COLOR_BGR2RGB)

    # ── 4) render: same style as apply_frame0.py --viz
    fig, ax = plt.subplots(figsize=(19, 11), tight_layout=True)
    ax.imshow(img)

    # Hood polygon (magenta)
    if frm_resp.get('hood_polygon'):
        poly = np.array(frm_resp['hood_polygon'], dtype=float)
        poly = np.vstack([poly, poly[:1]])
        ax.plot(poly[:, 0], poly[:, 1], '-', color='magenta',
                lw=1.8, alpha=0.9, label='hood mask polygon')

    # Tile grid
    cells = _tile_grid(IW, IH, CS)
    fused_flags = {t['tile_idx']: not t['skipped'] for t in frm_resp['tiles']}
    for b, (u0, v0) in enumerate(cells):
        col = '#4CAF50' if fused_flags.get(b, False) else '#E53935'
        ax.add_patch(Rectangle((u0, v0), CS, CS, fill=False,
                                ec=col, lw=0.7, alpha=0.35))

    # HAT / PRED / arrows / σ ellipses (per-tile representative grid points)
    all_uv_hat, all_uv_pred, all_sx, all_sy = [], [], [], []
    for t in frm_resp['tiles']:
        if t.get('skipped'): continue
        uv_hat = np.array(t['uv_hat'])
        mu     = np.array(t['mu_orig'])
        uv_pred = uv_hat + mu
        all_uv_hat.append(uv_hat); all_uv_pred.append(uv_pred)
        all_sx.append(t['sigma_x_orig']); all_sy.append(t['sigma_y_orig'])
    if all_uv_hat:
        uv_hat = np.concatenate(all_uv_hat, 0)
        uv_pred = np.concatenate(all_uv_pred, 0)
        sx = np.concatenate(all_sx, 0)
        sy = np.concatenate(all_sy, 0)
        ax.scatter(uv_hat[:, 0], uv_hat[:, 1], marker='x', c='red',
                    s=8, lw=0.6, label=f'HAT current calib ({len(uv_hat)})')
        ax.scatter(uv_pred[:, 0], uv_pred[:, 1], marker='o', c='cyan',
                    s=6, edgecolors='none', label='PRED (HAT + μ)')
        for i in range(len(uv_hat)):
            ax.plot([uv_hat[i, 0], uv_pred[i, 0]],
                    [uv_hat[i, 1], uv_pred[i, 1]],
                    color='yellow', lw=0.35, alpha=0.6)
        for i in range(0, len(uv_hat), 3):
            ax.add_patch(Ellipse((uv_pred[i, 0], uv_pred[i, 1]),
                                  width=2*sx[i], height=2*sy[i],
                                  fill=False, ec='cyan', lw=0.3, alpha=0.5))

    title = (f'{args.seq.name} · frame {fid} · '
             f'{sum(1 for t in frm_resp["tiles"] if not t.get("skipped"))}/40 tiles · '
             f'{frm_resp["n_points"]} pts   (seq pool F={seq_resp["n_frames_used"]} '
             f'k={seq_resp["k_dispersion"]})\n'
             f'API-corrected bias:  pitch {np.degrees(d["rot_y"]):+.4f}°±{np.degrees(sd["rot_y"]):.4f}°   '
             f'yaw {np.degrees(d["rot_z"]):+.4f}°±{np.degrees(sd["rot_z"]):.4f}°   '
             f'roll {np.degrees(d["rot_x"]):+.4f}°±{np.degrees(sd["rot_x"]):.4f}°   '
             f't=({d["mp_x"]*1000:+.1f},{d["mp_y"]*1000:+.1f},{d["mp_z"]*1000:+.1f}) mm')
    ax.set_title(title, fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(0, IW); ax.set_ylim(IH, 0); ax.axis('off')

    # Left / center / right zoom regions on top of the whole-frame overlay.
    # 512×512 crops covering horizon-height (where the pitch shift lives).
    zoom_ph = 800
    zoom_v0 = int(IH * 0.20)          # start ~20% down from the top
    zoom_pw = 900
    zoom_u_left   = int(IW * 0.05)
    zoom_u_center = int((IW - zoom_pw) / 2)
    zoom_u_right  = int(IW * 0.95) - zoom_pw
    for u0z, tag in [(zoom_u_left, 'L'),
                     (zoom_u_center, 'C'),
                     (zoom_u_right, 'R')]:
        ax.add_patch(Rectangle((u0z, zoom_v0), zoom_pw, zoom_ph, fill=False,
                                ec='#FFEB3B', lw=2.0, alpha=0.9))
        ax.text(u0z + 8, zoom_v0 + 30, tag, color='#FFEB3B',
                fontsize=16, fontweight='bold')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=110, bbox_inches='tight')
    print(f'[client] wrote {args.out}', flush=True)

    # ── 5) 3-panel zoom row (L / C / R)
    fig2, axes = plt.subplots(1, 3, figsize=(21, 8), tight_layout=True)
    for ax2, u0z, tag in [(axes[0], zoom_u_left,   'L (left)'),
                          (axes[1], zoom_u_center, 'C (center)'),
                          (axes[2], zoom_u_right,  'R (right)')]:
        ax2.imshow(img[zoom_v0:zoom_v0+zoom_ph, u0z:u0z+zoom_pw])
        # Draw points inside this zoom window, offset into local coords
        if all_uv_hat:
            m = ((uv_hat[:, 0] >= u0z) & (uv_hat[:, 0] < u0z + zoom_pw)
                 & (uv_hat[:, 1] >= zoom_v0) & (uv_hat[:, 1] < zoom_v0 + zoom_ph))
            uh = uv_hat[m]  - np.array([u0z, zoom_v0])
            up = uv_pred[m] - np.array([u0z, zoom_v0])
            ax2.scatter(uh[:, 0], uh[:, 1], marker='x', c='red',
                         s=32, lw=1.4, label=f'HAT  N={m.sum()}')
            ax2.scatter(up[:, 0], up[:, 1], marker='o', c='cyan',
                         s=18, edgecolors='navy', lw=0.5, label='PRED')
            for i in range(m.sum()):
                ax2.plot([uh[i, 0], up[i, 0]], [uh[i, 1], up[i, 1]],
                         color='yellow', lw=0.9, alpha=0.75)
            for i in range(0, m.sum(), 3):
                ax2.add_patch(Ellipse((up[i, 0], up[i, 1]),
                                       width=2*sx[m][i], height=2*sy[m][i],
                                       fill=False, ec='cyan', lw=0.5, alpha=0.6))
        ax2.set_title(f'{tag}  ·  u0={u0z} v0={zoom_v0} ({zoom_pw}x{zoom_ph})',
                       fontsize=11)
        ax2.set_xlim(0, zoom_pw); ax2.set_ylim(zoom_ph, 0); ax2.axis('off')
        ax2.legend(loc='upper right', fontsize=9)
    fig2.suptitle(f'zoom L / C / R (yellow boxes in the whole-frame panel)   ·   '
                   f'API pitch {np.degrees(d["rot_y"]):+.4f}°±{np.degrees(sd["rot_y"]):.4f}°',
                   fontsize=11)
    out_zoom = args.out.with_name(args.out.stem + '_zoom_LCR.png')
    plt.savefig(out_zoom, dpi=110, bbox_inches='tight')
    print(f'[client] wrote {out_zoom}', flush=True)

    # ── 6) tile-level detail: 3 tiles (row-major idx) — one per L/C/R column
    # Choose from the 8×5 grid (row-major = row*8 + col). Row 1 (v≈412) covers
    # buildings / traffic-signal poles where the pitch bias is most visible.
    tile_picks = [(1, 1), (1, 3), (1, 5)]  # (row, col)  L / C / R at row 1
    fig3, axes = plt.subplots(1, 3, figsize=(21, 8), tight_layout=True)
    for ax3, (r, c) in zip(axes, tile_picks):
        b = r * 8 + c
        tile = next((t for t in frm_resp['tiles'] if t['tile_idx'] == b), None)
        if tile is None or tile.get('skipped'):
            ax3.set_title(f'tile #{b}  (row {r}, col {c}) — SKIPPED', fontsize=11)
            ax3.axis('off')
            continue
        u0, v0 = tile['u0'], tile['v0']
        crop = img[v0:v0+CS, u0:u0+CS]
        ax3.imshow(crop, extent=[0, CS, CS, 0])
        uv_h = np.array(tile['uv_hat']) - np.array([u0, v0])
        mu   = np.array(tile['mu_orig'])
        uv_p = uv_h + mu
        sx_t = np.array(tile['sigma_x_orig'])
        sy_t = np.array(tile['sigma_y_orig'])
        N = len(uv_h)
        ax3.scatter(uv_h[:, 0], uv_h[:, 1], marker='x', c='red',
                    s=32, lw=1.4, label=f'HAT  N={N}')
        ax3.scatter(uv_p[:, 0], uv_p[:, 1], marker='o', c='cyan',
                    s=18, edgecolors='navy', lw=0.4, label='PRED')
        for i in range(N):
            ax3.plot([uv_h[i, 0], uv_p[i, 0]], [uv_h[i, 1], uv_p[i, 1]],
                     color='yellow', lw=0.9, alpha=0.75)
        for i in range(N):
            ax3.add_patch(Ellipse((uv_p[i, 0], uv_p[i, 1]),
                                   width=2*sx_t[i], height=2*sy_t[i],
                                   fill=False, ec='cyan', lw=0.5, alpha=0.7))
        sig_med = float(np.median(np.sqrt(sx_t * sy_t)))
        mu_norm_med = float(np.median(np.linalg.norm(mu, axis=1)))
        ax3.set_title(f'tile #{b}  (row {r}, col {c})  u0={u0} v0={v0}   '
                       f'|μ|₀.₅ = {mu_norm_med:.2f}px   σ₀.₅ = {sig_med:.2f}px',
                       fontsize=10)
        ax3.set_xlim(0, CS); ax3.set_ylim(CS, 0); ax3.axis('off')
        ax3.legend(loc='upper right', fontsize=9)
    fig3.suptitle('Tile-level detail (3 fused tiles across row 1 — signal / building height). '
                   'σ ellipses show per-point aperture — narrow on structured edges, wide on sky.',
                   fontsize=11)
    out_tiles = args.out.with_name(args.out.stem + '_tiles.png')
    plt.savefig(out_tiles, dpi=110, bbox_inches='tight')
    print(f'[client] wrote {out_tiles}', flush=True)


if __name__ == '__main__':
    main()
