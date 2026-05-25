"""Init / Fit 2-row comparison overlay using loom-style full-frame projection.

Reuses _tss4_calib_overlay.build_K_D_RT to load recalibration.json (the GT-ish
init calib used at training time), then applies the 13-DoF (cxy-locked) GN
fit on top by:
  K_fit  = diag(fx·(1+dfx), fy·(1+dfy), 1) with cxy from K_init
  D_fit  = k_fit (4 coeffs)
  R_fit  = R_rear2cam · R(omega)^{-1}   (drift = ω applied AFTER rear→cam)

For each chosen frame: stack INIT (top) and FIT (bottom) on the SAME canvas
with depth-coloured small/transparent dots.

Usage:
  /home/hfunaya/.pyenv/versions/3.10.4/bin/python \
      scripts/_debug/_tss4_init_fit_overlay.py \
        --json scripts/_debug/_outputs/<...>_13dof_<...>.json \
        --n-frames 4
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Reuse the loom-flow projector
from scripts._debug._tss4_calib_overlay import (
    load_recalib, build_K_D_RT, REAR_X_CUT, VEHICLE_ID, SEQ_ROOT,
)

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


def kb_project_with_p(pts_cam, fx, fy, cx, cy, k, p=None):
    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    valid = Z > 0.5
    out_uv = np.full((pts_cam.shape[0], 2), np.nan)
    if not valid.any():
        return out_uv, Z, valid
    Xv, Yv, Zv = X[valid], Y[valid], Z[valid]
    r = np.sqrt(Xv*Xv + Yv*Yv + 1e-24)
    theta = np.arctan2(r, np.maximum(Zv, 1e-9))
    poly = np.ones_like(theta)
    t2 = theta * theta
    tp = t2.copy()
    for ki in k:
        poly = poly + ki * tp
        tp = tp * t2
    theta_d = theta * poly
    Xr = Xv / np.maximum(r, 1e-9); Yr = Yv / np.maximum(r, 1e-9)
    Xp = theta_d * Xr; Yp = theta_d * Yr
    if p is not None:
        r2p = Xp*Xp + Yp*Yp
        du_t = 2*p[0]*Xp*Yp + p[1]*(r2p + 2*Xp*Xp)
        dv_t = p[0]*(r2p + 2*Yp*Yp) + 2*p[1]*Xp*Yp
        Xp = Xp + du_t
        Yp = Yp + dv_t
    out_uv[valid, 0] = fx * Xp + cx
    out_uv[valid, 1] = fy * Yp + cy
    return out_uv, Z, valid


def color_for_depth(z, zmax=80.0):
    """Vectorised: returns (N, 3) uint8 RGB. near=red→yellow→cyan."""
    t = np.clip(z / zmax, 0.0, 1.0)
    r = np.where(t < 0.5, 255.0, 255.0 * (1 - (t - 0.5) / 0.5))
    g = np.where(t < 0.5, 255.0 * (t / 0.5), 255.0)
    b = np.where(t < 0.5, 0.0, 255.0 * (t - 0.5) / 0.5)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def alpha_blend_dots(arr, uv, z, alpha=0.55, dot_size=0):
    """Tiny ALPHA-blended dots (radius=0 means single pixel).  Vectorised."""
    H, W, _ = arr.shape
    iu = np.round(uv[:, 0]).astype(np.int64)
    iv = np.round(uv[:, 1]).astype(np.int64)
    ok = (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)
    iu = iu[ok]; iv = iv[ok]; z = z[ok]
    if iu.size == 0:
        return
    cols = color_for_depth(z)  # (N, 3)
    arr_f = arr.astype(np.float32)
    for du in range(-dot_size, dot_size + 1):
        for dv in range(-dot_size, dot_size + 1):
            x = iu + du; y = iv + dv
            mm = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            if not mm.any():
                continue
            xx = x[mm]; yy = y[mm]; cc = cols[mm]
            arr_f[yy, xx] = (1 - alpha) * arr_f[yy, xx] + alpha * cc
    arr[...] = np.clip(arr_f, 0, 255).astype(np.uint8)


def stamp_text(arr, text, anchor=(8, 8), font_size=20):
    from PIL import ImageDraw, ImageFont
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', font_size)
    except Exception:
        font = ImageFont.load_default()
    pad = 6
    box_w = int(font.getlength(text)) + pad * 2
    box_h = font_size + 4 + pad * 2
    x0, y0 = anchor
    draw.rectangle((x0, y0, x0 + box_w, y0 + box_h), fill=(0, 0, 0))
    draw.text((x0 + pad, y0 + pad), text, font=font, fill=(255, 255, 255))
    arr[...] = np.asarray(img)


def render_one(seq_dir, frame_idx, calib, fit_params_list, out_path, args):
    """Stack INIT + N FITs (one per fit json) on a single canvas."""
    K, D, R, t, IW, IH = build_K_D_RT(calib)
    fx0 = float(K[0, 0]); fy0 = float(K[1, 1])
    cx0 = float(K[0, 2]); cy0 = float(K[1, 2])

    cam_files = sorted((seq_dir / 'tss4_fcm').glob('*.jpg'))
    lid_files = sorted((seq_dir / 'vls128_rear_axle').glob('*.npz'))
    cam_path = cam_files[frame_idx]
    lid_path = lid_files[frame_idx]
    img = np.asarray(Image.open(cam_path).convert('RGB')).copy()
    d = np.load(lid_path)
    pts_w = np.stack([d['xs'], d['ys'], d['zs']], axis=-1).astype(np.float64)
    pts_w = pts_w[pts_w[:, 0] > REAR_X_CUT]

    # rear-axle → init cam frame
    pts_cam_init = (R @ pts_w.T + t).T

    # ----- INIT projection (KB4 of recalibration.json) -----
    uv_init, z_init, val_init = kb_project_with_p(
        pts_cam_init, fx0, fy0, cx0, cy0, np.asarray(D))
    init_label = (f'INIT  fx={fx0:.1f} fy={fy0:.1f} cxy=({cx0:.1f},{cy0:.1f}) '
                  f'k={np.asarray(D).round(4).tolist()}')

    rows_uv_z_val = [(uv_init, z_init, val_init, init_label)]

    # ----- per-FIT projections -----
    for fi, fit_params in enumerate(fit_params_list, start=1):
        omega = np.asarray(fit_params['omega_deg'], dtype=np.float64)
        R_om = rodrigues(omega)
        dt = (np.asarray(fit_params['delta_t_m'], dtype=np.float64)
              if 'delta_t_m' in fit_params else np.zeros(3))
        pts_cam_fit = pts_cam_init @ R_om.T + dt[None, :]
        fx_f = float(fit_params['fx_fit']); fy_f = float(fit_params['fy_fit'])
        cx_f = float(fit_params['cx_fit']); cy_f = float(fit_params['cy_fit'])
        k_f = np.asarray(fit_params['dist_fit'], dtype=np.float64)
        p_f = (np.asarray(fit_params['tangential_p'], dtype=np.float64)
               if 'tangential_p' in fit_params else None)
        uv_fit, z_fit, val_fit = kb_project_with_p(
            pts_cam_fit, fx_f, fy_f, cx_f, cy_f, k_f, p=p_f)
        p_str = f'  p={p_f.round(4).tolist()}' if p_f is not None else ''
        dt_str = (f'  Δt=[{dt[0]*1000:+.0f},{dt[1]*1000:+.0f},'
                  f'{dt[2]*1000:+.0f}]mm' if np.any(dt != 0) else '')
        tag = f'FIT iter{fi}' if len(fit_params_list) > 1 else 'FIT'
        wrms = fit_params.get('final_wrms_px')
        wrms_str = f'  wrms={wrms:.3f}px' if wrms is not None else ''
        fit_label = (f'{tag}  ω={omega.round(3).tolist()}°{dt_str}  '
                     f'fx*={fx_f/fx0:.4f} fy*={fy_f/fy0:.4f}  '
                     f'cxy=({cx_f:.1f},{cy_f:.1f})  '
                     f'k={k_f.round(4).tolist()}{p_str}{wrms_str}')
        rows_uv_z_val.append((uv_fit, z_fit, val_fit, fit_label))

    # ----- compose canvas -----
    rendered_rows = []
    for (uv, z, val, label) in rows_uv_z_val:
        rgb = img.copy()
        alpha_blend_dots(rgb, uv[val], z[val],
                         alpha=args.alpha, dot_size=args.dot_size)
        if args.crop_size > 0:
            cu0, cv0, cs = args.crop_u0, args.crop_v0, args.crop_size
            cu1, cv1 = min(cu0 + cs, IW), min(cv0 + cs, IH)
            rgb = rgb[cv0:cv1, cu0:cu1].copy()
        stamp_text(rgb, label, font_size=22)
        rendered_rows.append(rgb)

    H_row, W_row = rendered_rows[0].shape[:2]
    gap = 8
    n_rows = len(rendered_rows)
    canvas = np.full((H_row * n_rows + gap * (n_rows - 1), W_row, 3),
                     0, dtype=np.uint8)
    for i, row in enumerate(rendered_rows):
        y0 = i * (H_row + gap)
        canvas[y0:y0 + H_row] = row
    Image.fromarray(canvas).save(out_path, quality=88)
    in_counts = ' '.join(
        f'fit{i}_in={int(np.isfinite(uv[:,0]).sum())}'
        for i, (uv, _, _, _) in enumerate(rows_uv_z_val[1:], start=1))
    init_in = int(np.isfinite(rows_uv_z_val[0][0][:, 0]).sum())
    print(f'[ovr] frame={frame_idx} seq={seq_dir.name} '
          f'init_in={init_in} {in_counts}  -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', type=Path, action='append', default=None,
                    help='GN fit json. Can be passed multiple times to stack '
                         'iter1 / iter2 / ... rows. If 0 jsons are given, '
                         'only the INIT row is rendered (useful for ITER0).')
    ap.add_argument('--n-frames', type=int, default=4)
    ap.add_argument('--out-dir', type=Path,
                    default=REPO / 'scripts/_debug/_outputs/_init_fit_overlay')
    ap.add_argument('--alpha', type=float, default=0.95)
    ap.add_argument('--dot-size', type=int, default=1,
                    help='dot half-width in px (0 = single pixel)')
    ap.add_argument('--seq-idx', type=int, default=0,
                    help='which sibling 248_* sequence to use (sorted)')
    ap.add_argument('--frames', type=str, default=None,
                    help='comma-separated explicit frame indices, overrides --n-frames')
    ap.add_argument('--crop-u0', type=int, default=0)
    ap.add_argument('--crop-v0', type=int, default=0)
    ap.add_argument('--crop-size', type=int, default=0,
                    help='crop window edge length in original-px (0 = no crop)')
    ap.add_argument('--out-suffix', type=str, default='',
                    help='extra suffix added to output filename (e.g. _t19crop)')
    ap.add_argument('--list-seqs', action='store_true',
                    help='list all sibling sequences and exit')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    calib = load_recalib(VEHICLE_ID)
    json_paths = list(args.json) if args.json else []
    fit_params_list = [json.loads(jp.read_text()) for jp in json_paths]
    print(f'[ovr] stacking INIT + {len(fit_params_list)} FIT row(s) '
          f'from: {[jp.name for jp in json_paths]}')

    # pick sibling 248_* sequence
    seq_dirs = sorted([p for p in SEQ_ROOT.iterdir()
                       if p.is_dir() and p.name.startswith(f'sequence={VEHICLE_ID}_')])
    if not seq_dirs:
        raise SystemExit(f'no sequences under {SEQ_ROOT}')
    if args.list_seqs:
        for i, p in enumerate(seq_dirs):
            n = len(list((p / 'tss4_fcm').glob('*.jpg')))
            print(f'  [{i:2d}] {p.name}  ({n} frames)')
        return
    seq = seq_dirs[args.seq_idx]
    cam_files = sorted((seq / 'tss4_fcm').glob('*.jpg'))
    n = len(cam_files)
    if args.frames:
        idxs = [int(x) for x in args.frames.split(',') if x.strip()]
    elif args.n_frames >= n:
        idxs = list(range(n))
    else:
        idxs = np.linspace(0, n - 1, args.n_frames).round().astype(int).tolist()
    print(f'[ovr] seq={seq.name} ({n} frames)  picking idx={idxs}')

    for fi in idxs:
        out = args.out_dir / (
            f'{seq.name}_frame{fi:04d}_init_vs_fit{args.out_suffix}.jpg')
        render_one(seq, fi, calib, fit_params_list, out, args)


if __name__ == '__main__':
    main()
