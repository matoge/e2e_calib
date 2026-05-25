"""For each of N TSS4 frames, stitch the 30 parent tiles onto a 3840×IH
canvas and overlay LiDAR projections under (init) vs (fit) calib.

Init  = K_full + dist  from the cached inst.
Fit   = K_full · (1+dfx, 1+dfy) + (cx+dcx, cy+dcy), dist=k_fit, plus a
        post-rotation R(ω) applied to the cam-frame XYZ.

Saves N pngs (one per frame). Use to visually verify the GN result.
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull, decode_inst_img

_D2R = np.pi / 180.0


def rodrigues(omega_deg):
    th = np.linalg.norm(omega_deg) * _D2R
    if th < 1e-12: return np.eye(3)
    axis = (omega_deg * _D2R) / th
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)


def kb_project(pts_cam, fx, fy, cx, cy, k, p=None):
    """KB radial (any order via len(k)) + optional Brown–Conrady tangential p=[p1,p2]."""
    X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    r = np.sqrt(X*X + Y*Y + 1e-24)
    theta = np.arctan2(r, np.maximum(Z, 1e-9))
    poly = np.ones_like(theta)
    t2 = theta * theta
    tp = t2.copy()
    for ki in k:
        poly = poly + ki * tp
        tp = tp * t2
    theta_d = theta * poly
    Xr = X / np.maximum(r, 1e-9); Yr = Y / np.maximum(r, 1e-9)
    Xp = theta_d * Xr; Yp = theta_d * Yr
    if p is not None:
        r2p = Xp*Xp + Yp*Yp
        Xp = Xp + 2*p[0]*Xp*Yp + p[1]*(r2p + 2*Xp*Xp)
        Yp = Yp + p[0]*(r2p + 2*Yp*Yp) + 2*p[1]*Xp*Yp
    u = fx * Xp + cx
    v = fy * Yp + cy
    return np.stack([u, v], axis=-1), Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', type=Path, required=True,
                    help='11-DoF fit json (has fx/fy/cx/cy fit + omega + k_fit)')
    ap.add_argument('--cache', type=Path,
                    default=Path('/mnt/fsx/tmp/hfunaya/e2e_calib_cache/tss4_v3_tiled'))
    ap.add_argument('--n-frames', type=int, default=4)
    ap.add_argument('--out-dir', type=Path,
                    default=REPO / 'scripts' / '_debug' / '_outputs' / '_proj_overlay')
    ap.add_argument('--max-pts', type=int, default=4000,
                    help='max LiDAR points to overlay (random subsample)')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    j = json.loads(args.json.read_text())
    fx0, fy0 = j['fx_init'], j['fy_init']
    cx0, cy0 = j['cx_init'], j['cy_init']
    fx_fit, fy_fit = j['fx_fit'], j['fy_fit']
    cx_fit, cy_fit = j['cx_fit'], j['cy_fit']
    # Init has 4 KB coeffs (legacy); fit may have any order or include 'tangential_p'.
    if 'dist_init_4' in j:
        k0 = np.array(j['dist_init_4'])
    else:
        k0 = np.array(j['dist_init'])
    k_fit = np.array(j['dist_fit'])
    p_fit = np.array(j['tangential_p']) if 'tangential_p' in j else None
    omega = np.array(j['omega_deg'])
    R_fit = rodrigues(omega)
    p_str = f'  p={p_fit.round(4).tolist()}' if p_fit is not None else ''
    print(f'[ovr] order={len(k_fit)}  omega={omega}°  '
          f'fx*={fx_fit/fx0:.4f} fy*={fy_fit/fy0:.4f}  '
          f'cx{cx_fit-cx0:+.1f} cy{cy_fit-cy0:+.1f}  k_fit={k_fit}{p_str}')

    ds_train = PandaSetCalibDatasetFull(
        cache_dir=args.cache, split='train',
        img_size=128, min_crop_px=128, max_crop_px=512,
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=16, center_band=0.0, preload=False)
    ds_val = PandaSetCalibDatasetFull(
        cache_dir=args.cache, split='val',
        img_size=128, min_crop_px=128, max_crop_px=512,
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=16, center_band=0.0, preload=False)

    # group cached tiles by (scene, frame), keep tile_u0/v0
    groups = defaultdict(list)  # key=(scene, frame) → list of (ds, fname)
    for ds in (ds_train, ds_val):
        for fn in ds.fnames:
            # filename pattern: <scene>_<frame>_t<NN>.pt
            stem = fn.replace('.pt', '')
            parts = stem.rsplit('_t', 1)
            base = parts[0]  # <scene>_<frame>
            groups[base].append((ds, fn))
    keys = sorted(groups.keys())
    # only keep groups with all 30 tiles
    full_keys = [k for k in keys if len(groups[k]) == 30]
    print(f'[ovr] {len(full_keys)}/{len(keys)} keys have all 30 tiles')
    # spread N across full_keys
    if args.n_frames >= len(full_keys):
        chosen = full_keys
    else:
        idxs = np.linspace(0, len(full_keys)-1, args.n_frames).astype(int)
        chosen = [full_keys[i] for i in idxs]

    for ki, key in enumerate(chosen):
        members = groups[key]
        # find tile origins + pts/T_gt/K (same per scene+frame)
        ds0, fn0 = members[0]
        idx0 = ds0.fnames.index(fn0)
        inst0 = ds0._load_inst(idx0)
        K = inst0['K_full'].numpy()
        T_gt = inst0['T_gt'].numpy() if 'T_gt' in inst0 else None
        pts = inst0['pts'].numpy()

        # Each LMDB entry stores only that tile's 512×512 jpg → stitch back.
        u_max = 0; v_max = 0
        tile_imgs = []  # (u0, v0, np.uint8 H,W,3)
        for ds, fn in members:
            inst = ds._load_inst(ds.fnames.index(fn))
            u0 = int(inst['tile_u0']); v0 = int(inst['tile_v0'])
            img = decode_inst_img(inst).permute(1, 2, 0).contiguous().numpy()
            tile_imgs.append((u0, v0, img))
            u_max = max(u_max, u0 + img.shape[1])
            v_max = max(v_max, v0 + img.shape[0])
        canvas = np.zeros((v_max, u_max, 3), dtype=np.uint8)
        for u0, v0, img in tile_imgs:
            canvas[v0:v0+img.shape[0], u0:u0+img.shape[1]] = img

        # project pts (world) → cam → KB
        if T_gt is None:
            print(f'[ovr] skip {key} (no T_gt)'); continue
        homo = np.column_stack([pts, np.ones(len(pts))])
        pts_cam = (T_gt @ homo.T)[:3].T  # (N, 3) in cam frame, metres
        # filter z>0
        front = pts_cam[:, 2] > 0.5
        pts_cam = pts_cam[front]
        if len(pts_cam) > args.max_pts:
            sub = np.random.choice(len(pts_cam), args.max_pts, replace=False)
            pts_cam = pts_cam[sub]

        # init projection (KB4)
        uv_init, _ = kb_project(pts_cam, fx0, fy0, cx0, cy0, k0)
        # fit: R(omega) → KB(k_fit) [+ optional tangential p_fit] with fitted intrinsics
        pts_cam_rot = pts_cam @ R_fit.T
        uv_fit, _ = kb_project(pts_cam_rot, fx_fit, fy_fit, cx_fit, cy_fit,
                               k_fit, p=p_fit)

        # plot: 2-row stack (top = init, bottom = fit)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        IW = u_max; IH = v_max
        aspect = IH / IW
        fig_w = 20
        fig_h = fig_w * aspect * 2 + 1.0
        fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=110)

        in_init = ((uv_init[:,0]>0) & (uv_init[:,0]<IW) &
                   (uv_init[:,1]>0) & (uv_init[:,1]<IH))
        in_fit  = ((uv_fit[:,0]>0)  & (uv_fit[:,0]<IW)  &
                   (uv_fit[:,1]>0)  & (uv_fit[:,1]<IH))

        # color by depth (Z) for both panels — same scale
        Zc = pts_cam[:, 2]
        z_lo = float(np.percentile(Zc, 2))
        z_hi = float(np.percentile(Zc, 98))

        ax = axes[0]
        ax.imshow(canvas, origin='upper')
        sc = ax.scatter(uv_init[in_init, 0], uv_init[in_init, 1],
                        s=2.5, c=Zc[in_init], cmap='turbo',
                        vmin=z_lo, vmax=z_hi, alpha=0.85)
        ax.set_title(f'{key}  INIT (K_full + dist_init)  '
                     f'fx={fx0:.1f} fy={fy0:.1f} cx={cx0:.1f} cy={cy0:.1f}',
                     fontsize=10)
        ax.set_xlim(0, IW); ax.set_ylim(IH, 0)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[1]
        ax.imshow(canvas, origin='upper')
        ax.scatter(uv_fit[in_fit, 0], uv_fit[in_fit, 1],
                   s=2.5, c=Zc[in_fit], cmap='turbo',
                   vmin=z_lo, vmax=z_hi, alpha=0.85)
        ax.set_title(f'FIT (11-DoF)  '
                     f'ω={omega.round(3).tolist()}°  '
                     f'fx*={fx_fit/fx0:.4f} fy*={fy_fit/fy0:.4f}  '
                     f'cx{cx_fit-cx0:+.1f} cy{cy_fit-cy0:+.1f}px  '
                     f'k={np.round(k_fit,3).tolist()}',
                     fontsize=10)
        ax.set_xlim(0, IW); ax.set_ylim(IH, 0)
        ax.set_xticks([]); ax.set_yticks([])

        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(),
                            fraction=0.012, pad=0.01)
        cbar.set_label('depth Z (m)', fontsize=9)

        out_path = args.out_dir / f'{key}_proj_init_vs_fit.png'
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'[ovr] [{ki+1}/{len(chosen)}] wrote {out_path}')


if __name__ == '__main__':
    main()
