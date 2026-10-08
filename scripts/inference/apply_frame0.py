"""Apply S3 (cam6-warmstart) ckpt to a single WovenSequence frame — no docker,
no cache. Reuses the cache builder helpers for calibration + LiDAR projection,
runs the model on 40 tiles the same way training did (share_pert + crop_grid),
fuses the per-tile info matrices via one GN, and prints the recovered 6-DoF.

Usage:
    python scripts/inference/apply_frame0.py \\
        --seq /home/hfunaya/git/loom/backend/assets/woven_sequence/unilab_001/test01/sequence=ip654-lidar0-1432519511400022000-1432519516399877000 \\
        --ckpt experiments/kmwv_s3_ba40_512r256_0901_1344 \\
        --frame-idx 0
"""
from __future__ import annotations
import argparse, json, sys, importlib.util
from pathlib import Path
import numpy as np
import torch
import cv2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts' / 'preprocessing'))

from build_woven_sequence_v3 import (
    _load_setting, _camera_calib_fcm, _load_metadata, _get_poses,
    _camera_delay_ms_for_frame, _pose_at_camera_time,
    _T_lidar_to_cam_at_camera_time, _load_pts_intensity,
)
from scripts.util.projection import project_lidar_into_image
from models.calibnet2 import CalibNet2, D_DIM
from scripts.ba.ba_torch import solve_pinhole_xyz, make_info_from_sigma_rho

DOF = ('omega_x','omega_y','omega_z','tx','ty','tz')
CS = 512      # crop side in original px
S  = 256      # model input side
GRID_N = 16   # queries per axis inside a tile


def _tile_grid(iw, ih, cs):
    nx = max(1, int(np.ceil(iw / cs)))
    ny = max(1, int(np.ceil(ih / cs)))
    sx = max(0, iw - cs) / max(nx - 1, 1)
    sy = max(0, ih - cs) / max(ny - 1, 1)
    cells = []
    for j in range(ny):
        for i in range(nx):
            u0 = int(round(sx * i)); v0 = int(round(sy * j))
            u0 = max(0, min(iw - cs, u0))
            v0 = max(0, min(ih - cs, v0))
            cells.append((u0, v0))
    return cells


def _stratified_grid(uv_loc_S, grid_n, max_per_cell=1):
    """uv_loc_S in [0, S). Return indices, up to max_per_cell per grid cell."""
    cell = np.clip((uv_loc_S / S * grid_n).astype(int), 0, grid_n - 1)
    cid = cell[:, 0] + cell[:, 1] * grid_n
    picks = []
    for c in np.unique(cid):
        idx = np.where(cid == c)[0]
        picks.extend(idx[:max_per_cell].tolist())
    return np.array(picks, dtype=int)


def _build_batch(img_full, uv_full, z_full, int_full, pts_cam_full, K, cells, dev):
    """Build one batch of B tiles ready to feed CalibNet2.

    Mirrors datasets/pandaset_full.py:
      * distorted_uvd: (B, N, 4) [u_local(0..S), v_local(0..S), d/100, intensity]
      * bucket_uvd:    (B, G², K_per_cell, 4) — pre-binned lidar, same units
      * bucket_valid:  (B, G², K_per_cell) bool
      * image:         (B, 3, S, S) float 0..1
      * key_padding_mask: (B, N) bool, True = pad
      * vfp: (B,) scalar = fx_orig * S/cs (per pandaset_full.py line 1802)
    """
    B = len(cells)
    Nq_max = GRID_N * GRID_N
    G2 = GRID_N * GRID_N
    K_per_cell = 8
    imgs = np.zeros((B, 3, S, S), dtype=np.float32)
    dist_uvd = np.zeros((B, Nq_max, 4), dtype=np.float32)
    bucket_uvd = np.zeros((B, G2, K_per_cell, 4), dtype=np.float32)
    bucket_valid = np.zeros((B, G2, K_per_cell), dtype=bool)
    key_padding_mask = np.ones((B, Nq_max), dtype=bool)
    vfp = np.zeros(B, dtype=np.float32)
    per_tile_data = []
    scale = S / CS   # 256 / 512 = 0.5

    for b, (u0, v0) in enumerate(cells):
        # 1. Points visible in this tile
        m = ((uv_full[:, 0] >= u0) & (uv_full[:, 0] < u0 + CS)
             & (uv_full[:, 1] >= v0) & (uv_full[:, 1] < v0 + CS)
             & (z_full > 0.5))
        n_in = int(m.sum())
        if n_in < 8:
            per_tile_data.append(None)
            continue

        # 2. Tile-local uv in model-input px (0..S)
        idx_in_full = np.where(m)[0]
        uv_loc_S = (uv_full[idx_in_full] - np.array([u0, v0], dtype=np.float32)) * scale
        d_norm = z_full[idx_in_full] / 100.0
        inten  = int_full[idx_in_full]

        # 3. Bucket (G², K_per_cell, 4) — same convention as trainer
        cu = np.clip((uv_loc_S[:, 0] / (S / GRID_N)).astype(int), 0, GRID_N - 1)
        cv = np.clip((uv_loc_S[:, 1] / (S / GRID_N)).astype(int), 0, GRID_N - 1)
        cid = cv * GRID_N + cu
        uvd_raw = np.column_stack([uv_loc_S, d_norm, inten]).astype(np.float32)
        # up to K_per_cell per cell (first K, no shuffle for determinism)
        order = np.argsort(cid, kind='stable')
        cid_s = cid[order]; uvd_s = uvd_raw[order]
        counts = np.bincount(cid_s, minlength=G2)
        starts = np.zeros(G2 + 1, dtype=np.int64); starts[1:] = counts.cumsum()
        intra = np.arange(len(cid_s)) - starts[cid_s]
        keep = intra < K_per_cell
        bucket_uvd[b, cid_s[keep], intra[keep]] = uvd_s[keep]
        bucket_valid[b, cid_s[keep], intra[keep]] = True

        # 4. Query = stratified grid sample (1 pt per cell)
        picks = _stratified_grid(uv_loc_S, GRID_N, max_per_cell=1)
        Nq = len(picks)
        dist_uvd[b, :Nq, 0] = uv_loc_S[picks, 0]
        dist_uvd[b, :Nq, 1] = uv_loc_S[picks, 1]
        dist_uvd[b, :Nq, 2] = d_norm[picks]
        dist_uvd[b, :Nq, 3] = inten[picks]
        key_padding_mask[b, :Nq] = False

        # 5. Image tile: crop → resize to S
        crop = img_full[v0:v0+CS, u0:u0+CS]
        crop_s = cv2.resize(crop, (S, S), interpolation=cv2.INTER_LINEAR)
        imgs[b] = crop_s.transpose(2, 0, 1).astype(np.float32) / 255.0

        # 6. vfp scalar
        vfp[b] = float(K[0, 0]) * S / CS

        per_tile_data.append(dict(
            u0=u0, v0=v0, idx=idx_in_full[picks],
            pts_cam=pts_cam_full[idx_in_full[picks]],
            uv_orig=uv_full[idx_in_full[picks]],
        ))

    return (torch.from_numpy(imgs).to(dev),
            torch.from_numpy(dist_uvd).to(dev),
            torch.from_numpy(vfp).to(dev),
            torch.from_numpy(bucket_uvd).to(dev),
            torch.from_numpy(bucket_valid).to(dev),
            torch.from_numpy(key_padding_mask).to(dev),
            per_tile_data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True, type=Path)
    ap.add_argument('--ckpt', required=True, type=Path,
                    help='experiment dir with best_model.pt + config.py')
    ap.add_argument('--frame-idx', type=int, default=0)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--viz', type=Path, default=None,
                    help='save whole-frame overlay PNG here')
    ap.add_argument('--hood-mask', type=Path, default=None,
                    help='PNG mask (nonzero = ego hood/bonnet). Points landing '
                         'in that region are dropped before tiling.')
    ap.add_argument('--hood-mask-root', type=Path,
                    default=Path('/home/hfunaya/git/loom/backend/assets/hood_masks'),
                    help='dir with car<car_id>.png; auto-resolves --hood-mask '
                         'from metadata.car_id when --hood-mask not given.')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = args.ckpt / 'best_model.pt'
    cfg_path = args.ckpt / 'config.py'
    spec = importlib.util.spec_from_file_location('_cfg', cfg_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cfg = mod.CFG
    print(f'ckpt={ck}  img_size={cfg["img_size"]} grid_n={cfg["grid_n"]} '
          f'n_iter={cfg["n_iter"]} use_info_head={cfg["use_info_head"]}', flush=True)

    # ------ raw sequence ------
    setting = _load_setting(args.seq)
    K, dist, R_cv, t_cv, IW, IH, delay_default = _camera_calib_fcm(setting)
    metadata = _load_metadata(args.seq)
    frame_ids, poses = _get_poses(metadata)
    fid = frame_ids[args.frame_idx]
    delay_ms = _camera_delay_ms_for_frame(metadata, fid, delay_default)
    print(f'seq={args.seq.name}  IH×IW={IH}×{IW}  frame={fid}  delay={delay_ms}ms', flush=True)

    pose_curr = poses[fid]
    pose_cam = _pose_at_camera_time(poses, frame_ids, args.frame_idx, delay_ms)
    T_cl = _T_lidar_to_cam_at_camera_time(pose_curr, pose_cam, R_cv, t_cv)

    pts_flu, intensity = _load_pts_intensity(args.seq, fid)
    pts_xyzi = np.column_stack([pts_flu.astype(np.float32), intensity.astype(np.float32)])
    _, pts_cam, uv_full, z_full, int_full = project_lidar_into_image(
        pts_xyzi, K, T_cl, IW, IH, is_fisheye=True, dist=dist, z_min=0.5)
    print(f'visible pts: {len(pts_cam)} / {len(pts_flu)}', flush=True)

    # ------ Hood mask: drop LiDAR pts that project onto the ego hood ------
    mask_path = args.hood_mask
    if mask_path is None:
        car_id = metadata.get('car_id')
        if car_id:
            cand = args.hood_mask_root / f'car{car_id}.png'
            if cand.exists():
                mask_path = cand
                print(f'auto-resolved hood mask (car_id={car_id}): {mask_path}',
                      flush=True)
    hood_img = None                          # for viz
    uv_dropped_by_hood = np.zeros((0, 2), dtype=np.float32)
    if mask_path is not None and Path(mask_path).exists():
        hood_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert hood_img is not None, f'failed to read {mask_path}'
        assert hood_img.shape == (IH, IW), (hood_img.shape, IH, IW,
            'hood mask must match image resolution')
        uu = uv_full[:, 0].round().astype(int).clip(0, IW - 1)
        vv = uv_full[:, 1].round().astype(int).clip(0, IH - 1)
        keep = hood_img[vv, uu] == 0
        uv_dropped_by_hood = uv_full[~keep].copy()
        n_before = len(pts_cam)
        pts_cam  = pts_cam[keep]
        uv_full  = uv_full[keep]
        z_full   = z_full[keep]
        int_full = int_full[keep]
        print(f'hood mask: dropped {n_before - len(pts_cam)} / {n_before} '
              f'pts ({100*(n_before-len(pts_cam))/max(n_before,1):.1f}%) '
              f'in hood region', flush=True)
    else:
        print('hood mask: (none applied)', flush=True)

    img_bgr = cv2.imread(str(args.seq / 'tss4_fcm' / f'{fid}.jpg'))
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    assert img.shape[:2] == (IH, IW), (img.shape, IH, IW)

    # ------ model ------
    model = CalibNet2(img_size=cfg['img_size'],
                      frustum_grid_n=cfg.get('grid_n', 16),
                      n_iter=cfg['n_iter'],
                      use_info_head=cfg['use_info_head'],
                      point_mlp_fourier_n_freq=cfg.get('point_mlp_fourier_n_freq', 0),
                      in_channels=3).to(dev).eval()
    sd = torch.load(ck, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model' in sd: sd = sd['model']
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f'loaded: missing={len(miss)} unexpected={len(unexp)}', flush=True)

    # ------ tiles ------
    cells = _tile_grid(IW, IH, CS)
    print(f'tile grid: {len(cells)} = {int(np.ceil(IW/CS))}x{int(np.ceil(IH/CS))}', flush=True)
    imgs, q_in, vfp, buck, bvalid, kpm, per_tile = _build_batch(
        img, uv_full, z_full, int_full, pts_cam, K, cells, dev)
    print(f'batch tensors: imgs={tuple(imgs.shape)}  q_in={tuple(q_in.shape)}', flush=True)

    # ------ forward ------
    with torch.no_grad():
        out = model(imgs, q_in, vfp=vfp,
                    bucket_uvd=buck, bucket_valid=bvalid,
                    key_padding_mask=kpm, mode='calib')
    per_pt, W_head = (out if isinstance(out, tuple) else (out, None))
    print(f'per_pt shape: {tuple(per_pt.shape)}  W_head: {None if W_head is None else tuple(W_head.shape)}',
          flush=True)

    # ------ Concatenate per-tile → outer GN ------
    # μ (px in model 256) → back to original px = × (CS/S) = × 2 (tile-local shift)
    # Since we're solving one δ for the whole frame using pts_cam in ORIGINAL cam frame,
    # target uv = uv_orig + μ_orig; solver expects duv defined as target − project(pts,K).
    all_pts, all_duv, all_W, all_valid = [], [], [], []
    for b, td in enumerate(per_tile):
        if td is None: continue
        Nq = len(td['idx'])
        mu_model  = per_pt[b, :Nq, :2].cpu().numpy()             # px in 256 model input
        mu_orig   = mu_model * (CS / S)                          # px in original camera frame
        # duv = (uv_orig + mu_orig) − uv_orig = mu_orig, since target = uv_orig + mu_orig
        # and project(pts_cam, K) = uv_orig by construction (KB4 projection we did earlier).
        # BUT: outer GN uses PINHOLE. To stay consistent with training, we linearise
        # around the CROP CENTER using tile-local K.
        # For a first pass we use the FULL K and full pts_cam. Model was trained tile-local
        # so residuals should still be small.
        all_pts.append(td['pts_cam'])
        all_duv.append(mu_orig)
        if W_head is not None:
            W_b = W_head[b, :Nq].cpu().numpy()                    # (Nq, 2, 2)
        else:
            sx = torch.exp(per_pt[b, :Nq, 2]).cpu().numpy() * (CS/S)
            sy = torch.exp(per_pt[b, :Nq, 3]).cpu().numpy() * (CS/S)
            # 2×2 diagonal precision
            W_b = np.zeros((Nq, 2, 2), dtype=np.float32)
            W_b[:, 0, 0] = 1.0 / (sx ** 2)
            W_b[:, 1, 1] = 1.0 / (sy ** 2)
        all_W.append(W_b)
        all_valid.append(np.ones(Nq, dtype=bool))

    pts_cam_all = np.concatenate(all_pts, 0)
    duv_all     = np.concatenate(all_duv, 0)
    W_all       = np.concatenate(all_W,   0)
    valid_all   = np.concatenate(all_valid, 0)
    print(f'total points: {len(pts_cam_all)}', flush=True)

    # ---- outer GN in fp64 ----
    pts_t = torch.from_numpy(pts_cam_all).double().unsqueeze(0).to(dev)      # (1,N,3)
    duv_t = torch.from_numpy(duv_all).double().unsqueeze(0).to(dev)          # (1,N,2)
    W_t   = torch.from_numpy(W_all).double().unsqueeze(0).to(dev)            # (1,N,2,2)
    K_t   = torch.from_numpy(K.astype(np.float64)).unsqueeze(0).to(dev)      # (1,3,3)
    v_t   = torch.from_numpy(valid_all).unsqueeze(0).to(dev)
    delta, H = solve_pinhole_xyz(pts_t, duv_t, W_t, K_t, DOF,
                                  valid=v_t, n_iter=10, damping=1e-3,
                                  robust='huber', huber_k=1.5)
    delta = delta[0].cpu().numpy()                # (6,)
    H     = H[0].cpu().numpy()                     # (6,6)
    cov   = np.linalg.inv(H)
    sd_axis = np.sqrt(np.diag(cov))

    # ------ Report ------
    print()
    print('=== S3 cam6-warmstart · applied to real sequence ===')
    print(f'  frame:            {fid}')
    print(f'  # tiles fused:    {sum(1 for t in per_tile if t is not None)}')
    print(f'  # points fused:   {len(pts_cam_all)}')
    print()
    # Solver returns angles in DEGREES, translations in METRES
    # (see ba_torch.pinhole_jacobian docstring: "angles in DEGREES, translations in METERS").
    print(f'  ω_x (pitch):  {delta[0]:+.4f}°   ± {sd_axis[0]:.4f}°')
    print(f'  ω_y (yaw)  :  {delta[1]:+.4f}°   ± {sd_axis[1]:.4f}°')
    print(f'  ω_z (roll) :  {delta[2]:+.4f}°   ± {sd_axis[2]:.4f}°')
    print(f'  t_x        :  {delta[3]*1000:+.2f} mm  ± {sd_axis[3]*1000:.2f} mm')
    print(f'  t_y        :  {delta[4]*1000:+.2f} mm  ± {sd_axis[4]*1000:.2f} mm')
    print(f'  t_z        :  {delta[5]*1000:+.2f} mm  ± {sd_axis[5]*1000:.2f} mm')

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, delta=delta, H=H, cov=cov,
                 pts_cam=pts_cam_all, duv=duv_all)
        print(f'\nwrote {args.out}')

    # ------ Visualization ------
    if args.viz:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, Rectangle

        # Per-point σ (px, orig scale) from per_pt log_sx/log_sy
        # (only for tiles that fused successfully)
        all_uv_orig, all_mu_orig, all_sigma_px, all_pts_cam = [], [], [], []
        for b, td in enumerate(per_tile):
            if td is None: continue
            Nq = len(td['idx'])
            mu_model = per_pt[b, :Nq, :2].cpu().numpy()
            log_sx = per_pt[b, :Nq, 2].cpu().numpy()
            log_sy = per_pt[b, :Nq, 3].cpu().numpy()
            sx_orig = np.exp(log_sx) * (CS / S)
            sy_orig = np.exp(log_sy) * (CS / S)
            all_uv_orig.append(td['uv_orig'])
            all_mu_orig.append(mu_model * (CS / S))
            all_sigma_px.append(np.column_stack([sx_orig, sy_orig]))
            all_pts_cam.append(td['pts_cam'])
        uv_orig = np.concatenate(all_uv_orig, 0)
        mu_orig = np.concatenate(all_mu_orig, 0)
        sig_px  = np.concatenate(all_sigma_px, 0)
        pts_c   = np.concatenate(all_pts_cam, 0)
        depth   = pts_c[:, 2]

        # PRED position = HAT + μ  (in original px)
        uv_pred = uv_orig + mu_orig

        fig, ax = plt.subplots(figsize=(19, 11), tight_layout=True)
        ax.imshow(img)

        # ---- Hood mask overlay (translucent magenta fill + polygon outline) ----
        if hood_img is not None:
            hood_rgba = np.zeros((IH, IW, 4), dtype=np.float32)
            hood_rgba[..., 0] = 1.0      # magenta = R
            hood_rgba[..., 2] = 0.8      # + a bit of blue
            hood_rgba[..., 3] = (hood_img > 0).astype(np.float32) * 0.25
            ax.imshow(hood_rgba)
            # polygon vertices, if present
            poly_json = Path(str(mask_path).replace('.png', '.polygon.json'))
            if poly_json.exists():
                pj = json.load(open(poly_json))
                pts = np.array(pj.get('polygon', []), dtype=float)
                if len(pts) >= 3:
                    pts = np.vstack([pts, pts[0:1]])
                    ax.plot(pts[:, 0], pts[:, 1], '-', color='magenta',
                            lw=1.8, alpha=0.9, label='hood mask polygon')

        # Tile grid overlay (only fused tiles in green, skipped in red)
        for b, (u0, v0) in enumerate(cells):
            color = '#4CAF50' if per_tile[b] is not None else '#E53935'
            ax.add_patch(Rectangle((u0, v0), CS, CS, fill=False,
                                    ec=color, lw=0.7, alpha=0.35))

        # DROPPED points (LiDAR that projected into the hood mask, ignored)
        if len(uv_dropped_by_hood) > 0:
            ax.scatter(uv_dropped_by_hood[:, 0], uv_dropped_by_hood[:, 1],
                        marker='x', c='magenta', s=6, lw=0.4, alpha=0.6,
                        label=f'DROPPED by hood mask ({len(uv_dropped_by_hood)})')

        # HAT (input projection, red ×)
        ax.scatter(uv_orig[:, 0], uv_orig[:, 1], marker='x', c='red',
                    s=8, lw=0.6, label=f'HAT current calib ({len(uv_orig)})')
        # PRED (model-corrected, cyan)
        ax.scatter(uv_pred[:, 0], uv_pred[:, 1], marker='o', c='cyan',
                    s=6, edgecolors='none', label='PRED (HAT + μ)')
        # Arrows HAT → PRED
        for i in range(len(uv_orig)):
            ax.plot([uv_orig[i, 0], uv_pred[i, 0]],
                    [uv_orig[i, 1], uv_pred[i, 1]],
                    color='yellow', lw=0.35, alpha=0.6)
        # σ ellipse
        for i in range(0, len(uv_orig), 3):
            ax.add_patch(Ellipse((uv_pred[i, 0], uv_pred[i, 1]),
                                  width=2*sig_px[i, 0], height=2*sig_px[i, 1],
                                  fill=False, ec='cyan', lw=0.3, alpha=0.5))

        title = (f'S3 cam6 · {args.seq.name} · frame {fid} · {len(pts_cam_all)} pts across {sum(1 for t in per_tile if t is not None)}/40 tiles\n'
                 f'inferred bias:  pitch {delta[0]:+.3f}°±{sd_axis[0]:.3f}°   '
                 f'yaw {delta[1]:+.3f}°±{sd_axis[1]:.3f}°   '
                 f'roll {delta[2]:+.3f}°±{sd_axis[2]:.3f}°   '
                 f't=({delta[3]*1000:+.1f},{delta[4]*1000:+.1f},{delta[5]*1000:+.1f}) mm')
        ax.set_title(title, fontsize=11)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(0, IW); ax.set_ylim(IH, 0); ax.axis('off')
        args.viz.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.viz, dpi=110, bbox_inches='tight')
        print(f'\nwrote viz: {args.viz}')


if __name__ == '__main__':
    main()
