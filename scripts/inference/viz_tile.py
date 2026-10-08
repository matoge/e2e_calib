"""Zoom into a specific tile of frame N and show the model's per-point μ / σ.

    python scripts/inference/viz_tile.py --seq <...> --ckpt <exp> \\
        --frame-idx 0 --tile-idx 11 --out docs/assets/.../tile11.png

Tile numbering is row-major over the 8×5 crop-grid (row 0 = top).
"""
from __future__ import annotations
import argparse, sys, importlib.util
from pathlib import Path
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts' / 'preprocessing'))
sys.path.insert(0, str(REPO / 'scripts' / 'inference'))

from build_woven_sequence_v3 import (
    _load_setting, _camera_calib_fcm, _load_metadata, _get_poses,
    _camera_delay_ms_for_frame, _pose_at_camera_time,
    _T_lidar_to_cam_at_camera_time, _load_pts_intensity,
)
from scripts.util.projection import project_lidar_into_image
from models.calibnet2 import CalibNet2
from apply_frame0 import CS, S, GRID_N, _tile_grid, _build_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True, type=Path)
    ap.add_argument('--ckpt', required=True, type=Path)
    ap.add_argument('--frame-idx', type=int, default=0)
    ap.add_argument('--tile-idxs', type=str, default='11',
                    help='comma-separated tile indices (row-major, 8×5 grid)')
    ap.add_argument('--hood-mask-root', type=Path,
                    default=Path('/home/hfunaya/git/loom/backend/assets/hood_masks'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = args.ckpt / 'best_model.pt'
    spec = importlib.util.spec_from_file_location('_cfg', args.ckpt / 'config.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cfg = mod.CFG

    setting = _load_setting(args.seq)
    K, dist, R_cv, t_cv, IW, IH, delay_default = _camera_calib_fcm(setting)
    metadata = _load_metadata(args.seq)
    frame_ids, poses = _get_poses(metadata)
    fid = frame_ids[args.frame_idx]
    delay_ms = _camera_delay_ms_for_frame(metadata, fid, delay_default)
    pose_curr = poses[fid]
    pose_cam = _pose_at_camera_time(poses, frame_ids, args.frame_idx, delay_ms)
    T_cl = _T_lidar_to_cam_at_camera_time(pose_curr, pose_cam, R_cv, t_cv)

    pts_flu, intensity = _load_pts_intensity(args.seq, fid)
    pts_xyzi = np.column_stack([pts_flu.astype(np.float32), intensity.astype(np.float32)])
    _, pts_cam, uv_full, z_full, int_full = project_lidar_into_image(
        pts_xyzi, K, T_cl, IW, IH, is_fisheye=True, dist=dist, z_min=0.5)

    # Hood mask
    car_id = metadata.get('car_id')
    hood_img = None
    if car_id:
        cand = args.hood_mask_root / f'car{car_id}.png'
        if cand.exists():
            hood_img = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
            uu = uv_full[:, 0].round().astype(int).clip(0, IW - 1)
            vv = uv_full[:, 1].round().astype(int).clip(0, IH - 1)
            keep = hood_img[vv, uu] == 0
            pts_cam, uv_full, z_full, int_full = (pts_cam[keep], uv_full[keep],
                                                   z_full[keep], int_full[keep])

    img_bgr = cv2.imread(str(args.seq / 'tss4_fcm' / f'{fid}.jpg'))
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    model = CalibNet2(img_size=cfg['img_size'],
                      frustum_grid_n=cfg.get('grid_n', 16),
                      n_iter=cfg['n_iter'],
                      use_info_head=cfg['use_info_head'],
                      point_mlp_fourier_n_freq=cfg.get('point_mlp_fourier_n_freq', 0),
                      in_channels=3).to(dev).eval()
    sd = torch.load(ck, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model' in sd: sd = sd['model']
    model.load_state_dict(sd, strict=False)

    cells = _tile_grid(IW, IH, CS)
    imgs, q_in, vfp, buck, bvalid, kpm, per_tile = _build_batch(
        img, uv_full, z_full, int_full, pts_cam, K, cells, dev)

    with torch.no_grad():
        out = model(imgs, q_in, vfp=vfp, bucket_uvd=buck, bucket_valid=bvalid,
                    key_padding_mask=kpm, mode='calib')
    per_pt, W_head = (out if isinstance(out, tuple) else (out, None))

    tile_ids = [int(x) for x in args.tile_idxs.split(',')]
    nx = int(np.ceil(IW / CS)); ny = int(np.ceil(IH / CS))
    n_panels = len(tile_ids)
    fig, axes = plt.subplots(2, n_panels, figsize=(6 * n_panels, 11),
                              tight_layout=True,
                              gridspec_kw={'height_ratios': [1, 4]})
    if n_panels == 1:
        axes = axes.reshape(2, 1)

    # Row 0: whole-frame thumbnail with selected tile boxed
    for ci, tid in enumerate(tile_ids):
        ax = axes[0, ci]
        ax.imshow(img)
        for b, (u0, v0) in enumerate(cells):
            col = '#FFEB3B' if b == tid else '#4CAF50' if per_tile[b] is not None else '#E53935'
            lw = 2.5 if b == tid else 0.5
            ax.add_patch(Rectangle((u0, v0), CS, CS, fill=False, ec=col,
                                    lw=lw, alpha=0.8 if b == tid else 0.3))
        r, c = tid // nx, tid % nx
        ax.set_title(f'tile #{tid}  (row {r}, col {c})', fontsize=10)
        ax.axis('off')

    # Row 1: zoom into each tile
    for ci, tid in enumerate(tile_ids):
        ax = axes[1, ci]
        u0, v0 = cells[tid]
        crop = img[v0:v0+CS, u0:u0+CS]
        ax.imshow(crop, extent=[0, CS, CS, 0])

        td = per_tile[tid]
        if td is None:
            ax.set_title(f'tile #{tid}: SKIPPED (fewer than 8 pts)', fontsize=10)
            ax.axis('off')
            continue

        Nq = len(td['idx'])
        mu_model = per_pt[tid, :Nq, :2].cpu().numpy()          # px in model-256
        log_sx = per_pt[tid, :Nq, 2].cpu().numpy()
        log_sy = per_pt[tid, :Nq, 3].cpu().numpy()
        # Everything below is in TILE-LOCAL 512-px coords (extent = [0,CS,CS,0])
        mu_tile = mu_model * (CS / S)
        sx_tile = np.exp(log_sx) * (CS / S)
        sy_tile = np.exp(log_sy) * (CS / S)
        uv_hat = td['uv_orig'] - np.array([u0, v0])
        uv_pred = uv_hat + mu_tile

        ax.scatter(uv_hat[:, 0], uv_hat[:, 1], marker='x', c='red', s=25,
                    lw=1.0, label=f'HAT (current calib) N={Nq}')
        ax.scatter(uv_pred[:, 0], uv_pred[:, 1], marker='o', c='cyan', s=20,
                    edgecolors='navy', lw=0.4, label='PRED (HAT + μ)')
        for i in range(Nq):
            ax.plot([uv_hat[i, 0], uv_pred[i, 0]],
                    [uv_hat[i, 1], uv_pred[i, 1]],
                    color='yellow', lw=0.9, alpha=0.7)
        for i in range(Nq):
            ax.add_patch(Ellipse((uv_pred[i, 0], uv_pred[i, 1]),
                                  width=2*sx_tile[i], height=2*sy_tile[i],
                                  fill=False, ec='cyan', lw=0.5, alpha=0.7))

        sig_med = float(np.median(np.sqrt(sx_tile * sy_tile)))
        mu_norm = float(np.median(np.linalg.norm(mu_tile, axis=1)))
        ax.set_title(f'tile #{tid} · u0={u0} v0={v0}  N={Nq}\n'
                      f'|μ| median = {mu_norm:.2f}px   σ median = {sig_med:.2f}px',
                      fontsize=10)
        ax.set_xlim(0, CS); ax.set_ylim(CS, 0)
        ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(f'Frame {args.frame_idx} ({fid})  ·  ckpt={args.ckpt.name}',
                  fontsize=11, y=1.005)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=140, bbox_inches='tight')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
