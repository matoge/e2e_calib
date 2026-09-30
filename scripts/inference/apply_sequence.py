"""Apply S3 ckpt to a stride'd subset of frames in a sequence, report per-frame
and joint 6-DoF. All frames observe the same rig δ, so per-frame (H_f, δ_f)
plus inverse-variance pool give the closed-form joint solve.

Usage:
    python scripts/inference/apply_sequence.py \\
        --seq <sequence dir> --ckpt experiments/kmwv_s3_ba40_512r256_0901_1344 \\
        --stride 5 --max-frames 10 \\
        --viz-dir docs/assets/2026-09-30_unilab001_apply/seq_stride5
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
from models.calibnet2 import CalibNet2
from scripts.ba.ba_torch import solve_pinhole_xyz

# reuse helpers from apply_frame0.py
sys.path.insert(0, str(REPO / 'scripts' / 'inference'))
from apply_frame0 import CS, S, GRID_N, _tile_grid, _stratified_grid, _build_batch, DOF


def process_frame(model, dev, args, seq, metadata, frame_ids, poses, K, dist,
                  R_cv, t_cv, IW, IH, delay_default, hood_img, frame_idx):
    fid = frame_ids[frame_idx]
    delay_ms = _camera_delay_ms_for_frame(metadata, fid, delay_default)
    pose_curr = poses[fid]
    pose_cam = _pose_at_camera_time(poses, frame_ids, frame_idx, delay_ms)
    T_cl = _T_lidar_to_cam_at_camera_time(pose_curr, pose_cam, R_cv, t_cv)

    pts_flu, intensity = _load_pts_intensity(seq, fid)
    if len(pts_flu) == 0:
        return None
    pts_xyzi = np.column_stack([pts_flu.astype(np.float32), intensity.astype(np.float32)])
    _, pts_cam, uv_full, z_full, int_full = project_lidar_into_image(
        pts_xyzi, K, T_cl, IW, IH, is_fisheye=True, dist=dist, z_min=0.5)
    n_vis = len(pts_cam)

    n_dropped = 0
    if hood_img is not None:
        uu = uv_full[:, 0].round().astype(int).clip(0, IW - 1)
        vv = uv_full[:, 1].round().astype(int).clip(0, IH - 1)
        keep = hood_img[vv, uu] == 0
        n_dropped = int((~keep).sum())
        pts_cam, uv_full, z_full, int_full = (pts_cam[keep], uv_full[keep],
                                                z_full[keep], int_full[keep])

    img_bgr = cv2.imread(str(seq / 'tss4_fcm' / f'{fid}.jpg'))
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    cells = _tile_grid(IW, IH, CS)
    imgs, q_in, vfp, buck, bvalid, kpm, per_tile = _build_batch(
        img, uv_full, z_full, int_full, pts_cam, K, cells, dev)

    with torch.no_grad():
        out = model(imgs, q_in, vfp=vfp, bucket_uvd=buck, bucket_valid=bvalid,
                    key_padding_mask=kpm, mode='calib')
    per_pt, W_head = (out if isinstance(out, tuple) else (out, None))

    all_pts, all_duv, all_W, all_valid = [], [], [], []
    for b, td in enumerate(per_tile):
        if td is None: continue
        Nq = len(td['idx'])
        mu_orig = per_pt[b, :Nq, :2].cpu().numpy() * (CS / S)
        all_pts.append(td['pts_cam']); all_duv.append(mu_orig)
        if W_head is not None:
            all_W.append(W_head[b, :Nq].cpu().numpy())
        else:
            sx = np.exp(per_pt[b, :Nq, 2].cpu().numpy()) * (CS/S)
            sy = np.exp(per_pt[b, :Nq, 3].cpu().numpy()) * (CS/S)
            W = np.zeros((Nq, 2, 2), dtype=np.float32)
            W[:, 0, 0] = 1.0/(sx**2); W[:, 1, 1] = 1.0/(sy**2)
            all_W.append(W)
        all_valid.append(np.ones(Nq, dtype=bool))
    if not all_pts:
        return None
    pts_cam_all = np.concatenate(all_pts, 0)
    duv_all     = np.concatenate(all_duv, 0)
    W_all       = np.concatenate(all_W,   0)
    valid_all   = np.concatenate(all_valid, 0)

    pts_t = torch.from_numpy(pts_cam_all).double().unsqueeze(0).to(dev)
    duv_t = torch.from_numpy(duv_all).double().unsqueeze(0).to(dev)
    W_t   = torch.from_numpy(W_all).double().unsqueeze(0).to(dev)
    K_t   = torch.from_numpy(K.astype(np.float64)).unsqueeze(0).to(dev)
    v_t   = torch.from_numpy(valid_all).unsqueeze(0).to(dev)
    delta, H = solve_pinhole_xyz(pts_t, duv_t, W_t, K_t, DOF,
                                  valid=v_t, n_iter=10, damping=1e-3)
    delta = delta[0].cpu().numpy(); H = H[0].cpu().numpy()

    return dict(fid=fid, frame_idx=frame_idx, delta=delta, H=H,
                n_vis=n_vis, n_dropped=n_dropped,
                n_pts_fused=len(pts_cam_all),
                n_tiles_fused=sum(1 for t in per_tile if t is not None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True, type=Path)
    ap.add_argument('--ckpt', required=True, type=Path)
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--max-frames', type=int, default=10)
    ap.add_argument('--hood-mask-root', type=Path,
                    default=Path('/home/hfunaya/git/loom/backend/assets/hood_masks'))
    ap.add_argument('--out', type=Path, default=None)
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

    hood_img = None
    car_id = metadata.get('car_id')
    if car_id:
        cand = args.hood_mask_root / f'car{car_id}.png'
        if cand.exists():
            hood_img = cv2.imread(str(cand), cv2.IMREAD_GRAYSCALE)
            print(f'hood mask: {cand.name}', flush=True)

    model = CalibNet2(img_size=cfg['img_size'],
                      frustum_grid_n=cfg.get('grid_n', 16),
                      n_iter=cfg['n_iter'],
                      use_info_head=cfg['use_info_head'],
                      point_mlp_fourier_n_freq=cfg.get('point_mlp_fourier_n_freq', 0),
                      in_channels=3).to(dev).eval()
    sd = torch.load(ck, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model' in sd: sd = sd['model']
    model.load_state_dict(sd, strict=False)
    print(f'ckpt loaded: {ck.name}', flush=True)

    idxs = list(range(0, len(frame_ids), args.stride))[:args.max_frames]
    print(f'processing {len(idxs)} frames (stride {args.stride}): {idxs}', flush=True)

    rows = []
    for i in idxs:
        r = process_frame(model, dev, args, args.seq, metadata, frame_ids, poses,
                          K, dist, R_cv, t_cv, IW, IH, delay_default, hood_img, i)
        if r is None: continue
        rows.append(r)
        d = r['delta']
        cov = np.linalg.inv(r['H'])
        sd_ax = np.sqrt(np.diag(cov))
        print(f'  frame {r["frame_idx"]:03d} ({r["fid"]}): '
              f'pitch {d[0]:+.4f}°±{sd_ax[0]:.4f}  '
              f'yaw {d[1]:+.4f}°±{sd_ax[1]:.4f}  '
              f'roll {d[2]:+.4f}°±{sd_ax[2]:.4f}  '
              f't=({d[3]*1000:+.1f},{d[4]*1000:+.1f},{d[5]*1000:+.1f})mm  '
              f'[{r["n_tiles_fused"]}/40 tiles, {r["n_pts_fused"]} pts]',
              flush=True)

    if len(rows) == 0:
        print('no frames processed'); return

    # ---- Joint inv-var pool ----
    H_sum = np.zeros((6, 6), dtype=np.float64)
    b_sum = np.zeros(6, dtype=np.float64)
    for r in rows:
        H_sum += r['H']
        b_sum += r['H'] @ r['delta']
    delta_j = np.linalg.solve(H_sum, b_sum)
    cov_j = np.linalg.inv(H_sum)
    sd_j = np.sqrt(np.diag(cov_j))

    # ---- χ² of joint residuals vs per-frame estimates → over-dispersion k ----
    chi2 = []
    for r in rows:
        e = r['delta'] - delta_j
        chi2.append(e @ r['H'] @ e / 6.0)
    k_est = float(np.median(chi2))

    F = len(rows)
    print()
    print(f'=== joint solve, F={F} frames ===')
    axis_names = ['ω_x pitch', 'ω_y yaw  ', 'ω_z roll ', 't_x      ', 't_y      ', 't_z      ']
    for i, name in enumerate(axis_names):
        unit = 'mm' if i >= 3 else '°'
        val = delta_j[i] * (1000 if i >= 3 else 1)
        sig = sd_j[i]  * (1000 if i >= 3 else 1)
        sig_k = sig * np.sqrt(k_est)
        print(f'  {name}: {val:+.4f} {unit}   sd_H {sig:.4f}   sd·√k {sig_k:.4f}  (k={k_est:.2f})')

    # Per-frame consistency check
    print()
    print(f'=== per-frame consistency (χ² of per-frame δ vs joint δ) ===')
    print(f'  median χ²/6 (over F={F}) = {k_est:.2f}   → factor {np.sqrt(k_est):.2f} on 1σ')
    print(f'  1σ sd on rot/trans (H⁻¹ raw / after k-correction):')
    for i, name in enumerate(['pitch°', 'yaw°  ', 'roll° ']):
        print(f'    {name}: {sd_j[i]:.5f}  /  {sd_j[i]*np.sqrt(k_est):.5f}')
    for i, name in enumerate(['tx mm ', 'ty mm ', 'tz mm ']):
        print(f'    {name}: {sd_j[i+3]*1000:.3f} /  {sd_j[i+3]*1000*np.sqrt(k_est):.3f}')

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out,
                 delta_joint=delta_j, H_joint=H_sum, k=k_est,
                 frame_idxs=np.array([r['frame_idx'] for r in rows]),
                 deltas=np.array([r['delta'] for r in rows]),
                 Hs=np.array([r['H'] for r in rows]))
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
