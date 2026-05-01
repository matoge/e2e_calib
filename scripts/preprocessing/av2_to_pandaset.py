"""Convert Argoverse 2 Sensor logs → PandaSet on-disk layout for cross-frame training.

Per log:
  /mnt/nvme6t/av2_ps/<log_id>/
    camera/<cam_name>/           # one subdir per AV2 camera (9 total)
      intrinsics.json            # {fx, fy, cx, cy, k1, k2, k3}  -- radial distortion captured
      poses.json                 # [{heading, position}]  cam→world at each sampled frame
      00.jpg -> <symlink>        # to the original AV2 image (saves disk)
      01.jpg, ...
    lidar/
      00.pkl, 01.pkl, ...        # DataFrame(x,y,z) in world frame, compensated to lidar ts

Frame schedule: use every K-th LiDAR timestamp (default K=2 → ~5 Hz, ~80 frames/log).
For each sampled LiDAR timestamp, each camera picks its nearest image by timestamp.

Distortion: AV2 radial (k1, k2, k3) is written to intrinsics.json. `ring_front_center`
is portrait (1550×2048); the other 8 cams are landscape (2048×1550). Both are
stored as-is — the dataset side (pandaset_pair._proj_cam) reads k1/k2/k3 and
applies the OpenCV radial model at projection time, so no image rewriting.

Coordinate conventions:
  AV2 cameras are OpenCV (X=right, Y=down, Z=forward). Same as PandaSet / Waymo-converted,
  so `_project` + `_proj_cam` in pandaset_pair work without any axis swap.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp


AV2_CAMERAS = [
    'ring_front_center',
    'ring_front_left',  'ring_front_right',
    'ring_side_left',   'ring_side_right',
    'ring_rear_left',   'ring_rear_right',
    'stereo_front_left', 'stereo_front_right',
]


def _se3_from_row(row):
    """Row with qw, qx, qy, qz, tx_m, ty_m, tz_m → 4x4 homogeneous SE3 (sensor→ego or ego→world)."""
    q = [row['qx'], row['qy'], row['qz'], row['qw']]  # scipy order
    R = Rotation.from_quat(q).as_matrix()
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [row['tx_m'], row['ty_m'], row['tz_m']]
    return M


def _slerp_ego_pose(city_df, ts_ns):
    """Interpolate ego→world SE3 at arbitrary timestamp using SLERP (rotation)
    and linear interp (translation). Clamps to [min, max] timestamp."""
    ts_arr = city_df['timestamp_ns'].values
    # clamp
    if ts_ns <= ts_arr[0]:
        return _se3_from_row(city_df.iloc[0])
    if ts_ns >= ts_arr[-1]:
        return _se3_from_row(city_df.iloc[-1])

    idx = np.searchsorted(ts_arr, ts_ns)
    t0, t1 = ts_arr[idx - 1], ts_arr[idx]
    alpha = (ts_ns - t0) / (t1 - t0)
    r0 = city_df.iloc[idx - 1]
    r1 = city_df.iloc[idx]
    # quat SLERP
    q0 = np.array([r0['qx'], r0['qy'], r0['qz'], r0['qw']])
    q1 = np.array([r1['qx'], r1['qy'], r1['qz'], r1['qw']])
    slerp = Slerp([0, 1], Rotation.from_quat(np.stack([q0, q1])))
    R_interp = slerp([alpha]).as_matrix()[0]
    t_interp = np.array([
        (1 - alpha) * r0['tx_m'] + alpha * r1['tx_m'],
        (1 - alpha) * r0['ty_m'] + alpha * r1['ty_m'],
        (1 - alpha) * r0['tz_m'] + alpha * r1['tz_m'],
    ])
    M = np.eye(4)
    M[:3, :3] = R_interp
    M[:3, 3] = t_interp
    return M


def _mat_to_heading_position(M):
    """4x4 SE3 → {heading: {x,y,z,w}, position: {x,y,z}} as used by PandaSet poses.json.
    Matches `_quat_pos_to_mat` in pandaset_pair: input represents cam→world."""
    R = M[:3, :3]; t = M[:3, 3]
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return {
        'heading':  {'x': float(q[0]), 'y': float(q[1]), 'z': float(q[2]), 'w': float(q[3])},
        'position': {'x': float(t[0]), 'y': float(t[1]), 'z': float(t[2])},
    }


def convert_log(log_dir: Path, out_root: Path, every_k: int = 2, symlink: bool = True):
    """Convert one AV2 log to PandaSet layout.

    every_k=2 → take every 2nd LiDAR frame (~5 Hz, ~80 frames per 20 s log).
    """
    log_id = log_dir.name
    out_dir = out_root / log_id
    have_poses  = (out_dir / 'camera' / AV2_CAMERAS[-1] / 'poses.json').exists()
    have_mc     = (out_dir / 'lidar' / '.mc_v2').exists()
    if have_poses and have_mc:
        return f'[skip] {log_id}'  # already done with per-point MC

    # --- log-level calib and ego-pose tables ---
    intr = pd.read_feather(log_dir / 'calibration/intrinsics.feather')
    extr = pd.read_feather(log_dir / 'calibration/egovehicle_SE3_sensor.feather')
    city = pd.read_feather(log_dir / 'city_SE3_egovehicle.feather').sort_values('timestamp_ns').reset_index(drop=True)

    # lookup helpers
    extr_by_name = {r['sensor_name']: _se3_from_row(r) for _, r in extr.iterrows()}
    cam_intr = {r['sensor_name']: r for _, r in intr.iterrows()}

    lidar_files = sorted((log_dir / 'sensors/lidar').iterdir())
    sampled_lidar = lidar_files[::every_k]
    N_FRAMES = len(sampled_lidar)

    # --- lidar (world frame, shared across all cams) ---
    # AV2 `.feather` lidar sweeps are in EGO-VEHICLE frame at each point's own
    # capture time (NOT motion-compensated to a single timestamp). Each point
    # carries `offset_ns` indicating its capture time relative to the sweep
    # start (file stem). Sweep span = ~103 ms across the rotating beam.
    #
    # Per-point motion compensation: for each pt, look up T_e2w(t_capture) from
    # the high-rate `city_SE3_egovehicle` table (~5 ms cadence) via SLERP +
    # linear translation, then transform that single point to world. Without
    # this, side cameras show 30-100 px misalignment at typical urban speeds.
    #
    # `lidar/.mc_v2` sentinel marks the directory as built with this code path;
    # if a log was previously preprocessed without per-point MC, the sentinel
    # is missing and we rebuild every pkl.
    lidar_out = out_dir / 'lidar'
    lidar_out.mkdir(parents=True, exist_ok=True)
    mc_sentinel = lidar_out / '.mc_v2'

    # vectorized ego-pose interpolators (reused across all frames in this log)
    city_ts_ns = city['timestamp_ns'].values
    city_quats = np.stack([city['qx'].values, city['qy'].values,
                            city['qz'].values, city['qw'].values], axis=1)
    city_slerp = Slerp(city_ts_ns, Rotation.from_quat(city_quats))
    city_tx, city_ty, city_tz = city['tx_m'].values, city['ty_m'].values, city['tz_m'].values

    needs_rebuild = not mc_sentinel.exists()
    lidar_ts_ns = []
    for fi, lf in enumerate(sampled_lidar):
        ts_ns = int(lf.stem)
        lidar_ts_ns.append(ts_ns)
        out_pkl = lidar_out / f'{fi:02d}.pkl'
        if out_pkl.exists() and not needs_rebuild:
            continue
        df = pd.read_feather(lf)
        pts_e     = df[['x', 'y', 'z']].values.astype(np.float64)
        offset_ns = df['offset_ns'].values.astype(np.int64)
        pts_ts_ns = ts_ns + offset_ns
        pts_ts_clipped = np.clip(pts_ts_ns, city_ts_ns[0], city_ts_ns[-1])
        Rs = city_slerp(pts_ts_clipped).as_matrix()              # (N, 3, 3)
        tx = np.interp(pts_ts_clipped, city_ts_ns, city_tx)
        ty = np.interp(pts_ts_clipped, city_ts_ns, city_ty)
        tz = np.interp(pts_ts_clipped, city_ts_ns, city_tz)
        pts_w = np.einsum('nij,nj->ni', Rs, pts_e) + np.stack([tx, ty, tz], axis=1)
        pd.DataFrame({'x': pts_w[:, 0].astype(np.float32),
                      'y': pts_w[:, 1].astype(np.float32),
                      'z': pts_w[:, 2].astype(np.float32)}).to_pickle(out_pkl)
    mc_sentinel.touch()

    # --- per-camera: intrinsics, poses, image symlinks ---
    for cam in AV2_CAMERAS:
        if cam not in extr_by_name or cam not in cam_intr:
            continue
        cam_dir = log_dir / 'sensors/cameras' / cam
        if not cam_dir.is_dir():
            continue

        out_cam = out_dir / 'camera' / cam
        out_cam.mkdir(parents=True, exist_ok=True)

        # intrinsics with distortion
        row = cam_intr[cam]
        (out_cam / 'intrinsics.json').write_text(json.dumps({
            'fx': float(row['fx_px']), 'fy': float(row['fy_px']),
            'cx': float(row['cx_px']), 'cy': float(row['cy_px']),
            'k1': float(row['k1']),   'k2': float(row['k2']),   'k3': float(row['k3']),
            'width': int(row['width_px']), 'height': int(row['height_px']),
        }, indent=2))

        # extrinsic cam→ego
        T_c2e = extr_by_name[cam]

        # enumerate camera images, build ts array
        img_files = sorted(cam_dir.glob('*.jpg'))
        img_ts = np.array([int(p.stem) for p in img_files], dtype=np.int64)

        poses = []
        for fi, target_ts in enumerate(lidar_ts_ns):
            # nearest camera image to this lidar timestamp
            idx = int(np.argmin(np.abs(img_ts - target_ts)))
            src_img = img_files[idx]
            cam_ts = int(src_img.stem)

            # ego→world at camera timestamp (better than lidar's for this cam's pose)
            T_e2w = _slerp_ego_pose(city, cam_ts)
            T_c2w = T_e2w @ T_c2e

            poses.append(_mat_to_heading_position(T_c2w))

            # link/copy image
            dst_img = out_cam / f'{fi:02d}.jpg'
            if dst_img.exists() or dst_img.is_symlink():
                continue
            if symlink:
                dst_img.symlink_to(src_img)
            else:
                os.link(src_img, dst_img)

        (out_cam / 'poses.json').write_text(json.dumps(poses, indent=2))

    return f'[ok]   {log_id}  ({N_FRAMES} frames × {len(AV2_CAMERAS)} cams)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/mnt/nvme6t/argoverse2/sensor/train')
    ap.add_argument('--dst', default='/mnt/nvme6t/av2_ps')
    ap.add_argument('--limit', type=int, default=0, help='0 = all logs')
    ap.add_argument('--every-k', type=int, default=2, help='take every k-th lidar frame')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    src = Path(args.src); dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    logs = sorted([p for p in src.iterdir() if p.is_dir()])
    if args.limit:
        logs = logs[:args.limit]
    print(f'converting {len(logs)} logs → {dst} (every_k={args.every_k}, workers={args.workers})')

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(convert_log, log, dst, args.every_k): log for log in logs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                print(f'  [{i}/{len(logs)}] {fut.result()}', flush=True)
            except Exception as e:
                log = futs[fut]
                print(f'  [{i}/{len(logs)}] [FAIL] {log.name}: {e!r}', flush=True)


if __name__ == '__main__':
    main()
