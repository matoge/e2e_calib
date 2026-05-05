"""V3 full-image cache for Argoverse 2 sensor dataset.

Per (log_id, cam_name, frame_ts) saves an inst .pt with:
  img         : (3, H, W) uint8 — undistorted JPEG-decoded camera frame
  pts_cam     : (N, 3) float32 — LiDAR pts in CAM frame, MOTION-COMPENSATED
                                  (per-point ego-pose interpolation,
                                   per project_dataset_alignment_audit memory)
  uv_full     : (N, 2) float32 — projected pixel UVs (in undistorted image)
  z_cam       : (N,)   float32 — depth in cam frame
  K_full      : (3, 3) float32 — intrinsic of UNDISTORTED camera
  IH, IW      : ints
  cuboids     : list of cuboid dicts (pos, dims, yaw) in CAM frame
  log_id, cam, ts                                    — provenance

Per-point MC required because AV2 LiDAR sweep is rotational (40m radius
~25-30ms per full revolution). Each point has a per-shot timestamp; ego
pose at shot time interpolated from city_SE3_egovehicle.feather, then
shot's world-coords transformed back to cam frame at IMAGE timestamp.
Without this, points lag/lead the image by up to ~30 cm at 30 km/h
(per memory project_dataset_alignment_audit).

Usage:
  python build_av2_v3.py --split train --max-logs 5     # smoke
  python build_av2_v3.py --split train                  # full
  python build_av2_v3.py --split val
"""
import argparse, sys, time, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
import torch
from PIL import Image

# AV2 SDK
from av2.datasets.sensor.sensor_dataloader import SensorDataloader
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.geometry.se3 import SE3
from av2.utils.io import read_feather

# Default paths
AV2_ROOT = Path('/mnt/nvme6t/argoverse2/sensor')
CACHE_OUT = Path('/mnt/nvme6t/e2e_calib_cache/av2_v3_tiled')


CAM_NAMES = [
    'ring_front_center', 'ring_front_left', 'ring_front_right',
    'ring_rear_left', 'ring_rear_right',
    'ring_side_left', 'ring_side_right',
    # stereo cameras dropped — different intrinsic, separate processing
]


def motion_compensate_lidar(sweep_df: pd.DataFrame, ego_pose_df: pd.DataFrame,
                              target_ts_ns: int) -> np.ndarray:
    """Per-point motion compensation: interpolate ego pose at each shot ts,
    transform shot's points world → ego(target_ts).

    sweep_df: lidar sweep parquet with columns x, y, z, intensity, timestamp_ns
    ego_pose_df: city_SE3_egovehicle.feather, columns timestamp_ns, qw,qx,qy,qz, tx,ty,tz
    target_ts_ns: image timestamp to compensate to

    Returns (N, 3) float32 in target ego frame.
    """
    # Per-shot timestamps + xyz
    pts_local = sweep_df[['x', 'y', 'z']].to_numpy().astype(np.float32)  # ego at shot time
    shot_ts = sweep_df['timestamp_ns'].to_numpy()  # per-point ego time

    # Build interpolated ego poses at each shot ts (from city_SE3_egovehicle table)
    # AV2 stores city ← ego, so per-shot world coords = T_city_ego(shot_ts) @ pt_local
    # Then world → target ego = T_city_ego(target_ts)^-1 @ pt_world
    # = T_target_ego ← city @ T_city_ego(shot_ts) @ pt_local

    # For simplicity and accuracy, interpolate at each unique shot_ts
    # (sweep typically has ~10000-100000 shots per revolution)
    eg = ego_pose_df.sort_values('timestamp_ns')
    eg_ts = eg['timestamp_ns'].to_numpy()
    eg_q = eg[['qw', 'qx', 'qy', 'qz']].to_numpy()
    eg_t = eg[['tx', 'ty', 'tz']].to_numpy()

    # Find target pose T_city_target
    idx_t = np.searchsorted(eg_ts, target_ts_ns)
    idx_t = min(max(idx_t, 1), len(eg_ts) - 1)
    # Linear interp between eg_ts[idx_t-1] and eg_ts[idx_t]
    a = (target_ts_ns - eg_ts[idx_t - 1]) / max(1, eg_ts[idx_t] - eg_ts[idx_t - 1])
    target_q = (1 - a) * eg_q[idx_t - 1] + a * eg_q[idx_t]
    target_q /= np.linalg.norm(target_q)
    target_t = (1 - a) * eg_t[idx_t - 1] + a * eg_t[idx_t]
    T_city_target = SE3.from_qwxyz_translation(target_q, target_t)

    # Per-shot interp (vectorized over shots)
    idx_s = np.searchsorted(eg_ts, shot_ts)
    idx_s = np.clip(idx_s, 1, len(eg_ts) - 1)
    a_s = (shot_ts - eg_ts[idx_s - 1]) / np.maximum(1, eg_ts[idx_s] - eg_ts[idx_s - 1])
    shot_q = (1 - a_s)[:, None] * eg_q[idx_s - 1] + a_s[:, None] * eg_q[idx_s]
    shot_q = shot_q / np.linalg.norm(shot_q, axis=1, keepdims=True)
    shot_t = (1 - a_s)[:, None] * eg_t[idx_s - 1] + a_s[:, None] * eg_t[idx_s]

    # Apply T_target ← city @ T_city ← shot per point
    # = T_target_inv @ T_city_shot @ pt_local
    # Build per-shot rotation matrices (qwxyz → R)
    # AV2 SE3 expects qwxyz; we do per-point rotation manually for speed
    qw, qx, qy, qz = shot_q[:, 0], shot_q[:, 1], shot_q[:, 2], shot_q[:, 3]
    R_shot = np.empty((len(shot_ts), 3, 3), dtype=np.float32)
    R_shot[:, 0, 0] = 1 - 2*(qy*qy + qz*qz)
    R_shot[:, 0, 1] = 2*(qx*qy - qz*qw)
    R_shot[:, 0, 2] = 2*(qx*qz + qy*qw)
    R_shot[:, 1, 0] = 2*(qx*qy + qz*qw)
    R_shot[:, 1, 1] = 1 - 2*(qx*qx + qz*qz)
    R_shot[:, 1, 2] = 2*(qy*qz - qx*qw)
    R_shot[:, 2, 0] = 2*(qx*qz - qy*qw)
    R_shot[:, 2, 1] = 2*(qy*qz + qx*qw)
    R_shot[:, 2, 2] = 1 - 2*(qx*qx + qy*qy)

    # pt in city = R_shot @ pt_local + shot_t
    pt_city = (R_shot @ pts_local[:, :, None]).squeeze(-1) + shot_t.astype(np.float32)
    # pt in target ego = T_target_inv @ pt_city
    pt_target = T_city_target.inverse().transform_point_cloud(pt_city)
    return pt_target.astype(np.float32)


def build_one_log(log_dir: Path, cache_out: Path, cam_names=CAM_NAMES,
                   tile_size=512):
    """Process one AV2 log: for each (cam, frame_ts) pair, produce inst .pt."""
    log_id = log_dir.name
    ego_pose_df = read_feather(log_dir / 'city_SE3_egovehicle.feather')
    annot_df = read_feather(log_dir / 'annotations.feather')

    # ... (cuboid loading, calibration parsing, per-cam loop) ...
    # TODO: implement
    raise NotImplementedError(
        f"build_av2_v3.py: per-log processing not yet implemented. "
        f"Pattern follows build_waymo_v3.py — for each (cam, frame_ts):\n"
        f"  1. Load camera frame (jpg from sensors/cameras/<cam>/<ts>.jpg)\n"
        f"  2. Find nearest LiDAR sweep parquet\n"
        f"  3. Apply motion_compensate_lidar() with target_ts = camera ts\n"
        f"  4. Project compensated pts via undistorted PinholeCamera\n"
        f"  5. Convert cuboids ego → cam frame\n"
        f"  6. Tile + save inst .pt at cache_out/inst/<idx>_t<tile>.pt\n"
        f"  Use av2.geometry.camera.pinhole_camera.PinholeCamera for undistort + proj"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    ap.add_argument('--max-logs', type=int, default=None)
    ap.add_argument('--out', default=str(CACHE_OUT))
    args = ap.parse_args()

    split_dir = AV2_ROOT / args.split
    cache_out = Path(args.out)
    cache_out.mkdir(parents=True, exist_ok=True)
    (cache_out / 'inst').mkdir(exist_ok=True)

    logs = sorted(split_dir.iterdir())
    if args.max_logs:
        logs = logs[:args.max_logs]
    print(f'Found {len(logs)} logs in {split_dir}')

    train_files, val_files = [], []
    for log_dir in logs:
        try:
            fnames = build_one_log(log_dir, cache_out)
            # 80/20 split per log (or use predefined av2 train/val split)
            split_at = int(len(fnames) * 0.8)
            train_files.extend(fnames[:split_at])
            val_files.extend(fnames[split_at:])
        except NotImplementedError as e:
            print(f'[stub] {log_dir.name}: {e}')
            break

    meta = {'train': train_files, 'val': val_files, 'cam': 'av2_ring'}
    torch.save(meta, cache_out / 'meta.pt')
    print(f'meta.pt saved: train={len(train_files)} val={len(val_files)}')


if __name__ == '__main__':
    main()
