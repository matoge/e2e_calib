"""Convert Waymo Open Dataset v2 segments to PandaSet-style on-disk layout.

Why: the cross-frame residual net trains from `PandaSetCrossFrameDataset`
which expects PandaSet's layout (camera/<cam>/{fi:02d}.jpg + poses.json +
intrinsics.json + lidar/{fi:02d}.pkl, all in world frame, OpenCV camera
convention). Converting once means Waymo data mixes into the same loader
without touching training code.

Output per segment (cap at MAX_FRAMES frames from the start):

  <out>/<seg>/
    camera/{front,front_left,front_right,side_left,side_right}_camera/
      intrinsics.json                       # {fx,fy,cx,cy}
      poses.json                            # [{heading:{x,y,z,w}, position:{x,y,z}}, ...]
      00.jpg, 01.jpg, ...                   # full-res JPG from camera_image parquet
    lidar/
      00.pkl, 01.pkl, ...                   # pandas DataFrame with x,y,z (world frame, meters)

Coordinate conventions:
  Waymo cam:  X=forward (depth), Y=left, Z=up
  PandaSet / OpenCV: X=right, Y=down, Z=forward (depth)
  R_opencv_from_waymo = [[0,-1,0],[0,0,-1],[1,0,0]]
  We store world→opencv_cam by composing world_from_waymocam @ waymocam_from_opencv.

Idempotent: re-running skips jpg/lidar files that already exist on disk; only
missing pieces get regenerated. poses.json / intrinsics.json are always
rewritten (cheap).
"""
import argparse
import io
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation

WAYMO_ROOT = Path('/mnt/nvme6t/waymo/training')
OUT_ROOT   = Path('/mnt/nvme6t/waymo_ps')

# Waymo v2 camera name enum → PandaSet-style directory name.
CAMERAS = {
    1: 'front_camera',
    2: 'front_left_camera',
    3: 'front_right_camera',
    4: 'side_left_camera',
    5: 'side_right_camera',
}
TOP_LIDAR  = 1   # 1=TOP, 2..5=side

# Rotation from OpenCV camera frame to Waymo camera frame.
# Waymo: X_w=forward, Y_w=left, Z_w=up
# OpenCV: X_o=right, Y_o=down, Z_o=forward
#   X_w =  Z_o
#   Y_w = -X_o
#   Z_w = -Y_o
R_WAYMOCAM_FROM_OPENCVCAM = np.array([
    [0,  0,  1],
    [-1, 0,  0],
    [0, -1,  0],
], dtype=np.float64)


def _decode_range_image(ri_vals, ri_shape, incl, az_correction=0.0):
    """Waymo range image → (N,3) XYZ in lidar sensor frame."""
    ri = np.array(ri_vals, dtype=np.float32).reshape(ri_shape)
    H, W = ri.shape[:2]
    r = ri[:, :, 0]
    valid = r > 0
    if valid.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32)
    cols = np.arange(W, dtype=np.float64)
    azimuth = np.pi * (1.0 - 2.0 * (cols + 0.5) / W) - az_correction
    cos_e = np.cos(incl[:, None]).astype(np.float32)
    sin_e = np.sin(incl[:, None]).astype(np.float32)
    cos_a = np.cos(azimuth[None, :]).astype(np.float32)
    sin_a = np.sin(azimuth[None, :]).astype(np.float32)
    xs = r * cos_e * cos_a
    ys = r * cos_e * sin_a
    zs = r * sin_e
    return np.stack([xs[valid], ys[valid], zs[valid]], axis=1)


def _load_cam_intr_extr(cam_cal_row):
    fu = float(cam_cal_row['[CameraCalibrationComponent].intrinsic.f_u'])
    fv = float(cam_cal_row['[CameraCalibrationComponent].intrinsic.f_v'])
    cu = float(cam_cal_row['[CameraCalibrationComponent].intrinsic.c_u'])
    cv = float(cam_cal_row['[CameraCalibrationComponent].intrinsic.c_v'])
    # extrinsic is T_vehicle_from_waymocam (row-major 4×4)
    T_veh_from_wcam = np.array(
        cam_cal_row['[CameraCalibrationComponent].extrinsic.transform'],
        dtype=np.float64).reshape(4, 4)
    return fu, fv, cu, cv, T_veh_from_wcam


def _load_lidar_cal(lid_cal_row):
    T_veh_from_lid = np.array(
        lid_cal_row['[LiDARCalibrationComponent].extrinsic.transform'],
        dtype=np.float64).reshape(4, 4)
    incl = np.array(lid_cal_row['[LiDARCalibrationComponent].beam_inclination.values'],
                    dtype=np.float64)
    if incl is None or len(incl) == 0:
        lo = float(lid_cal_row['[LiDARCalibrationComponent].beam_inclination.min'])
        hi = float(lid_cal_row['[LiDARCalibrationComponent].beam_inclination.max'])
        incl = np.linspace(lo, hi, 64)
    incl = incl[::-1].copy()
    az_correction = math.atan2(T_veh_from_lid[1, 0], T_veh_from_lid[0, 0])
    return T_veh_from_lid, incl, az_correction


def _T_world_from_opencvcam(T_world_from_waymocam):
    """Compose out the Waymo→OpenCV cam rotation so stored pose is for OpenCV cam."""
    M = np.eye(4)
    M[:3, :3] = R_WAYMOCAM_FROM_OPENCVCAM
    return T_world_from_waymocam @ M


def _mat_to_quat_pos(T):
    """Store as {heading: xyzw quaternion, position: xyz} (PandaSet format)."""
    q = Rotation.from_matrix(T[:3, :3]).as_quat()   # scipy returns xyzw
    t = T[:3, 3]
    return dict(heading=dict(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])),
                position=dict(x=float(t[0]), y=float(t[1]), z=float(t[2])))


def convert_segment(seg_name: str, out_root: Path, max_frames: int = 80,
                    jpg_quality: int = 90) -> tuple[str, str]:
    seg_out = out_root / seg_name

    # Fully-done shortcut: every camera has intrinsics + poses + last frame,
    # and last lidar pkl exists. Enables idempotent re-runs.
    def _fully_done() -> bool:
        for cname in CAMERAS.values():
            cd = seg_out / 'camera' / cname
            if not (cd / 'intrinsics.json').exists(): return False
            if not (cd / 'poses.json').exists():      return False
            if not (cd / f'{max_frames-1:02d}.jpg').exists(): return False
        return (seg_out / 'lidar' / f'{max_frames-1:02d}.pkl').exists()
    if _fully_done():
        return seg_name, 'skipped (all cams done)'

    # Load parquets
    pose_df = pd.read_parquet(WAYMO_ROOT / 'vehicle_pose' / f'{seg_name}.parquet')
    img_df  = pd.read_parquet(WAYMO_ROOT / 'camera_image' / f'{seg_name}.parquet')
    lid_df  = pd.read_parquet(WAYMO_ROOT / 'lidar'        / f'{seg_name}.parquet')
    ccal_df = pd.read_parquet(WAYMO_ROOT / 'camera_calibration' / f'{seg_name}.parquet')
    lcal_df = pd.read_parquet(WAYMO_ROOT / 'lidar_calibration'  / f'{seg_name}.parquet')

    lcal_row = lcal_df[lcal_df['key.laser_name'] == TOP_LIDAR].iloc[0]
    T_veh_from_lid, incl, az_corr = _load_lidar_cal(lcal_row)

    # Per-camera intrinsics + per-timestamp image index.
    cam_intr:  dict[int, tuple] = {}
    cam_imgs:  dict[int, pd.DataFrame] = {}
    for cid in CAMERAS:
        row = ccal_df[ccal_df['key.camera_name'] == cid]
        if row.empty:
            return seg_name, f'skipped (camera {cid} missing calibration)'
        cam_intr[cid] = _load_cam_intr_extr(row.iloc[0])
        sub = img_df[img_df['key.camera_name'] == cid]
        if sub.empty:
            return seg_name, f'skipped (camera {cid} missing images)'
        cam_imgs[cid] = sub.set_index('key.frame_timestamp_micros').sort_index()

    top_lid = lid_df[lid_df['key.laser_name'] == TOP_LIDAR].set_index('key.frame_timestamp_micros').sort_index()
    pose_ix = pose_df.set_index('key.frame_timestamp_micros').sort_index()

    # Common timestamps across all 5 cameras + lidar + pose. Waymo hardware-syncs
    # these, so in practice the intersection equals every source's frame count.
    ts_sets = [set(cam_imgs[cid].index) for cid in CAMERAS]
    ts_sets.append(set(top_lid.index))
    ts_sets.append(set(pose_ix.index))
    ts_common = sorted(set.intersection(*ts_sets))[:max_frames]
    if len(ts_common) < 10:
        return seg_name, f'skipped (only {len(ts_common)} common frames)'

    seg_out.mkdir(parents=True, exist_ok=True)
    (seg_out / 'lidar').mkdir(parents=True, exist_ok=True)

    # Prepare camera dirs + write intrinsics (rewrite is cheap).
    for cid, cname in CAMERAS.items():
        cd = seg_out / 'camera' / cname
        cd.mkdir(parents=True, exist_ok=True)
        fu, fv, cu, cv, _ = cam_intr[cid]
        (cd / 'intrinsics.json').write_text(
            json.dumps(dict(fx=fu, fy=fv, cx=cu, cy=cv), indent=2))

    cam_poses: dict[int, list] = {cid: [] for cid in CAMERAS}

    for fi, ts in enumerate(ts_common):
        # ── lidar (shared, decode once per frame) ─────────────────────────
        lid_path = seg_out / f'lidar/{fi:02d}.pkl'
        if not lid_path.exists():
            lid_row = top_lid.loc[ts]
            pose_row = pose_ix.loc[ts]
            pts_lid = _decode_range_image(
                lid_row['[LiDARComponent].range_image_return1.values'],
                lid_row['[LiDARComponent].range_image_return1.shape'],
                incl, az_corr)
            if len(pts_lid) == 0:
                pd.DataFrame(dict(x=[], y=[], z=[])).to_pickle(lid_path)
            else:
                h = np.hstack([pts_lid, np.ones((len(pts_lid), 1), dtype=pts_lid.dtype)])
                pts_veh = (T_veh_from_lid @ h.T).T[:, :3]
                T_world_from_veh = np.array(
                    pose_row['[VehiclePoseComponent].world_from_vehicle.transform'],
                    dtype=np.float64).reshape(4, 4)
                h2 = np.hstack([pts_veh, np.ones((len(pts_veh), 1), dtype=pts_veh.dtype)])
                pts_world = (T_world_from_veh @ h2.T).T[:, :3].astype(np.float32)
                pd.DataFrame(
                    dict(x=pts_world[:, 0], y=pts_world[:, 1], z=pts_world[:, 2])
                ).to_pickle(lid_path)

        # ── per-camera image + pose ───────────────────────────────────────
        for cid, cname in CAMERAS.items():
            img_row = cam_imgs[cid].loc[ts]
            jpg_path = seg_out / f'camera/{cname}/{fi:02d}.jpg'
            if not jpg_path.exists():
                img_bytes = img_row['[CameraImageComponent].image']
                Image.open(io.BytesIO(img_bytes)).convert('RGB').save(
                    jpg_path, quality=jpg_quality)
            # Pose per camera (world_from_waymocam → world_from_opencvcam).
            T_world_from_wcam = np.array(
                img_row['[CameraImageComponent].pose.transform'],
                dtype=np.float64).reshape(4, 4)
            T_world_from_cam = _T_world_from_opencvcam(T_world_from_wcam)
            cam_poses[cid].append(_mat_to_quat_pos(T_world_from_cam))

    # ── write per-camera poses.json ──
    for cid, cname in CAMERAS.items():
        (seg_out / f'camera/{cname}/poses.json').write_text(
            json.dumps(cam_poses[cid], indent=2))

    return seg_name, f'ok ({len(ts_common)} frames × {len(CAMERAS)} cams)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-segs', type=int, default=40,
                    help='cap number of segments to convert')
    ap.add_argument('--max-frames', type=int, default=80,
                    help='cap frames per segment (PandaSet is 80)')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out-root', default=str(OUT_ROOT))
    ap.add_argument('--offset', type=int, default=0,
                    help='skip this many segments (for chunked conversion)')
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_segs = sorted(p.stem for p in (WAYMO_ROOT / 'camera_image').glob('*.parquet')
                      if not p.stem.startswith('_'))
    segs = all_segs[args.offset:args.offset + args.max_segs]
    print(f'converting {len(segs)} segments → {out_root}', flush=True)

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(convert_segment, s, out_root, args.max_frames): s for s in segs}
        for fut in as_completed(futures):
            name, msg = fut.result()
            done += 1
            print(f'[{done}/{len(segs)}] {name}: {msg}', flush=True)
    print('done.')


if __name__ == '__main__':
    main()
