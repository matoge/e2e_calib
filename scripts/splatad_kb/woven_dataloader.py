"""Minimal dataloader for woven_sequence calibration scenes.

Produces tensors directly compatible with gsplat 1.5.3 `rasterization`
(camera_model="fisheye", radial_coeffs=k1..k4). No nerfstudio Cameras
indirection.

Per frame i:
  cam_intrinsics[i]   : (3,3) K (KB4 -- fx, fy, cx, cy)
  cam_dist[i]         : (4,)  k1..k4 (KB4 radial)
  cam_pose[i]         : (4,4) world-from-cam   (PS-cam axes: x=right, y=down, z=fwd)
  cam_t[i]            : float64 unix-sec
  img_path[i]         : Path to JPEG (original fisheye, full-resolution)
  mask_path[i]        : Path to dynamic mask (PNG, white=keep, black=skip), or None

Lidar:
  lid_pose[i]         : (4,4) world-from-lidar   (lidar = rear_axle, axes match rear_axle)
  lid_t[i]            : float64 unix-sec (sweep mid)
  pc[i]               : (N, 4 or 5) tensor (x,y,z,intensity[,t_offset_sec])
                        in lidar/rear_axle frame

World frame definition: PandaSet doesn't apply here. We use the rear_axle
frame OF FRAME 0 as the static world. cam/lidar poses are then per-frame
rigid motions of rear_axle relative to that frame-0 anchor (interpolated
from POSLV).

The math is identical to scripts/_debug/_crop_pinhole_proj_check.py for
frame 0 (where world == frame-0 rear_axle and the per-frame motion is
identity).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as _R

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'webui_kb_fit'))
import lidar_deskew as DK


RECALIB_DEFAULT = Path(os.environ.get(
    'WOVEN_RECALIB_JSON',
    '/home/hfunaya/git/loom/backend/assets/woven_sequence/'
    'llinking_26/recalibration.json'))

# vehicle (woven) axes -> opencv (PS-cam) axes
R_TO_RDF = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)


@dataclass
class Frame:
    """Single frame data."""
    idx: int
    img_path: Path
    mask_path: Optional[Path]
    K: np.ndarray            # (3,3) KB4 K
    dist: np.ndarray         # (4,) k1..k4
    width: int
    height: int
    cam_t2w: np.ndarray      # (4,4) world-from-PScam
    cam_t: float             # unix-sec
    lid_t2w: np.ndarray      # (4,4) world-from-lidar (= world-from-rearAxle)
    lid_t: float             # unix-sec, sweep mid
    pc: np.ndarray           # (N, 4 or 5) lidar-frame xyzir[t]


def _load_calib(recalib: Path, vehicle: str) -> dict:
    d = json.loads(recalib.read_text())[vehicle]
    fcm = d['fcm']
    poslv = d['poslv']
    fx = fy = float(fcm['kb']['focal_length'])
    cx, cy = fcm['cc']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([fcm['kb'][f'k{i}'] for i in (1, 2, 3, 4)],
                    dtype=np.float64)
    W, H = int(fcm['resolution'][0]), int(fcm['resolution'][1])

    mp_fcm = np.asarray(fcm['mp'], dtype=np.float64)
    roll, pitch, yaw = fcm['rot']
    R_fcm = _R.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
    mp_poslv = np.asarray(poslv['mp'], dtype=np.float64)
    roll_p, pitch_p, yaw_p = poslv['rot']
    R_poslv = _R.from_euler('zyx', [yaw_p, pitch_p, roll_p]).as_matrix()

    # static rear_axle->PS-cam
    R_w2c_static = R_TO_RDF @ R_fcm.T @ R_poslv
    t_w2c_static = R_TO_RDF @ R_fcm.T @ mp_poslv - R_TO_RDF @ mp_fcm
    T_pscam_from_rear_static = np.eye(4)
    T_pscam_from_rear_static[:3, :3] = R_w2c_static
    T_pscam_from_rear_static[:3, 3] = t_w2c_static
    T_rear_from_pscam_static = np.linalg.inv(T_pscam_from_rear_static)

    return dict(
        K=K, dist=dist, W=W, H=H,
        T_rear_from_pscam_static=T_rear_from_pscam_static,
    )


def load_woven_sequence(
    seq_dir: Path,
    vehicle: str = '248',
    recalib_json: Path = RECALIB_DEFAULT,
    masks_dir: Optional[Path] = None,
) -> List[Frame]:
    """Load all frames of a woven_sequence directory.

    seq_dir/
      tss4_fcm/*.jpg
      vls128_rear_axle/*.npz
      poslv/POS.parquet (or poslv.csv)

    masks_dir: optional dir with PNGs named <stem>.png matching jpg stems
               (white=keep, black=skip).

    Returns list of Frame, ordered by camera timestamp.
    """
    calib = _load_calib(recalib_json, vehicle)
    cam_files = sorted((seq_dir / 'tss4_fcm').glob('*.jpg'))
    lid_files = sorted((seq_dir / 'vls128_rear_axle').glob('*.npz'))
    if len(cam_files) != len(lid_files):
        raise RuntimeError(
            f'cam={len(cam_files)} != lid={len(lid_files)} in {seq_dir}')

    pq = seq_dir / 'poslv' / 'POS.parquet'
    csv = seq_dir / 'poslv' / 'poslv.csv'
    ts_a, e, n_, u, roll, pitch, head = DK.load_poslv_poses(
        pq if pq.is_file() else csv)

    # World = rear_axle frame at frame 0. So define
    #   T_rear0_from_world = I
    # Per-frame:  T_world_from_rear_i  = T_rear0_from_rear_i^{-1}
    # where T_rear0_from_rear_i is the relative POSLV motion between t_0 and t_i
    # in rear_axle frame.
    # We use POSLV as the "absolute" frame to interpolate, and anchor at t_0.
    cam_t0 = float(cam_files[0].stem.split('_')[1]) / 1000.0
    T_poslv_from_rear_at_t0 = DK.interp_pose(
        np.array([cam_t0]), ts_a, e, n_, u, roll, pitch, head)[0]
    T_rear0_from_poslv = np.linalg.inv(T_poslv_from_rear_at_t0)

    frames: List[Frame] = []
    for i, (jpg, npz) in enumerate(zip(cam_files, lid_files)):
        cam_t = float(jpg.stem.split('_')[1]) / 1000.0
        T_poslv_from_rear_i = DK.interp_pose(
            np.array([cam_t]), ts_a, e, n_, u, roll, pitch, head)[0]
        T_world_from_rear_i = T_rear0_from_poslv @ T_poslv_from_rear_i

        cam_t2w = T_world_from_rear_i @ calib['T_rear_from_pscam_static']

        # LiDAR: same world definition; pose at lidar sweep mid
        z = np.load(npz)
        sweep_end_s = float(z['timestamp_millisecond'])
        lid_t = sweep_end_s - 0.05  # mid
        T_poslv_from_rear_lid = DK.interp_pose(
            np.array([lid_t]), ts_a, e, n_, u, roll, pitch, head)[0]
        T_world_from_rear_lid = T_rear0_from_poslv @ T_poslv_from_rear_lid

        xs, ys, zs = z['xs'], z['ys'], z['zs']
        intensity = z['intensity'].astype(np.float32) / 255.0
        # per-point relative time (sweep_end-mid offset)
        if 'point_time_offset_us' in z.files:
            t_off = z['point_time_offset_us'].astype(np.float64) * 1e-6
            t_off = t_off - 0.05  # relative to mid
        else:
            t_off = np.zeros_like(xs, dtype=np.float64)
        pc = np.stack([xs, ys, zs, intensity, t_off], axis=-1).astype(np.float32)

        mask_path: Optional[Path] = None
        if masks_dir is not None:
            cand = masks_dir / (jpg.stem + '.png')
            if cand.is_file():
                mask_path = cand

        frames.append(Frame(
            idx=i,
            img_path=jpg,
            mask_path=mask_path,
            K=calib['K'],
            dist=calib['dist'],
            width=calib['W'],
            height=calib['H'],
            cam_t2w=cam_t2w,
            cam_t=cam_t,
            lid_t2w=T_world_from_rear_lid,
            lid_t=lid_t,
            pc=pc,
        ))
    return frames


def _self_test():
    import cv2
    SEQ = Path('/mnt/ecp-perception/woven_sequence/tss4_calib_raw_01/'
               '20230612_001946/sequence=248_20230612_001946_'
               '1686533186104-1686533191007')
    frames = load_woven_sequence(SEQ, vehicle='248')
    print(f'[loaded] {len(frames)} frames')
    f0 = frames[0]
    print(f'[f0] K=\n{f0.K}\ndist={f0.dist} WxH={f0.width}x{f0.height}')
    print(f'[f0] cam_t2w=\n{f0.cam_t2w}')
    print(f'[f0] lid_t2w=\n{f0.lid_t2w}')
    print(f'[f0] pc.shape={f0.pc.shape}')

    # Project with cv2.fisheye to verify against reference overlay
    img = cv2.imread(str(f0.img_path))
    pc_xyz = f0.pc[:, :3].astype(np.float64)  # lidar/rear_axle frame
    # frame 0: lid_t2w should be ~identity (small t-interpolation drift only)
    print(f'[f0] |lid_t2w - I|_F = {np.linalg.norm(f0.lid_t2w - np.eye(4)):.4e}')

    # world point = lid_t2w @ pc_xyz
    pts_w = (f0.lid_t2w[:3, :3] @ pc_xyz.T + f0.lid_t2w[:3, 3:4]).T
    # cam coords = inv(cam_t2w) @ pts_w
    T_w2c = np.linalg.inv(f0.cam_t2w)
    pts_c = (T_w2c[:3, :3] @ pts_w.T + T_w2c[:3, 3:4]).T
    Z = pts_c[:, 2]
    valid = Z > 0.5
    pcv = pts_c[valid]
    uv, _ = cv2.fisheye.projectPoints(
        pcv.reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
        f0.K, f0.dist)
    uv = uv.reshape(-1, 2)
    zin = Z[valid]
    in_b = ((uv[:, 0] >= 0) & (uv[:, 0] < f0.width)
            & (uv[:, 1] >= 0) & (uv[:, 1] < f0.height))
    uv = uv[in_b]
    zin = zin[in_b]

    over = img.copy()
    zclip = np.clip(zin, 0.5, 80.0)
    znorm = ((zclip - 0.5) / 79.5 * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(znorm.reshape(-1, 1),
                              cv2.COLORMAP_TURBO).reshape(-1, 3)
    for i in range(len(uv)):
        cv2.circle(over, (int(round(uv[i, 0])), int(round(uv[i, 1]))),
                   2, tuple(int(x) for x in cmap[i]), -1)
    out = Path('/raid/home/hfunaya/_overlay_DATALOADER_FRAME0.jpg')
    cv2.imwrite(str(out), over, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f'[wrote] {out}  pts in img = {len(uv)}')


if __name__ == '__main__':
    _self_test()
