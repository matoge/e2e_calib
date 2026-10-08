"""Smoke test for POST /calibrate/generic/frame — builds a payload from a
WovenSequence frame, sends it, and prints δ_cam. Also compares to the existing
/calibrate/frame result on the same seq/frame_idx so we can verify the two
paths give the same answer (they should, up to float32 rounding).
"""
from __future__ import annotations
import base64
import json
import sys
from pathlib import Path

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts' / 'inference'))

from calib_lib import load_frame_data
from scripts.preprocessing.build_woven_sequence_v3 import (
    _load_setting, _load_metadata, _get_poses,
    _camera_calib_fcm, _camera_delay_ms_for_frame, _pose_at_camera_time,
)


def _T_lidar_to_cam_at_camera_time(pose_curr, pose_cam, R_cv, t_cv):
    """Same math as calib_lib._T_lidar_to_cam_at_camera_time (kept here so this
    smoke test is self-contained and doesn't need to import the private helper)."""
    import numpy as _np
    R_world_lidar = pose_curr[:3, :3]
    t_world_lidar = pose_curr[:3, 3]
    R_world_cam = pose_cam[:3, :3] @ R_cv.T
    t_world_cam = pose_cam[:3, 3] - R_world_cam @ (-R_cv.T @ t_cv)
    R_cw = R_world_cam.T
    t_cw = -R_cw @ t_world_cam
    R_cl = R_cw @ R_world_lidar
    t_cl = R_cw @ t_world_lidar + t_cw
    T = _np.eye(4)
    T[:3, :3] = R_cl; T[:3, 3] = t_cl
    return T


def main():
    seq = Path('/home/hfunaya/git/loom/backend/assets/woven_sequence/'
               'unilab_001/test01/'
               'sequence=ip654-lidar0-1432519511400022000-1432519516399877000')
    frame_idx = 0
    api = "http://127.0.0.1:8501"

    setting = _load_setting(seq)
    K, dist, R_cv, t_cv, IW, IH, delay_default = _camera_calib_fcm(setting)
    md = _load_metadata(seq)
    fids, poses = _get_poses(md)
    fid = fids[frame_idx]
    delay_ms = _camera_delay_ms_for_frame(md, fid, delay_default)
    pose_cam = _pose_at_camera_time(poses, fids, frame_idx, delay_ms)
    T_cl = _T_lidar_to_cam_at_camera_time(poses[fid], pose_cam, R_cv, t_cv)

    # LiDAR points (LiDAR frame) — same schema every WovenSequence uses.
    pts_arr = np.load(seq / 'vls128_rear_axle' / f'{fid}.npz')
    pts_xyzi = np.column_stack([pts_arr['xs'], pts_arr['ys'], pts_arr['zs'],
                                  pts_arr['intensity']]).astype(np.float32)
    print(f'[smoke] loaded {pts_xyzi.shape[0]} LiDAR points from {fid}.npz')

    image_bytes = (seq / 'tss4_fcm' / f'{fid}.jpg').read_bytes()
    print(f'[smoke] image {len(image_bytes)/1024:.1f} KB, K.fc=[{K[0,0]:.1f},{K[1,1]:.1f}]')

    # Build the payload
    payload = dict(
        image_b64=base64.b64encode(image_bytes).decode('ascii'),
        points_xyzi_b64=base64.b64encode(pts_xyzi.tobytes()).decode('ascii'),
        camera=dict(
            resolution=[int(IW), int(IH)],
            fc=[float(K[0, 0]), float(K[1, 1])],
            cc=[float(K[0, 2]), float(K[1, 2])],
            dist_kb4=[float(x) for x in dist],
            T_cam_lidar=T_cl.tolist(),
            setting_rot_rad=[float(x) for x in setting['fcm']['rot']],
            setting_mp_m=[float(x) for x in setting['fcm']['mp']],
        ),
        frame_id=fid,
    )
    print(f'[smoke] payload size ≈ '
          f'{(len(payload["image_b64"]) + len(payload["points_xyzi_b64"]))/1024/1024:.2f} MB')

    r = requests.post(f'{api}/calibrate/generic/frame', json=payload, timeout=120)
    if r.status_code != 200:
        print(f'❌ POST failed {r.status_code}: {r.text[:500]}')
        return 1
    gen = r.json()
    print('[smoke] /calibrate/generic/frame →')
    for k in ['status', 'fid', 'n_tiles', 'n_points', 'n_hood_dropped',
              'delta_cam_deg', 'delta_cam_m', 'image_size']:
        print(f'    {k:20s} = {gen.get(k)}')
    if 'deltas' in gen:
        print(f'    deltas               = pitch {np.degrees(gen["deltas"]["rot_y"]):+.4f}°  '
              f'yaw {np.degrees(gen["deltas"]["rot_z"]):+.4f}°  '
              f'roll {np.degrees(gen["deltas"]["rot_x"]):+.4f}°')

    # Compare to the seq-based endpoint (same math but reads from disk).
    r2 = requests.post(f'{api}/calibrate/frame',
                        json=dict(seq_dir=str(seq), frame_idx=frame_idx,
                                  use_hood_mask=False), timeout=120)
    if r2.status_code != 200:
        print(f'⚠️  /calibrate/frame failed {r2.status_code}: {r2.text[:300]}')
        return 0
    ref = r2.json()
    print('\n[smoke] /calibrate/frame (reference, use_hood_mask=False) →')
    for k in ['fid', 'n_tiles', 'n_points',
              'delta_cam_deg', 'delta_cam_m']:
        print(f'    {k:20s} = {ref.get(k)}')

    # Difference — expect ≤ a few % or float32 rounding.
    d_gen = np.array(gen['delta_cam_deg'] + gen['delta_cam_m'])
    d_ref = np.array(ref['delta_cam_deg'] + ref['delta_cam_m'])
    max_abs = float(np.max(np.abs(d_gen - d_ref)))
    print(f'\n[smoke] max |δ_generic − δ_ref| = {max_abs:.6f}')
    if max_abs < 0.05:
        print('[smoke] ✅ PASS: generic path matches reference path.')
        return 0
    print('[smoke] ⚠️  DIFF > 0.05 — probably OK for KB4 fisheye but investigate.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
