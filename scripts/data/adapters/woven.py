"""Woven sequence (TSS4 FCM fisheye) → CalibFrame.

Wraps the existing build_woven_sequence_v3 helpers so the adapter goes
through the SAME projection pipeline (project_lidar_into_image with the
camera-shutter-time-corrected T_cl) as the cache builder. Adapter output
is what the validator gates; we never duplicate the time/extrinsic math.
"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# import compatibility for the no-future-annotations rule we adopted in
# the pipeline file: this adapter doesn't need it, but it MUST be safe to
# import inside ClearML pipeline tmp scripts.

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.calib_frame import CalibFrame
from scripts.util.projection import project_lidar_into_image
from scripts.preprocessing.build_woven_sequence_v3 import (
    _load_setting,
    _camera_calib_fcm,
    _load_metadata,
    _camera_delay_ms_for_frame,
    _get_poses,
    _load_pts_intensity,
    _pose_at_camera_time,
    _T_lidar_to_cam_at_camera_time,
    _load_cuboids_vehicle,
    _is_obj_per_point,
)


_INTENSITY_DIVISOR = 128.0   # ip664 sensor (same as kamikado), /128 quantum
_Z_MIN = 0.5


def list_frames(seq: Path) -> list:
    """Frame IDs available in the sequence (intersection of pose, lidar, jpg)."""
    seq = Path(seq)
    md = _load_metadata(seq)
    fids, _ = _get_poses(md)
    have_jpg = {p.stem for p in (seq / 'tss4_fcm').glob('*.jpg')}
    have_pts = {p.stem.replace('points_', '')
                for p in (seq / 'vls128_rear_axle').glob('*.npz')}
    return sorted(fid for fid in fids
                   if fid in have_jpg and fid in have_pts)


def load_frame(seq: Path, frame_id) -> CalibFrame:
    seq = Path(seq)
    setting = _load_setting(seq)
    K, dist, R_cv, t_cv, W, H, delay_default = _camera_calib_fcm(setting)
    md = _load_metadata(seq)
    delay_ms = _camera_delay_ms_for_frame(md, frame_id, delay_default)

    fids, poses = _get_poses(md)
    if frame_id not in poses:
        raise KeyError(f'frame {frame_id!r} has no pose in {seq}')
    idx = fids.index(frame_id)

    pts_flu, intensity_raw = _load_pts_intensity(seq, frame_id)

    img_path = seq / 'tss4_fcm' / f'{frame_id}.jpg'
    img = np.asarray(Image.open(img_path).convert('RGB'))

    if pts_flu.size == 0:
        cf = CalibFrame(
            img=img, K=K.astype(np.float64), is_fisheye=True,
            dist=dist.astype(np.float64),
            pts_cam=np.zeros((0, 3), np.float32),
            intensity=np.zeros((0,), np.float32),
            uv_full=np.zeros((0, 2), np.float32),
            z_cam=np.zeros((0,), np.float32),
            is_obj=np.zeros((0,), np.float32),
            scene_id=seq.name, frame_id=int(frame_id.split('_')[0])
                                      if isinstance(frame_id, str) and '_' in frame_id
                                      else int(frame_id),
            cam_id='tss4_fcm')
        cf.validate()
        return cf

    pose_curr = poses[frame_id]
    pose_cam  = _pose_at_camera_time(poses, fids, idx, delay_ms)
    T_cl = _T_lidar_to_cam_at_camera_time(pose_curr, pose_cam, R_cv, t_cv)

    pts_xyzi = np.column_stack(
        [pts_flu.astype(np.float32), intensity_raw.astype(np.float32)])
    _, pts_cam, uv_full, z_cam, intens_raw = project_lidar_into_image(
        pts_xyzi, K, T_cl, W, H,
        is_fisheye=True, dist=dist, z_min=_Z_MIN)

    intensity = np.clip(intens_raw / _INTENSITY_DIVISOR, 0.0, 1.0).astype(np.float32)

    # Cuboid → is_obj membership in vehicle FLU (matches build script).
    cubs = _load_cuboids_vehicle(seq, frame_id)
    if cubs:
        # Lift cam-FRD pts back to vehicle FLU for the membership test
        pts_vis_veh = ((np.linalg.inv(R_cv).astype(np.float32) @ pts_cam.T).T
                        + t_cv.astype(np.float32)[None, :])
        is_obj = _is_obj_per_point(pts_vis_veh, cubs).astype(np.float32)
    else:
        is_obj = np.zeros(len(pts_cam), dtype=np.float32)

    # Convert frame_id to an int for CalibFrame (build keeps a str, but
    # the validator/dataset needs an int handle).
    if isinstance(frame_id, str) and '_' in frame_id:
        fr_int = int(frame_id.split('_')[0])
    else:
        fr_int = int(frame_id)

    cf = CalibFrame(
        img=img, K=K.astype(np.float64), is_fisheye=True,
        dist=dist.astype(np.float64),
        pts_cam=pts_cam.astype(np.float32),
        intensity=intensity,
        uv_full=uv_full.astype(np.float32),
        z_cam=z_cam.astype(np.float32),
        is_obj=is_obj,
        cuboids=[],   # raw cubs list not yet normalised — caller can re-attach
        scene_id=seq.name, frame_id=fr_int, cam_id='tss4_fcm',
    )
    cf.validate()
    return cf
