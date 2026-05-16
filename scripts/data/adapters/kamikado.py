"""kamikado-san's woven scenes → CalibFrame.

Raw layout (one dir per scene, flat):
    SCENE/
      calib.calib          JSON: intrinsics + Kannala-Brandt coeffs + extrinsics V→S
      image_N.png          parent fisheye PNG (3840×2160)
      points_V_N.txt       LiDAR pts in VEHICLE FLU frame, `x y z intensity` lines

This adapter mirrors build_kamikado_v3.process_frame's projection logic
exactly (it now both go through scripts.util.projection.project_lidar_into_image),
so the resulting CalibFrame is byte-identical with what the legacy
build pipeline writes — except the output is the in-memory dataclass
instead of LMDB tile records.

Kamikado intensity normalisation: ip664 sensor reports values up to
~96 with a /128 quantum, so we divide by 128 and clip to [0,1] to match
the per-sensor normalisation that previously lived in
datasets/pandaset_full.py. The new pipeline performs this normalisation
once at adapter time and stores the [0,1] value in the cache.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np

_INTENSITY_DIVISOR = 128.0
from PIL import Image

from scripts.data.calib_frame import CalibFrame
from scripts.util.projection import project_lidar_into_image


_Z_MIN = 0.5

# Tile-layout defaults that match build_kamikado_v3 CLI defaults so a
# fresh CalibFrame → tile_cutter run produces tiles at the same origins
# (and the same JPEG quality) as the legacy cache.
TILE_LAYOUT = dict(
    tile_w=512, tile_h=512, stride=384, pad_px=64,
    y_start=600,         # skip the top 600 px sky band
    jpg_quality=95,
)


def _load_calib(scene_dir: Path):
    """Parse calib.calib → (K, dist4, T_SV).

    Mirrors build_kamikado_v3._load_calib exactly so the adapter is
    byte-identical with the cache builder.
    """
    from scipy.spatial.transform import Rotation
    j = json.loads((scene_dir / 'calib.calib').read_text())
    intr = j['calibration']['intrinsics']
    K = np.array(intr['camera_model']['pinhole_parameters']
                  ['matrix_image_camera']['matrix']).T
    K = np.ascontiguousarray(K, dtype=np.float64)
    dist = np.asarray(intr['distortion_model']['generic_fisheye_parameters']
                       ['coefficients'], dtype=np.float64)
    extr = j['calibration']['extrinsics']['transform_VS']
    quat = extr['so3']
    R_VS = Rotation.from_quat([quat['x'], quat['y'], quat['z'], quat['w']]).as_matrix()
    t_VS = np.asarray(extr['translation']['matrix'][0], dtype=np.float64)
    T_VS = np.eye(4); T_VS[:3, :3] = R_VS; T_VS[:3, 3] = t_VS
    T_SV = np.linalg.inv(T_VS)
    return K, dist, T_SV


def _read_points_V(p: Path) -> np.ndarray:
    """`x y z intensity` text → (N, 4) float32."""
    arr = np.loadtxt(p, comments='#', usecols=(0, 1, 2, 3), dtype=np.float32)
    return np.atleast_2d(arr)


def list_frames(scene_dir: Path) -> list[int]:
    """Return frame indices that have BOTH image_N.png and points_V_N.txt."""
    img_re = re.compile(r'^image_(\d+)\.png$')
    pts_re = re.compile(r'^points_V_(\d+)\.txt$')
    img_idx = {int(img_re.match(p.name).group(1)) for p in scene_dir.glob('image_*.png')
               if img_re.match(p.name)}
    pts_idx = {int(pts_re.match(p.name).group(1)) for p in scene_dir.glob('points_V_*.txt')
               if pts_re.match(p.name)}
    return sorted(img_idx & pts_idx)


def load_frame(scene_dir: Path | str, frame_idx: int) -> CalibFrame:
    """Load one (scene, frame) → CalibFrame, validated.

    Drops behind-camera and out-of-image points so the validator's
    invariants hold.
    """
    scene_dir = Path(scene_dir)
    K, dist, T_SV = _load_calib(scene_dir)

    img_path = scene_dir / f'image_{frame_idx}.png'
    pts_path = scene_dir / f'points_V_{frame_idx}.txt'
    img = np.asarray(Image.open(img_path).convert('RGB'))
    H, W = img.shape[:2]

    pts_V = _read_points_V(pts_path)
    if pts_V.size == 0:
        # Empty point cloud is a valid frame, just no LiDAR returns.
        cf = CalibFrame(
            img=img, K=K, is_fisheye=True, dist=dist,
            pts_cam=np.zeros((0, 3), np.float32),
            intensity=np.zeros((0,), np.float32),
            uv_full=np.zeros((0, 2), np.float32),
            z_cam=np.zeros((0,), np.float32),
            is_obj=np.zeros((0,), np.float32),
            scene_id=scene_dir.name, frame_id=int(frame_idx),
            cam_id='fcm')
        cf.validate()
        return cf

    _, pts_cam, uv_full, z_cam, intens_raw = project_lidar_into_image(
        pts_V, K, T_SV, W, H, is_fisheye=True, dist=dist, z_min=_Z_MIN)

    intensity = np.clip(intens_raw / _INTENSITY_DIVISOR, 0.0, 1.0).astype(np.float32)
    is_obj = np.zeros(len(pts_cam), dtype=np.float32)  # kamikado has no boxes

    # NOTE: scene_id matches the production cache exactly (no 'kamikado/'
    # prefix) so golden compares against the legacy build_kamikado_v3
    # output succeed.
    cf = CalibFrame(
        img=img, K=K, is_fisheye=True, dist=dist,
        pts_cam=pts_cam.astype(np.float32),
        intensity=intensity,
        uv_full=uv_full.astype(np.float32),
        z_cam=z_cam.astype(np.float32),
        is_obj=is_obj,
        scene_id=scene_dir.name, frame_id=int(frame_idx),
        cam_id='fcm',
    )
    cf.validate()
    return cf
