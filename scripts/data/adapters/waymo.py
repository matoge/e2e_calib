"""Waymo Open Dataset → CalibFrame.

Wraps the existing build_waymo_v3 helpers so the adapter goes through
the same per_cam_projections + unproject_pinhole pipeline as the cache
builder. Adapter output is what the validator gates; never duplicate the
projection math.

Waymo intensity is already roughly in [0,1] from the LCP component; we
clip to [0,1] just to enforce the CalibFrame contract.
"""
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.calib_frame import CalibFrame
from scripts.util.projection import unproject_pinhole
from scripts.preprocessing.build_waymo_v3 import (
    get_cam_calib, read_boxes_by_ts, boxes_cam_frame,
    list_frame_timestamps,
)
from datasets.waymo_lcp import (ALL_LASERS, ensure_lcp,
                                  read_lcp_all, read_range_all,
                                  per_cam_projections)
from datasets.waymo import WAYMO_DIR
import pyarrow.parquet as pq


_Z_MIN = 0.5

TILE_LAYOUT = dict(
    tile_w=512, tile_h=512, stride=384, pad_px=64,
    y_start=0,           # waymo doesn't have a sky band to skip
    jpg_quality=95,
)


def list_segs() -> list[str]:
    """All segment names available locally (camera_image parquets)."""
    return sorted(p.stem for p in (WAYMO_DIR / 'camera_image').glob('*.parquet'))


def list_frames(seg_name: str) -> list[int]:
    """Frame timestamps (microsecond ints) in the segment."""
    return list_frame_timestamps(seg_name)


def load_frame(seg_name: str, ts: int, cam_id: int = 1) -> CalibFrame:
    """Build a validated CalibFrame for (segment, timestamp, camera).

    cam_id matches Waymo's Camera Name enum (1=FRONT, 2=FRONT_LEFT,
    3=FRONT_RIGHT, 4=SIDE_LEFT, 5=SIDE_RIGHT).
    """
    calib_per_cam = get_cam_calib(seg_name)
    if cam_id not in calib_per_cam:
        raise KeyError(f'cam_id={cam_id} not in seg {seg_name} '
                        f'(have: {sorted(calib_per_cam.keys())})')
    K, T_c_v = calib_per_cam[cam_id]

    # JPEG bytes for this (seg, ts, cam_id)
    cam_df = pq.read_table(
        WAYMO_DIR / 'camera_image' / f'{seg_name}.parquet',
        filters=[('key.camera_name', '=', cam_id),
                 ('key.frame_timestamp_micros', '=', int(ts))]
    ).to_pandas()
    if len(cam_df) == 0:
        raise KeyError(f'no camera_image for seg={seg_name} ts={ts} cam={cam_id}')
    jpg_bytes = bytes(cam_df.iloc[0]['[CameraImageComponent].image'])
    img = np.asarray(Image.open(io.BytesIO(jpg_bytes)).convert('RGB'))
    H, W = img.shape[:2]

    # Pull pre-projected (uv, depth, intensity, laser) for this frame.
    lcp_path = ensure_lcp(seg_name)
    lcp_by_ts = read_lcp_all(lcp_path, lasers=ALL_LASERS)
    range_by_ts = read_range_all(WAYMO_DIR / 'lidar' / f'{seg_name}.parquet',
                                   lasers=ALL_LASERS)
    lcp_arrs = lcp_by_ts.get(int(ts), {})
    range_arrs = range_by_ts.get(int(ts), {})
    if not lcp_arrs or not range_arrs:
        raise KeyError(f'lidar empty for seg={seg_name} ts={ts}')
    per_cam = per_cam_projections(lcp_arrs, range_arrs, cams=[cam_id])
    proj = per_cam.get(cam_id)
    if proj is None or len(proj['uv']) == 0:
        # Empty point cloud is a valid frame.
        cf = CalibFrame(
            img=img, K=K.astype(np.float64), is_fisheye=False, dist=None,
            pts_cam=np.zeros((0, 3), np.float32),
            intensity=np.zeros((0,), np.float32),
            uv_full=np.zeros((0, 2), np.float32),
            z_cam=np.zeros((0,), np.float32),
            is_obj=np.zeros((0,), np.float32),
            scene_id=seg_name, frame_id=int(ts), cam_id=str(cam_id))
        cf.validate()
        return cf

    uv = proj['uv'].astype(np.float32)
    depth = proj['depth'].astype(np.float32)
    intensity_raw = proj.get('intensity',
                              np.zeros_like(depth)).astype(np.float32)
    pts_cam = unproject_pinhole(uv, depth, K)

    # Filter to in-image, z > z_min
    keep = ((depth > _Z_MIN)
            & (uv[:, 0] >= 0) & (uv[:, 0] < W)
            & (uv[:, 1] >= 0) & (uv[:, 1] < H))
    uv = uv[keep]; pts_cam = pts_cam[keep]
    depth = depth[keep]; intensity_raw = intensity_raw[keep]

    # Cuboid → is_obj (boxes in vehicle frame, then to cam frame for membership).
    boxes_by_ts = read_boxes_by_ts(seg_name)
    cuboids = boxes_cam_frame(boxes_by_ts.get(int(ts), []), T_c_v)
    if cuboids:
        from datasets.pandaset_full import _is_obj_per_point
        is_obj = _is_obj_per_point(pts_cam, cuboids).astype(np.float32)
    else:
        is_obj = np.zeros(len(pts_cam), dtype=np.float32)

    intensity = np.clip(intensity_raw, 0.0, 1.0).astype(np.float32)

    cf = CalibFrame(
        img=img, K=K.astype(np.float64), is_fisheye=False, dist=None,
        pts_cam=pts_cam.astype(np.float32),
        intensity=intensity,
        uv_full=uv.astype(np.float32),
        z_cam=depth.astype(np.float32),
        is_obj=is_obj,
        cuboids=cuboids,
        scene_id=seg_name, frame_id=int(ts), cam_id=str(cam_id),
    )
    cf.validate()
    return cf
