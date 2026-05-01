"""Waymo Open Dataset v2 — lidar_camera_projection (LCP) helpers.

Waymo provides per-pixel precomputed lidar→camera projections in
`gs://waymo_open_dataset_v_2_0_0/{split}/lidar_camera_projection/<seg>.parquet`.
These already account for ego-pose interpolation, lidar rotation, and camera
rolling shutter — using them removes the need for any manual cam-pose math
and is the only path to pixel-perfect alignment with v2 (the per-row pose
channel that was in v1 TFRecord is dropped from the v2 lidar parquet).

LCP encoding per range-image pixel:
    shape = (H_beams, W_azimuth, 6)
    channels = [cam_id_a, u_a, v_a, cam_id_b, u_b, v_b]
       — up to 2 cams per lidar pixel (overlapping FOVs)
       — cam_id == 0 means slot empty
       — cam_id ∈ {1:FRONT, 2:FRONT_LEFT, 3:FRONT_RIGHT, 4:SIDE_LEFT, 5:SIDE_RIGHT}

Range encoding per pixel (from the regular `lidar/<seg>.parquet`):
    shape = (H_beams, W_azimuth, ≥1)
    channel 0 = range (m); 0 means no return.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq

CAM_NAMES = {1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT', 4: 'SIDE_LEFT', 5: 'SIDE_RIGHT'}
ALL_LASERS = (1, 2, 3, 4, 5)
ALL_RETURNS = ('return1', 'return2')

GCS_BASE = 'gs://waymo_open_dataset_v_2_0_0'


def ensure_lcp(seg_name: str, split: str = 'training',
               local_root: str | None = None) -> Path:
    """Ensure the LCP parquet for `seg_name` is on local disk; download from GCS if not.
    Returns the local path.

    `local_root` defaults to env var `WAYMO_LCP_DIR` or `/mnt/nvme6t/waymo_lcp`.
    """
    if local_root is None:
        local_root = os.environ.get('WAYMO_LCP_DIR', '/mnt/nvme6t/waymo_lcp')
    p = Path(local_root) / split / f'{seg_name}.parquet'
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    src = f'{GCS_BASE}/{split}/lidar_camera_projection/{seg_name}.parquet'
    print(f'[waymo_lcp] downloading {src} → {p}', flush=True)
    subprocess.check_call(['gsutil', 'cp', src, str(p)])
    return p


def read_lcp_at_ts(lcp_path: Path | str, ts_micros: int,
                   lasers: Iterable[int] = ALL_LASERS) -> dict:
    """Read LCP parquet rows for one frame.

    Returns dict[laser_id] -> dict[return_name] -> ndarray (H, W, 6) float32.
    Empty slot (no row for that laser) is omitted.
    """
    df = pq.read_table(
        lcp_path,
        filters=[('key.frame_timestamp_micros', '=', int(ts_micros)),
                 ('key.laser_name', 'in', list(lasers))],
    ).to_pandas()
    out: dict[int, dict[str, np.ndarray]] = {}
    for _, r in df.iterrows():
        lid = int(r['key.laser_name'])
        out.setdefault(lid, {})
        for ret in ALL_RETURNS:
            vals  = r[f'[LiDARCameraProjectionComponent].range_image_{ret}.values']
            shape = r[f'[LiDARCameraProjectionComponent].range_image_{ret}.shape']
            if vals is None or len(vals) == 0:
                continue
            out[lid][ret] = np.asarray(vals, dtype=np.float32).reshape(shape)
    return out


def read_range_at_ts(lidar_parquet_path: Path | str, ts_micros: int,
                     lasers: Iterable[int] = ALL_LASERS) -> dict:
    """Read range_image (range channel only) for one frame.

    Returns dict[laser_id] -> dict[return_name] -> ndarray (H, W) float32 (range in m).
    """
    df = pq.read_table(
        lidar_parquet_path,
        filters=[('key.frame_timestamp_micros', '=', int(ts_micros)),
                 ('key.laser_name', 'in', list(lasers))],
    ).to_pandas()
    out: dict[int, dict[str, np.ndarray]] = {}
    for _, r in df.iterrows():
        lid = int(r['key.laser_name'])
        out.setdefault(lid, {})
        for ret in ALL_RETURNS:
            vals  = r[f'[LiDARComponent].range_image_{ret}.values']
            shape = r[f'[LiDARComponent].range_image_{ret}.shape']
            if vals is None or len(vals) == 0:
                continue
            arr = np.asarray(vals, dtype=np.float32).reshape(shape)
            out[lid][ret] = arr[:, :, 0]  # channel 0 = range
    return out


def read_lcp_all(lcp_path: Path | str,
                 lasers: Iterable[int] = ALL_LASERS) -> dict:
    """Read entire LCP parquet once, group by ts in memory.

    Returns dict[ts_micros] -> dict[laser] -> dict[return_name] -> ndarray (H,W,6).
    Use instead of calling read_lcp_at_ts() per frame (40x faster: one parquet
    metadata scan vs one per ts).
    """
    df = pq.read_table(
        lcp_path,
        filters=[('key.laser_name', 'in', list(lasers))],
    ).to_pandas()
    out: dict[int, dict[int, dict[str, np.ndarray]]] = {}
    for _, r in df.iterrows():
        ts  = int(r['key.frame_timestamp_micros'])
        lid = int(r['key.laser_name'])
        out.setdefault(ts, {}).setdefault(lid, {})
        for ret in ALL_RETURNS:
            vals  = r[f'[LiDARCameraProjectionComponent].range_image_{ret}.values']
            shape = r[f'[LiDARCameraProjectionComponent].range_image_{ret}.shape']
            if vals is None or len(vals) == 0:
                continue
            out[ts][lid][ret] = np.asarray(vals, dtype=np.float32).reshape(shape)
    return out


def read_range_all(lidar_parquet_path: Path | str,
                   lasers: Iterable[int] = ALL_LASERS) -> dict:
    """Read entire lidar parquet once, group by ts in memory.

    Returns dict[ts_micros] -> dict[laser] -> dict[return_name] -> ndarray (H,W) range in m.
    """
    df = pq.read_table(
        lidar_parquet_path,
        filters=[('key.laser_name', 'in', list(lasers))],
    ).to_pandas()
    out: dict[int, dict[int, dict[str, np.ndarray]]] = {}
    for _, r in df.iterrows():
        ts  = int(r['key.frame_timestamp_micros'])
        lid = int(r['key.laser_name'])
        out.setdefault(ts, {}).setdefault(lid, {})
        for ret in ALL_RETURNS:
            vals  = r[f'[LiDARComponent].range_image_{ret}.values']
            shape = r[f'[LiDARComponent].range_image_{ret}.shape']
            if vals is None or len(vals) == 0:
                continue
            arr = np.asarray(vals, dtype=np.float32).reshape(shape)
            out[ts][lid][ret] = arr[:, :, 0]
    return out


def per_cam_projections(lcp_arrs: dict, range_arrs: dict,
                        cams: Iterable[int] = tuple(CAM_NAMES.keys())) -> dict:
    """Bucket LCP pixels into per-camera (uv, depth) arrays.

    Args:
        lcp_arrs:   output of read_lcp_at_ts   (dict[laser][return] = (H,W,6))
        range_arrs: output of read_range_at_ts (dict[laser][return] = (H,W))
        cams: subset of camera ids to extract
    Returns:
        dict[cam_id] -> dict with keys
            uv:    (N, 2) float32  (u, v in pixel coords)
            depth: (N,)   float32  (lidar range in m, NOT cam-frame depth)
            laser: (N,)   uint8    (source laser id)
            ret:   (N,)   uint8    (1 = return1, 2 = return2)
    """
    cams = list(cams)
    buckets = {c: dict(uv=[], depth=[], laser=[], ret=[]) for c in cams}
    for lid, by_ret in lcp_arrs.items():
        rng_by_ret = range_arrs.get(lid, {})
        for ret_name, proj in by_ret.items():
            depth = rng_by_ret.get(ret_name)
            if depth is None:
                continue
            d_valid = depth > 0
            ret_idx = 1 if ret_name == 'return1' else 2
            # 2 projection slots per pixel
            for slot in (0, 3):
                cam_ids = proj[:, :, slot].astype(np.int32)
                u = proj[:, :, slot + 1]
                v = proj[:, :, slot + 2]
                for c in cams:
                    m = (cam_ids == c) & d_valid
                    if not m.any():
                        continue
                    buckets[c]['uv'   ].append(np.stack([u[m], v[m]], axis=1).astype(np.float32))
                    buckets[c]['depth'].append(depth[m])
                    n = int(m.sum())
                    buckets[c]['laser'].append(np.full(n, lid, dtype=np.uint8))
                    buckets[c]['ret'  ].append(np.full(n, ret_idx, dtype=np.uint8))
    out = {}
    for c, b in buckets.items():
        if not b['uv']:
            out[c] = dict(uv=np.zeros((0, 2), np.float32),
                          depth=np.zeros(0, np.float32),
                          laser=np.zeros(0, np.uint8),
                          ret=np.zeros(0, np.uint8))
        else:
            out[c] = dict(
                uv   =np.concatenate(b['uv'   ], axis=0),
                depth=np.concatenate(b['depth']),
                laser=np.concatenate(b['laser']),
                ret  =np.concatenate(b['ret']),
            )
    return out


def project_frame(seg_name: str, ts_micros: int,
                  waymo_root: str | Path = '/mnt/nvme6t/waymo/training',
                  split: str = 'training',
                  lcp_root: str = '/mnt/nvme6t/waymo_lcp',
                  cams: Iterable[int] = tuple(CAM_NAMES.keys()),
                  lasers: Iterable[int] = ALL_LASERS) -> dict:
    """One-call helper: ensure LCP, read it + range image, return per-cam projections.

    Convenience wrapper for vis / one-off use; for batch preprocessing prefer
    calling read_lcp_at_ts / read_range_at_ts / per_cam_projections directly so
    the parquet handles can be reused across frames.
    """
    waymo_root = Path(waymo_root)
    lcp = ensure_lcp(seg_name, split=split, local_root=lcp_root)
    lcp_arrs   = read_lcp_at_ts(lcp,                                ts_micros, lasers)
    range_arrs = read_range_at_ts(waymo_root/'lidar'/f'{seg_name}.parquet',
                                  ts_micros, lasers)
    return per_cam_projections(lcp_arrs, range_arrs, cams=cams)
