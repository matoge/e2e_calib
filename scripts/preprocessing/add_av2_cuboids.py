"""Post-processor: add annotations/cuboids/{fi:02d}.pkl.gz to converted AV2 PS scenes.

The original av2_to_pandaset.py skipped 3D box annotations. This script
reads each source log's annotations.feather, picks the boxes whose
timestamp_ns matches each PS frame's LiDAR timestamp, converts them to
the PandaSet pickle schema (yaw + position.{x,y,z} + dimensions.{x,y,z}
in WORLD coords), and writes them so pandaset_pair's is_obj path lights
up.

Usage:
    python scripts/preprocessing/add_av2_cuboids.py \
        --src /mnt/mininas/datasets/argoverse2/sensor/train \
        --dst /mnt/nvme6t/av2_ps \
        [--workers 8]
"""
from __future__ import annotations
import argparse, gzip, math, pickle, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


def _q_to_yaw(qw, qx, qy, qz):
    """Quaternion → yaw (rotation around world z-axis). Boxes are gravity-aligned."""
    # standard formula for yaw of a (w, x, y, z) quaternion
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _se3_from_row(row):
    """city_SE3_egovehicle row → 4×4 ego→world transform."""
    qw, qx, qy, qz = row['qw'], row['qx'], row['qy'], row['qz']
    tx, ty, tz = row['tx_m'], row['ty_m'], row['tz_m']
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = (tx, ty, tz)
    return T


def _slerp_ego_pose(city_df, ts_ns):
    """Nearest ego→world pose to ts_ns (linear interp between flanking rows)."""
    ts_arr = city_df['timestamp_ns'].values
    if ts_ns <= ts_arr[0]:
        return _se3_from_row(city_df.iloc[0])
    if ts_ns >= ts_arr[-1]:
        return _se3_from_row(city_df.iloc[-1])
    j = int(np.searchsorted(ts_arr, ts_ns))
    t0 = float(ts_arr[j-1]); t1 = float(ts_arr[j])
    f = (float(ts_ns) - t0) / (t1 - t0)
    T0 = _se3_from_row(city_df.iloc[j-1])
    T1 = _se3_from_row(city_df.iloc[j])
    # linear interp on translation, nlerp on rotation as a quick approx
    T = np.eye(4)
    T[:3, 3] = (1 - f) * T0[:3, 3] + f * T1[:3, 3]
    R0, R1 = T0[:3, :3], T1[:3, :3]
    R = (1 - f) * R0 + f * R1
    U, _, Vt = np.linalg.svd(R)
    T[:3, :3] = U @ Vt
    return T


def process_one(src_log_dir: Path, ps_scene_dir: Path):
    """Build annotations/cuboids/{fi:02d}.pkl.gz for one scene."""
    out_dir = ps_scene_dir / 'annotations' / 'cuboids'
    if out_dir.is_dir():
        existing = list(out_dir.glob('*.pkl.gz'))
        if existing:
            return f'[skip] {ps_scene_dir.name} ({len(existing)} files)'

    anno_path = src_log_dir / 'annotations.feather'
    city_path = src_log_dir / 'city_SE3_egovehicle.feather'
    if not anno_path.exists() or not city_path.exists():
        return f'[skip-no-src] {ps_scene_dir.name}'

    # frame index → lidar_ts_ns from PS lidar dir filenames + feather lookup
    # (we pick the same timestamps the PS converter picked: every K-th lidar file)
    src_lidar_dir = src_log_dir / 'sensors' / 'lidar'
    if not src_lidar_dir.is_dir():
        return f'[skip-no-lidar] {ps_scene_dir.name}'
    lidar_files = sorted(src_lidar_dir.iterdir())
    # match the PS converter: every_k=2 by default. Detect from PS lidar count.
    ps_lidar_count = len(list((ps_scene_dir / 'lidar').glob('*.pkl')))
    if ps_lidar_count == 0:
        return f'[skip-empty-ps] {ps_scene_dir.name}'
    every_k = max(1, len(lidar_files) // ps_lidar_count)
    sampled_ts = [int(p.stem) for p in lidar_files[::every_k]][:ps_lidar_count]

    anno = pd.read_feather(anno_path)
    city = pd.read_feather(city_path).sort_values('timestamp_ns').reset_index(drop=True)

    # Group annotations by timestamp_ns for quick lookup
    by_ts = dict(tuple(anno.groupby('timestamp_ns')))

    out_dir.mkdir(parents=True, exist_ok=True)
    for fi, ts_ns in enumerate(sampled_ts):
        # AV2 annotations are at exact LiDAR timestamps
        boxes_df = by_ts.get(ts_ns)
        if boxes_df is None:
            # try nearest annotation timestamp (within 50ms)
            ts_keys = np.array(list(by_ts.keys()), dtype=np.int64)
            if len(ts_keys) == 0:
                _write_empty(out_dir / f'{fi:02d}.pkl.gz'); continue
            j = int(np.argmin(np.abs(ts_keys - ts_ns)))
            if abs(ts_keys[j] - ts_ns) > 50_000_000:    # >50 ms = no match
                _write_empty(out_dir / f'{fi:02d}.pkl.gz'); continue
            boxes_df = by_ts[int(ts_keys[j])]

        # Transform each box ego → world via this frame's interpolated pose
        T_e2w = _slerp_ego_pose(city, ts_ns)
        R_e2w = T_e2w[:3, :3]
        t_e2w = T_e2w[:3,  3]

        rows = []
        for _, b in boxes_df.iterrows():
            p_ego = np.array([b['tx_m'], b['ty_m'], b['tz_m']], dtype=np.float64)
            p_world = R_e2w @ p_ego + t_e2w
            # Box yaw in ego frame; rotate by ego→world heading.
            # Combined yaw: atan2 of R_e2w @ R_ego_box's heading direction.
            yaw_ego = _q_to_yaw(b['qw'], b['qx'], b['qy'], b['qz'])
            heading = np.array([math.cos(yaw_ego), math.sin(yaw_ego), 0.0])
            heading_w = R_e2w @ heading
            yaw_world = math.atan2(heading_w[1], heading_w[0])
            rows.append({
                'yaw':          float(yaw_world),
                'position.x':   float(p_world[0]),
                'position.y':   float(p_world[1]),
                'position.z':   float(p_world[2]),
                'dimensions.x': float(b['length_m']),
                'dimensions.y': float(b['width_m']),
                'dimensions.z': float(b['height_m']),
                'category':     str(b.get('category', 'unknown')),
            })
        df_out = pd.DataFrame(rows)
        with gzip.open(out_dir / f'{fi:02d}.pkl.gz', 'wb') as f:
            pickle.dump(df_out, f)

    return f'[ok]   {ps_scene_dir.name}  frames={len(sampled_ts)}'


def _write_empty(path: Path):
    with gzip.open(path, 'wb') as f:
        pickle.dump(pd.DataFrame(columns=[
            'yaw', 'position.x', 'position.y', 'position.z',
            'dimensions.x', 'dimensions.y', 'dimensions.z', 'category'
        ]), f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='AV2 source root (...sensor/train)')
    ap.add_argument('--dst', required=True, help='AV2 PS-format root (e.g. /mnt/nvme6t/av2_ps)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit',   type=int, default=0, help='process only first N scenes')
    args = ap.parse_args()

    src = Path(args.src); dst = Path(args.dst)
    ps_scenes = sorted([d for d in dst.iterdir() if d.is_dir()])
    if args.limit:
        ps_scenes = ps_scenes[:args.limit]
    print(f'processing {len(ps_scenes)} scenes  ({args.workers} workers)')

    tasks = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for ps in ps_scenes:
            log_id = ps.name
            src_log = src / log_id
            if not src_log.is_dir():
                print(f'[skip-no-src-dir] {log_id}'); continue
            tasks.append(ex.submit(process_one, src_log, ps))
        for i, fut in enumerate(as_completed(tasks)):
            print(f'  [{i+1}/{len(tasks)}] {fut.result()}', flush=True)


if __name__ == '__main__':
    main()
