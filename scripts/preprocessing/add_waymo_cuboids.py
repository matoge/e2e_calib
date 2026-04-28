"""Post-processor: add annotations/cuboids/{fi:02d}.pkl.gz to converted Waymo PS scenes.

Source: /mnt/mininas/datasets/waymo/training/lidar_box/<segment>.parquet
        /mnt/mininas/datasets/waymo/training/vehicle_pose/<segment>.parquet
Boxes are in VEHICLE frame at frame_timestamp_micros — transform to WORLD via
the matching vehicle_pose row, then write PandaSet-schema pickles.
"""
from __future__ import annotations
import argparse, gzip, math, pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


def _flat16_to_T(flat16):
    """Waymo vehicle_pose row → 4×4 vehicle→world."""
    return np.asarray(flat16, dtype=np.float64).reshape(4, 4)


def process_one(box_parquet: Path, pose_parquet: Path, ps_scene_dir: Path):
    out_dir = ps_scene_dir / 'annotations' / 'cuboids'
    if out_dir.is_dir() and any(out_dir.glob('*.pkl.gz')):
        return f'[skip] {ps_scene_dir.name}'

    ps_lidar_files = sorted((ps_scene_dir / 'lidar').glob('*.pkl'))
    if not ps_lidar_files:
        return f'[skip-no-ps-lidar] {ps_scene_dir.name}'
    n_ps_frames = len(ps_lidar_files)

    # Load lidar timestamps from poses.json (any cam stores them) for mapping
    poses_json = ps_scene_dir / 'camera'
    cams = sorted(p for p in poses_json.iterdir() if p.is_dir()) if poses_json.is_dir() else []
    if not cams:
        return f'[skip-no-cam] {ps_scene_dir.name}'

    # Source boxes (in vehicle frame)
    box_df = pd.read_parquet(box_parquet)
    pose_df = pd.read_parquet(pose_parquet).reset_index()

    # Distinct frame timestamps the converter saw (lidar frames are sampled
    # in order — we take the first n_ps_frames unique timestamps).
    ts_col = 'key.frame_timestamp_micros'
    ts_uniq = sorted(box_df[ts_col].unique())[:n_ps_frames]
    if len(ts_uniq) < n_ps_frames:
        # Fall back to whatever boxes have; pad with empty
        pass

    # Map ts → ego→world flat16
    pose_ts_col = 'key.frame_timestamp_micros'
    pose_T_col  = '[VehiclePoseComponent].world_from_vehicle.transform'
    pose_by_ts = {row[pose_ts_col]: _flat16_to_T(row[pose_T_col]) for _, row in pose_df.iterrows()}

    box_grouped = dict(tuple(box_df.groupby(ts_col)))

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for fi in range(n_ps_frames):
        ts = ts_uniq[fi] if fi < len(ts_uniq) else None
        rows = []
        if ts is not None and ts in box_grouped:
            T = pose_by_ts.get(ts)
            if T is None:
                # nearest pose
                tk = np.array(list(pose_by_ts.keys()), dtype=np.int64)
                j = int(np.argmin(np.abs(tk - ts)))
                T = pose_by_ts[int(tk[j])]
            R = T[:3, :3]; t = T[:3, 3]
            for _, b in box_grouped[ts].iterrows():
                cx = b['[LiDARBoxComponent].box.center.x']
                cy = b['[LiDARBoxComponent].box.center.y']
                cz = b['[LiDARBoxComponent].box.center.z']
                heading = b['[LiDARBoxComponent].box.heading']
                lx = b['[LiDARBoxComponent].box.size.x']
                ly = b['[LiDARBoxComponent].box.size.y']
                lz = b['[LiDARBoxComponent].box.size.z']
                p_v = np.array([cx, cy, cz], dtype=np.float64)
                p_w = R @ p_v + t
                # heading is around vehicle z; rotate by ego→world
                hx = math.cos(heading); hy = math.sin(heading)
                hw = R @ np.array([hx, hy, 0.0])
                yaw_w = math.atan2(hw[1], hw[0])
                rows.append({
                    'yaw':          float(yaw_w),
                    'position.x':   float(p_w[0]),
                    'position.y':   float(p_w[1]),
                    'position.z':   float(p_w[2]),
                    'dimensions.x': float(lx),
                    'dimensions.y': float(ly),
                    'dimensions.z': float(lz),
                    'category':     int(b['[LiDARBoxComponent].type']),
                })
        df = pd.DataFrame(rows, columns=[
            'yaw', 'position.x', 'position.y', 'position.z',
            'dimensions.x', 'dimensions.y', 'dimensions.z', 'category'
        ])
        with gzip.open(out_dir / f'{fi:02d}.pkl.gz', 'wb') as f:
            pickle.dump(df, f)
        written += 1
    return f'[ok]   {ps_scene_dir.name}  frames={written}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/mnt/mininas/datasets/waymo/training')
    ap.add_argument('--dst', default='/mnt/nvme6t/waymo_ps')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit',   type=int, default=0)
    args = ap.parse_args()

    src = Path(args.src); dst = Path(args.dst)
    box_dir = src / 'lidar_box'
    pose_dir = src / 'vehicle_pose'
    ps_scenes = sorted([d for d in dst.iterdir() if d.is_dir()])
    if args.limit:
        ps_scenes = ps_scenes[:args.limit]
    print(f'processing {len(ps_scenes)} scenes  ({args.workers} workers)')

    tasks = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for ps in ps_scenes:
            seg = ps.name
            box = box_dir / f'{seg}.parquet'
            pose = pose_dir / f'{seg}.parquet'
            if not box.exists() or not pose.exists():
                print(f'[skip-no-src] {seg}'); continue
            tasks.append(ex.submit(process_one, box, pose, ps))
        for i, fut in enumerate(as_completed(tasks)):
            print(f'  [{i+1}/{len(tasks)}] {fut.result()}', flush=True)


if __name__ == '__main__':
    main()
