"""Add radar/ subdirectory to existing nuscenes_ps scenes.

Per scene:
  /mnt/nvme6t/nuscenes_ps/<scene_name>/
    radar/
      00.pkl, 01.pkl, ...   # merged radar (5 sensors) in world frame, per frame

nuScenes has 5 ARS 408 radars (front, FL, FR, BL, BR). We merge all 5 per
keyframe to maximize density (~50-200 pts/frame total). z is always 0
(planar 2.5D radar) — set to ego-mounted height (1.0 m) to avoid degenerate
all-zero scatter when the projection layer hits.

Usage:
    python scripts/preprocessing/nuscenes_radar_to_pandaset.py \
        --max-scenes 150 --workers 8
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from nuscenes.utils.data_classes import RadarPointCloud


NS_ROOT = Path('/mnt/nvme6t/nuscenes/data')
OUT_ROOT = Path('/mnt/nvme6t/nuscenes_ps')

RADAR_CHANNELS = [
    'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT',
]

# Default radar sensor mounting height above ground (rough). Used to give
# planar radar points a non-zero elevation so frame_token scatter doesn't
# all collapse onto a single image row.
RADAR_Z_DEFAULT = 1.0


def _quat_mat(q_wxyz):
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()


def _sensor_to_world(cal, ep):
    R_s = _quat_mat(cal['rotation']); t_s = np.array(cal['translation'])
    R_e = _quat_mat(ep['rotation']);  t_e = np.array(ep['translation'])
    T = np.eye(4); T[:3, :3] = R_e; T[:3, 3] = t_e
    S = np.eye(4); S[:3, :3] = R_s; S[:3, 3] = t_s
    return (T @ S).astype(np.float64)


_CTX = {}


def _worker_init(meta_dir: str):
    meta = Path(meta_dir)
    _CTX['samples']     = {s['token']: s for s in json.load(open(meta / 'sample.json'))}
    _CTX['sample_data'] = json.load(open(meta / 'sample_data.json'))
    _CTX['cal']         = {c['token']: c for c in json.load(open(meta / 'calibrated_sensor.json'))}
    _CTX['ego']         = {e['token']: e for e in json.load(open(meta / 'ego_pose.json'))}
    _CTX['scenes']      = {s['token']: s for s in json.load(open(meta / 'scene.json'))}
    sd_by = {}
    for d in _CTX['sample_data']:
        if not d['is_key_frame']:
            continue
        tok = d['sample_token']
        parts = d['filename'].split('/')
        if len(parts) < 2:
            continue
        channel = parts[1]
        if channel in RADAR_CHANNELS or channel == 'LIDAR_TOP':
            sd_by.setdefault(tok, {})[channel] = d
    _CTX['sd_by'] = sd_by
    RadarPointCloud.disable_filters()  # keep all radar returns, no quality filter


def _load_radar_world(sd, data_root):
    """Load one radar pcd and return points (N, 4) in world frame: x,y,z,rcs."""
    fp = Path(data_root) / sd['filename']
    pc = RadarPointCloud.from_file(str(fp))   # (18, N) — x,y,z,dyn,id,rcs,...
    pts_radar = pc.points[:3].T.astype(np.float32)               # (N, 3)
    rcs       = pc.points[5].astype(np.float32)                  # (N,)
    # radar z is always 0; keep as-is (we'll lift to RADAR_Z_DEFAULT in caller
    # if desired) — for the "planar tokenize" case, we put points in world
    # using their own z=0 which after sensor→world becomes the sensor mounting
    # height in world coords automatically (good).
    cal = _CTX['cal'][sd['calibrated_sensor_token']]
    ep  = _CTX['ego'][sd['ego_pose_token']]
    T_s2w = _sensor_to_world(cal, ep)
    h = np.hstack([pts_radar, np.ones((len(pts_radar), 1), dtype=pts_radar.dtype)])
    pts_w = (T_s2w @ h.T).T[:, :3].astype(np.float32)
    return np.column_stack([pts_w, rcs])  # (N, 4)


def convert_scene(args) -> tuple[str, str]:
    scene_token, out_root, max_frames, data_root = args
    scene = _CTX['scenes'][scene_token]
    scene_name = scene['name']
    seg_out = Path(out_root) / scene_name

    if not (seg_out / 'lidar' / '00.pkl').exists():
        return scene_name, 'skipped (no lidar/ — base layout missing)'

    radar_dir = seg_out / 'radar'
    radar_dir.mkdir(parents=True, exist_ok=True)

    # Use SAME frame index as nuscenes_to_pandaset.py: every keyframe sample
    # that has LIDAR_TOP + 6 cams. Filter by lidar/00.pkl existing already.
    sample_tok = scene['first_sample_token']
    frames = []
    while sample_tok and len(frames) < max_frames:
        sd_map = _CTX['sd_by'].get(sample_tok, {})
        if 'LIDAR_TOP' in sd_map and all(r in sd_map for r in RADAR_CHANNELS):
            frames.append(sd_map)
        sample_tok = _CTX['samples'][sample_tok]['next']

    if len(frames) < 10:
        return scene_name, f'skipped ({len(frames)} frames < 10)'

    n_done = 0
    for fi, sd_map in enumerate(frames):
        out_pkl = radar_dir / f'{fi:02d}.pkl'
        if out_pkl.exists():
            n_done += 1
            continue
        pts_all = []
        for ch in RADAR_CHANNELS:
            try:
                pts = _load_radar_world(sd_map[ch], data_root)
                pts_all.append(pts)
            except Exception as e:
                continue
        if not pts_all:
            continue
        merged = np.concatenate(pts_all, axis=0)
        pd.DataFrame(dict(x=merged[:, 0], y=merged[:, 1],
                           z=merged[:, 2], rcs=merged[:, 3])).to_pickle(out_pkl)
        n_done += 1

    return scene_name, f'ok ({n_done} frames, ~{len(frames)} expected)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-scenes', type=int, default=150)
    ap.add_argument('--max-frames', type=int, default=40)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--out-root', default=str(OUT_ROOT))
    ap.add_argument('--meta-dir', default=str(NS_ROOT / 'v1.0-trainval'))
    ap.add_argument('--data-root', default=str(NS_ROOT))
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    scenes = json.load(open(Path(args.meta_dir) / 'scene.json'))
    scenes_sorted = sorted(scenes, key=lambda s: s['name'])
    picked = scenes_sorted[args.offset:args.offset + args.max_scenes]
    tokens = [s['token'] for s in picked]
    print(f'adding radar/ to {len(tokens)} scenes → {out_root}', flush=True)

    done = 0
    with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(args.meta_dir,)) as ex:
        futures = {ex.submit(convert_scene,
                             (tok, str(out_root), args.max_frames, args.data_root)): tok
                   for tok in tokens}
        for fut in as_completed(futures):
            name, msg = fut.result()
            done += 1
            print(f'[{done}/{len(tokens)}] {name}: {msg}', flush=True)
    print('done.')


if __name__ == '__main__':
    main()
