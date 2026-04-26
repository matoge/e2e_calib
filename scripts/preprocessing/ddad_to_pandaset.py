"""Convert DDAD (TRI DGP format) scenes → PandaSet on-disk layout for cross-frame training.

Per scene:
  /mnt/nvme6t/ddad_ps/<scene_name>/
    camera/<CAMERA_NAME>/       # e.g. camera/CAMERA_01/, camera/CAMERA_05/, …
      intrinsics.json           # {fx, fy, cx, cy}
      poses.json                # [{heading, position}] cam→world per frame
      00.jpg, 01.jpg, ...       # re-saved as JPEG (PandaSet dataset reads .jpg)
    lidar/
      00.pkl, 01.pkl, ...       # world-frame LiDAR (pandas x, y, z)

DDAD scene JSON layout (from the DGP spec):
- samples[]: per-frame record with `datum_keys[7]` hashes and `calibration_key`
- data[]: flat list of datums, each has `key`, `id.name` (sensor), `datum.image` or
  `datum.point_cloud` containing `filename` + `pose` (world frame)
- calibration/<calibration_key>.json: parallel lists `names / extrinsics / intrinsics`
  indexed by sensor name; LIDAR has all-zero intrinsics.

DDAD camera frames are OpenCV (X=right, Y=down, Z=forward) — same projection
math as PandaSet / AV2 / NuScenes converted. Intrinsics have no radial
distortion term → intrinsics.json omits k1/k2/k3 and the dataset's
`_proj_cam` falls through to pure pinhole.
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation


def _pose_to_mat(pose):
    q = pose['rotation']
    t = pose['translation']
    R = Rotation.from_quat([q['qx'], q['qy'], q['qz'], q['qw']]).as_matrix()
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = [t['x'], t['y'], t['z']]
    return M


def _mat_to_heading_pos(M):
    q = Rotation.from_matrix(M[:3, :3]).as_quat()  # xyzw
    t = M[:3, 3]
    return dict(heading=dict(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])),
                position=dict(x=float(t[0]), y=float(t[1]), z=float(t[2])))


def convert_scene(args):
    scene_dir, out_root, max_frames, jpg_quality = args
    scene_dir = Path(scene_dir)
    out_root = Path(out_root)
    out_dir = out_root / scene_dir.name

    # idempotent skip (first cam's poses.json + first lidar.pkl present)
    if (out_dir / 'lidar' / '00.pkl').exists() and \
       any((out_dir / 'camera').glob('*/poses.json')):
        return f'[skip] {scene_dir.name}'

    scene_json_paths = list(scene_dir.glob('scene_*.json'))
    if not scene_json_paths:
        return f'[skip no-scene-json] {scene_dir.name}'
    sc = json.loads(scene_json_paths[0].read_text())

    calib_paths = list((scene_dir / 'calibration').glob('*.json'))
    if not calib_paths:
        return f'[skip no-calib] {scene_dir.name}'
    # pick the one referenced by first sample
    calib_key = sc['samples'][0].get('calibration_key', '')
    cal_path = scene_dir / 'calibration' / f'{calib_key}.json'
    if not cal_path.exists():
        cal_path = calib_paths[0]
    cal = json.loads(cal_path.read_text())

    # index data by key
    data_by_key = {d['key']: d for d in sc['data']}

    # per-camera intrinsics
    cam_names = []
    for i, name in enumerate(cal['names']):
        intr = cal['intrinsics'][i]
        if intr.get('fx', 0) == 0:
            continue  # LIDAR row
        cam_names.append(name)
        cam_dir = out_dir / 'camera' / name
        cam_dir.mkdir(parents=True, exist_ok=True)
        (cam_dir / 'intrinsics.json').write_text(json.dumps({
            'fx': float(intr['fx']), 'fy': float(intr['fy']),
            'cx': float(intr['cx']), 'cy': float(intr['cy']),
        }, indent=2))

    (out_dir / 'lidar').mkdir(parents=True, exist_ok=True)

    samples = sc['samples'][:max_frames] if max_frames > 0 else sc['samples']
    poses_by_cam = {cn: [] for cn in cam_names}

    for fi, sample in enumerate(samples):
        for dk in sample['datum_keys']:
            d = data_by_key.get(dk)
            if d is None:
                continue
            sensor = d['id']['name']
            if sensor == 'LIDAR':
                lnpz_path = scene_dir / d['datum']['point_cloud']['filename']
                lz = np.load(lnpz_path)
                arr = lz[list(lz.keys())[0]]
                pts = arr[:, :3].astype(np.float64)
                pose = d['datum']['point_cloud']['pose']
                T_s2w = _pose_to_mat(pose)
                pts_w = (T_s2w[:3, :3] @ pts.T + T_s2w[:3, 3:4]).T
                pd.DataFrame({'x': pts_w[:, 0].astype(np.float32),
                              'y': pts_w[:, 1].astype(np.float32),
                              'z': pts_w[:, 2].astype(np.float32)}).to_pickle(
                    out_dir / 'lidar' / f'{fi:02d}.pkl')
            elif sensor in cam_names:
                src = scene_dir / d['datum']['image']['filename']
                dst = out_dir / 'camera' / sensor / f'{fi:02d}.jpg'
                pose = d['datum']['image']['pose']
                poses_by_cam[sensor].append(_mat_to_heading_pos(_pose_to_mat(pose)))
                if not dst.exists():
                    Image.open(src).convert('RGB').save(dst, quality=jpg_quality)

    for cn, poses in poses_by_cam.items():
        (out_dir / 'camera' / cn / 'poses.json').write_text(
            json.dumps(poses, indent=2))

    return f'[ok]   {scene_dir.name}  ({len(samples)} frames × {len(cam_names)} cams)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/mnt/nvme6t/ddad_ext/ddad_train_val')
    ap.add_argument('--dst', default='/mnt/nvme6t/ddad_ps')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-frames', type=int, default=0,
                    help='cap frames per scene; 0 = all')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--jpg-quality', type=int, default=90)
    args = ap.parse_args()

    src = Path(args.src); dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    scenes = sorted(p for p in src.iterdir() if p.is_dir())
    if args.limit:
        scenes = scenes[:args.limit]
    print(f'converting {len(scenes)} DDAD scenes → {dst}  (workers={args.workers})')

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(convert_scene,
                          (str(s), str(dst), args.max_frames, args.jpg_quality)): s
                for s in scenes}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                print(f'  [{i}/{len(scenes)}] {fut.result()}', flush=True)
            except Exception as e:
                print(f'  [{i}/{len(scenes)}] [FAIL] {futs[fut].name}: {e!r}', flush=True)


if __name__ == '__main__':
    main()
