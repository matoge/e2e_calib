"""Convert nuScenes scenes → PandaSet on-disk layout for cross-frame training.

Per scene:
  /mnt/nvme6t/nuscenes_ps/<scene_name>/
    camera/front_camera/
      intrinsics.json        # {fx, fy, cx, cy}
      poses.json             # [{heading, position}] cam→world per frame
      00.jpg, 01.jpg, ...    # keyframe images for that cam
    camera/front_left_camera/ ... (6 cams total)
    lidar/
      00.pkl, 01.pkl, ...    # LIDAR_TOP points in world frame (shared across cams)

Cameras: the 6 ring cameras (front / front_left / front_right / back /
back_left / back_right). Frame index is anchored on LIDAR_TOP keyframes;
for each lidar keyframe we pick the *same-sample* CAM_XXX sample_data (they
are time-synced keyframes in nuScenes), so fi is consistent across all
cameras + lidar within a scene.

Coordinate conventions:
  nuScenes cameras are already OpenCV (X=right, Y=down, Z=forward) — same as
  PandaSet / Waymo-converted / AV2. `_project` works without any axis swap.
  nuScenes intrinsics are pinhole (no distortion), so the intrinsics.json
  omits k1/k2/k3 and the dataset side falls through to pure pinhole.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation


NS_ROOT = Path('/mnt/nvme6t/nuscenes/data')
OUT_ROOT = Path('/mnt/nvme6t/nuscenes_ps')

# nuScenes channel name → PandaSet camera subdir name
CAM_MAP = {
    'CAM_FRONT':       'front_camera',
    'CAM_FRONT_LEFT':  'front_left_camera',
    'CAM_FRONT_RIGHT': 'front_right_camera',
    'CAM_BACK':        'back_camera',
    'CAM_BACK_LEFT':   'back_left_camera',
    'CAM_BACK_RIGHT':  'back_right_camera',
}
CAM_CHANNELS = list(CAM_MAP.keys()) + ['LIDAR_TOP']


def _quat_mat(q_wxyz):
    # nuScenes stores [w,x,y,z]; scipy expects [x,y,z,w]
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()


def _sensor_to_world(cal, ep):
    R_s = _quat_mat(cal['rotation']); t_s = np.array(cal['translation'])
    R_e = _quat_mat(ep['rotation']);  t_e = np.array(ep['translation'])
    T = np.eye(4); T[:3, :3] = R_e; T[:3, 3] = t_e
    S = np.eye(4); S[:3, :3] = R_s; S[:3, 3] = t_s
    return (T @ S).astype(np.float64)


def _mat_to_quat_pos(T):
    q = Rotation.from_matrix(T[:3, :3]).as_quat()   # xyzw
    t = T[:3, 3]
    return dict(heading=dict(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3])),
                position=dict(x=float(t[0]), y=float(t[1]), z=float(t[2])))


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
        channel = parts[1]   # samples/<CHANNEL>/<file>
        if channel in CAM_CHANNELS:
            sd_by.setdefault(tok, {})[channel] = d
    _CTX['sd_by'] = sd_by


def convert_scene(args) -> tuple[str, str]:
    scene_token, out_root, max_frames, jpg_quality, data_root = args
    scene = _CTX['scenes'][scene_token]
    scene_name = scene['name']
    seg_out = Path(out_root) / scene_name

    # idempotency: all 6 cam poses.json + lidar/00.pkl must exist
    done = all((seg_out / 'camera' / CAM_MAP[ch] / 'poses.json').exists()
               for ch in CAM_MAP)
    if done and (seg_out / 'lidar' / '00.pkl').exists():
        return scene_name, 'skipped'

    # collect frames: for each sample (keyframe) that has LIDAR_TOP + all 6 cams
    sample_tok = scene['first_sample_token']
    frames = []
    while sample_tok and len(frames) < max_frames:
        sd_map = _CTX['sd_by'].get(sample_tok, {})
        if 'LIDAR_TOP' in sd_map and all(ch in sd_map for ch in CAM_MAP):
            frames.append(sd_map)
        sample_tok = _CTX['samples'][sample_tok]['next']

    if len(frames) < 10:
        return scene_name, f'skipped ({len(frames)} frames < 10)'

    seg_out.mkdir(parents=True, exist_ok=True)
    (seg_out / 'lidar').mkdir(parents=True, exist_ok=True)

    # intrinsics per camera (invariant per-scene, read from frame 0)
    for ch, cam_sub in CAM_MAP.items():
        cam_dir = seg_out / 'camera' / cam_sub
        cam_dir.mkdir(parents=True, exist_ok=True)
        sd0 = frames[0][ch]
        K = np.array(_CTX['cal'][sd0['calibrated_sensor_token']]['camera_intrinsic'],
                     dtype=np.float64)
        (cam_dir / 'intrinsics.json').write_text(
            json.dumps(dict(fx=float(K[0, 0]), fy=float(K[1, 1]),
                            cx=float(K[0, 2]), cy=float(K[1, 2])), indent=2))

    # per-cam pose lists, built as we walk frames
    poses_by_cam = {ch: [] for ch in CAM_MAP}

    for fi, sd_map in enumerate(frames):
        # lidar (shared across cams)
        lid = sd_map['LIDAR_TOP']
        cal_lid = _CTX['cal'][lid['calibrated_sensor_token']]
        ep_lid  = _CTX['ego'][lid['ego_pose_token']]
        T_lid2w = _sensor_to_world(cal_lid, ep_lid)
        bin_path = Path(data_root) / lid['filename']
        pts_lid = np.frombuffer(bin_path.read_bytes(),
                                 dtype=np.float32).reshape(-1, 5)[:, :3]
        h = np.hstack([pts_lid, np.ones((len(pts_lid), 1), dtype=pts_lid.dtype)])
        pts_w = (T_lid2w @ h.T).T[:, :3].astype(np.float32)
        pd.DataFrame(dict(x=pts_w[:, 0], y=pts_w[:, 1], z=pts_w[:, 2])).to_pickle(
            seg_out / f'lidar/{fi:02d}.pkl')

        # each camera: pose + image
        for ch, cam_sub in CAM_MAP.items():
            sd_cam = sd_map[ch]
            cal_cam = _CTX['cal'][sd_cam['calibrated_sensor_token']]
            ep_cam  = _CTX['ego'][sd_cam['ego_pose_token']]
            T_cam2w = _sensor_to_world(cal_cam, ep_cam)
            poses_by_cam[ch].append(_mat_to_quat_pos(T_cam2w))
            src = Path(data_root) / sd_cam['filename']
            dst = seg_out / 'camera' / cam_sub / f'{fi:02d}.jpg'
            if not dst.exists():
                Image.open(src).convert('RGB').save(dst, quality=jpg_quality)

    for ch, cam_sub in CAM_MAP.items():
        (seg_out / 'camera' / cam_sub / 'poses.json').write_text(
            json.dumps(poses_by_cam[ch], indent=2))

    return scene_name, f'ok ({len(frames)} frames × {len(CAM_MAP)} cams)'


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
    print(f'converting {len(tokens)} scenes (6 cams + lidar) → {out_root}', flush=True)

    done = 0
    with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_worker_init,
            initargs=(args.meta_dir,)) as ex:
        futures = {ex.submit(convert_scene,
                             (tok, str(out_root), args.max_frames, 90, args.data_root)): tok
                   for tok in tokens}
        for fut in as_completed(futures):
            name, msg = fut.result()
            done += 1
            print(f'[{done}/{len(tokens)}] {name}: {msg}', flush=True)
    print('done.')


if __name__ == '__main__':
    main()
