"""Convert nuScenes scenes → PandaSet on-disk layout for cross-frame training.

Per scene (capped at MAX_FRAMES keyframes, 2 Hz in nuScenes):
  /mnt/nvme6t/nuscenes_ps/<scene_name>/
    camera/front_camera/
      intrinsics.json        # {fx,fy,cx,cy}
      poses.json             # [{heading, position} for CAM_FRONT, cam→world]
      00.jpg, 01.jpg, ...    # CAM_FRONT keyframe images
    lidar/
      00.pkl, 01.pkl, ...    # LIDAR_TOP points in world frame (pandas df x,y,z)

Coordinate conventions:
  nuScenes CAM_FRONT is already OpenCV (X=right, Y=down, Z=forward) — same as
  PandaSet / Waymo-converted, so NO frame rotation needed here. K @ [X,Y,Z] just
  works with the existing `_project` helper.

Notes:
  - 2 Hz keyframes → 40 / scene × 20 sec. baseline=20 frames is 10 sec physical,
    larger than Waymo/PandaSet bl=20 (2 sec). Model treats as just-another-pair;
    we keep --max-frames configurable in case long baselines are too hard.
  - Pose GT quality is generally considered good on nuScenes, no per-segment
    yaw drift issue like Waymo.
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


def _quat_mat(q_wxyz):
    # nuScenes stores [w,x,y,z]; scipy expects [x,y,z,w]
    return Rotation.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()


def _sensor_to_world(cal, ep):
    """calibrated_sensor + ego_pose → 4×4 sensor→world."""
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


# global tables are set once per worker to avoid re-loading 300MB json
_CTX = {}


def _worker_init(meta_dir: str):
    meta = Path(meta_dir)
    _CTX['samples']     = {s['token']: s for s in json.load(open(meta / 'sample.json'))}
    _CTX['sample_data'] = json.load(open(meta / 'sample_data.json'))
    _CTX['cal']         = {c['token']: c for c in json.load(open(meta / 'calibrated_sensor.json'))}
    _CTX['ego']         = {e['token']: e for e in json.load(open(meta / 'ego_pose.json'))}
    _CTX['scenes']      = {s['token']: s for s in json.load(open(meta / 'scene.json'))}
    # build sd_by_sample lookup
    sd_by = {}
    for d in _CTX['sample_data']:
        if not d['is_key_frame']:
            continue
        tok = d['sample_token']
        sd_by.setdefault(tok, {})
        # channel via directory name. Filename is always "samples/<CHANNEL>/<file>"
        # where <file> itself can contain "__CAM_FRONT__timestamp" — so string
        # containment on "CAM_FRONT_" accidentally matches the file body too.
        parts = d['filename'].split('/')
        if len(parts) < 2:
            continue
        channel = parts[1]
        if channel in ('CAM_FRONT', 'LIDAR_TOP'):
            sd_by[tok][channel] = d
    _CTX['sd_by'] = sd_by


def convert_scene(args) -> tuple[str, str]:
    scene_token, out_root, max_frames, jpg_quality, data_root = args
    scene = _CTX['scenes'][scene_token]
    scene_name = scene['name']   # like "scene-0001"
    seg_out = Path(out_root) / scene_name

    # idempotency check
    if (seg_out / 'camera/front_camera/intrinsics.json').exists() and \
       (seg_out / 'camera/front_camera/poses.json').exists():
        return scene_name, 'skipped (already converted)'

    sample_tok = scene['first_sample_token']
    frames = []
    while sample_tok and len(frames) < max_frames:
        sd_map = _CTX['sd_by'].get(sample_tok, {})
        cam = sd_map.get('CAM_FRONT')
        lid = sd_map.get('LIDAR_TOP')
        if cam is None or lid is None:
            sample_tok = _CTX['samples'][sample_tok]['next']
            continue
        frames.append((cam, lid))
        sample_tok = _CTX['samples'][sample_tok]['next']

    if len(frames) < 10:
        return scene_name, f'skipped ({len(frames)} frames)'

    seg_out.mkdir(parents=True, exist_ok=True)
    (seg_out / 'camera/front_camera').mkdir(parents=True, exist_ok=True)
    (seg_out / 'lidar').mkdir(parents=True, exist_ok=True)

    # intrinsics from first frame's calibrated_sensor (invariant per scene for same cam)
    cam0, _ = frames[0]
    cal_cam0 = _CTX['cal'][cam0['calibrated_sensor_token']]
    K = np.array(cal_cam0['camera_intrinsic'], dtype=np.float64)
    (seg_out / 'camera/front_camera/intrinsics.json').write_text(
        json.dumps(dict(fx=float(K[0, 0]), fy=float(K[1, 1]),
                        cx=float(K[0, 2]), cy=float(K[1, 2])), indent=2))

    poses_out = []
    for fi, (cam, lid) in enumerate(frames):
        # camera pose
        cal_cam = _CTX['cal'][cam['calibrated_sensor_token']]
        ep_cam  = _CTX['ego'][cam['ego_pose_token']]
        T_cam2w = _sensor_to_world(cal_cam, ep_cam)
        poses_out.append(_mat_to_quat_pos(T_cam2w))

        # image
        src = Path(data_root) / cam['filename']
        im = Image.open(src).convert('RGB')
        im.save(seg_out / f'camera/front_camera/{fi:02d}.jpg', quality=jpg_quality)

        # lidar
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

    (seg_out / 'camera/front_camera/poses.json').write_text(
        json.dumps(poses_out, indent=2))

    return scene_name, f'ok ({len(frames)} frames)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-scenes', type=int, default=80,
                    help='cap number of scenes to convert')
    ap.add_argument('--max-frames', type=int, default=40,
                    help='cap frames per scene (nuScenes is 2Hz × ~40 keyframes)')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--offset', type=int, default=0,
                    help='skip this many scenes')
    ap.add_argument('--out-root', default=str(OUT_ROOT))
    ap.add_argument('--meta-dir', default=str(NS_ROOT / 'v1.0-trainval'),
                    help='nuScenes metadata json directory')
    ap.add_argument('--data-root', default=str(NS_ROOT),
                    help='directory containing samples/ and sweeps/')
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # load scene list just to pick tokens
    scenes = json.load(open(Path(args.meta_dir) / 'scene.json'))
    scenes_sorted = sorted(scenes, key=lambda s: s['name'])
    picked = scenes_sorted[args.offset:args.offset + args.max_scenes]
    tokens = [s['token'] for s in picked]
    print(f'converting {len(tokens)} scenes → {out_root}', flush=True)

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
