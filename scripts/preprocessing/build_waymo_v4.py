"""Waymo Open Dataset v2 → calibration cache (same inst schema as the kamikado / woven / PandaSet caches).

uv comes from Waymo's own projection, the same chain that produced lidar_camera_projection (cp points):

  TOP LiDAR range image return1 + lidar_calibration + lidar_pose (vehicle pose per LiDAR pixel) + vehicle_pose
      → v2 lidar_utils.convert_range_image_to_cartesian(pixel_pose, frame_pose)   points, vehicle frame @ frame time
      → frame pose                                                                world points (float64)
  camera_calibration (f, c, k1 k2 p1 p2 k3, extrinsic, rolling_shutter_direction)
      + [CameraImageComponent] pose / velocity / pose_timestamp / rolling_shutter_params
      → py_camera_model_ops.world_to_image                                        uv with distortion + rolling shutter

Checked against cp on 5 segments × 5 cameras: median 1.15–1.57 px, p90 ≤ 2.84 px (GLOBAL_SHUTTER instead: up to 25 px).

Training is pinhole with one pose per image, so each point is stored as the 3D point that the pinhole K projects
exactly onto Waymo's uv: pts = z · K⁻¹ [u, v, 1], z = depth of the point in the camera at pose_timestamp.
Distortion and rolling shutter are folded into the point's direction; K·pts reproduces uv_full.

Per inst: pts (N,3) camera frame (OpenCV: x right, y down, z forward), uv_full, z_cam, intensity (range-image
channel 1, clipped to [0,1]), K_full (f_u f_v c_u c_v), jpg_bytes (original JPEG), IW, IH, R_gt = I, cam_pos = 0, is_obj = 0.
Points kept: world_to_image ok, z > max(Z_MIN, --min-depth, default 2 m), up to FOV_PAD px outside the image.
Splits: training/ → train, validation/ → val (different segments).

Runs in the `waymo` conda env (tensorflow + waymo-open-dataset-tf-2-12-0 + torch cpu):
    python scripts/preprocessing/build_waymo_v4.py --src /mnt/ssd2t/work/waymo_v2 --out <cache> --stride 10
    python scripts/preprocessing/convert_tile_cache_to_lmdb.py --cache <cache>      (neurad env)
"""
import argparse, io
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

FOV_PAD = 256
Z_MIN = 0.1
GID_STRIDE = 10000          # gid block per segment (frames × cameras)
TOP_LIDAR = 1
CAMERAS = {1: 'FRONT', 2: 'FRONT_LEFT', 3: 'FRONT_RIGHT', 4: 'SIDE_LEFT', 5: 'SIDE_RIGHT'}
CI, CC = '[CameraImageComponent].', '[CameraCalibrationComponent].'
INTR = ['f_u', 'f_v', 'c_u', 'c_v', 'k1', 'k2', 'p1', 'p2', 'k3']


def process_seg(args):
    split_dir, seg, out_dir, gid0, cams, stride, max_frames, min_depth = args
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    from waymo_open_dataset import v2
    from waymo_open_dataset.v2.perception import lidar as v2_lidar
    from waymo_open_dataset.v2.perception.utils import lidar_utils
    from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops

    W = Path(split_dir)
    inst_dir = Path(out_dir) / 'inst'
    inst_dir.mkdir(parents=True, exist_ok=True)
    rd = lambda c, **k: pq.read_table(W / c / f'{seg}.parquet', **k).to_pandas()
    top = [('key.laser_name', '=', TOP_LIDAR)]
    lcal = v2.LiDARCalibrationComponent.from_dict(rd('lidar_calibration', filters=top).iloc[0])
    ccal = rd('camera_calibration').set_index('key.camera_name')
    vpose = rd('vehicle_pose').set_index('key.frame_timestamp_micros', drop=False)
    lid_ts = set(pq.read_table(W / 'lidar' / f'{seg}.parquet', columns=['key.frame_timestamp_micros'],
                               filters=top).column(0).to_pylist())
    img_cols = [c for c in pq.read_schema(W / 'camera_image' / f'{seg}.parquet').names]
    img_ts = set(pq.read_table(W / 'camera_image' / f'{seg}.parquet', columns=['key.frame_timestamp_micros']).column(0).to_pylist())
    ts_all = sorted(lid_ts & set(vpose.index) & img_ts)[::stride]
    if max_frames:
        ts_all = ts_all[:max_frames]
    # one read per component for just these frames (a read per frame grew pyarrow's memory to 17 GB → OOM)
    sel = top + [('key.frame_timestamp_micros', 'in', ts_all)]
    key = ['key.segment_context_name', 'key.frame_timestamp_micros', 'key.laser_name']
    lid = rd('lidar', filters=sel, columns=key + [f'[LiDARComponent].range_image_return1.{x}' for x in ('values', 'shape')]
             ).set_index('key.frame_timestamp_micros', drop=False)
    lpo = rd('lidar_pose', filters=sel, columns=key + [f'[LiDARPoseComponent].range_image_return1.{x}' for x in ('values', 'shape')]
             ).set_index('key.frame_timestamp_micros', drop=False)
    img = rd('camera_image', columns=img_cols, filters=[('key.frame_timestamp_micros', 'in', ts_all),
                                                        ('key.camera_name', 'in', list(cams))])
    written, gid = [], gid0
    for fi, ts in enumerate(ts_all):
        # range images built directly: from_dict wants the return2 columns we do not read
        ri1 = v2_lidar.RangeImage(values=lid.loc[ts, '[LiDARComponent].range_image_return1.values'],
                                  shape=lid.loc[ts, '[LiDARComponent].range_image_return1.shape'])
        pri = v2_lidar.PoseRangeImage(values=lpo.loc[ts, '[LiDARPoseComponent].range_image_return1.values'],
                                      shape=lpo.loc[ts, '[LiDARPoseComponent].range_image_return1.shape'])
        fpose = v2.VehiclePoseComponent.from_dict(vpose.loc[ts])
        xyz = lidar_utils.convert_range_image_to_cartesian(ri1, lcal, pri,
                                                            fpose).numpy().astype(np.float64)
        ri = ri1.tensor.numpy()
        m = ri[..., 0] > 0
        inten = ri[..., 1][m].astype(np.float32)
        T_world_from_veh = np.asarray(fpose.world_from_vehicle.transform, np.float64).reshape(4, 4)
        pw = xyz[m] @ T_world_from_veh[:3, :3].T + T_world_from_veh[:3, 3]
        imgs = img[img['key.frame_timestamp_micros'] == ts].set_index('key.camera_name')
        for cid in cams:
            if cid not in imgs.index:
                continue
            r, c = imgs.loc[cid], ccal.loc[cid]
            T_veh_from_wcam = np.asarray(c[CC + 'extrinsic.transform'], np.float64).reshape(4, 4)
            cim = (list(r[CI + 'pose.transform'])
                   + [r[CI + f'velocity.{a}_velocity.{x}'] for a in ('linear', 'angular') for x in 'xyz']
                   + [r[CI + 'pose_timestamp'], r[CI + 'rolling_shutter_params.shutter'],
                      r[CI + 'rolling_shutter_params.camera_trigger_time'],
                      r[CI + 'rolling_shutter_params.camera_readout_done_time']])
            uvo = py_camera_model_ops.world_to_image(
                tf.constant(T_veh_from_wcam, tf.float64),
                tf.constant([c[CC + 'intrinsic.' + k] for k in INTR], tf.float64),
                tf.constant([c[CC + 'width'], c[CC + 'height'], c[CC + 'rolling_shutter_direction']], tf.int32),
                tf.constant(cim, tf.float64), tf.constant(pw, tf.float64)).numpy()
            u, v, ok = uvo[:, 0], uvo[:, 1], uvo[:, 2] > 0
            # depth: the point in the camera at pose_timestamp (waymo cam: x fwd, y left, z up)
            T_wcam_from_world = np.linalg.inv(np.asarray(r[CI + 'pose.transform'], np.float64).reshape(4, 4) @ T_veh_from_wcam)
            z = (pw @ T_wcam_from_world[:3, :3].T + T_wcam_from_world[:3, 3])[:, 0]
            jpg = bytes(r[CI + 'image'])
            with Image.open(io.BytesIO(jpg)) as im:
                IW, IH = im.size
            in_img = ok & (z > 0.5) & (u >= 0) & (u < IW) & (v >= 0) & (v < IH)
            if int(in_img.sum()) < 64:
                continue
            keep = ok & (z > max(Z_MIN, min_depth)) & (u >= -FOV_PAD) & (u < IW + FOV_PAD) & (v >= -FOV_PAD) & (v < IH + FOV_PAD)
            fu, fv, cu, cv = (float(c[CC + 'intrinsic.' + k]) for k in INTR[:4])
            zk = z[keep]
            pc = np.stack([(u[keep] - cu) / fu * zk, (v[keep] - cv) / fv * zk, zk], 1)       # K · pc = (u, v)
            K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]], np.float32)
            inst = dict(
                cam_pos=torch.zeros(3), R_gt=torch.eye(3), T_gt=torch.eye(4), K_full=torch.from_numpy(K),
                cuboids=[], scene=seg, cam=CAMERAS[cid], frame=int(fi),
                jpg_bytes=jpg, IH=int(IH), IW=int(IW),
                pts=torch.from_numpy(pc.astype(np.float32)),
                uv_full=torch.from_numpy(np.stack([u[keep], v[keep]], 1).astype(np.float32)),
                z_cam=torch.from_numpy(zk.astype(np.float32)),
                is_obj=torch.zeros(int(keep.sum())),
                intensity=torch.from_numpy(np.clip(inten[keep], 0.0, 1.0).astype(np.float32)),
            )
            fname = f'{gid:08d}.pt'
            torch.save(inst, inst_dir / fname)
            written.append(fname)
            gid += 1
    return seg, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='root holding training/ and validation/ (Waymo v2 parquet)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cams', default='1,2,3,4,5')
    ap.add_argument('--stride', type=int, default=10, help='frame stride (10 → 1 Hz from 10 Hz)')
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--max-segs', type=int, default=None)
    ap.add_argument('--val-max-segs', type=int, default=None, help='cap on validation segments (default: --max-segs)')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--min-depth', type=float, default=2.0,
                    help='drop points closer than this camera-frame z [m]: at z = 2 m a 0.2 m translation already moves a\n'
                         'point ~200 px (f ~ 2000 px), and near returns include the ego body')
    a = ap.parse_args()
    cams = tuple(int(c) for c in a.cams.split(','))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs, meta = [], {'train': [], 'val': []}
    k = 0
    for split, key in [('training', 'train'), ('validation', 'val')]:
        cap = a.val_max_segs if (key == 'val' and a.val_max_segs is not None) else a.max_segs
        segs = sorted(p.stem for p in (Path(a.src) / split / 'lidar').glob('*.parquet'))[:cap]
        for seg in segs:
            jobs.append((key, (str(Path(a.src) / split), seg, str(out), k * GID_STRIDE, cams, a.stride, a.max_frames,
                               a.min_depth)))
            k += 1
    print(f'{len(jobs)} segments ({sum(j[0] == "train" for j in jobs)} train) cams={cams} stride={a.stride}', flush=True)
    split_of = {j[1][1]: j[0] for j in jobs}
    # a fresh process per segment: pyarrow keeps the ~9 GB peak of a segment's reads, and it piled up across segments
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for n, (seg, written) in enumerate(pool.imap_unordered(process_seg, [j[1] for j in jobs]), 1):
            meta[split_of[seg]].extend(written)
            print(f'  [{n}/{len(jobs)}] {split_of[seg]} {seg}: {len(written)} insts', flush=True)
    meta = {'train': sorted(meta['train']), 'val': sorted(meta['val']), 'cam': 'waymo_5cam', 'is_fisheye': False}
    torch.save(meta, out / 'meta.pt')
    print(f'meta.pt saved: train={len(meta["train"])} val={len(meta["val"])}', flush=True)


if __name__ == '__main__':
    main()
