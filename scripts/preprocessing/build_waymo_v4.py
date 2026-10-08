"""Waymo Open Dataset v2 → calibration cache (same inst schema as the kamikado / woven / PandaSet caches).

Unlike build_waymo_v3.py this does NOT use lidar_camera_projection. v3 back-projected the LCP uv with the
LiDAR range used as the camera z (range ≥ z: median +4–7 %, up to +20 %), so translation-induced shifts were
under-estimated. Here the LiDAR points are real 3D points and the projection is our own pinhole, exactly as
for PandaSet / nuScenes:

  TOP LiDAR range image (return1) + lidar_calibration  → points in the vehicle frame at LiDAR time
  vehicle_pose (LiDAR time) and [CameraImageComponent].pose (vehicle pose at the camera shutter)
      + camera extrinsic (camera_calibration)          → points in the camera frame at shutter time
  pinhole K (camera_calibration f_u f_v c_u c_v)       → uv_full

Range-image decode and calibration loading are the helpers already verified in waymo_to_pandaset.py.
Lens distortion is not modelled (pinhole), and the per-pixel pose of the spinning LiDAR is not applied.

Per inst: pts (N,3) camera frame (OpenCV: x right, y down, z forward), uv_full, z_cam, intensity (range-image
channel 1, clipped to [0,1]), K_full, jpg_bytes (original JPEG), IW, IH, R_gt = I, cam_pos = 0, is_obj = 0.
Points kept: z > Z_MIN and up to FOV_PAD px outside the image (same rule as the other builders).
Splits: training/ → train, validation/ → val (different segments).

    python scripts/preprocessing/build_waymo_v4.py --src /mnt/ssd2t/work/waymo_v2 --out <cache> --stride 10
    python scripts/preprocessing/convert_tile_cache_to_lmdb.py --cache <cache>
"""
import argparse, io, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.preprocessing.waymo_to_pandaset import (_decode_range_image, _load_cam_intr_extr,  # noqa: E402
                                                      _load_lidar_cal, CAMERAS, TOP_LIDAR)

FOV_PAD = 256
Z_MIN = 0.1
GID_STRIDE = 10000          # gid block per segment (frames × cameras)


def _top_lidar_pts(row, incl, az_corr, T_veh_from_lid):
    """Range image return1 → (N,3) vehicle-frame points + (N,) intensity, same mask as _decode_range_image."""
    vals = row['[LiDARComponent].range_image_return1.values']
    shape = row['[LiDARComponent].range_image_return1.shape']
    pts_lid = _decode_range_image(vals, shape, incl, az_corr)
    ri = np.asarray(vals, np.float32).reshape(shape)
    inten = ri[:, :, 1][ri[:, :, 0] > 0].astype(np.float32)
    pts_veh = pts_lid @ T_veh_from_lid[:3, :3].T + T_veh_from_lid[:3, 3]
    return pts_veh.astype(np.float64), inten


def process_seg(args):
    split_dir, seg, out_dir, gid0, cams, stride, max_frames = args
    W = Path(split_dir)
    inst_dir = Path(out_dir) / 'inst'
    inst_dir.mkdir(parents=True, exist_ok=True)
    lcal = pd.read_parquet(W / 'lidar_calibration' / f'{seg}.parquet')
    T_veh_from_lid, incl, az_corr = _load_lidar_cal(lcal[lcal['key.laser_name'] == TOP_LIDAR].iloc[0])
    ccal = pd.read_parquet(W / 'camera_calibration' / f'{seg}.parquet')
    cam_cal = {int(r['key.camera_name']): _load_cam_intr_extr(r) for _, r in ccal.iterrows()}
    vpose = pd.read_parquet(W / 'vehicle_pose' / f'{seg}.parquet').set_index('key.frame_timestamp_micros')
    lid = pq.read_table(W / 'lidar' / f'{seg}.parquet',
                        columns=['key.frame_timestamp_micros', '[LiDARComponent].range_image_return1.values',
                                 '[LiDARComponent].range_image_return1.shape'],
                        filters=[('key.laser_name', '=', TOP_LIDAR)]).to_pandas().set_index('key.frame_timestamp_micros')
    img = pq.read_table(W / 'camera_image' / f'{seg}.parquet',
                        columns=['key.frame_timestamp_micros', 'key.camera_name', '[CameraImageComponent].image',
                                 '[CameraImageComponent].pose.transform'],
                        filters=[('key.camera_name', 'in', list(cams))]).to_pandas()
    ts_all = sorted(set(lid.index) & set(vpose.index) & set(img['key.frame_timestamp_micros']))[::stride]
    if max_frames:
        ts_all = ts_all[:max_frames]
    img = img.set_index(['key.frame_timestamp_micros', 'key.camera_name'])
    written, gid = [], gid0
    for fi, ts in enumerate(ts_all):
        pts_veh, inten = _top_lidar_pts(lid.loc[ts], incl, az_corr, T_veh_from_lid)
        T_world_from_veh = np.asarray(vpose.loc[ts, '[VehiclePoseComponent].world_from_vehicle.transform'],
                                      np.float64).reshape(4, 4)
        for cid in cams:
            if (ts, cid) not in img.index:
                continue
            r = img.loc[(ts, cid)]
            fu, fv, cu, cv, T_veh_from_wcam = cam_cal[cid]
            T_world_from_vshut = np.asarray(r['[CameraImageComponent].pose.transform'], np.float64).reshape(4, 4)
            T_wcam_from_veh = np.linalg.inv(T_world_from_vshut @ T_veh_from_wcam) @ T_world_from_veh
            P = pts_veh @ T_wcam_from_veh[:3, :3].T + T_wcam_from_veh[:3, 3]           # waymo cam: x fwd, y left, z up
            pc = np.stack([-P[:, 1], -P[:, 2], P[:, 0]], 1)                              # OpenCV cam
            jpg = bytes(r['[CameraImageComponent].image'])
            with Image.open(io.BytesIO(jpg)) as im:
                IW, IH = im.size
            z = pc[:, 2]
            zs = np.where(z > 1e-6, z, 1e-6)
            u = fu * pc[:, 0] / zs + cu
            v = fv * pc[:, 1] / zs + cv
            in_img = (z > 0.5) & (u >= 0) & (u < IW) & (v >= 0) & (v < IH)
            if int(in_img.sum()) < 64:
                continue
            keep = (z > Z_MIN) & (u >= -FOV_PAD) & (u < IW + FOV_PAD) & (v >= -FOV_PAD) & (v < IH + FOV_PAD)
            K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]], np.float32)
            inst = dict(
                cam_pos=torch.zeros(3), R_gt=torch.eye(3), T_gt=torch.eye(4), K_full=torch.from_numpy(K),
                cuboids=[], scene=seg, cam=CAMERAS[cid], frame=int(fi),
                jpg_bytes=jpg, IH=int(IH), IW=int(IW),
                pts=torch.from_numpy(pc[keep].astype(np.float32)),
                uv_full=torch.from_numpy(np.stack([u[keep], v[keep]], 1).astype(np.float32)),
                z_cam=torch.from_numpy(z[keep].astype(np.float32)),
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
    a = ap.parse_args()
    cams = tuple(int(c) for c in a.cams.split(','))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    jobs, meta = [], {'train': [], 'val': []}
    k = 0
    for split, key in [('training', 'train'), ('validation', 'val')]:
        cap = a.val_max_segs if (key == 'val' and a.val_max_segs is not None) else a.max_segs
        segs = sorted(p.stem for p in (Path(a.src) / split / 'lidar').glob('*.parquet'))[:cap]
        for seg in segs:
            jobs.append((key, (str(Path(a.src) / split), seg, str(out), k * GID_STRIDE, cams, a.stride, a.max_frames)))
            k += 1
    print(f'{len(jobs)} segments ({sum(j[0] == "train" for j in jobs)} train) cams={cams} stride={a.stride}', flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(process_seg, j[1]): j[0] for j in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            seg, written = f.result()
            meta[futs[f]].extend(written)
            print(f'  [{n}/{len(jobs)}] {futs[f]} {seg}: {len(written)} insts', flush=True)
    meta = {'train': sorted(meta['train']), 'val': sorted(meta['val']), 'cam': 'waymo_5cam', 'is_fisheye': False}
    torch.save(meta, out / 'meta.pt')
    print(f'meta.pt saved: train={len(meta["train"])} val={len(meta["val"])}', flush=True)


if __name__ == '__main__':
    main()
