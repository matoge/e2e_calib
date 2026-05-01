"""Inject 3D cuboids into existing Waymo V3 cache by reading lidar_box.parquet.

build_waymo_v3.py originally wrote cuboids=[] (3D box not loaded). This script
re-opens every cached inst, looks up matching boxes from
WAYMO_DIR/lidar_box/<seg>.parquet at (seg, ts), transforms each box from
vehicle frame to the camera frame (using camera_calibration extrinsic), and
saves an axis-aligned cam-frame AABB into inst['cuboids'].

The pandaset_full.py is_obj test uses yaw-rotated AABB membership; we set
yaw=0 here and over-bound by computing the AABB of the rotated box's 8
corners projected to cam frame. This adds a few percent of false positives
near vehicle corners but reliably tags vehicle points as obj — sufficient
for the obj/bg pivot 50/50 sampling that calib training depends on.
"""
import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pyarrow.parquet as pq
import torch

from datasets.waymo import WAYMO_DIR


def _box_corners_vehicle(pos: np.ndarray, dims: np.ndarray, yaw: float) -> np.ndarray:
    l, w, h = dims
    sx = np.array([+l/2, +l/2, -l/2, -l/2, +l/2, +l/2, -l/2, -l/2], dtype=np.float32)
    sy = np.array([+w/2, -w/2, +w/2, -w/2, +w/2, -w/2, +w/2, -w/2], dtype=np.float32)
    sz = np.array([+h/2, +h/2, +h/2, +h/2, -h/2, -h/2, -h/2, -h/2], dtype=np.float32)
    cy, sn = np.cos(yaw), np.sin(yaw)
    rx = cy * sx - sn * sy
    ry = sn * sx + cy * sy
    return pos[None, :] + np.stack([rx, ry, sz], axis=-1)  # (8, 3)


def _process_seg(args):
    seg, files, _ = args
    box_path = WAYMO_DIR / 'lidar_box' / f'{seg}.parquet'
    cc_path  = WAYMO_DIR / 'camera_calibration' / f'{seg}.parquet'
    if not box_path.exists() or not cc_path.exists():
        return seg, 0, 0
    box_df = pq.read_table(box_path).to_pandas()
    cc_df  = pq.read_table(cc_path).to_pandas()

    # cam_id → T_c_v (vehicle to cam): extrinsic.transform stores T_v_c
    T_c_v_per_cam = {}
    for _, r in cc_df.iterrows():
        cid = int(r['key.camera_name'])
        T_v_c = np.asarray(r['[CameraCalibrationComponent].extrinsic.transform']).reshape(4, 4)
        T_c_v_per_cam[cid] = np.linalg.inv(T_v_c).astype(np.float32)

    # Waymo's CAMERA frame convention is x=forward, y=left, z=up (axes aligned to
    # vehicle). build_waymo_v3.py recovers pts in cv2 convention from K @ p_cam:
    # z=forward, x=right, y=down. We need to bring cuboid coords into the SAME
    # cv2 convention so the pandaset_full is_obj test (cam-frame AABB) matches.
    M_w2cv = np.array([[ 0, -1,  0],   # cv2_x = -waymo_y  (right = -left)
                       [ 0,  0, -1],   # cv2_y = -waymo_z  (down  = -up)
                       [ 1,  0,  0]],  # cv2_z = +waymo_x  (forward stays)
                      dtype=np.float32)

    # Group boxes by ts
    box_by_ts = {}
    cols = ['[LiDARBoxComponent].box.center.x',
            '[LiDARBoxComponent].box.center.y',
            '[LiDARBoxComponent].box.center.z',
            '[LiDARBoxComponent].box.size.x',
            '[LiDARBoxComponent].box.size.y',
            '[LiDARBoxComponent].box.size.z',
            '[LiDARBoxComponent].box.heading']
    for _, r in box_df.iterrows():
        ts = int(r['key.frame_timestamp_micros'])
        cx, cy, cz, sx, sy, sz, h = r[cols]
        box_by_ts.setdefault(ts, []).append({
            'pos_v': np.array([cx, cy, cz], dtype=np.float32),
            'dims_v': np.array([sx, sy, sz], dtype=np.float32),
            'heading': float(h),
        })

    n_updated = 0
    n_boxes_total = 0
    for fpath in files:
        try: inst = torch.load(fpath, weights_only=False)
        except Exception: continue
        ts = int(inst.get('ts', inst.get('frame', -1)))
        cam_id = int(inst.get('cam_id', inst.get('cam', 0)))
        T_c_v = T_c_v_per_cam.get(cam_id)
        if T_c_v is None: continue
        boxes_at_ts = box_by_ts.get(ts, [])
        cuboids = []
        for b in boxes_at_ts:
            corners_v = _box_corners_vehicle(b['pos_v'], b['dims_v'], b['heading'])
            # corners → waymo-cam frame (x=fwd, y=left, z=up)
            ones = np.ones((corners_v.shape[0], 1), dtype=np.float32)
            corners_cam_w = (T_c_v @ np.concatenate([corners_v, ones], axis=-1).T).T[:, :3]
            # waymo-cam → cv2-cam (z=fwd, x=right, y=down) so coords match stored pts
            corners_cam = corners_cam_w @ M_w2cv.T
            # Skip boxes entirely behind the camera
            if corners_cam[:, 2].max() < 0.5: continue
            mn = corners_cam.min(axis=0); mx = corners_cam.max(axis=0)
            pos_cam = ((mn + mx) / 2.0).astype(np.float32)
            dims_cam = (mx - mn).astype(np.float32)
            # Skip behind-cam centers (likely ego-vehicle or cam-blocked)
            if pos_cam[2] < 0.5: continue
            cuboids.append({'pos': pos_cam, 'dims': dims_cam, 'yaw': 0.0})
        inst['cuboids'] = cuboids
        torch.save(inst, fpath)
        n_updated += 1
        n_boxes_total += len(cuboids)
    return seg, n_updated, n_boxes_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default='/mnt/nvme6t/e2e_calib_cache/waymo_v3_full')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    inst_dir = Path(args.cache) / 'inst'
    files = sorted(inst_dir.glob('*.pt'))
    print(f'cache: {args.cache}, {len(files)} insts to scan', flush=True)

    # Group files by seg (read each .pt header to get seg)
    print('grouping by seg...', flush=True)
    seg_to_files = {}
    for i, f in enumerate(files):
        try: inst = torch.load(f, weights_only=False)
        except Exception: continue
        seg = inst.get('seg', inst.get('scene'))
        if seg is None: continue
        seg_to_files.setdefault(seg, []).append(f)
        if (i + 1) % 5000 == 0:
            print(f'  scanned {i+1}/{len(files)}', flush=True)
    print(f'segs: {len(seg_to_files)}', flush=True)

    argv = [(seg, fs, None) for seg, fs in seg_to_files.items()]
    import time; t0 = time.time()
    n_done_segs = 0; n_inst_updated = 0; n_box_total = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process_seg, a) for a in argv]
        for fut in as_completed(futs):
            seg, n_upd, n_box = fut.result()
            n_done_segs += 1; n_inst_updated += n_upd; n_box_total += n_box
            if n_done_segs % 10 == 0 or n_done_segs == len(argv):
                print(f'  [{n_done_segs}/{len(argv)}]  insts updated={n_inst_updated}  '
                      f'boxes injected={n_box_total}  ({time.time()-t0:.0f}s)', flush=True)
    print(f'done: {n_inst_updated} insts updated, {n_box_total} boxes total '
          f'(avg {n_box_total/max(1,n_inst_updated):.1f} per inst)', flush=True)


if __name__ == '__main__':
    main()
