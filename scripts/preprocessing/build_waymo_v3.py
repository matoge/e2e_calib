"""V3 full-image cache for Waymo (lcp-based clean projection).

Per (seg, frame_ts, cam_id) saves an inst .pt with:
  img         : (3, H, W) uint8 — JPEG decoded from camera_image parquet
  pts_cam     : (N, 3) float32 — pts in CAM frame (recovered from lcp uv+depth)
  uv_full     : (N, 2) float32 — lcp UVs (already pixel-clean per Waymo)
  z_cam       : (N,)   float32 — depth in cam frame
  K_full      : (3, 3) float32 — intrinsic
  IH, IW      : ints
  laser       : (N,)   int8    — lidar id per pt (1..5)
  seg, ts, cam_id                        — provenance

NO pose stored: perturbation at __getitem__ uses pt_cam directly:
    pt_cam_off = (R_off @ pt_cam.T + t_off).T
    uv_off = K @ pt_cam_off / pt_cam_off[:,2]

is_obj will be added later (needs cuboid → cam transform, deferred).

Usage:
  python build_waymo_v3.py --max-segs 5      # smoke
  python build_waymo_v3.py                   # full 798 segments
"""
import argparse, sys, time, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd, pyarrow.parquet as pq
import torch
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

from datasets.waymo_lcp import (CAM_NAMES, ALL_LASERS, ensure_lcp,
                                  read_lcp_at_ts, read_range_at_ts, per_cam_projections)
from datasets.waymo import WAYMO_DIR


def get_cam_intrinsics(seg_name: str):
    """Per-camera intrinsics for a segment. Returns dict[cam_id] -> K(3,3)."""
    cc = pd.read_parquet(WAYMO_DIR / 'camera_calibration' / f'{seg_name}.parquet')
    out = {}
    for _, r in cc.iterrows():
        cam_id = int(r['key.camera_name'])
        intr   = np.asarray(r['[CameraCalibrationComponent].intrinsic.f_u'])  # might be different
        # Try standard fields
        try:
            fu, fv = float(r['[CameraCalibrationComponent].intrinsic.f_u']), float(r['[CameraCalibrationComponent].intrinsic.f_v'])
            cu, cv = float(r['[CameraCalibrationComponent].intrinsic.c_u']), float(r['[CameraCalibrationComponent].intrinsic.c_v'])
            K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]], dtype=np.float32)
            out[cam_id] = K
        except KeyError:
            # Fallback: print actual columns
            print(f"calibration columns: {list(r.index)}")
            raise
    return out


def list_frame_timestamps(seg_name: str) -> list[int]:
    """All frame timestamps in a segment, from TOP_LIDAR (laser=1)."""
    df = pq.read_table(WAYMO_DIR / 'lidar' / f'{seg_name}.parquet',
                        columns=['key.frame_timestamp_micros'],
                        filters=[('key.laser_name', '=', 1)]).to_pandas()
    return sorted(df['key.frame_timestamp_micros'].unique().tolist())


def process_seg(args_tuple):
    """Optimized: read each parquet ONCE per segment, loop frames in memory."""
    (seg_name, out_dir, max_frames, gid_start, cams_keep, stride) = args_tuple
    out_dir = Path(out_dir)
    inst_dir = out_dir / 'inst'
    inst_dir.mkdir(parents=True, exist_ok=True)

    K_per_cam = get_cam_intrinsics(seg_name)
    lcp_path = ensure_lcp(seg_name)

    cam_filter = [('key.camera_name', 'in', list(cams_keep))]
    cam_df = pq.read_table(WAYMO_DIR / 'camera_image' / f'{seg_name}.parquet',
                            filters=cam_filter).to_pandas()
    cam_by_ts = {}
    for _, r in cam_df.iterrows():
        ts = int(r['key.frame_timestamp_micros'])
        cam_id = int(r['key.camera_name'])
        cam_by_ts.setdefault(ts, {})[cam_id] = bytes(r['[CameraImageComponent].image'])

    lidar_path = WAYMO_DIR / 'lidar' / f'{seg_name}.parquet'

    ts_list = sorted(cam_by_ts.keys())
    if stride > 1:
        ts_list = ts_list[::stride]
    if max_frames:
        ts_list = ts_list[:max_frames]

    written = 0
    gid = gid_start
    for ts in ts_list:
        lcp_arrs   = read_lcp_at_ts(lcp_path,   ts, lasers=ALL_LASERS)
        range_arrs = read_range_at_ts(lidar_path, ts, lasers=ALL_LASERS)
        per_cam    = per_cam_projections(lcp_arrs, range_arrs, cams=cams_keep)

        for cam_id, proj in per_cam.items():
            uv = proj['uv'].astype(np.float32)
            depth = proj['depth'].astype(np.float32)
            laser = proj['laser'].astype(np.int8)
            if cam_id not in cam_by_ts.get(ts, {}) or len(uv) < 16:
                continue
            jpg_bytes = cam_by_ts[ts][cam_id]
            img = np.array(Image.open(io.BytesIO(jpg_bytes)).convert('RGB'))
            IH, IW = img.shape[:2]
            K = K_per_cam[cam_id]
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            xc = (uv[:, 0] - cx) * depth / fx
            yc = (uv[:, 1] - cy) * depth / fy
            zc = depth
            pts_cam = np.stack([xc, yc, zc], axis=-1).astype(np.float32)

            # Pandaset-compat schema: pts treated as if in WORLD frame,
            # T_gt = identity (cam at origin). Perturbation logic still works.
            inst = dict(
                jpg_bytes = jpg_bytes,
                IH=int(IH), IW=int(IW),
                pts      = torch.from_numpy(pts_cam),                 # treat as world
                cam_pos  = torch.zeros(3, dtype=torch.float32),
                R_gt     = torch.eye(3, dtype=torch.float32),
                T_gt     = torch.eye(4, dtype=torch.float32),
                K_full   = torch.from_numpy(K),
                cuboids  = [],
                # legacy waymo extras (kept for vis convenience)
                pts_cam   = torch.from_numpy(pts_cam),
                uv_full   = torch.from_numpy(uv),
                z_cam     = torch.from_numpy(zc),
                laser     = torch.from_numpy(laser),
                # provenance
                scene = seg_name, cam = int(cam_id), frame = int(ts),
                seg=seg_name, ts=int(ts), cam_id=int(cam_id),
            )
            torch.save(inst, inst_dir / f'{gid:08d}.pt')
            gid += 1
            written += 1
    return seg_name, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',        default='/mnt/nvme6t/e2e_calib_cache/waymo_v3_full')
    ap.add_argument('--workers',    type=int, default=4)
    ap.add_argument('--max-segs',   type=int, default=None)
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--cams', default='1', help='comma-separated cam ids; default front-only "1"; all = "1,2,3,4,5"')
    ap.add_argument('--stride', type=int, default=1, help='frame stride (5 → 2Hz from 10Hz)')
    args = ap.parse_args()
    cams_keep = tuple(int(c) for c in args.cams.split(','))

    segs_dir = WAYMO_DIR / 'lidar'
    segs = sorted(f.stem for f in segs_dir.glob('*.parquet'))
    if args.max_segs:
        segs = segs[:args.max_segs]
    print(f'segments: {len(segs)} | out: {args.out} | workers: {args.workers}', flush=True)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    gid_stride = 5000   # ~200 frames × 5 cams = 1000 per seg, 5x buffer

    argv = [(seg, args.out, args.max_frames, i * gid_stride, cams_keep, args.stride)
            for i, seg in enumerate(segs)]
    t0 = time.time()
    written_total = 0
    if args.workers <= 1:
        for a in argv:
            seg, n = process_seg(a)
            written_total += n
            print(f"  {seg}: +{n}  total={written_total} ({time.time()-t0:.0f}s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_seg, a): a[0] for a in argv}
            done = 0
            for fut in as_completed(futs):
                seg, n = fut.result()
                written_total += n
                done += 1
                print(f"  [{done}/{len(segs)}] {seg}: +{n}  total={written_total} ({time.time()-t0:.0f}s)", flush=True)
    print(f"done: {written_total} insts → {args.out}", flush=True)

    # train/val split by segment (val = first 15%)
    out = Path(args.out)
    fnames = sorted(p.name for p in (out / 'inst').glob('*.pt'))
    import random
    rng = random.Random(42)
    seg_list = list(set(s for s in segs)); seg_list.sort(); rng.shuffle(seg_list)
    n_val_segs = max(1, int(len(seg_list) * 0.15))
    val_segs = set(seg_list[:n_val_segs])
    train_files, val_files = [], []
    for f in fnames:
        try: inst = torch.load(out / 'inst' / f, weights_only=False)
        except Exception: continue
        if inst.get('seg') in val_segs: val_files.append(f)
        else: train_files.append(f)
    torch.save({'train': train_files, 'val': val_files}, out / 'meta.pt')
    print(f'meta.pt: train={len(train_files)} val={len(val_files)}', flush=True)


if __name__ == '__main__':
    main()
