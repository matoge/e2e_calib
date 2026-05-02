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

Cuboids are read from lidar_box.parquet, transformed vehicle → waymo cam
→ cv2 cam (to match stored pts frame), and stored as axis-aligned AABB
(yaw=0) in cam frame. Matches scripts/preprocessing/inject_waymo_cuboids.py
logic so build + inject produce identical output.

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
                                  read_lcp_all, read_range_all, per_cam_projections)
from datasets.waymo import WAYMO_DIR


def get_cam_calib(seg_name: str):
    """Per-camera intrinsics + extrinsics for a segment.

    Returns dict[cam_id] -> (K(3,3), T_c_v(4,4)) where T_c_v is vehicle → waymo cam.
    The stored extrinsic is T_v_c (waymo cam → vehicle), so we invert.
    """
    cc = pd.read_parquet(WAYMO_DIR / 'camera_calibration' / f'{seg_name}.parquet')
    out = {}
    for _, r in cc.iterrows():
        cam_id = int(r['key.camera_name'])
        fu = float(r['[CameraCalibrationComponent].intrinsic.f_u'])
        fv = float(r['[CameraCalibrationComponent].intrinsic.f_v'])
        cu = float(r['[CameraCalibrationComponent].intrinsic.c_u'])
        cv = float(r['[CameraCalibrationComponent].intrinsic.c_v'])
        K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]], dtype=np.float32)
        T_v_c = np.asarray(r['[CameraCalibrationComponent].extrinsic.transform']).reshape(4, 4)
        T_c_v = np.linalg.inv(T_v_c).astype(np.float32)
        out[cam_id] = (K, T_c_v)
    return out


def _box_corners_vehicle(pos: np.ndarray, dims: np.ndarray, yaw: float) -> np.ndarray:
    """Official Waymo corner ordering (matches box_utils.get_upright_3d_box_corners):
    bottom face CCW (-h2): [+l,+w], [-l,+w], [-l,-w], [+l,-w]
    top face CCW (+h2):    same xy ordering.
    """
    l, w, h = dims
    l2, w2, h2 = l / 2, w / 2, h / 2
    cor = np.array([[ l2,  w2, -h2], [-l2,  w2, -h2], [-l2, -w2, -h2], [ l2, -w2, -h2],
                    [ l2,  w2,  h2], [-l2,  w2,  h2], [-l2, -w2,  h2], [ l2, -w2,  h2]],
                   dtype=np.float32)
    cy, sn = np.cos(yaw), np.sin(yaw)
    R = np.array([[cy, -sn, 0], [sn, cy, 0], [0, 0, 1]], dtype=np.float32)
    return (R @ cor.T).T + pos[None, :]


# waymo cam (x=fwd, y=left, z=up) → cv2 cam (x=right, y=down, z=fwd)
# cv2_x = -waymo_y, cv2_y = -waymo_z, cv2_z = +waymo_x.
_M_W2CV = np.array([[ 0, -1,  0],
                    [ 0,  0, -1],
                    [ 1,  0,  0]], dtype=np.float32)


def read_boxes_by_ts(seg_name: str) -> dict:
    """Read lidar_box.parquet once per seg, grouped by ts."""
    box_path = WAYMO_DIR / 'lidar_box' / f'{seg_name}.parquet'
    if not box_path.exists():
        return {}
    box_df = pq.read_table(box_path).to_pandas()
    cols = ['[LiDARBoxComponent].box.center.x',
            '[LiDARBoxComponent].box.center.y',
            '[LiDARBoxComponent].box.center.z',
            '[LiDARBoxComponent].box.size.x',
            '[LiDARBoxComponent].box.size.y',
            '[LiDARBoxComponent].box.size.z',
            '[LiDARBoxComponent].box.heading']
    out = {}
    for _, r in box_df.iterrows():
        ts = int(r['key.frame_timestamp_micros'])
        cx, cy, cz, sx, sy, sz, h = r[cols]
        out.setdefault(ts, []).append({
            'pos_v':   np.array([cx, cy, cz], dtype=np.float32),
            'dims_v':  np.array([sx, sy, sz], dtype=np.float32),
            'heading': float(h),
        })
    return out


def boxes_cam_frame(boxes_at_ts: list, T_c_v: np.ndarray) -> list:
    """Transform vehicle-frame boxes → cv2-cam-frame AABB (yaw=0) list.

    Matches scripts/preprocessing/inject_waymo_cuboids.py exactly.
    """
    cuboids = []
    for b in boxes_at_ts:
        corners_v = _box_corners_vehicle(b['pos_v'], b['dims_v'], b['heading'])
        ones = np.ones((corners_v.shape[0], 1), dtype=np.float32)
        corners_cam_w = (T_c_v @ np.concatenate([corners_v, ones], axis=-1).T).T[:, :3]
        # Skip boxes with any corner behind (or too near) the camera plane
        if (corners_cam_w[:, 0] <= 0.5).any():
            continue
        corners_cam = corners_cam_w @ _M_W2CV.T
        if corners_cam[:, 2].max() < 0.5:
            continue
        mn = corners_cam.min(axis=0); mx = corners_cam.max(axis=0)
        pos_cam  = ((mn + mx) / 2.0).astype(np.float32)
        dims_cam = (mx - mn).astype(np.float32)
        if pos_cam[2] < 0.5:
            continue
        cuboids.append({'pos': pos_cam, 'dims': dims_cam, 'yaw': 0.0})
    return cuboids


def list_frame_timestamps(seg_name: str) -> list[int]:
    """All frame timestamps in a segment, from TOP_LIDAR (laser=1)."""
    df = pq.read_table(WAYMO_DIR / 'lidar' / f'{seg_name}.parquet',
                        columns=['key.frame_timestamp_micros'],
                        filters=[('key.laser_name', '=', 1)]).to_pandas()
    return sorted(df['key.frame_timestamp_micros'].unique().tolist())


def process_seg(args_tuple):
    """Optimized: read each parquet ONCE per segment, loop frames in memory."""
    (seg_name, out_dir, max_frames, gid_start, cams_keep, stride,
     tile_layout) = args_tuple
    out_dir = Path(out_dir)
    inst_dir = out_dir / 'inst'
    inst_dir.mkdir(parents=True, exist_ok=True)

    calib_per_cam = get_cam_calib(seg_name)            # cam_id → (K, T_c_v)
    boxes_by_ts   = read_boxes_by_ts(seg_name)          # ts → [box dicts]
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

    # Read entire LCP + lidar parquets ONCE per seg (was: re-open + filter scan
    # per ts × 20 frames = ~8s/seg of metadata overhead). Now ~1-2s/seg total.
    lcp_by_ts   = read_lcp_all(lcp_path,   lasers=ALL_LASERS)
    range_by_ts = read_range_all(lidar_path, lasers=ALL_LASERS)

    ts_list = sorted(cam_by_ts.keys())
    if stride > 1:
        ts_list = ts_list[::stride]
    if max_frames:
        ts_list = ts_list[:max_frames]

    written = 0
    gid = gid_start
    for ts in ts_list:
        lcp_arrs   = lcp_by_ts.get(ts, {})
        range_arrs = range_by_ts.get(ts, {})
        if not lcp_arrs or not range_arrs:
            continue
        per_cam    = per_cam_projections(lcp_arrs, range_arrs, cams=cams_keep)

        for cam_id, proj in per_cam.items():
            uv = proj['uv'].astype(np.float32)
            depth = proj['depth'].astype(np.float32)
            laser = proj['laser'].astype(np.int8)
            if cam_id not in cam_by_ts.get(ts, {}) or len(uv) < 16:
                continue
            jpg_bytes = cam_by_ts[ts][cam_id]
            # Skip full JPEG decode; PIL.Image.open is lazy — .size reads only
            # the header for IW/IH. Was: 50ms × 5 cam × 20 frame = 5s/seg wasted.
            with Image.open(io.BytesIO(jpg_bytes)) as _im:
                IW, IH = _im.size
            K, T_c_v = calib_per_cam[cam_id]
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            xc = (uv[:, 0] - cx) * depth / fx
            yc = (uv[:, 1] - cy) * depth / fy
            zc = depth
            pts_cam = np.stack([xc, yc, zc], axis=-1).astype(np.float32)

            # Transform vehicle-frame boxes at this ts → cv2-cam-frame AABBs
            cuboids = boxes_cam_frame(boxes_by_ts.get(ts, []), T_c_v)

            # is_obj on visible pts (cuboid AABB membership in cam frame)
            from datasets.pandaset_full import _is_obj_per_point
            is_obj_vis = _is_obj_per_point(pts_cam, cuboids).astype(np.float32)

            common_inst = dict(
                cam_pos  = torch.zeros(3, dtype=torch.float32),
                R_gt     = torch.eye(3, dtype=torch.float32),
                T_gt     = torch.eye(4, dtype=torch.float32),
                K_full   = torch.from_numpy(K),
                cuboids  = cuboids,
                scene = seg_name, cam = int(cam_id), frame = int(ts),
                seg=seg_name, ts=int(ts), cam_id=int(cam_id),
            )

            if tile_layout is None:
                inst = dict(common_inst)
                inst.update(dict(
                    jpg_bytes = jpg_bytes,
                    IH=int(IH), IW=int(IW),
                    pts      = torch.from_numpy(pts_cam),
                    pts_cam  = torch.from_numpy(pts_cam),
                    uv_full  = torch.from_numpy(uv),
                    z_cam    = torch.from_numpy(zc),
                    is_obj   = torch.from_numpy(is_obj_vis),
                    laser    = torch.from_numpy(laser),
                ))
                torch.save(inst, inst_dir / f'{gid:08d}.pt')
                gid += 1
                written += 1
            else:
                from scripts.preprocessing._tile_split import cut_inst_to_tiles
                tw, th, st, pad, y0, q = tile_layout
                tile_files = cut_inst_to_tiles(
                    jpg_bytes=jpg_bytes, IW=int(IW), IH=int(IH),
                    pts_vis=pts_cam, uv_vis=uv, z_vis=zc,
                    is_obj_vis=is_obj_vis, common_inst=common_inst,
                    tile_w=tw, tile_h=th, stride=st, pad_px=pad,
                    y_start=y0, jpg_quality=q,
                    out_dir=inst_dir, gid_base=gid)
                gid += 1
                written += len(tile_files)
    return seg_name, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',        default='/mnt/nvme6t/e2e_calib_cache/waymo_v3_full')
    ap.add_argument('--workers',    type=int, default=4)
    ap.add_argument('--max-segs',   type=int, default=None)
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--cams', default='1', help='comma-separated cam ids; default front-only "1"; all = "1,2,3,4,5"')
    ap.add_argument('--stride', type=int, default=1, help='frame stride (5 → 2Hz from 10Hz)')
    ap.add_argument('--tile', action='store_true')
    ap.add_argument('--tile-w',       type=int, default=512)
    ap.add_argument('--tile-h',       type=int, default=512)
    ap.add_argument('--tile-stride',  type=int, default=384)
    ap.add_argument('--tile-pad',     type=int, default=64)
    ap.add_argument('--tile-y-start', type=int, default=200,
                    help='Waymo front-cam 1280 tall — skip ~200 sky')
    ap.add_argument('--tile-jpg-q',   type=int, default=90)
    args = ap.parse_args()
    tile_layout = None
    if args.tile:
        tile_layout = (args.tile_w, args.tile_h, args.tile_stride,
                       args.tile_pad, args.tile_y_start, args.tile_jpg_q)
        print(f'TILE mode: tile={args.tile_w}×{args.tile_h} stride={args.tile_stride} '
              f'pad={args.tile_pad} y_start={args.tile_y_start}', flush=True)
    cams_keep = tuple(int(c) for c in args.cams.split(','))

    segs_dir = WAYMO_DIR / 'lidar'
    segs = sorted(f.stem for f in segs_dir.glob('*.parquet'))
    if args.max_segs:
        segs = segs[:args.max_segs]
    print(f'segments: {len(segs)} | out: {args.out} | workers: {args.workers}', flush=True)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    gid_stride = 5000   # ~200 frames × 5 cams = 1000 per seg, 5x buffer

    argv = [(seg, args.out, args.max_frames, i * gid_stride, cams_keep, args.stride,
             tile_layout)
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
