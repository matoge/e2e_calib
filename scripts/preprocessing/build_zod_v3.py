"""V3 tile cache for ZOD Frames (front cam, VLS-128, Kannala-Brandt fisheye).

Per ZOD frame produces tile insts with:
  jpg_bytes      : tile JPEG (re-encoded, lossy)
  IH, IW         : tile dims
  tile_u0, tile_v0 : tile origin in parent image
  tile_id
  pts            : (N, 3) float32 — lidar pts in CAM frame (single core sweep,
                                    no per-shot motion comp on this build)
  uv_full        : (N, 2) float32 — projected pixel UVs via Kannala-Brandt
                                    (in parent-image coords; subtract tile_u0,v0)
  z_cam          : (N,)   float32
  is_obj         : (N,)   float32
  in_box         : (N,)   float32
  K_full         : (3, 4) float32 — fisheye K (fx, fy, cx, cy)
  distortion     : (4,)   float32 — Kannala k1..k4
  is_fisheye     : True   bool
  cuboids        : list  of dicts (pos, dims, yaw) in CAM frame
  cam_pos, R_gt, T_gt — provenance (R_gt = identity, T_gt = identity)
  scene = frame_id, cam = 'front', frame = int(frame_id)

Design — workers DO NOT import zod SDK to avoid 1-2GB info DB load per worker
(that path OOMed earlier with 8 workers × ZodFrames(version='full') = 12GB+).
Workers read directly from frame_dir/calibration.json + lidar_velodyne/*.npy +
camera_front_dnat/*.jpg + annotations/object_detection.json.

Usage:
  python build_zod_v3.py --max-frames 100        # smoke
  python build_zod_v3.py --tile --max-frames 1000  # 1K
  python build_zod_v3.py --tile                  # all extracted
"""
import argparse, sys, time, io, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch

ZOD_ROOT  = Path('/mnt/nvme6t/zod/frames')
CACHE_OUT = Path('/mnt/nvme6t/e2e_calib_cache/zod_v3_tiled')


def _project_kannala(pts_cam: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Kannala-Brandt forward projection. pts_cam: (N,3) cam frame, x=right y=down z=forward."""
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(r, np.maximum(z, 1e-6))
    k1, k2, k3, k4 = dist
    t2 = theta * theta
    theta_d = theta * (1.0 + k1 * t2 + k2 * t2 ** 2 + k3 * t2 ** 3 + k4 * t2 ** 4)
    r_safe = np.where(r > 1e-9, r, 1.0)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = fx * (theta_d * x / r_safe) + cx
    v = fy * (theta_d * y / r_safe) + cy
    return np.stack([u, v], axis=-1).astype(np.float32)


def _is_obj_per_point(pts_cam: np.ndarray, cuboids: list) -> np.ndarray:
    N = len(pts_cam)
    if not cuboids or N == 0:
        return np.zeros(N, dtype=np.float32)
    M = len(cuboids)
    poss = np.stack([np.asarray(c['pos'],  dtype=np.float32) for c in cuboids])
    dims = np.stack([np.asarray(c['dims'], dtype=np.float32) for c in cuboids])
    yaws = np.fromiter((float(c['yaw']) for c in cuboids), dtype=np.float32, count=M)
    cy_, sy_ = np.cos(yaws), np.sin(yaws)
    R = np.zeros((M, 3, 3), dtype=np.float32)
    R[:, 0, 0] = cy_;  R[:, 0, 1] = sy_
    R[:, 1, 0] = -sy_; R[:, 1, 1] = cy_
    R[:, 2, 2] = 1.0
    delta = pts_cam.astype(np.float32, copy=False)[None, :, :] - poss[:, None, :]
    local = np.einsum('mij,mnj->mni', R, delta)
    half = (dims * 0.5)[:, None, :]
    inside = np.all(np.abs(local) <= half, axis=-1)
    return inside.any(axis=0).astype(np.float32)


def _frame_motion_metrics(frame_dir: Path):
    """Return (yaw_rate_deg_per_s, speed_kmh) at scan center.
    Returns (None, None) on missing data."""
    try:
        from scipy.spatial.transform import Rotation
        em = json.loads((frame_dir / "ego_motion.json").read_text())
        ts = np.array(em['timestamps'])
        poses = np.array(em['poses'])
        vels = np.array(em['velocities'])
        rots = Rotation.from_matrix(poses[:, :3, :3])
        yaws = rots.as_euler('zyx', degrees=True)[:, 0]
        mid = len(ts) // 2
        if mid < 1 or mid+1 >= len(ts): return None, None
        yaw_rate = abs(yaws[mid+1] - yaws[mid-1]) / max(ts[mid+1] - ts[mid-1], 1e-6)
        speed_kmh = float(np.sqrt(vels[mid, 0]**2 + vels[mid, 1]**2)) * 3.6
        return float(yaw_rate), float(speed_kmh)
    except Exception:
        return None, None


def _load_calib(frame_dir: Path):
    """Read calibration.json directly. Returns (K, dist, T_vc, T_vl, IW, IH)."""
    calib = json.loads((frame_dir / "calibration.json").read_text())["FC"]
    K = np.asarray(calib["intrinsics"], dtype=np.float32)[:3, :3]
    dist = np.asarray(calib["distortion"], dtype=np.float32)
    T_vc = np.asarray(calib["extrinsics"], dtype=np.float32)
    T_vl = np.asarray(calib["lidar_extrinsics"], dtype=np.float32)
    IW = int(calib["image_dimensions"][0])
    IH = int(calib["image_dimensions"][1])
    return K, dist, T_vc, T_vl, IW, IH


def _load_lidar(frame_dir: Path):
    """Single core sweep → (pts (N,3) xyz lidar local, ts (N,) per-point us int).
    Returns (None, None) on missing data."""
    lidars = sorted((frame_dir / "lidar_velodyne").glob("*.npy"))
    if not lidars:
        return None, None
    sweep = np.load(lidars[len(lidars) // 2], allow_pickle=False)
    if sweep.dtype.names and 'x' in sweep.dtype.names:
        pts = np.stack([sweep['x'], sweep['y'], sweep['z']], axis=1).astype(np.float32)
        # per-point timestamp: integer microseconds, RELATIVE to camera shutter time
        ts = sweep['timestamp'].astype(np.int64) if 'timestamp' in sweep.dtype.names else None
    else:
        pts = sweep[:, :3].astype(np.float32)
        ts = None
    return pts, ts


def _ego_motion_apply(pts_lidar, pt_ts_us, frame_dir, T_vl, cam_ts_unix):
    """Per-point motion compensation: bring pts (lidar local) to ego frame at cam_ts.

    Args:
      pts_lidar: (N, 3) lidar-local xyz
      pt_ts_us: (N,) int per-point ts (us, RELATIVE to cam shutter)
      frame_dir: ZOD frame dir
      T_vl: 4x4 vehicle←lidar extrinsic
      cam_ts_unix: cam shutter ts (Unix seconds, float)
    Returns:
      pts_veh_at_cam_ts: (N, 3) — lidar pts in vehicle frame at cam_ts
    """
    from scipy.spatial.transform import Rotation, Slerp
    em = json.loads((frame_dir / "ego_motion.json").read_text())
    poses = np.array(em['poses'])
    ts_em = np.array(em['timestamps'])
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    translations = poses[:, :3, 3]
    slerp = Slerp(ts_em, rotations)

    def ego_world_pose(t):
        t_c = np.clip(t, ts_em[0], ts_em[-1])
        R = slerp(t_c).as_matrix()
        tr = np.array([np.interp(t_c, ts_em, translations[:, i]) for i in range(3)])
        P = np.eye(4); P[:3, :3] = R; P[:3, 3] = tr
        return P

    P_we_cam = ego_world_pose(cam_ts_unix)
    P_we_cam_inv = np.linalg.inv(P_we_cam)

    # Bin per-point ts to 1ms resolution to limit slerp calls
    pt_abs_ts = cam_ts_unix + pt_ts_us.astype(np.float64) * 1e-6
    binned = np.round(pt_abs_ts * 1000.0) * 1e-3
    unique_ts, inv_idx = np.unique(binned, return_inverse=True)
    deltas = np.zeros((len(unique_ts), 4, 4), dtype=np.float64)
    for i, t in enumerate(unique_ts):
        deltas[i] = P_we_cam_inv @ ego_world_pose(t)

    # ego at scan time = T_vl @ pt_lidar
    homo = np.column_stack([pts_lidar, np.ones(len(pts_lidar), dtype=np.float64)])
    pts_e_at_t = (T_vl @ homo.T).T  # (N, 4) ego frame at the per-point scan time
    deltas_per_pt = deltas[inv_idx]  # (N, 4, 4)
    pts_veh_at_cam = np.einsum('nij,nj->ni', deltas_per_pt, pts_e_at_t)[:, :3]
    return pts_veh_at_cam.astype(np.float32)


def _camera_shutter_ts(frame_dir: Path) -> float:
    """Parse Unix shutter ts from camera_front_dnat filename ISO part."""
    import re, datetime
    img = sorted((frame_dir / "camera_front_dnat").glob("*.jpg"))[0]
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\.(\d+)Z?\.jpg', img.name)
    if not m:
        return None
    y, mo, d, h, mi, s, frac_str = m.groups()
    frac_us = int(frac_str.ljust(6, '0')[:6])
    dt = datetime.datetime(int(y), int(mo), int(d), int(h), int(mi), int(s),
                           frac_us, tzinfo=datetime.timezone.utc)
    return dt.timestamp()


def _quat_to_yaw(qw, qx, qy, qz):
    """Quaternion → yaw (rotation about Z axis). For ZOD orientation."""
    return float(np.arctan2(2.0 * (qw * qz + qx * qy),
                            1.0 - 2.0 * (qy * qy + qz * qz)))


def _load_cuboids_in_cam(frame_dir: Path, T_inv: np.ndarray):
    """Read object_detection.json (ZOD format) and transform vehicle → cam.

    ZOD format: list of objects with `properties.location_3d.coordinates`,
    `size_3d_{length,width,height}`, `orientation_3d_q{w,x,y,z}`. Only items
    with 3D bbox fields are kept (lane / 2D-only entries skipped).

    Filter out tiny / zero-extent items (poles, bollards) where size_3d_length
    < 0.3m — those have ill-defined yaw and confuse the AABB membership test.
    """
    ann_path = frame_dir / "annotations" / "object_detection.json"
    if not ann_path.exists():
        return []
    try:
        data = json.loads(ann_path.read_text())
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    cubs = []
    for o in data:
        props = o.get('properties', {})
        loc = props.get('location_3d')
        if not loc or 'coordinates' not in loc:
            continue
        size_l = props.get('size_3d_length')
        size_w = props.get('size_3d_width')
        size_h = props.get('size_3d_height')
        if size_l is None or size_w is None or size_h is None:
            continue
        # skip poles/bollards/very-small items where dims < 0.3m
        if min(size_l, size_w, size_h) < 0.3:
            continue
        qw = props.get('orientation_3d_qw', 1.0)
        qx = props.get('orientation_3d_qx', 0.0)
        qy = props.get('orientation_3d_qy', 0.0)
        qz = props.get('orientation_3d_qz', 0.0)
        pos_v = np.asarray(loc['coordinates'], dtype=np.float32)
        if pos_v.size != 3:
            continue
        # Vehicle-frame: x=fwd, y=left, z=up; ZOD size_3d_length is along x (fwd),
        # size_3d_width along y (left), size_3d_height along z (up).
        # In CAM frame (cv2: x=right, y=down, z=fwd): the AABB rotated about Z
        # in `_is_obj_per_point` is WRONG (yaw is about vehicle-Z=up, which maps
        # to cam ≈ -Y, not Z). Workaround: store cuboid in vehicle frame and
        # the membership test runs in vehicle frame too. We add a flag so
        # downstream knows.
        # Map dims: vehicle (l, w, h) along (x_fwd, y_left, z_up). With yaw
        # rotation about Z, AABB extents are (l/2, w/2, h/2). Correct as-is.
        dims = np.asarray([size_l, size_w, size_h], dtype=np.float32)
        yaw = _quat_to_yaw(qw, qx, qy, qz)
        # Keep cuboid in VEHICLE frame so yaw axis (=Z up) matches the AABB
        # rotation in `_is_obj_per_point`. The membership test must therefore
        # be done in vehicle frame as well — see `process_frame` which converts
        # pts_vis (cam frame) → pts_veh via T_vc before calling.
        pos_cam = pos_v.astype(np.float32)  # actually vehicle frame; name kept for compat
        cubs.append({
            'pos':  pos_cam.astype(np.float32),
            'dims': dims,
            'yaw':  float(yaw),
        })
    return cubs


def process_frame(args_tuple):
    """Process one ZOD frame using direct file reads (no zod SDK)."""
    fid, src_root, out_dir, gid_start, tile_layout = args_tuple
    out_dir = Path(out_dir)
    inst_dir = out_dir / 'inst'
    inst_dir.mkdir(parents=True, exist_ok=True)

    try:
        frame_dir = Path(src_root) / "single_frames" / fid
        if not frame_dir.exists():
            return fid, 0
        K, dist, T_vc, T_vl, IW, IH = _load_calib(frame_dir)
        pts_lidar, pt_ts = _load_lidar(frame_dir)
        if pts_lidar is None or len(pts_lidar) == 0:
            return fid, 0

        # Per-point motion compensation: bring lidar pts to vehicle frame at
        # camera shutter time. Without MC, ZOD VLS-128's 115ms scan span causes
        # ~5-30 px projection drift at non-front edges (visible as "left shift"
        # on moving vehicle). Bins per-point ts to 1ms (cheap) and SLERP+lerp
        # ego pose at each bin from the 22-pose ego_motion.json track.
        cam_ts = _camera_shutter_ts(frame_dir)
        if cam_ts is not None and pt_ts is not None:
            pts_veh = _ego_motion_apply(pts_lidar, pt_ts, frame_dir, T_vl, cam_ts)
        else:
            # fallback: no MC
            N = pts_lidar.shape[0]
            homo = np.column_stack([pts_lidar, np.ones(N, dtype=np.float32)])
            pts_veh = (T_vl @ homo.T).T[:, :3].astype(np.float32)

        # vehicle frame at cam_ts → cam frame
        N = pts_veh.shape[0]
        T_cv = np.linalg.inv(T_vc)
        pts_veh_h = np.column_stack([pts_veh, np.ones(N, dtype=np.float32)])
        pts_cam = (T_cv @ pts_veh_h.T).T[:, :3].astype(np.float32)

        # Kannala project
        uv = _project_kannala(pts_cam, K, dist)
        z = pts_cam[:, 2].astype(np.float32)
        valid = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < IW) \
                & (uv[:, 1] >= 0) & (uv[:, 1] < IH)
        if valid.sum() < 64:
            return fid, 0
        pts_vis = pts_cam[valid]
        uv_vis  = uv[valid]
        z_vis   = z[valid]

        # cuboids stored in VEHICLE frame (yaw axis = vehicle Z = up).
        # AABB membership test must run in vehicle frame too — convert visible
        # pts_cam → pts_veh via T_vc and use those for is_obj.
        cubs = _load_cuboids_in_cam(frame_dir, T_cv)  # actually vehicle frame
        pts_vis_h = np.column_stack([pts_vis, np.ones(len(pts_vis), dtype=np.float32)])
        pts_vis_veh = (T_vc @ pts_vis_h.T).T[:, :3].astype(np.float32)
        is_obj_vis = _is_obj_per_point(pts_vis_veh, cubs)

        # image jpg bytes (raw, no decode)
        jpgs = sorted((frame_dir / "camera_front_dnat").glob("*.jpg"))
        if not jpgs:
            return fid, 0
        jpg_bytes = jpgs[0].read_bytes()

        common_inst = dict(
            cam_pos    = torch.zeros(3, dtype=torch.float32),
            R_gt       = torch.eye(3, dtype=torch.float32),
            T_gt       = torch.eye(4, dtype=torch.float32),
            K_full     = torch.from_numpy(K),
            distortion = torch.from_numpy(dist),
            is_fisheye = True,
            cuboids    = cubs,
            scene = fid, cam = 'front', frame = int(fid),
        )

        if tile_layout is None:
            inst = dict(common_inst)
            inst.update(dict(
                jpg_bytes = jpg_bytes,
                IH=int(IH), IW=int(IW),
                pts     = torch.from_numpy(pts_vis),
                uv_full = torch.from_numpy(uv_vis),
                z_cam   = torch.from_numpy(z_vis),
                is_obj  = torch.from_numpy(is_obj_vis),
            ))
            torch.save(inst, inst_dir / f'{gid_start:08d}.pt')
            return fid, 1
        else:
            from scripts.preprocessing._tile_split import cut_inst_to_tiles
            tw, th, st, pad, y0, q = tile_layout
            tile_files = cut_inst_to_tiles(
                jpg_bytes=jpg_bytes, IW=int(IW), IH=int(IH),
                pts_vis=pts_vis, uv_vis=uv_vis, z_vis=z_vis,
                is_obj_vis=is_obj_vis, common_inst=common_inst,
                tile_w=tw, tile_h=th, stride=st, pad_px=pad,
                y_start=y0, jpg_quality=q,
                out_dir=inst_dir, gid_base=gid_start)
            return fid, len(tile_files)
    except Exception as e:
        return fid, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src',         default=str(ZOD_ROOT))
    ap.add_argument('--out',         default=str(CACHE_OUT))
    ap.add_argument('--workers',     type=int, default=8)
    ap.add_argument('--max-frames',  type=int, default=None)
    ap.add_argument('--val-frac',    type=float, default=0.15)
    ap.add_argument('--max-yaw-rate', type=float, default=None,
                    help='exclude frames with yaw rate (deg/s, abs) above this')
    ap.add_argument('--max-speed-kmh', type=float, default=None,
                    help='exclude frames with speed (km/h) above this')
    ap.add_argument('--tile', action='store_true')
    ap.add_argument('--tile-w',       type=int, default=512)
    ap.add_argument('--tile-h',       type=int, default=512)
    ap.add_argument('--tile-stride',  type=int, default=384)
    ap.add_argument('--tile-pad',     type=int, default=64)
    ap.add_argument('--tile-y-start', type=int, default=400,
                    help='ZOD image 2168 tall — skip ~400 sky')
    ap.add_argument('--tile-jpg-q',   type=int, default=90)
    args = ap.parse_args()

    tile_layout = None
    if args.tile:
        tile_layout = (args.tile_w, args.tile_h, args.tile_stride,
                       args.tile_pad, args.tile_y_start, args.tile_jpg_q)
        print(f'TILE mode: tile={args.tile_w}×{args.tile_h} stride={args.tile_stride} '
              f'pad={args.tile_pad} y_start={args.tile_y_start}', flush=True)

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'inst').mkdir(exist_ok=True)

    # Discover frames by scanning single_frames/, filtered to those that
    # actually have a lidar npy file (extracted lidar core may cover only a
    # subset of frames).
    sf_dir = src / 'single_frames'
    if not sf_dir.exists():
        print(f'ERROR: {sf_dir} not found', flush=True)
        sys.exit(1)
    print(f'scanning {sf_dir} ...', flush=True)
    all_ids = sorted(p.name for p in sf_dir.iterdir() if p.is_dir())
    print(f'  {len(all_ids)} total frame dirs', flush=True)

    # Filter to frames with at least 1 lidar npy
    filtered = []
    for fid in all_ids:
        lidar_dir = sf_dir / fid / 'lidar_velodyne'
        if lidar_dir.is_dir() and any(lidar_dir.glob('*.npy')):
            filtered.append(fid)
    print(f'  {len(filtered)} frames have lidar data', flush=True)
    all_ids = filtered

    # Motion-quality filter (yaw rate + speed) for stable-driving curation.
    # Drops frames where ego dynamics produce unrecoverable lidar/camera drift
    # (e.g., 023445: 26 km/h × 3°/s yaw → 20 cm misalignment at 30 m).
    if args.max_yaw_rate is not None or args.max_speed_kmh is not None:
        print(f'  applying motion filter (max_yaw_rate={args.max_yaw_rate}, '
              f'max_speed_kmh={args.max_speed_kmh}) ...', flush=True)
        # parallel scan of ego_motion.json per frame
        from concurrent.futures import ProcessPoolExecutor as _PPE
        def _check(fid):
            yaw, spd = _frame_motion_metrics(sf_dir / fid)
            if yaw is None: return None
            if args.max_yaw_rate is not None and yaw >= args.max_yaw_rate: return None
            if args.max_speed_kmh is not None and spd >= args.max_speed_kmh: return None
            return fid
        with _PPE(max_workers=12) as ex:
            kept = [r for r in ex.map(_check, all_ids, chunksize=200) if r]
        print(f'  {len(kept)}/{len(all_ids)} pass motion filter '
              f'({len(kept)/len(all_ids)*100:.1f}%)', flush=True)
        all_ids = kept

    if args.max_frames:
        all_ids = all_ids[:args.max_frames]
    print(f'frames to process: {len(all_ids)} | out: {out} | workers: {args.workers}', flush=True)

    gid_stride = 100
    argv = [(fid, str(src), str(out), i * gid_stride, tile_layout)
            for i, fid in enumerate(all_ids)]

    t0 = time.time()
    written_total = 0
    if args.workers <= 1:
        for a in argv:
            fid, n = process_frame(a)
            written_total += n
            print(f"  {fid}: +{n}  total={written_total}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_frame, a): a[0] for a in argv}
            done = 0
            for fut in as_completed(futs):
                fid, n = fut.result()
                written_total += n
                done += 1
                if done % 50 == 0 or done == len(all_ids):
                    print(f"  [{done}/{len(all_ids)}] {fid}: +{n}  "
                          f"total={written_total} ({time.time()-t0:.0f}s)", flush=True)
    print(f"done: {written_total} insts → {out}", flush=True)

    # Build train/val split (deterministic first-N as val)
    n_val = max(1, int(len(all_ids) * args.val_frac))
    val_ids = set(all_ids[:n_val])
    train_files, val_files = [], []
    for f in sorted(p.name for p in (out / 'inst').glob('*.pt')):
        scene = f.split('_')[0]  # gid prefix isn't scene; need to peek
        # Quick scene lookup via filename alone is not reliable since gid is
        # frame index. Load once to get scene.
        try:
            inst = torch.load(out / 'inst' / f, weights_only=False)
            if str(inst.get('scene', '')) in val_ids:
                val_files.append(f)
            else:
                train_files.append(f)
        except Exception:
            train_files.append(f)
    meta = {'train': train_files, 'val': val_files,
            'cam': 'front', 'is_fisheye': True}
    torch.save(meta, out / 'meta.pt')
    print(f'meta.pt saved: train={len(train_files)} val={len(val_files)}', flush=True)


if __name__ == '__main__':
    main()
