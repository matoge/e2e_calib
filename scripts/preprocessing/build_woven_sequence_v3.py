"""V3 tile cache for Woven Sequence (TSS4 FCM fisheye).

Mirrors build_zod_v3 / build_kamikado_v3 schema so the same dataset class,
augmentation pipeline, and visualisation work unchanged. The lidar payload
stored per tile is in CAM-FRD frame at *camera shutter time* — i.e., the
exact same temporal alignment the loom backend (`simple_api.py
::_compute_projected_points`) uses to draw the live frontend overlay.
That means downstream training-time augmentation (random R/t/Δfx/Δfy
applied to `pts`, then re-projected via the cached `K_full + distortion`
Kannala-Brandt model) reproduces the projection the human sees in the
loom UI.

Inputs (one sequence dir, layout from /srv/loom/backend/assets):
    SEQ/
      setting-<ip>.json    intrinsics + extrinsics + per-camera delays
      metadata.json        poses (prefer gicped_poses) + camera_delays_ms
      tss4_fcm/<frame>.jpg
      vls128_rear_axle/<frame>.npz   (xs, ys, zs, intensity, ...)
      saved_annotations/<frame>.json (per-frame box list, optional)

Per-tile inst:
    pts          (N, 3) float32  cam-FRD lidar at camera shutter time
    intensity    (N,)   float32  raw lidar intensity (column from npz)
    uv_full      (N, 2) float32  Kannala-Brandt projection in PARENT coords
    z_cam        (N,)   float32  pts[:, 2]
    is_obj       (N,)   float32  in any 3D-box flag
    in_box       (N,)   float32  pt is inside this tile's pixel bounds
    K_full       (3, 3) float32  parent-image fisheye intrinsic
    distortion   (4,)   float32  Kannala-Brandt k1..k4
    is_fisheye   True   bool
    cuboids      list of {pos:(3,), dims:(3,), yaw:float, label:str}
                          pos in cam-FRD; dims = (l, w, h) → (x_local, y_local, z_local)
                          yaw rotates around local z.  (Visualisation-only —
                          the trainer reads is_obj.)
    cam_pos      (3,)   float32  zeros (cuboid frame is cam-FRD already)
    R_gt         (3, 3) float32  identity
    T_gt         (4, 4) float32  identity
    jpg_bytes    bytes  tile JPEG (re-encoded q=jpg_q)
    IH, IW       int    tile dims
    tile_u0, tile_v0, tile_id
    scene = sequence_dir.name, cam = 'tss4_fcm', frame = int(frame_idx)
"""
import argparse
import io
import json
import math
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.preprocessing._tile_split import cut_inst_to_tiles  # noqa: E402


# ─── conventions ────────────────────────────────────────────────────────────
# Vehicle / lidar (rear axle): ISO8855 FLU, x=fwd, y=left, z=up.
# Camera (FRD output frame): x=right, y=down, z=fwd. We rotate vehicle→cam
# with R_to_rdf = [[0,-1,0],[0,0,-1],[1,0,0]] (matches backend's get_rtotal).
R_TO_RDF = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)


# ─── calib / pose loaders ───────────────────────────────────────────────────
def _load_setting(seq: Path) -> dict:
    files = sorted(seq.glob('setting-*.json'))
    if not files:
        raise FileNotFoundError(f'no setting-*.json in {seq}')
    with open(files[0]) as f:
        s = json.load(f)
    return s[0] if isinstance(s, list) and s else s


def _camera_calib_fcm(setting: dict):
    """Returns (K, dist4, R_cam_from_veh, t_cam_in_veh, W, H, delay_ms_default).

    Mirrors backend/projection_utils.get_rtotal:
        rot = [roll, pitch, yaw] (stored)
        R_camera_to_vehicle = from_euler('zyx', [yaw, pitch, roll])
        R_cam_from_veh = R_TO_RDF @ inv(R_camera_to_vehicle)
        t_cam_in_veh   = mp
    """
    cam = setting['fcm']
    fx, fy = cam['fc']
    cx, cy = cam['cc']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    kb = cam['kb']
    dist = np.array([kb['k1'], kb['k2'], kb['k3'], kb['k4']], dtype=np.float32)
    W, H = int(cam['resolution'][0]), int(cam['resolution'][1])
    roll, pitch, yaw = cam['rot']
    R_cv = Rotation.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
    R_cam_from_veh = (R_TO_RDF @ np.linalg.inv(R_cv)).astype(np.float32)
    t_cam_in_veh = np.array(cam['mp'], dtype=np.float32)
    delay_default = float(cam.get('camera_delay_ms', 0.0))
    return K, dist, R_cam_from_veh, t_cam_in_veh, W, H, delay_default


def _load_metadata(seq: Path) -> dict:
    with open(seq / 'metadata.json') as f:
        return json.load(f)


def _get_poses(metadata: dict):
    if 'gicped_poses' in metadata:
        gd = metadata['gicped_poses']
        pd = gd['poses'] if isinstance(gd, dict) and 'poses' in gd else gd
    else:
        pd = metadata['poses']
    frame_ids = sorted(pd.keys())
    poses = {fid: np.array(pd[fid], dtype=np.float64) for fid in frame_ids}
    return frame_ids, poses


def _frame_interval_ms(ts_a: int, ts_b: int) -> float:
    if max(ts_a, ts_b) >= 1e15:  # ns timestamps
        return abs(ts_a - ts_b) / 1_000_000.0
    return float(abs(ts_a - ts_b))


def _camera_delay_ms_for_frame(metadata: dict, frame_id: str,
                                default: float) -> float:
    """Per-frame camera delay (ms). Falls back to setting.fcm.camera_delay_ms.

    Schema (priority order, mirrors simple_api._get_camera_delay_ms):
      metadata.camera_delays_ms.tss4_fcm.<frame_idx>      per-camera/per-frame
      metadata.camera_delays_ms.<frame_idx>                global per-frame
      metadata.camera_delay_ms                              scalar fallback
      setting-*.json fcm.camera_delay_ms                   final fallback
    """
    cd = metadata.get('camera_delays_ms')
    fidx = str(int(frame_id.split('_')[0]))
    if isinstance(cd, dict):
        for_cam = cd.get('tss4_fcm')
        if isinstance(for_cam, dict) and fidx in for_cam:
            return float(for_cam[fidx])
        if fidx in cd and not isinstance(cd[fidx], dict):
            return float(cd[fidx])
    if 'camera_delay_ms' in metadata:
        return float(metadata['camera_delay_ms'])
    return float(default)


def _pose_at_camera_time(poses: dict, frame_ids: list, current_idx: int,
                          camera_delay_ms: float) -> np.ndarray:
    """Slerp/lerp pose backwards by camera_delay_ms from frame_ids[current_idx].

    Matches simple_api._apply_camera_delay_to_pointcloud's pose interp:
      - normal:        interpolate between (prev, curr) at t = 1 - delay/dt
      - first frame:   extrapolate backwards along (curr, next) at t = -delay/dt_next
    """
    fid = frame_ids[current_idx]
    pose_curr = poses[fid]
    ts_curr = int(fid.split('_')[1])

    if current_idx == 0:
        if len(frame_ids) < 2:
            return pose_curr
        ts_next = int(frame_ids[1].split('_')[1])
        dt = _frame_interval_ms(ts_curr, ts_next)
        t_off = -camera_delay_ms / max(dt, 1e-6)
        return _interp_se3(pose_curr, poses[frame_ids[1]], t_off)

    pose_prev = poses[frame_ids[current_idx - 1]]
    ts_prev = int(frame_ids[current_idx - 1].split('_')[1])
    dt = _frame_interval_ms(ts_curr, ts_prev)
    t_off = 1.0 - (camera_delay_ms / max(dt, 1e-6))
    if t_off < 0 and current_idx + 1 < len(frame_ids):
        # Camera is later than lidar — interp/extrap forward.
        pose_next = poses[frame_ids[current_idx + 1]]
        ts_next = int(frame_ids[current_idx + 1].split('_')[1])
        dt_n = _frame_interval_ms(ts_next, ts_curr)
        t_fwd = abs(camera_delay_ms) / max(dt_n, 1e-6)
        return _interp_se3(pose_curr, pose_next, t_fwd)
    return _interp_se3(pose_prev, pose_curr, t_off)


def _interp_se3(pose_a: np.ndarray, pose_b: np.ndarray, t: float) -> np.ndarray:
    """Linear translation, slerp rotation. t may extrapolate (t<0 or t>1).

    Slerp in scipy is closed on [t0, t1]; for outside-range t we lift it to
    R_a · (R_a^{-1} R_b)^t which is well-defined for any t.
    """
    R_a = Rotation.from_matrix(pose_a[:3, :3])
    R_b = Rotation.from_matrix(pose_b[:3, :3])
    if 0.0 <= t <= 1.0:
        slerp = Slerp([0, 1], Rotation.concatenate([R_a, R_b]))
        R_t = slerp(t)
    else:
        delta = R_a.inv() * R_b
        rotvec = delta.as_rotvec() * t
        R_t = R_a * Rotation.from_rotvec(rotvec)
    trans = (1.0 - t) * pose_a[:3, 3] + t * pose_b[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R_t.as_matrix()
    out[:3, 3] = trans
    return out


# ─── lidar / projection ─────────────────────────────────────────────────────
def _load_pts_intensity(seq: Path, frame_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (pts (N,3) rear-axle FLU, intensity (N,))."""
    with np.load(seq / 'vls128_rear_axle' / f'{frame_id}.npz') as f:
        xs = f['xs'].astype(np.float32)
        ys = f['ys'].astype(np.float32)
        zs = f['zs'].astype(np.float32)
        intensity = (f['intensity'].astype(np.float32)
                      if 'intensity' in f.files
                      else np.zeros_like(xs, dtype=np.float32))
    return np.stack([xs, ys, zs], axis=1), intensity


def _project_kannala(pts_cam: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
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


def _lidar_to_cam_at_camera_time(pts_lidar_flu: np.ndarray,
                                  pose_curr: np.ndarray,
                                  pose_camera: np.ndarray,
                                  R_cam_from_veh: np.ndarray,
                                  t_cam_in_veh: np.ndarray) -> np.ndarray:
    """Take lidar pts in vehicle FLU at lidar time, return cam-FRD at camera time.

    Backend formula:  T_lidar_to_camera = inv(pose_camera) @ pose_curr
    applied to homogeneous pts in vehicle frame, then rotate to cam-FRD.
    """
    N = pts_lidar_flu.shape[0]
    homo = np.column_stack([pts_lidar_flu.astype(np.float64),
                             np.ones(N, dtype=np.float64)])
    T_lc = np.linalg.inv(pose_camera) @ pose_curr  # vehicle FLU @ camera time
    pts_veh_at_cam = (T_lc @ homo.T).T[:, :3]
    # Now express in camera FRD (mirror backend: X_cam = R_total @ X_veh + tvec
    # where R_total = R_to_rdf @ inv(R_cv) and tvec = -R_total @ mp).
    return ((R_cam_from_veh @ (pts_veh_at_cam - t_cam_in_veh[None, :]).T).T
            ).astype(np.float32)


# ─── annotations ────────────────────────────────────────────────────────────
def _load_cuboids_vehicle(seq: Path, frame_id: str) -> list[dict]:
    """Read saved_annotations/<frame>.json → list of cuboids in vehicle FLU
    at camera shutter time (uses `3dbb_rear_axle_camera_frame` if present,
    else falls back to `3dbb_rear_axle`).

    Boxes are yaw-only (pitch=roll=0); `direction = [cos_yaw, sin_yaw]`
    fully describes their planar heading. We keep the box in vehicle frame
    here; the build pipeline converts to camera FRD only when calling
    is_obj_per_point on cam-FRD pts (one-shot transform of cuboids in the
    test, not stored).
    """
    p = seq / 'saved_annotations' / f'{frame_id}.json'
    if not p.exists():
        return []
    try:
        with open(p) as f:
            d = json.load(f)
    except Exception:
        return []
    out = []
    for det in d.get('details', []) or []:
        if det.get('type') != 'box':
            continue
        attrs = det.get('attributes') or {}
        cf = attrs.get('3dbb_rear_axle_camera_frame') or attrs.get('3dbb_rear_axle')
        if not cf:
            continue
        center_veh = np.asarray(cf['center_meter'], dtype=np.float32)
        size = np.asarray(cf['size_meter'], dtype=np.float32)
        direction = cf.get('direction')
        if isinstance(direction, (list, tuple)) and len(direction) >= 2:
            yaw = float(np.arctan2(float(direction[1]), float(direction[0])))
        else:
            yaw = 0.0
        out.append({
            'pos':    center_veh,        # vehicle FLU (x=fwd y=left z=up)
            'dims':   size,              # (l, w, h) along (x, y, z) of veh frame
            'yaw':    yaw,               # rotation about veh-z (up)
            'label':  str(det.get('label', '')),
        })
    return out


def _is_obj_per_point(pts_cam_frd: np.ndarray, cubs: list) -> np.ndarray:
    """Axis-aligned-box test in each cuboid's local frame, OR'd. Mirrors
    build_zod_v3 logic but applied in cam-FRD."""
    N = len(pts_cam_frd)
    if not cubs or N == 0:
        return np.zeros(N, dtype=np.float32)
    M = len(cubs)
    poss = np.stack([np.asarray(c['pos'],  dtype=np.float32) for c in cubs])
    dims = np.stack([np.asarray(c['dims'], dtype=np.float32) for c in cubs])
    yaws = np.fromiter((float(c['yaw']) for c in cubs), dtype=np.float32, count=M)
    cy_, sy_ = np.cos(yaws), np.sin(yaws)
    R = np.zeros((M, 3, 3), dtype=np.float32)
    R[:, 0, 0] = cy_;  R[:, 0, 1] = sy_
    R[:, 1, 0] = -sy_; R[:, 1, 1] = cy_
    R[:, 2, 2] = 1.0
    delta = pts_cam_frd.astype(np.float32, copy=False)[None, :, :] - poss[:, None, :]
    local = np.einsum('mij,mnj->mni', R, delta)
    half = (dims * 0.5)[:, None, :]
    inside = np.all(np.abs(local) <= half, axis=-1)
    return inside.any(axis=0).astype(np.float32)


# ─── per-frame worker ───────────────────────────────────────────────────────
def _frame_idx_from_jpg(p: Path) -> str:
    return p.stem  # "0123_175044..."


def process_frame(args_tuple):
    (seq_str, frame_id, current_idx, frame_ids_str, gid_start,
     tile_layout, jpg_q) = args_tuple
    seq = Path(seq_str)
    out_dir = Path(tile_layout['out'])
    inst_dir = out_dir / 'inst'
    inst_dir.mkdir(parents=True, exist_ok=True)

    try:
        setting = _load_setting(seq)
        K, dist, R_cv, t_cv, W, H, delay_default = _camera_calib_fcm(setting)
        metadata = _load_metadata(seq)
        camera_delay_ms = _camera_delay_ms_for_frame(
            metadata, frame_id, delay_default)

        frame_ids, poses = _get_poses(metadata)
        if frame_id not in poses:
            return frame_id, 0
        idx = frame_ids.index(frame_id)

        pts_flu, intensity = _load_pts_intensity(seq, frame_id)
        if pts_flu.size == 0:
            return frame_id, 0

        pose_curr = poses[frame_id]
        pose_cam = _pose_at_camera_time(poses, frame_ids, idx, camera_delay_ms)

        pts_cam = _lidar_to_cam_at_camera_time(
            pts_flu, pose_curr, pose_cam, R_cv.astype(np.float64),
            t_cv.astype(np.float64))
        uv = _project_kannala(pts_cam, K, dist)
        z = pts_cam[:, 2].astype(np.float32)
        valid = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < W) \
                & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if int(valid.sum()) < 64:
            return frame_id, 0

        pts_vis  = pts_cam[valid]
        uv_vis   = uv[valid]
        z_vis    = z[valid]
        int_vis  = intensity[valid]

        cubs = _load_cuboids_vehicle(seq, frame_id)
        # is_obj test runs in vehicle FLU (boxes are stored there). Lift the
        # visible cam-FRD pts back to vehicle FLU for the membership test.
        pts_vis_veh = ((np.linalg.inv(R_cv).astype(np.float32) @ pts_vis.T).T
                        + t_cv.astype(np.float32)[None, :])
        is_obj_vis = _is_obj_per_point(pts_vis_veh, cubs)

        img_path = seq / 'tss4_fcm' / f'{frame_id}.jpg'
        if not img_path.exists():
            return frame_id, 0
        jpg_bytes = img_path.read_bytes()

        common_inst = dict(
            cam_pos    = torch.zeros(3, dtype=torch.float32),
            R_gt       = torch.eye(3, dtype=torch.float32),
            T_gt       = torch.eye(4, dtype=torch.float32),
            K_full     = torch.from_numpy(np.ascontiguousarray(K)),
            distortion = torch.from_numpy(dist),
            is_fisheye = True,
            cuboids    = cubs,
            scene = seq.name, cam = 'tss4_fcm', frame = int(frame_id.split('_')[0]),
        )

        if not tile_layout.get('tile'):
            inst = dict(common_inst)
            inst.update(dict(
                jpg_bytes = jpg_bytes,
                IH=H, IW=W,
                pts       = torch.from_numpy(pts_vis),
                intensity = torch.from_numpy(int_vis),
                uv_full   = torch.from_numpy(uv_vis),
                z_cam     = torch.from_numpy(z_vis),
                is_obj    = torch.from_numpy(is_obj_vis),
            ))
            torch.save(inst, inst_dir / f'{gid_start:08d}.pt')
            return frame_id, 1

        tile_files = cut_inst_to_tiles(
            jpg_bytes=jpg_bytes, IW=W, IH=H,
            pts_vis=pts_vis, uv_vis=uv_vis, z_vis=z_vis,
            is_obj_vis=is_obj_vis,
            extra_per_point={'intensity': int_vis},
            common_inst=common_inst,
            tile_w=tile_layout['tw'], tile_h=tile_layout['th'],
            stride=tile_layout['st'], pad_px=tile_layout['pad'],
            y_start=tile_layout['y0'], jpg_quality=jpg_q,
            out_dir=inst_dir, gid_base=gid_start)
        return frame_id, len(tile_files)
    except Exception:
        import traceback; traceback.print_exc()
        return frame_id, -1


# ─── driver ─────────────────────────────────────────────────────────────────
def _enumerate_frames(seq: Path) -> list[str]:
    """Frames that have all of {jpg, npz, pose} at minimum."""
    md = _load_metadata(seq)
    frame_ids, _ = _get_poses(md)
    have = []
    for fid in frame_ids:
        if (seq / 'tss4_fcm' / f'{fid}.jpg').exists() \
           and (seq / 'vls128_rear_axle' / f'{fid}.npz').exists():
            have.append(fid)
    return have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True,
                    help='dir holding sequence=*/ subdirs (or one sequence dir)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-frames-per-seq', type=int, default=None)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--tile', action='store_true')
    ap.add_argument('--tile-w',       type=int, default=512)
    ap.add_argument('--tile-h',       type=int, default=512)
    ap.add_argument('--tile-stride',  type=int, default=384)
    ap.add_argument('--tile-pad',     type=int, default=64)
    ap.add_argument('--tile-y-start', type=int, default=400)
    ap.add_argument('--tile-jpg-q',   type=int, default=92)
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    (out / 'inst').mkdir(parents=True, exist_ok=True)

    if (src / 'metadata.json').exists():
        sequences = [src]
    else:
        # Accept either `sequence=*` subdirs (woven_sequence layout) or any
        # subdir containing a metadata.json (lets a staging dir of symlinks
        # point at picked sequences).
        sequences = sorted(p for p in src.iterdir()
                            if p.is_dir() and (p / 'metadata.json').exists())
    print(f'{len(sequences)} sequence(s) under {src}', flush=True)
    if not sequences:
        sys.exit(1)

    tile_layout = dict(
        out=str(out), tile=args.tile,
        tw=args.tile_w, th=args.tile_h, st=args.tile_stride,
        pad=args.tile_pad, y0=args.tile_y_start,
    )

    tasks = []
    GID_PER_FRAME = 100
    gid_cursor = 0
    for seq in sequences:
        try:
            fids = _enumerate_frames(seq)
        except Exception as e:
            print(f'  [{seq.name}] skip: {e}', flush=True)
            continue
        if args.max_frames_per_seq:
            fids = fids[:args.max_frames_per_seq]
        print(f'  [{seq.name}] {len(fids)} frames', flush=True)
        # frame_ids list is needed by worker for index-based pose lookup.
        for i, fid in enumerate(fids):
            tasks.append((str(seq), fid, i, None,  # frame_ids re-fetched in worker
                           gid_cursor, tile_layout, args.tile_jpg_q))
            gid_cursor += GID_PER_FRAME

    print(f'total frames to process: {len(tasks)}', flush=True)
    if not tasks:
        return

    written = 0
    if args.workers <= 1:
        for a in tasks:
            _, n = process_frame(a)
            written += max(0, n)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_frame, a): a for a in tasks}
            done = 0
            for fut in as_completed(futs):
                _, n = fut.result()
                written += max(0, n)
                done += 1
                if done % 40 == 0 or done == len(tasks):
                    print(f'  [{done}/{len(tasks)}] written={written}',
                          flush=True)
    print(f'done: {written} insts → {out}', flush=True)

    val_seqs = set(s.name for s in sequences[:max(1, int(len(sequences)
                                                          * args.val_frac))])
    inst_dir = out / 'inst'
    train_files, val_files = [], []
    for f in sorted(p.name for p in inst_dir.glob('*.pt')):
        try:
            inst = torch.load(inst_dir / f, weights_only=False)
            (val_files if str(inst.get('scene', '')) in val_seqs
             else train_files).append(f)
        except Exception:
            train_files.append(f)
    meta = {'train': train_files, 'val': val_files,
            'cam': 'tss4_fcm', 'is_fisheye': True}
    torch.save(meta, out / 'meta.pt')
    print(f'meta.pt saved: train={len(train_files)} val={len(val_files)} '
          f'(val sequences = {sorted(val_seqs)})', flush=True)


if __name__ == '__main__':
    main()
