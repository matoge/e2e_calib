"""Waymo raw-parquet 5-camera LiDAR overlay using the trusted preprocessor pipeline.

Reuses:
  - waymo_to_pandaset._decode_range_image / _T_world_from_opencvcam / R_WAYMOCAM_FROM_OPENCVCAM
  - pandaset_pair._project (OpenCV pinhole, the production projection used by all training)

Per-camera shutter pose: each [CameraImageComponent].pose is T_world_from_waymocam at the
camera's pose_timestamp (different across cams ~25-75ms vs lidar). Convert to OpenCV cam
exactly the way the preprocessor stores poses.json, then build T_w2c and project.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import io, numpy as np, pandas as pd
import pyarrow.parquet as pq
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from datasets.waymo import WAYMO_DIR, TOP_LIDAR
from scripts.preprocessing.waymo_to_pandaset import (
    _decode_range_image, _load_cam_intr_extr, _load_lidar_cal,
    _T_world_from_opencvcam,
)
from datasets.pandaset_pair import _project

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--seg-idx', type=int, default=0)
ap.add_argument('--frame', type=int, default=20)
ap.add_argument('--use-return2', action='store_true', help='add range_image_return2 points')
ap.add_argument('--all-lasers', action='store_true', help='use all 5 lidars not just TOP')
ap.add_argument('--motion-comp', action='store_true', help='per-azimuth lidar motion compensation')
args = ap.parse_args()

CAMS  = {1:'FRONT', 2:'FRONT_LEFT', 3:'FRONT_RIGHT', 4:'SIDE_LEFT', 5:'SIDE_RIGHT'}
FRAME = args.frame
MAX_D = 60.0


segs = sorted(f.stem for f in (WAYMO_DIR/'lidar').glob('*.parquet'))
seg  = segs[args.seg_idx]
print(f'segment[{args.seg_idx}]: {seg}', flush=True)

# ── camera + lidar calibrations (static) ────────────────────────────────────
cam_cal_df = pd.read_parquet(WAYMO_DIR/'camera_calibration'/f'{seg}.parquet')
cam_cal = {}
for _, r in cam_cal_df.iterrows():
    cid = int(r['key.camera_name'])
    if cid not in CAMS: continue
    fu, fv, cu, cv, T_veh_from_wcam = _load_cam_intr_extr(r)
    IW = int(r['[CameraCalibrationComponent].width'])
    IH = int(r['[CameraCalibrationComponent].height'])
    cam_cal[cid] = dict(fu=fu, fv=fv, cu=cu, cv=cv, IW=IW, IH=IH,
                         T_veh_from_wcam=T_veh_from_wcam)

lid_cal_df = pd.read_parquet(WAYMO_DIR/'lidar_calibration'/f'{seg}.parquet')
LASER_IDS = [1,2,3,4,5] if args.all_lasers else [TOP_LIDAR]

# ── lidar at frame ─────────────────────────────────────────────────────────
print('reading lidar (filtered)...', flush=True)
lidar_df = pq.read_table(
    WAYMO_DIR/'lidar'/f'{seg}.parquet',
    filters=[('key.laser_name', 'in', LASER_IDS)],
).to_pandas()
ts_list = sorted(lidar_df[lidar_df['key.laser_name']==TOP_LIDAR]['key.frame_timestamp_micros'].unique())
ts_lidar = ts_list[min(FRAME, len(ts_list)-1)]

pts_veh_parts = []
def _decode_with_per_col_time(ri_vals, ri_shape, incl, az_correction, t_frame, t_per_col):
    """Decode range image; returns (pts_lidar (N,3), t_capture (N,)). pts in lidar
    sensor frame; t_capture per point computed from azimuth column (linear sweep)."""
    ri = np.array(ri_vals, dtype=np.float32).reshape(ri_shape)
    H, W = ri.shape[:2]
    r = ri[:,:,0]; valid = r > 0
    if valid.sum() == 0:
        return np.zeros((0,3), dtype=np.float32), np.zeros(0, dtype=np.float64)
    cols = np.arange(W, dtype=np.float64)
    azimuth = np.pi * (1.0 - 2.0 * (cols + 0.5) / W) - az_correction
    cos_e = np.cos(incl[:,None]).astype(np.float32)
    sin_e = np.sin(incl[:,None]).astype(np.float32)
    cos_a = np.cos(azimuth[None,:]).astype(np.float32)
    sin_a = np.sin(azimuth[None,:]).astype(np.float32)
    xs = r*cos_e*cos_a; ys = r*cos_e*sin_a; zs = r*sin_e
    # per-col capture time (broadcast to per-pixel)
    t_col = t_frame + cols * t_per_col   # (W,)
    t_pixel = np.broadcast_to(t_col[None,:], (H, W))
    return (np.stack([xs[valid], ys[valid], zs[valid]], axis=1).astype(np.float32),
            t_pixel[valid].astype(np.float64))


def _interp_world_from_veh(pose_ix, t):
    """SLERP rotation + linear translation of T_world_from_veh at query time t (us)."""
    from scipy.spatial.transform import Rotation, Slerp
    keys = pose_ix.index.values   # sorted ts (us)
    i = np.searchsorted(keys, t)
    i = max(1, min(i, len(keys)-1))
    t0, t1 = keys[i-1], keys[i]
    a = (t - t0) / max(t1 - t0, 1)
    T0 = np.array(pose_ix.iloc[i-1]['[VehiclePoseComponent].world_from_vehicle.transform']).reshape(4,4)
    T1 = np.array(pose_ix.iloc[i]  ['[VehiclePoseComponent].world_from_vehicle.transform']).reshape(4,4)
    rots = Rotation.from_matrix(np.stack([T0[:3,:3], T1[:3,:3]]))
    R_t = Slerp([0.0, 1.0], rots)([a]).as_matrix()[0]
    t_t = (1-a)*T0[:3,3] + a*T1[:3,3]
    T = np.eye(4); T[:3,:3] = R_t; T[:3,3] = t_t
    return T


def _safe_load_lidar_cal(cal_row, n_beams):
    import math
    T = np.array(cal_row['[LiDARCalibrationComponent].extrinsic.transform'], dtype=np.float64).reshape(4,4)
    iv = cal_row['[LiDARCalibrationComponent].beam_inclination.values']
    if iv is None or (hasattr(iv, '__len__') and len(iv) == 0):
        lo = float(cal_row['[LiDARCalibrationComponent].beam_inclination.min'])
        hi = float(cal_row['[LiDARCalibrationComponent].beam_inclination.max'])
        incl = np.linspace(lo, hi, n_beams)
    else:
        incl = np.array(iv, dtype=np.float64)
    incl = incl[::-1].copy()
    az_corr = math.atan2(T[1,0], T[0,0])
    return T, incl, az_corr

pts_veh_parts = []
pts_t_parts   = []
SWEEP_US = 100000  # 10Hz, 100ms full sweep
for lid in LASER_IDS:
    cal = lid_cal_df[lid_cal_df['key.laser_name']==lid].iloc[0]
    sub = lidar_df[(lidar_df['key.laser_name']==lid) & (lidar_df['key.frame_timestamp_micros']==ts_lidar)]
    if len(sub)==0: continue
    lr = sub.iloc[0]
    H_beams = int(lr['[LiDARComponent].range_image_return1.shape'][0])
    T_veh_from_lid, incl, az_corr = _safe_load_lidar_cal(cal, H_beams)
    for ret in (['return1', 'return2'] if args.use_return2 else ['return1']):
        vals = lr[f'[LiDARComponent].range_image_{ret}.values']
        shape = lr[f'[LiDARComponent].range_image_{ret}.shape']
        if len(vals)==0: continue
        W = int(shape[1])
        t_per_col = SWEEP_US / W
        pts_lid, t_pix = _decode_with_per_col_time(vals, shape, incl, az_corr, ts_lidar, t_per_col)
        if len(pts_lid)==0: continue
        h = np.hstack([pts_lid, np.ones((len(pts_lid),1), dtype=pts_lid.dtype)])
        pv = (T_veh_from_lid @ h.T).T[:,:3]
        pts_veh_parts.append(pv)
        pts_t_parts.append(t_pix)
        print(f'  laser={lid} {ret}: {len(pv)} pts  Δt_span={(t_pix.max()-t_pix.min())/1000:.1f}ms', flush=True)
pts_veh = np.concatenate(pts_veh_parts, axis=0).astype(np.float32)
pts_t   = np.concatenate(pts_t_parts, axis=0)
print(f'total: {len(pts_veh)} pts  veh-z[{pts_veh[:,2].min():.2f},{pts_veh[:,2].max():.2f}]')

# vehicle_pose for interpolation
print('reading vehicle_pose...', flush=True)
pose_ix = pq.read_table(WAYMO_DIR/'vehicle_pose'/f'{seg}.parquet').to_pandas().set_index('key.frame_timestamp_micros').sort_index()
T_world_from_veh_lidar = np.array(
    pose_ix.loc[ts_lidar]['[VehiclePoseComponent].world_from_vehicle.transform'],
    dtype=np.float64).reshape(4,4)

if args.motion_comp:
    print('applying per-azimuth motion compensation...', flush=True)
    # group by unique capture time (per-col), apply pose, accumulate
    unique_t, inv = np.unique(pts_t, return_inverse=True)
    pts_world = np.empty_like(pts_veh, dtype=np.float32)
    for i, t in enumerate(unique_t):
        T = _interp_world_from_veh(pose_ix, t)
        mask = (inv == i)
        pv = pts_veh[mask]
        h2 = np.hstack([pv, np.ones((len(pv),1), dtype=pv.dtype)])
        pts_world[mask] = (T @ h2.T).T[:,:3].astype(np.float32)
    print(f'  motion-comp using {len(unique_t)} unique col-times')
else:
    h2 = np.hstack([pts_veh, np.ones((len(pts_veh),1), dtype=pts_veh.dtype)])
    pts_world = (T_world_from_veh_lidar @ h2.T).T[:,:3].astype(np.float32)
print(f'pts_world: {len(pts_world)}  ts_lidar={ts_lidar}', flush=True)

# ── camera images at this frame ────────────────────────────────────────────
print(f'reading camera_image at ts={ts_lidar}...', flush=True)
cam_df = pq.read_table(
    WAYMO_DIR/'camera_image'/f'{seg}.parquet',
    filters=[('key.frame_timestamp_micros', '=', ts_lidar)],
).to_pandas()

fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
for ax, (cid, cname) in zip(axes, CAMS.items()):
    if cid not in cam_cal:
        ax.axis('off'); continue
    rows = cam_df[cam_df['key.camera_name']==cid]
    if len(rows) == 0:
        ax.set_title(f'{cname}: no row'); ax.axis('off'); continue
    cr = rows.iloc[0]
    img = Image.open(io.BytesIO(bytes(cr['[CameraImageComponent].image']))).convert('RGB')
    cc = cam_cal[cid]

    # interpolate vehicle_pose at cam's pose_timestamp (consistent with lidar interp)
    pose_ts_us = float(cr['[CameraImageComponent].pose_timestamp']) * 1e6
    T_world_from_veh_shutter = _interp_world_from_veh(pose_ix, pose_ts_us)
    T_world_from_wcam = T_world_from_veh_shutter @ cc['T_veh_from_wcam']
    T_world_from_cam_ocv = _T_world_from_opencvcam(T_world_from_wcam)
    T_w2c = np.linalg.inv(T_world_from_cam_ocv)

    K = np.array([[cc['fu'], 0, cc['cu']],
                  [0, cc['fv'], cc['cv']],
                  [0,        0,        1]], dtype=np.float64)
    uv, z = _project(pts_world, T_w2c, K, None)
    vis = (z > 0.5) & (uv[:,0]>=0) & (uv[:,0]<cc['IW']) & (uv[:,1]>=0) & (uv[:,1]<cc['IH'])

    pose_ts = float(cr['[CameraImageComponent].pose_timestamp'])*1e6
    dt_ms = (pose_ts - ts_lidar)/1e3

    ax.imshow(img)
    ax.scatter(uv[vis,0], uv[vis,1], c=z[vis].clip(0, MAX_D),
               cmap='turbo', s=0.5, alpha=0.25, vmin=0, vmax=MAX_D)
    ax.set_title(f'{cname}\n{vis.sum()} pts  Δt={dt_ms:+.1f}ms', fontsize=10)
    ax.set_xlim(0, cc['IW']); ax.set_ylim(cc['IH'], 0); ax.axis('off')

fig.suptitle(f'Waymo RAW seg={seg[:30]}… frame={FRAME} (preprocessor pipeline + pandaset_pair._project)', fontsize=11)
plt.tight_layout()
out = 'vis_waymo_raw_5cam.png'
plt.savefig(out, dpi=80, bbox_inches='tight')
print(f'saved → {out}')
