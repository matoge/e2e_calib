"""PandaSet single-cam zoom WITH cam-shutter-time pose correction.

PandaSet's poses.json[fi] for camera X is `T_world_from_veh(lidar_ts[fi]) @ T_veh_from_cam_static`.
That is, the vehicle pose was queried at LIDAR time, not at the camera's actual shutter time.
This causes a fixed time-offset error of ego_speed × Δt_cam_lidar visible especially on
side cameras (Δt up to ~100ms → ~0.77m at urban speeds).

Fix: SLERP-interpolate the cam's own poses.json across adjacent frames to get the cam's
world pose AT cam shutter time. Since the per-frame data are spaced by ~100ms (lidar Hz),
and the cam's Δt within a frame is ≤100ms, neighboring frames suffice.
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import json, gzip, pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='/mnt/nvme6t/pandaset')
ap.add_argument('--scene', default='001')
ap.add_argument('--frame', type=int, default=30)
ap.add_argument('--cam', default='left_camera')
ap.add_argument('--max-d', type=float, default=40.0)
ap.add_argument('--out', default='vis_panda_zoom_tc.png')
ap.add_argument('--no-tc', action='store_true', help='disable time-correction (baseline)')
args = ap.parse_args()

SC = Path(args.root) / args.scene
fi = args.frame

# lidar pkl
lidar_pkl = SC / 'lidar' / f'{fi:02d}.pkl'
if not lidar_pkl.exists():
    lidar_pkl = SC / 'lidar' / f'{fi:02d}.pkl.gz'
_open = gzip.open if lidar_pkl.suffix == '.gz' else open
df = pickle.load(_open(lidar_pkl, 'rb'))
if 'd' in df.columns: df = df[df['d'] == 0]
pts_world = df[['x','y','z']].values.astype(np.float32)

cam_dir = SC / 'camera' / args.cam
poses     = json.load(open(cam_dir / 'poses.json'))
intr      = json.load(open(cam_dir / 'intrinsics.json'))
cam_ts_all  = json.load(open(cam_dir / 'timestamps.json'))
lid_ts_all  = json.load(open(SC / 'lidar' / 'timestamps.json'))
K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]])

def pose_to_mat(p):
    q = [p['heading']['x'], p['heading']['y'], p['heading']['z'], p['heading']['w']]
    M = np.eye(4); M[:3,:3] = Rotation.from_quat(q).as_matrix()
    M[:3,3] = [p['position']['x'], p['position']['y'], p['position']['z']]
    return M

# baseline: just use poses.json[fi] (= queried at lidar_ts)
T_baseline = pose_to_mat(poses[fi])

if args.no_tc:
    T_w_from_cam = T_baseline
    print(f'[no-tc] using poses.json[{fi}] directly  (lidar-time pose)')
else:
    # time-corrected: SLERP cam world pose to cam_ts time
    # poses[i] effective time = lid_ts_all[i]; we want time = cam_ts_all[fi]
    target_t = cam_ts_all[fi]
    eff_ts = np.array(lid_ts_all)  # times at which poses were sampled
    # build SLERP across all frames
    quats = []; trans = []
    for p in poses:
        q = [p['heading']['x'], p['heading']['y'], p['heading']['z'], p['heading']['w']]
        quats.append(q)
        trans.append([p['position']['x'], p['position']['y'], p['position']['z']])
    quats = np.array(quats); trans = np.array(trans)
    slerp = Slerp(eff_ts, Rotation.from_quat(quats))
    t_clip = float(np.clip(target_t, eff_ts[0], eff_ts[-1]))
    R_t = slerp([t_clip]).as_matrix()[0]
    # linear interp translation
    tx = np.interp(t_clip, eff_ts, trans[:, 0])
    ty = np.interp(t_clip, eff_ts, trans[:, 1])
    tz = np.interp(t_clip, eff_ts, trans[:, 2])
    T_w_from_cam = np.eye(4); T_w_from_cam[:3,:3] = R_t; T_w_from_cam[:3,3] = [tx, ty, tz]
    print(f'[tc] target_t = cam_ts[{fi}] = {target_t}, lidar_ts = {lid_ts_all[fi]}, Δt = {(target_t-lid_ts_all[fi])*1000:+.1f}ms')
    print(f'    pose shift: {np.linalg.norm(T_w_from_cam[:3,3] - T_baseline[:3,3])*1000:.0f}mm')

T_cam_from_w = np.linalg.inv(T_w_from_cam)
img = np.array(Image.open(cam_dir / f'{fi:02d}.jpg'))
IH, IW = img.shape[:2]
pc = (T_cam_from_w[:3,:3] @ pts_world.T + T_cam_from_w[:3,3:]).T
z = pc[:,2]; zc = np.clip(z, 1e-6, None)
u = K[0,0]*pc[:,0]/zc + K[0,2]; v = K[1,1]*pc[:,1]/zc + K[1,2]
vis = (z > 0.5) & (u >= 0) & (u < IW) & (v >= 0) & (v < IH)
print(f'{args.cam}: {vis.sum()}/{len(z)} visible')

fig, ax = plt.subplots(figsize=(IW/120, IH/120))
ax.imshow(img)
log_z = np.log10(np.clip(z[vis], 0.5, args.max_d))
ax.scatter(u[vis], v[vis], c=log_z, cmap='turbo', s=3, alpha=0.55,
           vmin=np.log10(0.5), vmax=np.log10(args.max_d))
ax.set_title(f'{args.cam}  scene={args.scene} frame={fi}  {"TIME-CORRECTED" if not args.no_tc else "BASELINE (poses.json as-is)"}', fontsize=11)
ax.axis('off')
plt.tight_layout()
plt.savefig(args.out, dpi=120, bbox_inches='tight')
print(f'saved → {args.out}')
