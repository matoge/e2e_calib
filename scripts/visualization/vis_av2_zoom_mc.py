"""AV2 single-cam zoom WITH per-point lidar motion compensation.

Reads raw lidar feather (NOT preprocessed pkl) so we have per-point offset_ns.
For each lidar pt:
  t_capture = lidar_sweep_start_ns + offset_ns
  pt_world  = T_e2w(t_capture) @ pt_ego
Cam pose at cam shutter (from preprocessed poses.json, already time-aware).
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

ap = argparse.ArgumentParser()
ap.add_argument('--ps-root', default='/mnt/nvme6t/av2_ps')
ap.add_argument('--raw-root', default='/mnt/mininas/datasets/argoverse2/sensor/train')
ap.add_argument('--scene', default=None)
ap.add_argument('--frame', type=int, default=30)
ap.add_argument('--cam', default='ring_side_left')
ap.add_argument('--max-d', type=float, default=40.0)
ap.add_argument('--out', default='vis_av2_zoom_mc.png')
ap.add_argument('--no-mc', action='store_true', help='disable motion-comp (baseline)')
args = ap.parse_args()

PS  = Path(args.ps_root)
RAW = Path(args.raw_root)
SCENE = args.scene or sorted(p.name for p in PS.iterdir() if p.is_dir() and not p.name.startswith('_'))[0]
print(f'scene={SCENE}  cam={args.cam}  frame={args.frame}  motion_comp={not args.no_mc}', flush=True)

# raw lidar feather (need per-pt offset_ns) — find by frame index → ts mapping in PS
ps_sc  = PS / SCENE
raw_sc = RAW / SCENE

# match preprocessed frame index to raw lidar feather by sorted timestamp
lidar_files_raw = sorted((raw_sc / 'sensors/lidar').glob('*.feather'))
lidar_ts_raw = np.array([int(p.stem) for p in lidar_files_raw], dtype=np.int64)
# preprocessor used every K-th raw lidar (default K=2 → ~5Hz)
ps_n = len(sorted((ps_sc/'lidar').glob('*.pkl')))
K_stride = len(lidar_files_raw) // ps_n   # approx
print(f'raw lidar count: {len(lidar_files_raw)}  preprocessed: {ps_n}  stride≈{K_stride}', flush=True)

# pick raw idx (assume frame i in PS = raw idx i*K_stride or matching)
raw_idx = args.frame * K_stride
sweep_path = lidar_files_raw[raw_idx]
sweep_start_ns = int(sweep_path.stem)
print(f'sweep_start_ns = {sweep_start_ns}  ({sweep_path.name})', flush=True)

df = pd.read_feather(sweep_path)
pts_ego  = df[['x','y','z']].values.astype(np.float64)
offset_ns = df['offset_ns'].values.astype(np.int64)
pts_ts = sweep_start_ns + offset_ns  # per-point absolute capture time
print(f'pts: {len(pts_ego)}  ts span: {(pts_ts.max()-pts_ts.min())/1e6:.1f}ms', flush=True)

# city_SE3_egovehicle for high-rate ego pose
city = pd.read_feather(raw_sc / 'city_SE3_egovehicle.feather').sort_values('timestamp_ns').reset_index(drop=True)
city_ts = city['timestamp_ns'].values
print(f'ego pose entries: {len(city)}  span: {(city_ts.max()-city_ts.min())/1e6:.1f}ms  (median Δ: {np.median(np.diff(city_ts))/1e6:.2f}ms)', flush=True)

# Build SLERP for the time window we need
t_min = max(pts_ts.min(), city_ts.min())
t_max = min(pts_ts.max(), city_ts.max())
mask = (city_ts >= t_min - int(20e6)) & (city_ts <= t_max + int(20e6))
city_sub = city[mask].reset_index(drop=True)
print(f'using {len(city_sub)} ego pose samples in window', flush=True)

quats = np.stack([city_sub['qx'].values, city_sub['qy'].values, city_sub['qz'].values, city_sub['qw'].values], axis=1)
trans = np.stack([city_sub['tx_m'].values, city_sub['ty_m'].values, city_sub['tz_m'].values], axis=1)
slerp = Slerp(city_sub['timestamp_ns'].values, Rotation.from_quat(quats))

if args.no_mc:
    # baseline: single ego pose at sweep_start
    R0 = slerp([sweep_start_ns]).as_matrix()[0]
    t0 = np.stack([np.interp(sweep_start_ns, city_sub['timestamp_ns'], trans[:, i]) for i in range(3)])
    pts_world = (R0 @ pts_ego.T).T + t0[None, :]
else:
    # per-point: SLERP-interp ego pose at each pt's capture time, vectorized via groupby unique ts
    pts_ts_clip = np.clip(pts_ts, city_sub['timestamp_ns'].iloc[0], city_sub['timestamp_ns'].iloc[-1])
    Rs = slerp(pts_ts_clip).as_matrix()  # (N, 3, 3)
    ts_x = np.interp(pts_ts_clip, city_sub['timestamp_ns'], trans[:, 0])
    ts_y = np.interp(pts_ts_clip, city_sub['timestamp_ns'], trans[:, 1])
    ts_z = np.interp(pts_ts_clip, city_sub['timestamp_ns'], trans[:, 2])
    Ts = np.stack([ts_x, ts_y, ts_z], axis=1)
    # apply: pts_world[i] = R_i @ pts_ego[i] + t_i
    pts_world = np.einsum('nij,nj->ni', Rs, pts_ego) + Ts

pts_world = pts_world.astype(np.float32)

# cam side: use preprocessed PS poses.json + intrinsics
cam_dir = ps_sc / 'camera' / args.cam
intr = json.load(open(cam_dir / 'intrinsics.json'))
poses = json.load(open(cam_dir / 'poses.json'))
K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0,0,1]])
dist = np.array([intr.get('k1',0), intr.get('k2',0), intr.get('k3',0)])
p = poses[args.frame]
q = [p['heading']['x'], p['heading']['y'], p['heading']['z'], p['heading']['w']]
T = np.eye(4); T[:3,:3] = Rotation.from_quat(q).as_matrix(); T[:3,3] = [p['position']['x'],p['position']['y'],p['position']['z']]
Tinv = np.linalg.inv(T)

img = np.array(Image.open(cam_dir / f'{args.frame:02d}.jpg'))
IH, IW = img.shape[:2]
cam = (Tinv[:3,:3] @ pts_world.T + Tinv[:3,3:]).T
z = cam[:,2]; zc = np.clip(z, 1e-6, None)
xn = cam[:,0]/zc; yn = cam[:,1]/zc
r2 = xn*xn + yn*yn
f = 1 + dist[0]*r2 + dist[1]*r2*r2 + dist[2]*r2*r2*r2
xn *= f; yn *= f
u = K[0,0]*xn + K[0,2]; v = K[1,1]*yn + K[1,2]
vis = (z > 0.5) & (u >= 0) & (u < IW) & (v >= 0) & (v < IH)
print(f'visible: {vis.sum()}/{len(z)}', flush=True)

fig, ax = plt.subplots(figsize=(IW/120, IH/120))
ax.imshow(img)
log_z = np.log10(np.clip(z[vis], 0.5, args.max_d))
ax.scatter(u[vis], v[vis], c=log_z, cmap='turbo', s=3, alpha=0.6,
           vmin=np.log10(0.5), vmax=np.log10(args.max_d))
ax.set_title(f'{args.cam}  scene={SCENE[:8]}  frame={args.frame}  '
             f'{"per-pt MC" if not args.no_mc else "single-pose (baseline)"}  {vis.sum()} pts', fontsize=10)
ax.axis('off')
plt.tight_layout()
plt.savefig(args.out, dpi=120, bbox_inches='tight')
print(f'saved → {args.out}')
