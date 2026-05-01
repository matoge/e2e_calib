"""PandaSet single-cam zoom — check pole/edge alignment at full resolution."""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import json, gzip, pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='/mnt/nvme6t/pandaset')
ap.add_argument('--scene', default='001')
ap.add_argument('--frame', type=int, default=30)
ap.add_argument('--cam', default='left_camera')
ap.add_argument('--max-d', type=float, default=40.0)
ap.add_argument('--out', default='vis_panda_zoom.png')
args = ap.parse_args()

SC = Path(args.root) / args.scene
lidar_pkl = SC / 'lidar' / f'{args.frame:02d}.pkl'
if not lidar_pkl.exists():
    lidar_pkl = SC / 'lidar' / f'{args.frame:02d}.pkl.gz'
_open = gzip.open if lidar_pkl.suffix == '.gz' else open
print(f'lidar pkl: {lidar_pkl}', flush=True)
df = pickle.load(_open(lidar_pkl, 'rb'))
if 'd' in df.columns:
    df = df[df['d'] == 0]
pts_world = df[['x','y','z']].values.astype(np.float32)

cam_dir = SC / 'camera' / args.cam
poses = json.load(open(cam_dir/'poses.json'))
intr = json.load(open(cam_dir/'intrinsics.json'))
K = np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]])

p = poses[args.frame]
q = [p['heading']['x'], p['heading']['y'], p['heading']['z'], p['heading']['w']]
T = np.eye(4); T[:3,:3] = Rotation.from_quat(q).as_matrix(); T[:3,3] = [p['position']['x'],p['position']['y'],p['position']['z']]
Tinv = np.linalg.inv(T)

img = np.array(Image.open(cam_dir / f'{args.frame:02d}.jpg'))
IH, IW = img.shape[:2]
pc = (Tinv[:3,:3] @ pts_world.T + Tinv[:3,3:]).T
z = pc[:,2]; zc = np.clip(z, 1e-6, None)
u = K[0,0]*pc[:,0]/zc + K[0,2]; v = K[1,1]*pc[:,1]/zc + K[1,2]
vis = (z > 0.5) & (u >= 0) & (u < IW) & (v >= 0) & (v < IH)
print(f'{args.cam}: {vis.sum()}/{len(z)} pts  IW={IW} IH={IH}')

cam_ts = json.load(open(cam_dir/'timestamps.json'))[args.frame]
lid_ts = json.load(open(SC/'lidar/timestamps.json'))[args.frame]
print(f'cam_ts={cam_ts:.4f}  lidar_ts={lid_ts:.4f}  Δ={(cam_ts-lid_ts)*1000:+.1f}ms')

fig, ax = plt.subplots(figsize=(IW/120, IH/120))
ax.imshow(img)
log_z = np.log10(np.clip(z[vis], 0.5, args.max_d))
ax.scatter(u[vis], v[vis], c=log_z, cmap='turbo', s=3, alpha=0.55,
           vmin=np.log10(0.5), vmax=np.log10(args.max_d))
ax.set_title(f'{args.cam}  scene={args.scene}  frame={args.frame}  Δt_cam_lidar={(cam_ts-lid_ts)*1000:+.1f}ms', fontsize=10)
ax.axis('off')
plt.tight_layout()
plt.savefig(args.out, dpi=120, bbox_inches='tight')
print(f'saved → {args.out}')
