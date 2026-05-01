"""AV2 single-cam zoom — verify pole alignment at full resolution."""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import json, pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='/mnt/nvme6t/av2_ps')
ap.add_argument('--scene', default=None)
ap.add_argument('--frame', type=int, default=30)
ap.add_argument('--cam', default='ring_front_center')
ap.add_argument('--max-d', type=float, default=40.0)
ap.add_argument('--out', default='vis_av2_zoom.png')
args = ap.parse_args()

ROOT = Path(args.root)
SCENE = args.scene or sorted(p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith('_'))[0]
sc = ROOT / SCENE

lidar_pkl = sc / 'lidar' / f'{args.frame:02d}.pkl'
df = pickle.load(open(lidar_pkl, 'rb'))
pts_world = df[['x', 'y', 'z']].values.astype(np.float32)

cam_dir = sc / 'camera' / args.cam
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
print(f'{args.cam}: {vis.sum()} / {len(z)} pts visible, IW×IH={IW}×{IH}')

fig, ax = plt.subplots(figsize=(IW/120, IH/120))
ax.imshow(img)
log_z = np.log10(np.clip(z[vis], 0.5, args.max_d))
ax.scatter(u[vis], v[vis], c=log_z, cmap='turbo', s=3, alpha=0.6,
           vmin=np.log10(0.5), vmax=np.log10(args.max_d))
ax.set_title(f'{args.cam}  scene={SCENE[:8]}  frame={args.frame}  {vis.sum()} pts', fontsize=10)
ax.axis('off')
plt.tight_layout()
plt.savefig(args.out, dpi=120, bbox_inches='tight')
print(f'saved → {args.out}')
