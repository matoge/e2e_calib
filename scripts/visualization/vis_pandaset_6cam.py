"""Multi-camera LiDAR overlay — verify every camera projects cleanly.

Works on any dataset using OmniRe/PandaSet layout:
  {root}/{scene}/lidar/NN.pkl                     (world-frame x,y,z)
  {root}/{scene}/camera/{cam}/{NN.jpg, poses.json, intrinsics.json}

Usage:
  python vis_pandaset_6cam.py                     # default PandaSet 001 frame 0
  python vis_pandaset_6cam.py --root /mnt/nvme6t/waymo_ps --scene 10017090...
  python vis_pandaset_6cam.py --root /mnt/nvme6t/nuscenes_ps --scene scene-0001
"""
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
ap.add_argument('--scene', default=None, help='scene/segment name (default: first)')
ap.add_argument('--frame', type=int, default=0)
ap.add_argument('--max-d', type=float, default=60.0)
ap.add_argument('--out', default=None, help='output PNG (default: vis_<basename>_6cam.png)')
ap.add_argument('--cams', nargs='+', default=None, help='subset of cam dir names')
args = ap.parse_args()

ROOT  = Path(args.root)
SCENE = args.scene if args.scene else sorted(p.name for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith('_'))[0]
FRAME = args.frame
MAX_D = args.max_d
OUT   = args.out or f'vis_{ROOT.name}_6cam.png'


def quat_pos_to_mat(heading, position):
    q = [heading['x'], heading['y'], heading['z'], heading['w']]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q).as_matrix()
    T[:3, 3]  = [position['x'], position['y'], position['z']]
    return T


def project(pts_world, pose_mat, K):
    T_inv = np.linalg.inv(pose_mat)
    pts_cam = (T_inv[:3, :3] @ pts_world.T + T_inv[:3, 3:]).T
    z = pts_cam[:, 2]
    uv = (K @ pts_cam.T)[:2] / z
    return uv.T, z


sc_dir      = ROOT / SCENE
lidar_files = sorted((sc_dir / 'lidar').glob('*.pkl*'))
_open = gzip.open if lidar_files[FRAME].suffix == '.gz' else open
lidar_df    = pickle.load(_open(lidar_files[FRAME], 'rb'))
# PandaSet stores lidar pts as DataFrame with 'd' (device id 0=primary, 1=secondary)
if hasattr(lidar_df, 'columns') and 'd' in lidar_df.columns:
    lidar_df = lidar_df[lidar_df['d'] == 0]
pts_world   = lidar_df[['x', 'y', 'z']].values.astype(np.float32)

if args.cams:
    CAMS = args.cams
else:
    CAMS = sorted(p.name for p in (sc_dir / 'camera').iterdir() if p.is_dir())

n_cams = len(CAMS)
n_cols = 3 if n_cams <= 6 else 4 if n_cams <= 8 else 5
n_rows = (n_cams + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
axes = np.atleast_1d(axes).ravel()
for ax in axes[n_cams:]:
    ax.axis('off')
for ax, cam in zip(axes, CAMS):
    cam_dir = sc_dir / 'camera' / cam
    with open(cam_dir / 'poses.json') as f:     poses = json.load(f)
    with open(cam_dir / 'intrinsics.json') as f: intr = json.load(f)
    K = np.array([[intr['fx'], 0, intr['cx']],
                  [0, intr['fy'], intr['cy']],
                  [0, 0, 1]], dtype=np.float64)
    pose_mat = quat_pos_to_mat(poses[FRAME]['heading'], poses[FRAME]['position'])
    img_paths = sorted(cam_dir.glob('*.jpg')) + sorted(cam_dir.glob('*.png'))
    img = Image.open(img_paths[FRAME]).convert('RGB')
    IW, IH = img.size
    uv, z = project(pts_world, pose_mat, K)
    vis = (z > 0.5) & (uv[:,0] >= 0) & (uv[:,0] < IW) & (uv[:,1] >= 0) & (uv[:,1] < IH)
    ax.imshow(img)
    ax.scatter(uv[vis,0], uv[vis,1], c=z[vis].clip(0, MAX_D),
               cmap='jet_r', s=0.8, alpha=0.7, vmin=0, vmax=MAX_D)
    ax.set_title(f'{cam}   {vis.sum()} pts   {IW}×{IH}', fontsize=9)
    ax.axis('off')

fig.suptitle(f'{ROOT.name}  scene={SCENE}  frame={FRAME:02d}  — LiDAR overlay on {n_cams} cameras', fontsize=11)
plt.tight_layout()
plt.savefig(OUT, dpi=90, bbox_inches='tight')
print(f'Saved → {OUT}')
