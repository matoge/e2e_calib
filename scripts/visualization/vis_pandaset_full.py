"""Visualize full-resolution PandaSet image with LiDAR overlay."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import json, pickle, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation

PANDASET = '/mnt/mininas/datasets/pandaset'
OUT      = 'vis_ps_full.png'
SCENE    = '001'
FRAME    = 0
MAX_D    = 60.0   # depth clip for colormap


def quat_pos_to_mat(heading, position):
    q = [heading['x'], heading['y'], heading['z'], heading['w']]
    R = Rotation.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = [position['x'], position['y'], position['z']]
    return T.astype(np.float32)


def project(pts_world, pose_mat, K):
    T_inv = np.linalg.inv(pose_mat)
    pts_cam = (T_inv[:3, :3] @ pts_world.T + T_inv[:3, 3:]).T
    z = pts_cam[:, 2]
    uv = (K @ pts_cam.T)[:2] / z
    return uv.T, z


sc_dir  = Path(PANDASET) / SCENE
cam_dir = sc_dir / 'camera' / 'front_camera'

with open(cam_dir / 'poses.json') as f:
    all_poses = json.load(f)
with open(cam_dir / 'intrinsics.json') as f:
    intr = json.load(f)

K = np.array([[intr['fx'], 0, intr['cx']],
              [0, intr['fy'], intr['cy']],
              [0, 0, 1]], dtype=np.float64)

cam_pose = all_poses[FRAME]
pose_mat = quat_pos_to_mat(cam_pose['heading'], cam_pose['position'])

lidar_files = sorted((sc_dir / 'lidar').glob('*.pkl'))
lidar_df = pickle.load(open(lidar_files[FRAME], 'rb'))
lidar_df = lidar_df[lidar_df['d'] == 0]
pts_world = lidar_df[['x', 'y', 'z']].values.astype(np.float32)

img = Image.open(cam_dir / f'{FRAME:02d}.jpg').convert('RGB')
IW, IH = img.size

uv, z = project(pts_world, pose_mat, K)
vis = (z > 0.5) & (uv[:,0] >= 0) & (uv[:,0] < IW) & (uv[:,1] >= 0) & (uv[:,1] < IH)
uv_vis = uv[vis]
z_vis  = z[vis].clip(0, MAX_D)

fig, ax = plt.subplots(figsize=(IW/100, IH/100), dpi=100)
ax.imshow(img)
sc = ax.scatter(uv_vis[:,0], uv_vis[:,1],
                c=z_vis, cmap='jet_r', s=1.5, alpha=0.8,
                vmin=0, vmax=MAX_D)
plt.colorbar(sc, ax=ax, fraction=0.015, pad=0.01, label='depth (m)')
ax.set_title(f'Scene {SCENE}  Frame {FRAME:02d}  —  {vis.sum()} LiDAR pts', fontsize=10)
ax.axis('off')
plt.tight_layout(pad=0.2)
plt.savefig(OUT, dpi=100, bbox_inches='tight')
plt.close()
print(f"Saved → {OUT}  ({vis.sum()} pts, img {IW}×{IH})")
