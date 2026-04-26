"""PandaSet 6-camera LiDAR overlay — verify every camera projects cleanly."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import json, gzip, pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation

PANDASET = '/mnt/nvme6t/pandaset'
SCENE    = '001'
FRAME    = 0
MAX_D    = 60.0
OUT      = 'vis_ps_6cam.png'

CAMS = ['front_left_camera', 'front_camera', 'front_right_camera',
        'left_camera',       'back_camera',  'right_camera']


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


sc_dir      = Path(PANDASET) / SCENE
lidar_files = sorted((sc_dir / 'lidar').glob('*.pkl*'))
_open = gzip.open if lidar_files[FRAME].suffix == '.gz' else open
lidar_df    = pickle.load(_open(lidar_files[FRAME], 'rb'))
lidar_df    = lidar_df[lidar_df['d'] == 0]
pts_world   = lidar_df[['x', 'y', 'z']].values.astype(np.float32)

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
for ax, cam in zip(axes.ravel(), CAMS):
    cam_dir = sc_dir / 'camera' / cam
    with open(cam_dir / 'poses.json') as f:     poses = json.load(f)
    with open(cam_dir / 'intrinsics.json') as f: intr = json.load(f)
    K = np.array([[intr['fx'], 0, intr['cx']],
                  [0, intr['fy'], intr['cy']],
                  [0, 0, 1]], dtype=np.float64)
    pose_mat = quat_pos_to_mat(poses[FRAME]['heading'], poses[FRAME]['position'])
    img = Image.open(cam_dir / f'{FRAME:02d}.jpg').convert('RGB')
    IW, IH = img.size
    uv, z = project(pts_world, pose_mat, K)
    vis = (z > 0.5) & (uv[:,0] >= 0) & (uv[:,0] < IW) & (uv[:,1] >= 0) & (uv[:,1] < IH)
    ax.imshow(img)
    ax.scatter(uv[vis,0], uv[vis,1], c=z[vis].clip(0, MAX_D),
               cmap='jet_r', s=0.8, alpha=0.7, vmin=0, vmax=MAX_D)
    ax.set_title(f'{cam}   {vis.sum()} pts   {IW}×{IH}', fontsize=9)
    ax.axis('off')

fig.suptitle(f'PandaSet scene {SCENE} frame {FRAME:02d} — LiDAR overlay on 6 cameras', fontsize=11)
plt.tight_layout()
plt.savefig(OUT, dpi=90, bbox_inches='tight')
print(f'Saved → {OUT}')
