"""DDAD 6-camera LiDAR overlay — uses per-sensor world pose at capture time.

DDAD provides:
  scene_<hash>.json           — list of "data" items, each datum has its own
                                 pose (T_world_from_sensor at sensor capture time).
  calibration/<hash>.json     — names + per-sensor intrinsics + static extrinsics.
  point_cloud/LIDAR/<ts>.npz  — pts in LIDAR sensor frame, (N, 4) [X,Y,Z,intensity].
  rgb/CAMERA_NN/<ts>.png      — already rectified (no distortion).

Time-aware pose handling is inherent: each datum's `pose` is at its own
timestamp, so cam-vs-lidar timing differences are automatically resolved.
"""
import sys, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='/mnt/mininas/datasets/ddad/ddad_train_val')
ap.add_argument('--scene', default='000000')
ap.add_argument('--frame', type=int, default=20)
ap.add_argument('--max-d', type=float, default=40.0)
ap.add_argument('--out', default='vis_ddad_6cam.png')
args = ap.parse_args()

SC = Path(args.root) / args.scene

# scene JSON contains all data items
scene_jsons = list(SC.glob('scene_*.json'))
print(f'scene jsons: {[p.name for p in scene_jsons]}', flush=True)
scene = json.load(open(scene_jsons[0]))

# calib
calib_files = list((SC / 'calibration').glob('*.json'))
calib = json.load(open(calib_files[0]))
names = calib['names']
print(f'sensors: {names}', flush=True)

def quat_xyzw(q):
    return [q['qx'], q['qy'], q['qz'], q['qw']]
def pose_to_mat(pose):
    R = Rotation.from_quat(quat_xyzw(pose['rotation'])).as_matrix()
    t = np.array([pose['translation']['x'], pose['translation']['y'], pose['translation']['z']])
    M = np.eye(4); M[:3,:3] = R; M[:3,3] = t
    return M

# index data items by their key, then group by sample (frame)
data_by_key = {d['key']: d for d in scene['data']}
samples = scene['samples']
print(f'total frames: {len(samples)}', flush=True)
fi = min(args.frame, len(samples)-1)
print(f'using frame {fi}', flush=True)
frame = {data_by_key[k]['id']['name']: data_by_key[k] for k in samples[fi]['datum_keys']}

# lidar pts in lidar sensor frame → world via lidar's pose
lidar_d = frame['LIDAR']
lidar_npz = np.load(SC / lidar_d['datum']['point_cloud']['filename'])
pts_lid = lidar_npz[lidar_npz.files[0]][:, :3].astype(np.float64)
T_w_from_lid = pose_to_mat(lidar_d['datum']['point_cloud']['pose'])
pts_world = (T_w_from_lid[:3,:3] @ pts_lid.T + T_w_from_lid[:3,3:]).T.astype(np.float32)
print(f'lidar pts: {len(pts_world)}  ts={lidar_d["id"]["timestamp"]}', flush=True)

# cams: 6 of them
cam_names = [n for n in names if n.startswith('CAMERA_')]
n_cams = len(cam_names)
n_cols = 3; n_rows = (n_cams + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
axes = np.atleast_1d(axes).ravel()
for ax in axes[n_cams:]:
    ax.axis('off')

for ax, cam in zip(axes, cam_names):
    if cam not in frame:
        ax.set_title(f'{cam}: no data'); ax.axis('off'); continue
    cam_d = frame[cam]
    cam_idx = names.index(cam)
    intr = calib['intrinsics'][cam_idx]
    K = np.array([[intr['fx'], intr['skew'], intr['cx']],
                  [0, intr['fy'], intr['cy']],
                  [0, 0, 1]])
    img = np.array(Image.open(SC / cam_d['datum']['image']['filename']))
    IH, IW = img.shape[:2]
    T_w_from_cam = pose_to_mat(cam_d['datum']['image']['pose'])
    T_cam_from_w = np.linalg.inv(T_w_from_cam)
    pc = (T_cam_from_w[:3,:3] @ pts_world.T + T_cam_from_w[:3,3:]).T
    z = pc[:,2]
    zc = np.clip(z, 1e-6, None)
    u = K[0,0]*pc[:,0]/zc + K[0,2]
    v = K[1,1]*pc[:,1]/zc + K[1,2]
    vis = (z > 0.5) & (u >= 0) & (u < IW) & (v >= 0) & (v < IH)
    ax.imshow(img)
    log_z = np.log10(np.clip(z[vis], 0.5, args.max_d))
    ax.scatter(u[vis], v[vis], c=log_z, cmap='turbo', s=2.0, alpha=0.5,
               vmin=np.log10(0.5), vmax=np.log10(args.max_d))
    ax.set_title(f'{cam}  {vis.sum()} pts  {IW}×{IH}', fontsize=10)
    ax.axis('off')

fig.suptitle(f'DDAD scene={args.scene} frame={fi} — per-sensor world pose at capture', fontsize=11)
plt.tight_layout()
plt.savefig(args.out, dpi=140, bbox_inches='tight')
print(f'saved → {args.out}')
