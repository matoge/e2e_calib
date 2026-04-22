"""PandaSet projection on turning scenes — check if LiDAR points align with camera
during high yaw-rate frames. If pose compensation is imperfect, thin vertical
features (poles, signs, traffic lights) show visible horizontal smear.

Outputs experiments/turn_projection/<scene>_f<fi>.png — raw projection of LiDAR
points colored by capture time within the scan. Expected signature of mis-comp:
time-color bleeds horizontally across thin structures during turns.
"""
import json, pickle
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/mnt/mininas/datasets/pandaset')
OUT  = Path('experiments/turn_projection'); OUT.mkdir(parents=True, exist_ok=True)

# picks: (scene, frame) — first two are top-yaw, last is low-yaw control
PICKS = [
    ('040', 55),  # 22.3 deg/s
    ('029', 46),  # 19.6 deg/s
    ('039',  6),  # 14.1 deg/s
    ('015', 30),  #  3.1 deg/s (near-straight control)
]

def load_pose(scene, fi):
    poses = json.load(open(ROOT/scene/'camera'/'front_camera'/'poses.json'))
    p = poses[fi]
    q = [p['heading']['x'], p['heading']['y'], p['heading']['z'], p['heading']['w']]
    R = Rotation.from_quat(q).as_matrix()
    t = np.array([p['position']['x'], p['position']['y'], p['position']['z']])
    M = np.eye(4); M[:3,:3] = R; M[:3,3] = t
    return M, poses  # cam world pose + full list

def project_points(pts_world, cam_pose_mat, K):
    T = np.linalg.inv(cam_pose_mat)
    pts_cam = T[:3,:3] @ pts_world.T + T[:3,3:]
    z = pts_cam[2]
    uv = (K @ pts_cam)[:2] / z
    return uv.T, z

def plot_one(scene, fi, ax_img, ax_zoom):
    img = np.array(Image.open(ROOT/scene/'camera'/'front_camera'/f'{fi:02d}.jpg'))
    H, W = img.shape[:2]

    intr = json.load(open(ROOT/scene/'camera'/'front_camera'/'intrinsics.json'))
    K = np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]])

    cam_pose, _ = load_pose(scene, fi)

    lidar = pickle.load(open(ROOT/scene/'lidar'/f'{fi:02d}.pkl','rb'))
    pts_w = lidar[['x','y','z']].values
    t_pt  = lidar['t'].values
    dev   = lidar['d'].values

    # only Pandar64 (dev 0) — higher temporal granularity
    m = dev == 0
    pts_w, t_pt = pts_w[m], t_pt[m]

    uv, z = project_points(pts_w, cam_pose, K)
    in_img = (z > 0.5) & (uv[:,0] >= 0) & (uv[:,0] < W) & (uv[:,1] >= 0) & (uv[:,1] < H)
    uv, z, t_pt = uv[in_img], z[in_img], t_pt[in_img]

    # color by time-within-scan (0 → scan start, 1 → scan end)
    t_norm = (t_pt - t_pt.min()) / max(t_pt.max() - t_pt.min(), 1e-6)

    # yaw rate for title
    poses = json.load(open(ROOT/scene/'camera'/'front_camera'/'poses.json'))
    q_prev = [poses[max(0,fi-1)]['heading'][k] for k in 'xyzw']
    q_curr = [poses[fi]['heading'][k] for k in 'xyzw']
    Rp, Rc = Rotation.from_quat(q_prev), Rotation.from_quat(q_curr)
    yaw_rate = np.rad2deg(np.linalg.norm((Rc*Rp.inv()).as_rotvec()) / 0.1)

    # full view
    ax_img.imshow(img)
    # subsample for visibility
    N = len(uv); step = max(1, N // 20000)
    ax_img.scatter(uv[::step,0], uv[::step,1], c=t_norm[::step], cmap='plasma',
                   s=0.8, alpha=0.55, edgecolors='none')
    ax_img.set_title(f'scene {scene} f{fi:02d}  yaw_rate {yaw_rate:.1f}°/s',
                     fontsize=10, fontweight='bold')
    ax_img.set_xticks([]); ax_img.set_yticks([])

    # zoom: find a vertical structure near image center with many points in narrow column
    # heuristic: pick column with highest point density in central 60% of image
    cx0, cx1 = int(W*0.25), int(W*0.75)
    col_mask = (uv[:,0] >= cx0) & (uv[:,0] < cx1) & (uv[:,1] < H*0.7)
    if col_mask.sum() > 100:
        hist, bins = np.histogram(uv[col_mask,0], bins=60)
        peak_u = (bins[np.argmax(hist)] + bins[np.argmax(hist)+1]) / 2
    else:
        peak_u = W / 2
    zoom_w = 400; zoom_h = 600
    u0 = max(0, int(peak_u - zoom_w/2))
    v0 = int(H * 0.1)
    u1 = min(W, u0 + zoom_w); v1 = min(H, v0 + zoom_h)

    ax_zoom.imshow(img[v0:v1, u0:u1])
    in_zoom = (uv[:,0] >= u0) & (uv[:,0] < u1) & (uv[:,1] >= v0) & (uv[:,1] < v1)
    ax_zoom.scatter(uv[in_zoom,0] - u0, uv[in_zoom,1] - v0,
                    c=t_norm[in_zoom], cmap='plasma', s=3, alpha=0.75, edgecolors='none')
    ax_zoom.set_title(f'zoom — color = time within scan (0→100ms)', fontsize=9)
    ax_zoom.set_xticks([]); ax_zoom.set_yticks([])

    return yaw_rate


fig, axes = plt.subplots(len(PICKS), 2, figsize=(14, 3.2*len(PICKS)), dpi=110)
for row, (scene, fi) in enumerate(PICKS):
    print(f'scene {scene} f{fi} ...', flush=True)
    plot_one(scene, fi, axes[row,0], axes[row,1])

fig.suptitle('PandaSet LiDAR→camera projection: turning vs straight\n'
             'If motion compensation is perfect, time-color should not bleed along motion direction',
             fontsize=12, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0,0,1,0.975])
out = OUT / 'turn_vs_straight.png'
plt.savefig(out, dpi=110, bbox_inches='tight')
plt.close(fig)
print(f'wrote {out}')
