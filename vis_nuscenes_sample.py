"""Visualize nuScenes sample: CAM_FRONT + LiDAR overlay + cone/barrier 3D boxes."""
import json, struct, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT   = Path('/mnt/backup/nuscenes')
SAMPLE = 'b26e791522294bec90f86fd72226e35c'
TARGET = {'movable_object.trafficcone', 'movable_object.barrier'}

# ── load meta ────────────────────────────────────────────────────────────────
sample_data = {sd['token']: sd for sd in json.load(open(ROOT/'v1.0-mini/sample_data.json'))}
sd_by_sample = {}
for sd in sample_data.values():
    sd_by_sample.setdefault(sd['sample_token'], []).append(sd)
anns        = json.load(open(ROOT/'v1.0-mini/sample_annotation.json'))
ann_by_sample = {}
for a in anns: ann_by_sample.setdefault(a['sample_token'], []).append(a)
insts  = {i['token']: i['category_token'] for i in json.load(open(ROOT/'v1.0-mini/instance.json'))}
cats   = {c['token']: c['name'] for c in json.load(open(ROOT/'v1.0-mini/category.json'))}
cal    = {c['token']: c for c in json.load(open(ROOT/'v1.0-mini/calibrated_sensor.json'))}
sensors= {s['token']: s for s in json.load(open(ROOT/'v1.0-mini/sensor.json'))}
ego_poses = {e['token']: e for e in json.load(open(ROOT/'v1.0-mini/ego_pose.json'))}

def quat_to_mat(q):
    """[w,x,y,z] → 4×4"""
    return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()

def sensor_to_world(cal_rec, ego_pose_rec):
    R_s = quat_to_mat(cal_rec['rotation'])
    t_s = np.array(cal_rec['translation'])
    R_e = quat_to_mat(ego_pose_rec['rotation'])
    t_e = np.array(ego_pose_rec['translation'])
    T = np.eye(4); T[:3,:3]=R_e; T[:3,3]=t_e
    S = np.eye(4); S[:3,:3]=R_s; S[:3,3]=t_s
    return T @ S   # world ← sensor

# ── find CAM_FRONT and LIDAR_TOP for this sample ─────────────────────────────
cam_sd = lidar_sd = None
for sd in sd_by_sample[SAMPLE]:
    ch = sensors[cal[sd['calibrated_sensor_token']]['sensor_token']]['channel']
    if ch == 'CAM_FRONT':   cam_sd   = sd
    if ch == 'LIDAR_TOP':   lidar_sd = sd

# ── camera params ─────────────────────────────────────────────────────────────
cam_cal  = cal[cam_sd['calibrated_sensor_token']]
K        = np.array(cam_cal['camera_intrinsic'])
R_cam_s  = quat_to_mat(cam_cal['rotation'])
t_cam_s  = np.array(cam_cal['translation'])
ep_cam   = ego_poses[cam_sd['ego_pose_token']]
T_cam_world = np.eye(4)
T_cam_world[:3,:3] = R_cam_s; T_cam_world[:3,3] = t_cam_s
# world→cam = inv(ego→world @ sensor→ego)
T_ego_cam   = np.eye(4); T_ego_cam[:3,:3]=quat_to_mat(ep_cam['rotation']); T_ego_cam[:3,3]=np.array(ep_cam['translation'])
T_w2cam     = np.linalg.inv(T_ego_cam @ T_cam_world)

img_path = ROOT / cam_sd['filename']
img = Image.open(img_path)
IW, IH = img.size

# ── LiDAR → camera ────────────────────────────────────────────────────────────
lidar_cal = cal[lidar_sd['calibrated_sensor_token']]
ep_lidar  = ego_poses[lidar_sd['ego_pose_token']]
T_ego_lidar = np.eye(4); T_ego_lidar[:3,:3]=quat_to_mat(lidar_cal['rotation']); T_ego_lidar[:3,3]=np.array(lidar_cal['translation'])
T_ego_world = np.eye(4); T_ego_world[:3,:3]=quat_to_mat(ep_lidar['rotation']); T_ego_world[:3,3]=np.array(ep_lidar['translation'])
T_lidar2world = T_ego_world @ T_ego_lidar

# read .pcd.bin
bin_path = ROOT / lidar_sd['filename']
pts_raw  = np.frombuffer(bin_path.read_bytes(), dtype=np.float32).reshape(-1,5)
pts_lidar = pts_raw[:,:3]
pts_world_h = np.hstack([pts_lidar, np.ones((len(pts_lidar),1))])
pts_world   = (T_lidar2world @ pts_world_h.T).T[:,:3]
pts_cam_h   = (T_w2cam @ np.hstack([pts_world, np.ones((len(pts_world),1))]).T).T
z = pts_cam_h[:,2]
uv = (K @ pts_cam_h[:,:3].T)[:2] / z
uv = uv.T
vis = (z>0.5)&(uv[:,0]>=0)&(uv[:,0]<IW)&(uv[:,1]>=0)&(uv[:,1]<IH)

# ── 3D box corners ─────────────────────────────────────────────────────────────
def box_corners(t, s, q):
    """world-frame 8 corners. nuScenes size=[width,length,height], quat=[w,x,y,z]"""
    w,l,h = s   # nuScenes: size=[width, length, height]
    cx = np.array([[1,1,-1,-1, 1, 1,-1,-1],
                   [1,-1,-1, 1, 1,-1,-1, 1],
                   [1,1, 1, 1,-1,-1,-1,-1]], dtype=float) * np.array([[l/2],[w/2],[h/2]])
    R = Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
    return (R @ cx + np.array(t)[:,None]).T   # (8,3)

EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def project_box(corners_world, T_w2cam, K):
    h = np.hstack([corners_world, np.ones((8,1))])
    cam = (T_w2cam @ h.T).T
    z = cam[:,2]
    if (z<=0).any(): return None
    uv = (K @ cam[:,:3].T)[:2] / z
    return uv.T   # (8,2)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16,9), dpi=100)
ax.imshow(img)
ax.scatter(uv[vis,0], uv[vis,1], c=z[vis].clip(0,50), cmap='jet_r',
           s=1, alpha=0.7, vmin=0, vmax=50)

colors = {'movable_object.trafficcone': 'cyan',
          'movable_object.barrier':     'orange'}
counts = {'movable_object.trafficcone': 0, 'movable_object.barrier': 0}

for a in ann_by_sample[SAMPLE]:
    cat = cats.get(insts.get(a['instance_token'],''),'')
    if cat not in TARGET: continue
    corners = box_corners(a['translation'], a['size'], a['rotation'])
    uv_box  = project_box(corners, T_w2cam, K)
    if uv_box is None: continue
    col = colors[cat]
    for i,j in EDGES:
        ax.plot([uv_box[i,0],uv_box[j,0]], [uv_box[i,1],uv_box[j,1]],
                color=col, lw=1.2, alpha=0.85)
    counts[cat] += 1

ax.set_xlim(0, IW); ax.set_ylim(IH, 0)
ax.set_title(f'nuScenes  cone={counts["movable_object.trafficcone"]}  '
             f'barrier={counts["movable_object.barrier"]}  '
             f'LiDAR pts={vis.sum()}', fontsize=11)
ax.axis('off')
plt.tight_layout(pad=0.1)
plt.savefig('vis_nuscenes.png', dpi=100, bbox_inches='tight')
plt.close()
print(f"Saved → vis_nuscenes.png")
