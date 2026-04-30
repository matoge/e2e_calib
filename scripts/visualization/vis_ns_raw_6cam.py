"""nuScenes raw-data 6-camera LiDAR overlay — verifies per-sensor ego_pose lookup.

Each sample_data has its own ego_pose_token (timestamp-aligned). So even though
LIDAR_TOP captures at 20Hz and CAMs at 12Hz, every cam pose is interpolated by
nuScenes ingestion at the cam's own timestamp. We project:
  pts_world = T_world_ego(t_lidar) @ T_ego_lidar @ pts_lidar
  uv        = K @ T_cam_ego_inv @ T_world_ego_inv(t_cam) @ pts_world
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import orjson, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation
def _load(p):
    print(f'  loading {p.name} ({p.stat().st_size/1e6:.0f} MB)...', flush=True)
    return orjson.loads(p.read_bytes())

ROOT = Path('/mnt/mininas/datasets/nuscenes/data')
VER  = 'v1.0-trainval'

CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT']
MAX_D = 60.0

meta = ROOT / VER
print(f'loading meta from {meta}', flush=True)
samples  = _load(meta/'sample.json')
sensors  = {s['token']: s for s in _load(meta/'sensor.json')}
cal_t    = {c['token']: c for c in _load(meta/'calibrated_sensor.json')}
sd_all   = _load(meta/'sample_data.json')
ego_pose = {e['token']: e for e in _load(meta/'ego_pose.json')}

# pick scene-0001 first sample
scene_first = samples[0]
sample_tok  = scene_first['token']
print(f'sample_token={sample_tok}  scene={scene_first["scene_token"][:8]}…', flush=True)

# index sample_data by (sample_token, channel)
sd_by_sc = {}
for sd in sd_all:
    if not sd['is_key_frame']: continue
    ch = sensors[cal_t[sd['calibrated_sensor_token']]['sensor_token']]['channel']
    sd_by_sc[(sd['sample_token'], ch)] = sd


def qmat(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

def sensor2world(cal_rec, ego_rec):
    R_s = qmat(cal_rec['rotation']); t_s = np.array(cal_rec['translation'])
    R_e = qmat(ego_rec['rotation']);  t_e = np.array(ego_rec['translation'])
    T_e = np.eye(4); T_e[:3,:3]=R_e; T_e[:3,3]=t_e
    S   = np.eye(4); S[:3,:3]=R_s;   S[:3,3]=t_s
    return T_e @ S


# lidar → world (using LIDAR's own ego_pose)
lidar_sd = sd_by_sc[(sample_tok, 'LIDAR_TOP')]
lidar_cal = cal_t[lidar_sd['calibrated_sensor_token']]
T_l2w = sensor2world(lidar_cal, ego_pose[lidar_sd['ego_pose_token']])
pts_l = np.frombuffer((ROOT/lidar_sd['filename']).read_bytes(),
                      dtype=np.float32).reshape(-1,5)[:, :3]
pts_w = (T_l2w[:3,:3] @ pts_l.T + T_l2w[:3,3:]).T
print(f'lidar pts: {len(pts_w)}  ts_lidar={lidar_sd["timestamp"]}', flush=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
for ax, ch in zip(axes.ravel(), CAMS):
    cam_sd = sd_by_sc[(sample_tok, ch)]
    cam_cal = cal_t[cam_sd['calibrated_sensor_token']]
    K = np.array(cam_cal['camera_intrinsic'])
    T_cam2w = sensor2world(cam_cal, ego_pose[cam_sd['ego_pose_token']])  # cam's own ego_pose
    T_w2cam = np.linalg.inv(T_cam2w)
    img = Image.open(ROOT / cam_sd['filename']).convert('RGB')
    IW, IH = img.size

    cam = (T_w2cam[:3,:3] @ pts_w.T + T_w2cam[:3,3:]).T
    z = cam[:,2]
    uv = (K @ cam.T)[:2] / z
    uv = uv.T
    vis = (z > 0.5) & (uv[:,0]>=0) & (uv[:,0]<IW) & (uv[:,1]>=0) & (uv[:,1]<IH)

    dt_ms = (cam_sd['timestamp'] - lidar_sd['timestamp']) / 1e3
    ax.imshow(img)
    ax.scatter(uv[vis,0], uv[vis,1], c=z[vis].clip(0, MAX_D),
               cmap='turbo', s=1.2, alpha=0.7, vmin=0, vmax=MAX_D)
    ax.set_title(f'{ch}   {vis.sum()} pts   Δt={dt_ms:+.1f}ms', fontsize=9)
    ax.axis('off')

fig.suptitle(f'nuScenes RAW  sample={sample_tok[:12]}…  (per-sensor ego_pose)', fontsize=11)
plt.tight_layout()
out = 'vis_ns_raw_6cam.png'
plt.savefig(out, dpi=80, bbox_inches='tight')
print(f'saved → {out}')
