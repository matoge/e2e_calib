"""CAM_FRONT + LIDAR_TOP projection + 3D car boxes (strict FOV filter)."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path('/mnt/backup/nuscenes')
VER  = 'v1.0-mini'
SAMPLES = [
    '6e004695af1549c398a52b5c95e22060',
    'ff007cb7b78443e6887401d694f0d369',
    'ba2491f55619445e9d2e472f39f3b01b',
]

meta = ROOT / VER
sd_all    = json.load(open(meta/'sample_data.json'))
cal       = {c['token']: c for c in json.load(open(meta/'calibrated_sensor.json'))}
sensors   = {s['token']: s for s in json.load(open(meta/'sensor.json'))}
ego_poses = {e['token']: e for e in json.load(open(meta/'ego_pose.json'))}
anns      = json.load(open(meta/'sample_annotation.json'))
ann_by_sample = {}
for a in anns: ann_by_sample.setdefault(a['sample_token'], []).append(a)
insts  = {i['token']: i for i in json.load(open(meta/'instance.json'))}
cats   = {c['token']: c['name'] for c in json.load(open(meta/'category.json'))}

sd_by_sc = {}
for sd in sd_all:
    if not sd['is_key_frame']: continue
    ch = sensors[cal[sd['calibrated_sensor_token']]['sensor_token']]['channel']
    sd_by_sc[(sd['sample_token'], ch)] = sd


def qmat(q):
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

def sensor2world(cal_rec, ego_rec):
    R_s = qmat(cal_rec['rotation']); t_s = np.array(cal_rec['translation'])
    R_e = qmat(ego_rec['rotation']);  t_e = np.array(ego_rec['translation'])
    T_e = np.eye(4); T_e[:3,:3]=R_e; T_e[:3,3]=t_e
    S   = np.eye(4); S[:3,:3]=R_s;   S[:3,3]=t_s
    return T_e @ S

def project(pts_w, T_w2c, K):
    cam = (T_w2c[:3,:3] @ pts_w.T + T_w2c[:3,3:]).T
    z   = cam[:,2]
    uv  = (K @ cam.T)[:2] / z
    return uv.T, z

def box_corners(t, size_wlh, q):
    w, l, h = size_wlh
    c = np.array([[ l/2, l/2,-l/2,-l/2, l/2, l/2,-l/2,-l/2],
                  [ w/2,-w/2,-w/2, w/2, w/2,-w/2,-w/2, w/2],
                  [ h/2, h/2, h/2, h/2,-h/2,-h/2,-h/2,-h/2]])
    R = qmat(q)
    return (R @ c + np.array(t)[:,None]).T

EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


fig, axes = plt.subplots(len(SAMPLES), 1, figsize=(16, 9*len(SAMPLES)), dpi=80)
if len(SAMPLES) == 1: axes = [axes]

for ax, sample_tok in zip(axes, SAMPLES):
    cam_sd   = sd_by_sc[(sample_tok, 'CAM_FRONT')]
    lidar_sd = sd_by_sc[(sample_tok, 'LIDAR_TOP')]

    cam_cal = cal[cam_sd['calibrated_sensor_token']]
    K       = np.array(cam_cal['camera_intrinsic'])
    T_cam2w = sensor2world(cam_cal, ego_poses[cam_sd['ego_pose_token']])
    T_w2cam = np.linalg.inv(T_cam2w)
    img = Image.open(ROOT / cam_sd['filename']).convert('RGB')
    IW, IH = img.size

    lidar_cal = cal[lidar_sd['calibrated_sensor_token']]
    T_l2w = sensor2world(lidar_cal, ego_poses[lidar_sd['ego_pose_token']])
    pts_l = np.frombuffer((ROOT/lidar_sd['filename']).read_bytes(),
                          dtype=np.float32).reshape(-1,5)[:, :3]
    pts_w = (T_l2w[:3,:3] @ pts_l.T + T_l2w[:3,3:]).T

    uv, z = project(pts_w, T_w2cam, K)
    vis = (z > 0.5) & (uv[:,0] >= 0) & (uv[:,0] < IW) & (uv[:,1] >= 0) & (uv[:,1] < IH)

    ax.imshow(img)
    sc = ax.scatter(uv[vis,0], uv[vis,1], c=z[vis], cmap='turbo',
                    s=3, alpha=0.7, vmin=0, vmax=60)

    COLORS = {
        'vehicle.car':          'yellow',
        'vehicle.truck':        'orange',
        'vehicle.bus.rigid':    'red',
        'vehicle.bus.bendy':    'red',
        'vehicle.construction': 'magenta',
        'vehicle.trailer':      'pink',
        'vehicle.motorcycle':   'cyan',
        'vehicle.bicycle':      'lime',
    }
    counts = {k: 0 for k in COLORS}
    for a in ann_by_sample[sample_tok]:
        cat = cats.get(insts[a['instance_token']]['category_token'], '')
        if cat not in COLORS: continue
        corners_w = box_corners(a['translation'], a['size'], a['rotation'])
        uv_box, z_box = project(corners_w, T_w2cam, K)
        if (z_box <= 0.5).any(): continue
        in_img = ((uv_box[:,0] >= 0) & (uv_box[:,0] < IW) &
                  (uv_box[:,1] >= 0) & (uv_box[:,1] < IH))
        if in_img.sum() < 4: continue
        for i, j in EDGES:
            ax.plot([uv_box[i,0], uv_box[j,0]], [uv_box[i,1], uv_box[j,1]],
                    color=COLORS[cat], lw=1.5, alpha=0.9)
        counts[cat] += 1

    ax.set_xlim(0, IW); ax.set_ylim(IH, 0); ax.axis('off')
    label = " ".join(f"{k.split('.')[-1]}={v}" for k,v in counts.items() if v)
    ax.set_title(f"sample={sample_tok[:12]}…  lidar={int(vis.sum())}/{len(pts_l)}  {label}",
                 fontsize=13)

cbar = fig.colorbar(sc, ax=axes, shrink=0.6, label='depth (m)')
out = 'vis_ns_full_proj.png'
plt.savefig(out, dpi=80, bbox_inches='tight')
print(f"saved → {out}")
