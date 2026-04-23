"""Show what __getitem__ would output if bus/truck were included as target cats.

Replicates the cache-build crop + __getitem__ perturbation logic inline for
vehicle.truck and vehicle.bus.* annotations so we can see the 64x64 training
crops before committing to a cache rebuild.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path('/mnt/backup/nuscenes')
VER  = 'v1.0-mini'
TARGET_CATS = {'vehicle.truck', 'vehicle.bus.rigid', 'vehicle.bus.bendy'}
N_SHOW = 12
BBOX_SCALE = 1.5
MAX_OFFSET_M = 0.20
MAX_ROT_DEG  = 0.5

meta = ROOT / VER
sd_all    = json.load(open(meta/'sample_data.json'))
cal       = {c['token']: c for c in json.load(open(meta/'calibrated_sensor.json'))}
sensors   = {s['token']: s for s in json.load(open(meta/'sensor.json'))}
ego_poses = {e['token']: e for e in json.load(open(meta/'ego_pose.json'))}
anns      = json.load(open(meta/'sample_annotation.json'))
insts     = {i['token']: i for i in json.load(open(meta/'instance.json'))}
cats      = {c['token']: c['name'] for c in json.load(open(meta/'category.json'))}

sd_by_sc = {}
for sd in sd_all:
    if not sd['is_key_frame']: continue
    ch = sensors[cal[sd['calibrated_sensor_token']]['sensor_token']]['channel']
    sd_by_sc[(sd['sample_token'], ch)] = sd


def qmat(q): return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
def s2w(c, e):
    R_s=qmat(c['rotation']); t_s=np.array(c['translation'])
    R_e=qmat(e['rotation']);  t_e=np.array(e['translation'])
    T=np.eye(4); T[:3,:3]=R_e; T[:3,3]=t_e
    S=np.eye(4); S[:3,:3]=R_s; S[:3,3]=t_s
    return T@S
def proj(pts_w, T_w2c, K):
    cam = (T_w2c[:3,:3] @ pts_w.T + T_w2c[:3,3:]).T
    z = cam[:,2]; uv = (K @ cam.T)[:2]/z
    return uv.T, z
def bcor(t, wlh, q):
    w,l,h = wlh
    c = np.array([[ l/2, l/2,-l/2,-l/2, l/2, l/2,-l/2,-l/2],
                  [ w/2,-w/2,-w/2, w/2, w/2,-w/2,-w/2, w/2],
                  [ h/2, h/2, h/2, h/2,-h/2,-h/2,-h/2,-h/2]])
    return (qmat(q) @ c + np.array(t)[:,None]).T


# ── collect candidate (sample, annotation) pairs ─────────────────────────────
rng = np.random.default_rng(0)
candidates = []
for a in anns:
    cat = cats.get(insts[a['instance_token']]['category_token'], '')
    if cat not in TARGET_CATS: continue
    if (a['sample_token'], 'CAM_FRONT') not in sd_by_sc: continue
    candidates.append((a['sample_token'], a, cat))
print(f"candidates with target cat: {len(candidates)}")


# ── build (crop, dist_uv, true_uv, is_obj) for an annotation ────────────────
def process(sample_tok, a):
    cam_sd   = sd_by_sc[(sample_tok, 'CAM_FRONT')]
    lidar_sd = sd_by_sc[(sample_tok, 'LIDAR_TOP')]
    cam_cal = cal[cam_sd['calibrated_sensor_token']]
    K = np.array(cam_cal['camera_intrinsic'])
    T_c2w = s2w(cam_cal, ego_poses[cam_sd['ego_pose_token']])
    T_w2c = np.linalg.inv(T_c2w).astype(np.float32)
    cp = T_c2w[:3,3].astype(np.float32)
    img = Image.open(ROOT / cam_sd['filename']).convert('RGB')
    IW, IH = img.size

    lidar_cal = cal[lidar_sd['calibrated_sensor_token']]
    T_l2w = s2w(lidar_cal, ego_poses[lidar_sd['ego_pose_token']])
    pts_l = np.frombuffer((ROOT/lidar_sd['filename']).read_bytes(),
                          dtype=np.float32).reshape(-1,5)[:, :3]
    pts_w = (T_l2w[:3,:3] @ pts_l.T + T_l2w[:3,3:]).T.astype(np.float32)
    uv_gt, z_gt = proj(pts_w, T_w2c, K)
    vis = (z_gt > 0.5) & (uv_gt[:,0]>=0) & (uv_gt[:,0]<IW) & (uv_gt[:,1]>=0) & (uv_gt[:,1]<IH)
    if vis.sum() < 8: return None
    pts_vis = pts_w[vis]; uv_vis = uv_gt[vis]

    # 2D bbox from 3D box corners
    corners_w = bcor(a['translation'], a['size'], a['rotation'])
    uv_box, z_box = proj(corners_w, T_w2c, K)
    front = z_box > 0.1
    if front.sum() < 4: return None
    uv_box = uv_box[front]
    u_min, v_min = uv_box[:,0].min(), uv_box[:,1].min()
    u_max, v_max = uv_box[:,0].max(), uv_box[:,1].max()
    uc, vc = (u_min+u_max)/2, (v_min+v_max)/2
    if not (0 <= uc < IW and 0 <= vc < IH): return None
    bw, bh = u_max-u_min, v_max-v_min
    crop_size = max(max(bw,bh)*BBOX_SCALE, 32)
    u0 = float(np.clip(uc-crop_size/2, 0, IW-crop_size))
    v0 = float(np.clip(vc-crop_size/2, 0, IH-crop_size))
    crop_size = float(crop_size)

    box = (int(u0), int(v0), int(u0+crop_size), int(v0+crop_size))
    img_64 = np.array(img.crop(box).resize((64,64), Image.BILINEAR))

    # ROI gather (same 32×32 grid as cache)
    GS = 32; margin = crop_size*1.5; roi_w = crop_size + 2*margin
    in_roi = ((uv_vis[:,0]>=u0-margin) & (uv_vis[:,0]<u0+crop_size+margin) &
              (uv_vis[:,1]>=v0-margin) & (uv_vis[:,1]<v0+crop_size+margin))
    pts_roi = pts_vis[in_roi]; uv_roi = uv_vis[in_roi]
    if len(pts_roi) < 8: return None
    gu = u0-margin + (np.arange(GS)+0.5)*(roi_w/GS)
    gv = v0-margin + (np.arange(GS)+0.5)*(roi_w/GS)
    ggu, ggv = np.meshgrid(gu, gv); ggu=ggu.ravel(); ggv=ggv.ravel()
    d2 = (uv_roi[:,0][None]-ggu[:,None])**2 + (uv_roi[:,1][None]-ggv[:,None])**2
    g_sel = sorted(set(d2.argmin(axis=0).tolist()))  # axis=0 because we flipped order
    # correct argmin axis: we want for each grid cell, nearest pt idx
    d2 = (uv_roi[:,0][None]-ggu[:,None])**2 + (uv_roi[:,1][None]-ggv[:,None])**2  # (G, N_roi)
    g_sel = sorted(set(d2.argmin(axis=1).tolist()))  # per grid cell → pt idx
    pts_samp = pts_roi[g_sel]

    # perturbation (mimic __getitem__)
    R_gt = T_c2w[:3,:3]
    t_delta = (np.random.rand(3)*2-1) * MAX_OFFSET_M
    ypr     = (np.random.rand(3)*2-1) * MAX_ROT_DEG
    R_off   = R_gt @ Rotation.from_euler('zyx', ypr, degrees=True).as_matrix()
    cp_off  = cp + t_delta
    T_off   = np.eye(4, dtype=np.float32)
    T_off[:3,:3] = R_off.T
    T_off[:3, 3] = -(R_off.T @ cp_off)
    pcam_off = T_off[:3,:3] @ pts_samp.T + T_off[:3,3:]
    z_off = pcam_off[2]
    uv_off = ((K @ pcam_off)[:2] / z_off).T
    in_crop = ((uv_off[:,0]>=u0) & (uv_off[:,0]<u0+crop_size) &
               (uv_off[:,1]>=v0) & (uv_off[:,1]<v0+crop_size) &
               (z_off > 0.5))
    if in_crop.sum() < 8: return None
    scale = 64/crop_size
    uv_off_c = np.stack([(uv_off[in_crop,0]-u0)*scale,
                          (uv_off[in_crop,1]-v0)*scale], axis=1)
    pts_sel  = pts_samp[in_crop]

    # 16×16 grid selection (same as __getitem__)
    sel = []
    cell = 64/16
    for gi in range(16):
        for gj in range(16):
            d2 = ((uv_off_c[:,0]-(gj+.5)*cell)**2 + (uv_off_c[:,1]-(gi+.5)*cell)**2)
            sel.append(int(d2.argmin()))
    sel = sorted(set(sel))
    uv_off_c = uv_off_c[sel]; pts_sel = pts_sel[sel]

    # true projection (GT, same subset)
    pc_gt = T_w2c[:3,:3] @ pts_sel.T + T_w2c[:3,3:]
    uv_gt2 = ((K @ pc_gt)[:2] / pc_gt[2]).T
    uv_gt_c = np.stack([(uv_gt2[:,0]-u0)*scale, (uv_gt2[:,1]-v0)*scale], axis=1)

    # is_obj: 3D cuboid check
    pos = np.array(a['translation'], dtype=np.float32)
    wlh = np.array(a['size'], dtype=np.float32)
    obj_yaw = Rotation.from_quat([a['rotation'][1], a['rotation'][2],
                                   a['rotation'][3], a['rotation'][0]]).as_euler('zyx')[0]
    cy, sy = np.cos(obj_yaw), np.sin(obj_yaw)
    R_o = np.array([[cy,sy,0],[-sy,cy,0],[0,0,1]], dtype=np.float32)
    pts_local = (R_o @ (pts_sel - pos).T).T
    half = wlh/2
    # FIXED: local_x = length, local_y = width (nuScenes Box convention)
    is_obj = ((np.abs(pts_local[:,0])<=half[1]) &
              (np.abs(pts_local[:,1])<=half[0]) &
              (np.abs(pts_local[:,2])<=half[2]))
    return img_64, uv_gt_c, uv_off_c, is_obj, crop_size


# pick samples
rng.shuffle(candidates)
results = []
for sample_tok, a, cat in candidates:
    r = process(sample_tok, a)
    if r is None: continue
    results.append((cat, r))
    if len(results) >= N_SHOW: break

print(f"produced {len(results)} crops")

rows, cols = 3, 4
fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4), dpi=120)
for ax, (cat, r) in zip(axes.ravel(), results):
    img_64, uv_gt, uv_off, is_obj, cs = r
    ax.imshow(img_64)
    ax.scatter(uv_off[:,0], uv_off[:,1], c='red',  s=14, alpha=0.7, label='dist')
    ax.scatter(uv_gt[:,0],  uv_gt[:,1],  c='lime', s=14, alpha=0.7, label='true')
    if is_obj.any():
        ax.scatter(uv_gt[is_obj,0], uv_gt[is_obj,1], facecolors='none',
                   edgecolors='yellow', s=70, lw=1.0)
    ax.set_xlim(0,64); ax.set_ylim(64,0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{cat.split('.')[-1]}  N={len(uv_gt)}  obj={int(is_obj.sum())}  "
                 f"crop={cs:.0f}px", fontsize=9)

axes[0,0].legend(fontsize=7, loc='lower right')
fig.suptitle(f"bus / truck proposed target cats — __getitem__ simulation", fontsize=12)
plt.tight_layout()
out = 'vis_ns_bus_truck.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"saved → {out}")
