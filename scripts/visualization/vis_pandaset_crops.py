"""Visualize all object crops for one frame + run inference."""
import json, pickle, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation
from dataset_pandaset import (_quat_pos_to_mat, _project, _bbox2d_of_cuboid,
                               USEFUL_LABELS)
from model_depth import CalibNetDepth

PANDASET  = '/mnt/mininas/datasets/pandaset'
SCENE     = '001'
FRAME     = 0
CKPT      = 'experiments/ps_v6/best_model.pt'
BBOX_SCALE = 1.5
IMG_SIZE   = 64
DEVICE     = torch.device('cuda')
MAX_OFFSET = 0.20   # metres
MAX_ROT    = 0.5    # degrees

# ── load model ────────────────────────────────────────────────────────────────
model = CalibNetDepth(img_size=IMG_SIZE, in_channels=3, n_layers=3,
                      use_convnext=True, use_frustum=False).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
model.eval()

# ── load scene data ───────────────────────────────────────────────────────────
sc_dir  = Path(PANDASET) / SCENE
cam_dir = sc_dir / 'camera' / 'front_camera'
with open(cam_dir / 'poses.json') as f:  all_poses = json.load(f)
with open(cam_dir / 'intrinsics.json') as f:  intr = json.load(f)

K = np.array([[intr['fx'], 0, intr['cx']],
              [0, intr['fy'], intr['cy']], [0,0,1]], dtype=np.float64)
IW, IH = 1920, 1080

cam_pose = all_poses[FRAME]
pose_mat = _quat_pos_to_mat(cam_pose['heading'], cam_pose['position'])
cam_pos  = np.array([cam_pose['position']['x'], cam_pose['position']['y'],
                     cam_pose['position']['z']], dtype=np.float32)
R_gt = Rotation.from_quat([cam_pose['heading']['x'], cam_pose['heading']['y'],
                            cam_pose['heading']['z'], cam_pose['heading']['w']]).as_matrix().astype(np.float32)
T_gt = np.linalg.inv(pose_mat).astype(np.float32)

lidar_df = pickle.load(open(sorted((sc_dir/'lidar').glob('*.pkl'))[FRAME], 'rb'))
lidar_df = lidar_df[lidar_df['d'] == 0]
pts_world = lidar_df[['x','y','z']].values.astype(np.float32)
uv_gt_all, z_gt_all = _project(pts_world, pose_mat, K)
vis = (z_gt_all > 0.5) & (uv_gt_all[:,0] >= 0) & (uv_gt_all[:,0] < IW) & \
      (uv_gt_all[:,1] >= 0) & (uv_gt_all[:,1] < IH)
pts_vis = pts_world[vis]

img_full = Image.open(cam_dir / f'{FRAME:02d}.jpg').convert('RGB')
cuboids  = pickle.load(open(sorted((sc_dir/'annotations'/'cuboids').glob('*.pkl'))[FRAME],'rb'))
cuboids  = cuboids[cuboids['label'].isin(USEFUL_LABELS)]

# ── build per-object crops ────────────────────────────────────────────────────
crops = []
for _, obj in cuboids.iterrows():
    pos  = np.array([obj['position.x'], obj['position.y'], obj['position.z']])
    dims = np.array([obj['dimensions.x'], obj['dimensions.y'], obj['dimensions.z']])
    yaw  = obj['yaw']
    bbox = _bbox2d_of_cuboid(pos, dims, yaw, pose_mat, K)
    if bbox is None: continue
    u_min, v_min, u_max, v_max = bbox
    uc, vc = (u_min+u_max)/2, (v_min+v_max)/2
    if not (0 <= uc < IW and 0 <= vc < IH): continue

    bw, bh    = u_max-u_min, v_max-v_min
    crop_size = max(max(bw,bh)*BBOX_SCALE, 32)
    half      = crop_size/2
    u0 = float(np.clip(uc-half, 0, IW-crop_size))
    v0 = float(np.clip(vc-half, 0, IH-crop_size))
    scale = IMG_SIZE / crop_size

    box    = (int(u0), int(v0), int(u0+crop_size), int(v0+crop_size))
    img_64 = np.array(img_full.crop(box).resize((IMG_SIZE,IMG_SIZE),Image.BILINEAR), dtype=np.uint8)

    # is_obj: 3D cuboid check
    c_y, s_y = np.cos(yaw), np.sin(yaw)
    R_obj = np.array([[c_y,s_y,0],[-s_y,c_y,0],[0,0,1]], dtype=np.float32)
    pts_local = (R_obj @ (pts_vis - pos).T).T
    half3 = dims/2
    is_obj_mask = ((np.abs(pts_local[:,0])<=half3[0]) &
                   (np.abs(pts_local[:,1])<=half3[1]) &
                   (np.abs(pts_local[:,2])<=half3[2]))

    # random perturbation
    t_delta = (np.random.rand(3)*2-1) * MAX_OFFSET
    ypr     = (np.random.rand(3)*2-1) * MAX_ROT
    R_off   = R_gt @ Rotation.from_euler('zyx', ypr, degrees=True).as_matrix()
    cp_off  = cam_pos + t_delta
    T_off   = np.eye(4, dtype=np.float32)
    T_off[:3,:3] = R_off.T; T_off[:3,3] = -(R_off.T @ cp_off)

    pts_cam_off = T_off[:3,:3] @ pts_vis.T + T_off[:3,3:]
    z_off = pts_cam_off[2]
    uv_off = ((K @ pts_cam_off)[:2] / z_off).T

    in_crop = ((uv_off[:,0]>=u0)&(uv_off[:,0]<u0+crop_size)&
               (uv_off[:,1]>=v0)&(uv_off[:,1]<v0+crop_size)&(z_off>0.5))
    if in_crop.sum() < 4: continue

    uv_off_c = np.stack([(uv_off[in_crop,0]-u0)*scale,
                          (uv_off[in_crop,1]-v0)*scale], axis=1)
    pts_sel  = pts_vis[in_crop]
    pts_cam_gt = T_gt[:3,:3] @ pts_sel.T + T_gt[:3,3:]
    uv_gt_c = ((K @ pts_cam_gt)[:2] / pts_cam_gt[2]).T
    uv_gt_c = np.stack([(uv_gt_c[:,0]-u0)*scale,
                         (uv_gt_c[:,1]-v0)*scale], axis=1)

    dist_m = (np.linalg.norm(pts_sel - cam_pos, axis=1) / 100.0).astype(np.float32)
    is_obj = is_obj_mask[in_crop].astype(np.float32)

    dist_uvd = np.stack([uv_off_c[:,0], uv_off_c[:,1], dist_m, is_obj], axis=1).astype(np.float32)
    true_uvd = np.stack([uv_gt_c[:,0],  uv_gt_c[:,1],  dist_m, is_obj], axis=1).astype(np.float32)

    # inference
    img_t = torch.from_numpy(img_64).permute(2,0,1).float().div(255).unsqueeze(0).to(DEVICE)
    uvd_t = torch.from_numpy(dist_uvd[:,:3]).unsqueeze(0).to(DEVICE)
    pad_t = torch.zeros(1, len(dist_uvd), dtype=torch.bool, device=DEVICE)
    with torch.no_grad():
        params = model(img_t, uvd_t, key_padding_mask=pad_t)[0].cpu().float().numpy()
    pred_uv = dist_uvd[:,:2] + params[:,:2]

    crops.append(dict(img=img_64, true_uv=uv_gt_c, dist_uv=uv_off_c,
                      pred_uv=pred_uv, is_obj=is_obj,
                      u0=u0, v0=v0, crop_size=crop_size, label=obj['label']))

print(f"{len(crops)} crops")

# ── figure 1: full image with boxes ──────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(19.2, 10.8), dpi=100)
ax1.imshow(img_full)
uv_all, z_all = _project(pts_vis, pose_mat, K)
ax1.scatter(uv_all[:,0], uv_all[:,1], c=z_all.clip(0,60), cmap='jet_r',
            s=1, alpha=0.6, vmin=0, vmax=60)
for i, cr in enumerate(crops):
    rect = patches.Rectangle((cr['u0'],cr['v0']), cr['crop_size'], cr['crop_size'],
                               lw=1.5, edgecolor='yellow', facecolor='none')
    ax1.add_patch(rect)
    ax1.text(cr['u0']+4, cr['v0']+16, f"{i} {cr['label'][:3]}", color='yellow', fontsize=7)
ax1.axis('off'); ax1.set_title(f'Scene {SCENE}  Frame {FRAME:02d}  — {len(crops)} objects', fontsize=11)
plt.tight_layout(pad=0.1)
fig1.savefig('vis_ps_fullframe.png', dpi=100, bbox_inches='tight')
plt.close(fig1)

# ── figure 2: per-crop grid ───────────────────────────────────────────────────
nc = len(crops)
cols = min(nc, 6)
rows = (nc + cols - 1) // cols
fig2, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3), dpi=96)
axes = np.array(axes).flatten() if nc > 1 else [axes]

for i, cr in enumerate(crops):
    ax = axes[i]
    ax.imshow(cr['img'])
    is_o = cr['is_obj'] > 0.5
    for mask, col_gt, col_d, col_p, lbl in [
        (is_o,   'lime',  'red',    'cyan',       'obj'),
        (~is_o,  'yellow','orange', 'deepskyblue','bg'),
    ]:
        if not mask.any(): continue
        ax.scatter(cr['true_uv'][mask,0], cr['true_uv'][mask,1],
                   c=col_gt, s=12, marker='x', lw=1.2, zorder=3)
        ax.scatter(cr['dist_uv'][mask,0], cr['dist_uv'][mask,1],
                   c=col_d,  s=6,  alpha=0.5, zorder=2)
        ax.scatter(cr['pred_uv'][mask,0], cr['pred_uv'][mask,1],
                   c=col_p,  s=12, marker='+', lw=1.4, zorder=4)
    err_b = np.linalg.norm(cr['dist_uv'][is_o]-cr['true_uv'][is_o],axis=1).mean() if is_o.any() else float('nan')
    err_a = np.linalg.norm(cr['pred_uv'][is_o]-cr['true_uv'][is_o],axis=1).mean() if is_o.any() else float('nan')
    ax.set_title(f"{i} {cr['label'][:6]}\n{err_b:.2f}→{err_a:.2f}px", fontsize=6)
    ax.axis('off')
for ax in axes[len(crops):]: ax.axis('off')

plt.suptitle(f'Scene {SCENE}  Frame {FRAME:02d}  — green=GT, red=dist, cyan=pred (obj only)',
             fontsize=9)
plt.tight_layout(pad=0.3)
fig2.savefig('vis_ps_crops.png', dpi=96, bbox_inches='tight')
plt.close(fig2)
print("Saved → vis_ps_fullframe.png  vis_ps_crops.png")
