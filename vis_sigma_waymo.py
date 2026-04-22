"""Waymo per-position CalibNet σ/μ statistics. Same methodology as PandaSet.
Decodes raw parquet range images, projects top-LiDAR onto front cam, 6x3 crop grid.
"""
import io, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_depth import CalibNetDepth
from dataset_waymo import _load_cam_cal, _load_lidar_cal, _decode_range_image, _project

WAYMO_DIR = Path('/mnt/nvme6t/waymo/training')
DEVICE = torch.device('cpu')
CKPT = 'experiments/all_v3_mc/best_model.pt'

model = CalibNetDepth(img_size=64, in_channels=3, n_layers=4,
                       use_convnext=True, use_frustum=True).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
model.eval()

SEGS = [
    '10017090168044687777_6380_000_6400_000',
    '10023947602400723454_1120_000_1140_000',
    '1005081002024129653_5313_150_5333_150',
]
FRAMES_PER_SEG = 8
CROP_SIZE = 320
NCOL, NROW = 6, 3
IMG_W, IMG_H = 1920, 1280
u_centers = np.linspace(CROP_SIZE/2+20, IMG_W-CROP_SIZE/2-20, NCOL)
v_centers = np.linspace(CROP_SIZE/2+20, IMG_H-CROP_SIZE/2-20, NROW)

FRONT_CAM, TOP_LIDAR = 1, 1

def load_seg(seg):
    fu, fv, cu, cv, IW, IH, T_veh_from_cam = _load_cam_cal(seg)
    T_cam_from_veh = np.linalg.inv(T_veh_from_cam).astype(np.float64)
    T_veh_from_lidar, incl, az_corr = _load_lidar_cal(seg)
    lidar_df = pd.read_parquet(WAYMO_DIR/'lidar'/f'{seg}.parquet')
    cam_df   = pd.read_parquet(WAYMO_DIR/'camera_image'/f'{seg}.parquet')
    lidar_top = lidar_df[lidar_df['key.laser_name']==TOP_LIDAR]
    cam_front = cam_df[cam_df['key.camera_name']==FRONT_CAM]
    return fu, fv, cu, cv, T_cam_from_veh, T_veh_from_lidar, incl, az_corr, lidar_top, cam_front

def decode_frame(seg_data, ts):
    fu, fv, cu, cv, T_cam_from_veh, T_veh_from_lidar, incl, az_corr, lidar_top, cam_front = seg_data
    lr = lidar_top[lidar_top['key.frame_timestamp_micros']==ts]
    cr = cam_front[cam_front['key.frame_timestamp_micros']==ts]
    if lr.empty or cr.empty: return None
    lr, cr = lr.iloc[0], cr.iloc[0]
    pts_sensor = _decode_range_image(
        lr['[LiDARComponent].range_image_return1.values'],
        lr['[LiDARComponent].range_image_return1.shape'], incl, az_corr)
    if len(pts_sensor) == 0: return None
    # sensor → vehicle
    R, t = T_veh_from_lidar[:3,:3], T_veh_from_lidar[:3,3]
    pts_veh = (R @ pts_sensor.T).T + t
    uv, depth = _project(pts_veh, T_cam_from_veh, fu, fv, cu, cv)
    img_bytes = cr['[CameraImageComponent].image']
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert('RGB'))
    # compute euclidean range in cam frame for training-convention depth
    h = np.hstack([pts_veh, np.ones((len(pts_veh),1))])
    pts_cam = (T_cam_from_veh @ h.T).T[:,:3]
    rng = np.linalg.norm(pts_cam, axis=1) / 100.0
    H, W = img.shape[:2]
    ok = (depth>0.5)&(uv[:,0]>=0)&(uv[:,0]<W)&(uv[:,1]>=0)&(uv[:,1]<H)
    return img, uv[ok], depth[ok], rng[ok]

recs = []
t0 = time.time()
for seg in SEGS:
    print(f'loading {seg[:30]}...', flush=True)
    sd = load_seg(seg)
    timestamps = sorted(sd[8]['key.frame_timestamp_micros'].unique())
    # sample evenly
    idxs = np.linspace(0, len(timestamps)-1, FRAMES_PER_SEG).astype(int)
    for i in idxs:
        ts = timestamps[i]
        out = decode_frame(sd, ts)
        if out is None: continue
        img, uv_f, _, rng_f = out
        H, W = img.shape[:2]
        for uc in u_centers:
            for vc in v_centers:
                u0, v0 = int(uc-CROP_SIZE/2), int(vc-CROP_SIZE/2)
                u1, v1 = u0+CROP_SIZE, v0+CROP_SIZE
                if u1 > W or v1 > H: continue
                crop = img[v0:v1, u0:u1]
                img64 = np.array(Image.fromarray(crop).resize((64,64), Image.BILINEAR))
                m = (uv_f[:,0]>=u0)&(uv_f[:,0]<u1)&(uv_f[:,1]>=v0)&(uv_f[:,1]<v1)
                if m.sum()<8: continue
                u_c = (uv_f[m,0]-u0)/CROP_SIZE*64.0
                v_c = (uv_f[m,1]-v0)/CROP_SIZE*64.0
                d_c = rng_f[m]
                uvd = np.stack([u_c,v_c,d_c],axis=1)
                if len(uvd)>256:
                    idx = np.random.default_rng(int(ts)).permutation(len(uvd))[:256]
                    uvd = uvd[idx]
                with torch.no_grad():
                    im_b = torch.from_numpy(img64).permute(2,0,1).float().unsqueeze(0)/255.0
                    uvd_b = torch.from_numpy(uvd).float().unsqueeze(0)
                    pr = model(im_b, uvd_b).squeeze(0).numpy()
                recs.append((uc, vc, pr[:,0].mean(), pr[:,1].mean(),
                             np.exp(pr[:,2]).mean(), np.exp(pr[:,3]).mean(), seg))
    print(f'  {len(recs)} records so far ({time.time()-t0:.0f}s)')

rec = np.array([r[:6] for r in recs], dtype=float)
seg_arr = np.array([r[6] for r in recs])
print(f'\ntotal: {len(rec)} crops')
print(f'global mean μx={rec[:,2].mean():+.3f}  μy={rec[:,3].mean():+.3f}  '
      f'σx={rec[:,4].mean():.3f}  σy={rec[:,5].mean():.3f}')

# per-segment breakdown — middle row only (cleanest)
print('\n=== per-segment middle-row (v=640) μ_x profile ===')
for seg in SEGS:
    print(f'\n{seg[:35]}:')
    m_seg = seg_arr == seg
    for uc in u_centers:
        m = m_seg & (np.abs(rec[:,0]-uc)<1) & (np.abs(rec[:,1]-v_centers[1])<1)
        if m.sum()==0: continue
        mx = rec[m,2].mean(); my = rec[m,3].mean()
        sx = rec[m,4].mean()
        print(f'  u={uc:6.0f}  μx={mx:+.3f}  μy={my:+.3f}  σx={sx:.2f}  n={m.sum()}')

# 6×3 grids
grid_mx = np.full((NROW,NCOL), np.nan); grid_my = np.full_like(grid_mx, np.nan)
grid_sx = np.full_like(grid_mx, np.nan); grid_sy = np.full_like(grid_mx, np.nan); grid_n = np.zeros_like(grid_mx, dtype=int)
for iu, uc in enumerate(u_centers):
    for iv, vc in enumerate(v_centers):
        m = (np.abs(rec[:,0]-uc)<1)&(np.abs(rec[:,1]-vc)<1)
        if m.sum()==0: continue
        grid_mx[iv,iu], grid_my[iv,iu] = rec[m,2].mean(), rec[m,3].mean()
        grid_sx[iv,iu], grid_sy[iv,iu] = rec[m,4].mean(), rec[m,5].mean()
        grid_n[iv,iu] = m.sum()

fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=110)
for ax, data, title, cmap, fmt in [
    (axes[0,0], grid_sx, 'mean σ_x', 'viridis', '{:.2f}'),
    (axes[0,1], grid_sy, 'mean σ_y', 'viridis', '{:.2f}'),
    (axes[1,0], grid_mx, 'mean μ_x', 'RdBu_r', '{:+.2f}'),
    (axes[1,1], grid_my, 'mean μ_y', 'RdBu_r', '{:+.2f}'),
]:
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    if 'μ' in title:
        a = max(abs(vmin), abs(vmax)); vmin, vmax = -a, a
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto',
                   extent=[0, IMG_W, IMG_H, 0])
    for iu, uc in enumerate(u_centers):
        for iv, vc in enumerate(v_centers):
            if grid_n[iv,iu]>0:
                ax.text(uc, vc, fmt.format(data[iv,iu]), ha='center', va='center',
                         color='white', fontsize=9, fontweight='bold')
    ax.set_title(title+' (64-crop px)', fontsize=11, fontweight='bold')
    ax.set_xlabel('u'); ax.set_ylabel('v')
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.suptitle(f'Waymo front-cam CalibNet per-position — {len(SEGS)} segs × '
             f'{FRAMES_PER_SEG} frames × {NCOL}×{NROW} grid  (n={int(grid_n.sum())})',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96])
out = Path('experiments/turn_projection/sigma_by_position_waymo.png')
plt.savefig(out, dpi=110, bbox_inches='tight')
plt.close(fig)
print(f'wrote {out}')
