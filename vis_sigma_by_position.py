"""Per-image-position σ statistics.
Grid of crops across the image × many frames → aggregate predicted σx, σy by crop center.
If edges have larger σ → model is more uncertain there (distortion / edge effects).
"""
import json, pickle, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_depth import CalibNetDepth

ROOT = Path('/mnt/mininas/datasets/pandaset')
DEVICE = torch.device('cpu')
CKPT = 'experiments/all_v3_mc/best_model.pt'

model = CalibNetDepth(img_size=64, in_channels=3, n_layers=4,
                       use_convnext=True, use_frustum=True).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
model.eval()

# straight scenes (yaw < 2°/s) — isolate position effect from turn effect
SCENES_FRAMES = [
    ('015', list(range(0, 80, 4))),   # 20 frames slow straight
    ('021', list(range(0, 80, 4))),   # 20 frames fast straight
    ('023', list(range(0, 80, 4))),
]

# 6 × 4 crop grid of 300×300 over 1920×1080 full-res image
CROP_SIZE = 320
NCOL, NROW = 6, 3
IMG_W, IMG_H = 1920, 1080
# place crop centers uniformly with some margin
u_centers = np.linspace(CROP_SIZE/2+20, IMG_W-CROP_SIZE/2-20, NCOL)
v_centers = np.linspace(CROP_SIZE/2+20, IMG_H-CROP_SIZE/2-20, NROW)

records = []  # (u_center, v_center, r, mu_x, mu_y, sigma_x, sigma_y)

t0 = time.time()
for scene, frames in SCENES_FRAMES:
    intr = json.load(open(ROOT/scene/'camera'/'front_camera'/'intrinsics.json'))
    K = np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]])
    poses = json.load(open(ROOT/scene/'camera'/'front_camera'/'poses.json'))
    for fi in frames:
        img = np.array(Image.open(ROOT/scene/'camera'/'front_camera'/f'{fi:02d}.jpg'))
        H, W = img.shape[:2]
        p = poses[fi]
        R = Rotation.from_quat([p['heading'][k] for k in 'xyzw']).as_matrix()
        cam = np.eye(4); cam[:3,:3]=R; cam[:3,3]=[p['position'][k] for k in 'xyz']
        Tinv = np.linalg.inv(cam)
        lidar = pickle.load(open(ROOT/scene/'lidar'/f'{fi:02d}.pkl','rb'))
        pts_w = lidar[['x','y','z']].values
        pc = Tinv[:3,:3] @ pts_w.T + Tinv[:3,3:]
        z = pc[2]; uv = (K @ pc)[:2] / z
        rng_norm = np.linalg.norm(pc, axis=0) / 100.0
        ok = (z>0.5) & (uv[0]>=0) & (uv[0]<W) & (uv[1]>=0) & (uv[1]<H)
        uv_f, rng_f = uv.T[ok], rng_norm[ok]

        for uc in u_centers:
            for vc in v_centers:
                u0, v0 = int(uc - CROP_SIZE/2), int(vc - CROP_SIZE/2)
                u1, v1 = u0 + CROP_SIZE, v0 + CROP_SIZE
                crop = img[v0:v1, u0:u1]
                img64 = np.array(Image.fromarray(crop).resize((64, 64), Image.BILINEAR))
                m = (uv_f[:,0]>=u0) & (uv_f[:,0]<u1) & (uv_f[:,1]>=v0) & (uv_f[:,1]<v1)
                if m.sum() < 8: continue
                u_c = (uv_f[m,0] - u0) / CROP_SIZE * 64.0
                v_c = (uv_f[m,1] - v0) / CROP_SIZE * 64.0
                d_c = rng_f[m]
                uvd = np.stack([u_c, v_c, d_c], axis=1)
                if len(uvd) > 256:
                    idx = np.random.default_rng(fi).permutation(len(uvd))[:256]
                    uvd = uvd[idx]
                with torch.no_grad():
                    im_b = torch.from_numpy(img64).permute(2,0,1).float().unsqueeze(0)/255.0
                    uvd_b = torch.from_numpy(uvd).float().unsqueeze(0)
                    pr = model(im_b.to(DEVICE), uvd_b.to(DEVICE)).squeeze(0).cpu().numpy()
                mu_x, mu_y = pr[:,0].mean(), pr[:,1].mean()
                sx = float(np.exp(pr[:,2]).mean()); sy = float(np.exp(pr[:,3]).mean())
                r = float(np.hypot(uc-W/2, vc-H/2))
                records.append((uc, vc, r, mu_x, mu_y, sx, sy, scene, fi))
    print(f'{scene}: {len(records)} records so far ({time.time()-t0:.0f}s)')

rec = np.array([r[:7] for r in records], dtype=float)
print(f'\ntotal: {len(rec)} crop observations')
print(f'global mean: μx={rec[:,3].mean():+.3f}  μy={rec[:,4].mean():+.3f}  '
      f'σx={rec[:,5].mean():.3f}  σy={rec[:,6].mean():.3f}')

# aggregate per (crop_center_u, crop_center_v)
uniq_u = np.unique(rec[:,0]); uniq_v = np.unique(rec[:,1])
grid_sx = np.full((len(uniq_v), len(uniq_u)), np.nan)
grid_sy = np.full_like(grid_sx, np.nan)
grid_mx = np.full_like(grid_sx, np.nan)
grid_my = np.full_like(grid_sx, np.nan)
grid_n  = np.zeros_like(grid_sx, dtype=int)
for iu, uc in enumerate(uniq_u):
    for iv, vc in enumerate(uniq_v):
        m = (rec[:,0]==uc)&(rec[:,1]==vc)
        if m.sum()==0: continue
        grid_sx[iv,iu] = rec[m,5].mean()
        grid_sy[iv,iu] = rec[m,6].mean()
        grid_mx[iv,iu] = rec[m,3].mean()
        grid_my[iv,iu] = rec[m,4].mean()
        grid_n[iv,iu]  = m.sum()

fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=110)
for ax, data, title, cmap, fmt in [
    (axes[0,0], grid_sx, 'mean σ_x (64-crop px)', 'viridis', '{:.2f}'),
    (axes[0,1], grid_sy, 'mean σ_y (64-crop px)', 'viridis', '{:.2f}'),
    (axes[1,0], grid_mx, 'mean μ_x (64-crop px)', 'RdBu_r', '{:+.2f}'),
    (axes[1,1], grid_my, 'mean μ_y (64-crop px)', 'RdBu_r', '{:+.2f}'),
]:
    vmin = np.nanmin(data); vmax = np.nanmax(data)
    if 'μ' in title:
        a = max(abs(vmin), abs(vmax)); vmin, vmax = -a, a
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto',
                   extent=[0, IMG_W, IMG_H, 0])
    for iu, uc in enumerate(uniq_u):
        for iv, vc in enumerate(uniq_v):
            if grid_n[iv,iu]>0:
                ax.text(uc, vc, fmt.format(data[iv,iu]), ha='center', va='center',
                         color='white', fontsize=8, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('image u'); ax.set_ylabel('image v')
    plt.colorbar(im, ax=ax, shrink=0.8)

total_n = int(grid_n.sum())
fig.suptitle(f'Per-position CalibNet statistics — 3 straight scenes × '
             f'{sum(len(f) for _,f in SCENES_FRAMES)} frames × {NCOL}×{NROW} crop grid  '
             f'(n={total_n} crops)', fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96])
out = Path('experiments/turn_projection/sigma_by_position.png')
plt.savefig(out, dpi=110, bbox_inches='tight')
plt.close(fig)
print(f'wrote {out}')

# also radial plot: σ vs distance-from-center
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=110)
for ax, col, label in [(axes[0], 5, 'σ_x'), (axes[1], 6, 'σ_y')]:
    r_bins = np.linspace(0, rec[:,2].max()+1, 10)
    centers = 0.5*(r_bins[:-1]+r_bins[1:])
    means = [rec[(rec[:,2]>=r_bins[i])&(rec[:,2]<r_bins[i+1]),col].mean() for i in range(len(centers))]
    stds  = [rec[(rec[:,2]>=r_bins[i])&(rec[:,2]<r_bins[i+1]),col].std() for i in range(len(centers))]
    counts = [((rec[:,2]>=r_bins[i])&(rec[:,2]<r_bins[i+1])).sum() for i in range(len(centers))]
    ax.errorbar(centers, means, yerr=stds, fmt='o-', capsize=4, lw=2)
    for c, m, n in zip(centers, means, counts):
        ax.annotate(f'n={n}', (c, m), textcoords='offset points', xytext=(4,4), fontsize=7)
    ax.set_xlabel('distance from image center (px)'); ax.set_ylabel(f'mean {label}')
    ax.set_title(f'{label} vs radial position', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
plt.tight_layout()
out2 = Path('experiments/turn_projection/sigma_radial.png')
plt.savefig(out2, dpi=110, bbox_inches='tight')
plt.close(fig)
print(f'wrote {out2}')
