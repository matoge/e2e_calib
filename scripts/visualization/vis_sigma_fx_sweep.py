"""Sweep fx scale and see if μ_x radial pattern disappears.
If the hypothesis (fx too large by ~0.4%) is right, there should be a scale
that minimizes the radial slope of μ_x.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import json, pickle, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.model_depth import CalibNetDepth

ROOT = Path('/mnt/mininas/datasets/pandaset')
DEVICE = torch.device('cpu')
CKPT = 'experiments/all_v3_mc/best_model.pt'

model = CalibNetDepth(img_size=64, in_channels=3, n_layers=4,
                       use_convnext=True, use_frustum=True).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
model.eval()

SCENES_FRAMES = [
    ('015', list(range(0, 80, 4))),
    ('021', list(range(0, 80, 4))),
    ('023', list(range(0, 80, 4))),
]

CROP_SIZE = 320
NCOL, NROW = 6, 3
IMG_W, IMG_H = 1920, 1080
u_centers = np.linspace(CROP_SIZE/2+20, IMG_W-CROP_SIZE/2-20, NCOL)
v_centers = np.linspace(CROP_SIZE/2+20, IMG_H-CROP_SIZE/2-20, NROW)

def run(fx_scale, fy_scale=None):
    if fy_scale is None: fy_scale = fx_scale
    recs = []
    for scene, frames in SCENES_FRAMES:
        intr = json.load(open(ROOT/scene/'camera'/'front_camera'/'intrinsics.json'))
        fx, fy, cx, cy = intr['fx']*fx_scale, intr['fy']*fy_scale, intr['cx'], intr['cy']
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
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
                    u0, v0 = int(uc-CROP_SIZE/2), int(vc-CROP_SIZE/2)
                    u1, v1 = u0+CROP_SIZE, v0+CROP_SIZE
                    crop = img[v0:v1, u0:u1]
                    img64 = np.array(Image.fromarray(crop).resize((64,64), Image.BILINEAR))
                    m = (uv_f[:,0]>=u0)&(uv_f[:,0]<u1)&(uv_f[:,1]>=v0)&(uv_f[:,1]<v1)
                    if m.sum()<8: continue
                    u_c = (uv_f[m,0]-u0)/CROP_SIZE*64.0
                    v_c = (uv_f[m,1]-v0)/CROP_SIZE*64.0
                    d_c = rng_f[m]
                    uvd = np.stack([u_c,v_c,d_c],axis=1)
                    if len(uvd)>256:
                        idx = np.random.default_rng(fi).permutation(len(uvd))[:256]
                        uvd = uvd[idx]
                    with torch.no_grad():
                        im_b = torch.from_numpy(img64).permute(2,0,1).float().unsqueeze(0)/255.0
                        uvd_b = torch.from_numpy(uvd).float().unsqueeze(0)
                        pr = model(im_b, uvd_b).squeeze(0).numpy()
                    recs.append((uc, vc, pr[:,0].mean(), pr[:,1].mean(),
                                 np.exp(pr[:,2]).mean(), np.exp(pr[:,3]).mean()))
    return np.array(recs)

# 5-point sweep around expected optimum
scales = [1.0000, 0.9970, 0.9957, 0.9943, 0.9900]  # baseline + 0.3/0.43/0.57/1.0% down
all_rec = {}
t0 = time.time()
for s in scales:
    rec = run(s)
    all_rec[s] = rec
    # middle-row μ_x radial slope
    mid = rec[np.isclose(rec[:,1], u_centers[0]*0+v_centers[1]), :]  # v=540
    # group by u
    uniq_u = np.unique(rec[:,0])
    mid_mu = np.array([rec[(np.abs(rec[:,1]-v_centers[1])<1)&(np.abs(rec[:,0]-u)<1),2].mean()
                        for u in uniq_u])
    slope = np.polyfit(uniq_u - IMG_W/2, mid_mu, 1)[0]
    # global abs μ_x
    print(f'fx_scale={s:.4f}  mid-row μ_x slope={slope*1000:+.3f} per 1000px  '
          f'|μ_x|_mid_mean={np.abs(mid_mu).mean():.3f}  '
          f'global μ_x mean={rec[:,2].mean():+.3f}  ({time.time()-t0:.0f}s)')

# plot the middle-row μ_x curves for each scale
fig, ax = plt.subplots(1, 1, figsize=(9, 5), dpi=110)
uniq_u = np.unique(all_rec[1.0][:,0])
for s in scales:
    rec = all_rec[s]
    mid_mu = np.array([rec[(np.abs(rec[:,1]-v_centers[1])<1)&(np.abs(rec[:,0]-u)<1),2].mean()
                        for u in uniq_u])
    label = f'fx × {s}  ({(s-1)*100:+.2f}%)'
    ax.plot(uniq_u, mid_mu, 'o-', label=label, lw=2, markersize=8)
ax.axhline(0, color='k', lw=0.8, alpha=0.5)
ax.axvline(IMG_W/2, color='gray', lw=0.8, ls='--', alpha=0.5, label=f'image center u={IMG_W/2:.0f}')
ax.set_xlabel('image u (crop center)'); ax.set_ylabel('mean μ_x (64-crop px)')
ax.set_title('Middle-row μ_x vs image position — sweep fx scale', fontsize=12, fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
out = Path('experiments/turn_projection/fx_sweep.png')
plt.savefig(out, dpi=110, bbox_inches='tight')
plt.close(fig)
print(f'wrote {out}')
