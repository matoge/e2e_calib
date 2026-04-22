"""Fast straight (021, ~52 km/h) vs slow straight (015, ~27 km/h) — isolate speed-dependent
vs static misalignment. Same projection pipeline, same depth-coloring, same crop regions."""
import json, pickle
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/mnt/mininas/datasets/pandaset')
OUT = Path('experiments/turn_projection'); OUT.mkdir(parents=True, exist_ok=True)

def render(scene, fi, ax, label):
    img = np.array(Image.open(ROOT/scene/'camera'/'front_camera'/f'{fi:02d}.jpg'))
    H, W = img.shape[:2]
    intr = json.load(open(ROOT/scene/'camera'/'front_camera'/'intrinsics.json'))
    K = np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]])
    poses = json.load(open(ROOT/scene/'camera'/'front_camera'/'poses.json'))
    p = poses[fi]
    R = Rotation.from_quat([p['heading'][k] for k in 'xyzw']).as_matrix()
    cam = np.eye(4); cam[:3,:3]=R; cam[:3,3]=[p['position'][k] for k in 'xyz']
    Tinv = np.linalg.inv(cam)
    lidar = pickle.load(open(ROOT/scene/'lidar'/f'{fi:02d}.pkl','rb'))
    pts_w = lidar[['x','y','z']].values; dev = lidar['d'].values
    pc = Tinv[:3,:3] @ pts_w.T + Tinv[:3,3:]
    z = pc[2]; uv = (K @ pc)[:2] / z
    ok = (z>0.5)&(uv[0]>=0)&(uv[0]<W)&(uv[1]>=0)&(uv[1]<H)&(dev==0)
    u_, v_, z_ = uv[0][ok], uv[1][ok], z[ok]
    # ego speed
    if fi>0:
        p0 = poses[fi-1]['position']
        v = np.linalg.norm([p['position'][k]-p0[k] for k in 'xyz']) / 0.1
    else: v = 0.0
    ax.imshow(img)
    ax.scatter(u_, v_, c=z_, cmap='turbo', vmin=3, vmax=60, s=2.5, alpha=0.85, edgecolors='none')
    ax.set_xlim(0,W); ax.set_ylim(H,0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{label}  scene {scene} f{fi:02d}  v={v*3.6:.0f} km/h', fontsize=12, fontweight='bold')

def render_zooms(scene, fi, axes, bands):
    """Render 3 zoom crops: far-left edge, center, far-right edge — to check radial pattern."""
    img = np.array(Image.open(ROOT/scene/'camera'/'front_camera'/f'{fi:02d}.jpg'))
    H, W = img.shape[:2]
    intr = json.load(open(ROOT/scene/'camera'/'front_camera'/'intrinsics.json'))
    K = np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]])
    poses = json.load(open(ROOT/scene/'camera'/'front_camera'/'poses.json'))
    p = poses[fi]
    R = Rotation.from_quat([p['heading'][k] for k in 'xyzw']).as_matrix()
    cam = np.eye(4); cam[:3,:3]=R; cam[:3,3]=[p['position'][k] for k in 'xyz']
    Tinv = np.linalg.inv(cam)
    lidar = pickle.load(open(ROOT/scene/'lidar'/f'{fi:02d}.pkl','rb'))
    pts_w = lidar[['x','y','z']].values; dev = lidar['d'].values
    pc = Tinv[:3,:3] @ pts_w.T + Tinv[:3,3:]
    z = pc[2]; uv = (K @ pc)[:2] / z
    ok = (z>0.5)&(uv[0]>=0)&(uv[0]<W)&(uv[1]>=0)&(uv[1]<H)&(dev==0)
    u_, v_, z_ = uv[0][ok], uv[1][ok], z[ok]
    for ax, (u0,u1,v0,v1,name) in zip(axes, bands):
        ax.imshow(img[v0:v1, u0:u1])
        m = (u_>=u0)&(u_<u1)&(v_>=v0)&(v_<v1)
        ax.scatter(u_[m]-u0, v_[m]-v0, c=z_[m], cmap='turbo', vmin=3, vmax=60,
                   s=16, alpha=0.9, edgecolors='white', linewidths=0.3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=10, fontweight='bold')

# Full frames
fig, axes = plt.subplots(2, 1, figsize=(20, 11), dpi=100)
render('021', 30, axes[0], 'FAST')
render('015', 30, axes[1], 'SLOW')
fig.suptitle('Fast straight (52 km/h) vs slow straight (27 km/h) — pinhole projection',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.97])
out = OUT / 'fast_vs_slow_full.png'
plt.savefig(out, dpi=100, bbox_inches='tight')
plt.close(fig)
print(f'wrote {out}')

# Zoom: left-edge, center, right-edge of each scene
# image is 1920x1080 → center=960,540; left edge=100-500, right=1400-1800
bands = [
    (  50, 450, 350, 850, 'LEFT edge'),
    ( 760,1160, 350, 850, 'CENTER'),
    (1470,1870, 350, 850, 'RIGHT edge'),
]
fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=110)
render_zooms('021', 30, axes[0], bands)
render_zooms('015', 30, axes[1], bands)
for ax, txt in zip([axes[0,0], axes[1,0]], ['FAST 021 f30 (52 km/h)', 'SLOW 015 f30 (27 km/h)']):
    ax.set_ylabel(txt, fontsize=11, fontweight='bold')
    ax.yaxis.label.set_color('darkred' if 'FAST' in txt else 'darkblue')
fig.suptitle('Projection at LEFT-edge / CENTER / RIGHT-edge — '
             'radial misalignment = lens distortion;  '
             'horizontal bleed growing with speed = motion-comp residual',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.96])
out = OUT / 'fast_vs_slow_zooms.png'
plt.savefig(out, dpi=110, bbox_inches='tight')
plt.close(fig)
print(f'wrote {out}')
