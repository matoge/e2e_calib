"""scene 040 f55 (22.3°/s yaw). Hero target: right streetlight pole — a clean
thin vertical. Color LiDAR projections by depth. If motion comp is good:
pole points sit on the pole at consistent near depth, background at far depth.
If not: horizontal bleed / depth mixing on the pole."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import json, pickle
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/mnt/mininas/datasets/pandaset')
SCENE, FI = '040', 55

img = np.array(Image.open(ROOT/SCENE/'camera'/'front_camera'/f'{FI:02d}.jpg'))
H, W = img.shape[:2]
intr = json.load(open(ROOT/SCENE/'camera'/'front_camera'/'intrinsics.json'))
K = np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]])
poses = json.load(open(ROOT/SCENE/'camera'/'front_camera'/'poses.json'))
p = poses[FI]
R = Rotation.from_quat([p['heading'][k] for k in 'xyzw']).as_matrix()
cam_pose = np.eye(4); cam_pose[:3,:3]=R; cam_pose[:3,3]=[p['position'][k] for k in 'xyz']
Tinv = np.linalg.inv(cam_pose)

lidar = pickle.load(open(ROOT/SCENE/'lidar'/f'{FI:02d}.pkl','rb'))
pts_w = lidar[['x','y','z']].values
dev   = lidar['d'].values
t_pt  = lidar['t'].values

def proj(mask):
    pw = pts_w[mask]; tp = t_pt[mask]
    pc = Tinv[:3,:3] @ pw.T + Tinv[:3,3:]
    z  = pc[2]
    uv = (K @ pc)[:2] / z
    ok = (z>0.5) & (uv[0]>=0) & (uv[0]<W) & (uv[1]>=0) & (uv[1]<H)
    return uv.T[ok], z[ok], tp[ok]

uv0, z0, t0 = proj(dev==0)  # Pandar64, ~25ms FOV sweep
uv1, z1, t1 = proj(dev==1)  # PandarGT, near-instant

# Hero zoom: right streetlight pole (x~1510-1570 full res, y~40-650)
ZOOMS = [
    (1470, 1620,  20, 720, 'right streetlight (near, ~15 m)'),
    (  90,  280, 150, 720, 'left traffic light + pole'),
    (1040, 1240, 300, 600, 'pedestrian sign (mid-distance)'),
]

fig = plt.figure(figsize=(18, 10), dpi=130)
gs = fig.add_gridspec(2, 4, height_ratios=[1, 2])

# top: overview
ax_full = fig.add_subplot(gs[0, :])
ax_full.imshow(img)
step = max(1, len(uv0)//40000)
vmin, vmax = 5, 60
sc = ax_full.scatter(uv0[::step,0], uv0[::step,1], c=z0[::step], cmap='turbo',
                     vmin=vmin, vmax=vmax, s=1.2, alpha=0.8, edgecolors='none')
# boxes over zooms
for (u0,u1,v0,v1,_) in ZOOMS:
    ax_full.plot([u0,u1,u1,u0,u0],[v0,v0,v1,v1,v0],'w-',lw=1.2,alpha=0.85)
ax_full.set_title(f'scene {SCENE}  f{FI}  yaw 22.3°/s — LiDAR (Pandar64) colored by depth [m]',
                   fontsize=11, fontweight='bold')
ax_full.set_xticks([]); ax_full.set_yticks([])
cb = fig.colorbar(sc, ax=ax_full, shrink=0.6, pad=0.01); cb.set_label('depth (m)')

# bottom row: three tight zooms, each image + overlay stacked
for col, (u0,u1,v0,v1,name) in enumerate(ZOOMS):
    ax = fig.add_subplot(gs[1, col+1 if col==2 else col])
    # each column gets one plot — put image and overlay side-by-side inside
    sub = img[v0:v1, u0:u1]
    ax.imshow(sub)
    m64 = (uv0[:,0]>=u0)&(uv0[:,0]<u1)&(uv0[:,1]>=v0)&(uv0[:,1]<v1)
    mGT = (uv1[:,0]>=u0)&(uv1[:,0]<u1)&(uv1[:,1]>=v0)&(uv1[:,1]<v1)
    ax.scatter(uv0[m64,0]-u0, uv0[m64,1]-v0, c=z0[m64], cmap='turbo',
                vmin=vmin, vmax=vmax, s=14, alpha=0.9, edgecolors='none', marker='o')
    ax.scatter(uv1[mGT,0]-u0, uv1[mGT,1]-v0, c=z1[mGT], cmap='turbo',
                vmin=vmin, vmax=vmax, s=30, alpha=0.95, edgecolors='white', linewidths=0.5, marker='^')
    ax.set_title(f'{name}\nP64 (●) n={m64.sum()}  ·  PandarGT (▲) n={mGT.sum()}',
                  fontsize=9, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

# empty slot placeholder used — fix layout by giving col 0 a label
fig.text(0.02, 0.02,
         'near depth on pole + far depth on background = alignment OK.\n'
         'horizontal bleed of near-depth points into background column = residual motion-comp error.',
         fontsize=9, color='#444')

plt.tight_layout(rect=[0,0.03,1,1])
out = Path('experiments/turn_projection/tight_040_f55.png')
plt.savefig(out, dpi=130, bbox_inches='tight')
plt.close(fig)
print(f'wrote {out}')
