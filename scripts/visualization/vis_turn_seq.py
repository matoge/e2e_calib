"""Scene 006 left turn — 5 FPS projection sequence (every 2nd frame of 10 FPS data).
Pixel-exact 1:1 renders, LiDAR colored by depth."""
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
SCENE = '006'
FRAMES = list(range(0, 80, 2))  # 40 frames @ 5 FPS = full 8 s scene
OUT = Path('experiments/turn_projection/seq_006'); OUT.mkdir(parents=True, exist_ok=True)

intr = json.load(open(ROOT/SCENE/'camera'/'front_camera'/'intrinsics.json'))
K = np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]])
poses = json.load(open(ROOT/SCENE/'camera'/'front_camera'/'poses.json'))

def render_one(fi, ax):
    img = np.array(Image.open(ROOT/SCENE/'camera'/'front_camera'/f'{fi:02d}.jpg'))
    H, W = img.shape[:2]
    p = poses[fi]
    R = Rotation.from_quat([p['heading'][k] for k in 'xyzw']).as_matrix()
    cam = np.eye(4); cam[:3,:3]=R; cam[:3,3]=[p['position'][k] for k in 'xyz']
    Tinv = np.linalg.inv(cam)
    lidar = pickle.load(open(ROOT/SCENE/'lidar'/f'{fi:02d}.pkl','rb'))
    pts_w = lidar[['x','y','z']].values; dev = lidar['d'].values
    pc = Tinv[:3,:3] @ pts_w.T + Tinv[:3,3:]
    z = pc[2]; uv = (K @ pc)[:2] / z
    ok = (z>0.5)&(uv[0]>=0)&(uv[0]<W)&(uv[1]>=0)&(uv[1]<H)&(dev==0)
    u_, v_, z_ = uv[0][ok], uv[1][ok], z[ok]
    # yaw rate from pose diff
    if fi > 0:
        q0 = [poses[fi-1]['heading'][k] for k in 'xyzw']
        q1 = [p['heading'][k] for k in 'xyzw']
        R0, R1 = Rotation.from_quat(q0), Rotation.from_quat(q1)
        yaw = np.rad2deg((R1*R0.inv()).as_rotvec()[2]) / 0.1
    else:
        yaw = 0.0
    ax.imshow(img)
    ax.scatter(u_, v_, c=z_, cmap='turbo', vmin=3, vmax=60, s=1.8, alpha=0.85, edgecolors='none')
    ax.set_xlim(0,W); ax.set_ylim(H,0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'f{fi:02d}  t={fi*0.1:.1f}s  yaw={yaw:+.1f}°/s',
                  fontsize=11, fontweight='bold')

# 40 frames in 8 rows of 5
cols = 5
rows = (len(FRAMES) + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols*6.5, rows*3.7), dpi=90)
for ax, fi in zip(axes.flat, FRAMES):
    render_one(fi, ax)
for k in range(len(FRAMES), rows*cols):
    axes.flat[k].axis('off')
fig.suptitle(f'PandaSet scene {SCENE} — full scene at 5 FPS (every 2nd frame, 0–8 s)',
             fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.985])
out = OUT / 'grid_006_full.png'
plt.savefig(out, dpi=90, bbox_inches='tight')
plt.close(fig)
print(f'wrote {out}')

# also full-res per-frame for GIF / flip through
for fi in FRAMES:
    img = np.array(Image.open(ROOT/SCENE/'camera'/'front_camera'/f'{fi:02d}.jpg'))
    H, W = img.shape[:2]
    p = poses[fi]
    R = Rotation.from_quat([p['heading'][k] for k in 'xyzw']).as_matrix()
    cam = np.eye(4); cam[:3,:3]=R; cam[:3,3]=[p['position'][k] for k in 'xyz']
    Tinv = np.linalg.inv(cam)
    lidar = pickle.load(open(ROOT/SCENE/'lidar'/f'{fi:02d}.pkl','rb'))
    pts_w = lidar[['x','y','z']].values; dev = lidar['d'].values
    pc = Tinv[:3,:3] @ pts_w.T + Tinv[:3,3:]
    z = pc[2]; uv = (K @ pc)[:2] / z
    ok = (z>0.5)&(uv[0]>=0)&(uv[0]<W)&(uv[1]>=0)&(uv[1]<H)&(dev==0)
    u_, v_, z_ = uv[0][ok], uv[1][ok], z[ok]
    dpi=100; fig = plt.figure(figsize=(W/dpi,H/dpi),dpi=dpi)
    ax = fig.add_axes([0,0,1,1]); ax.imshow(img); ax.set_axis_off()
    ax.scatter(u_, v_, c=z_, cmap='turbo', vmin=3, vmax=60, s=3, alpha=0.85, edgecolors='none')
    ax.set_xlim(0,W); ax.set_ylim(H,0)
    plt.savefig(OUT / f'f{fi:02d}.png', dpi=dpi)
    plt.close(fig)
print(f'wrote {len(FRAMES)} per-frame pngs to {OUT}')
