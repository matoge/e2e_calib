"""Side-by-side projection comparison: baseline fx vs corrected fx (−0.57%).
Focus on scenes with thin vertical features (poles) at image edges — that's where
focal-length error is most visible."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import json, pickle
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/mnt/mininas/datasets/pandaset')
OUT = Path('experiments/turn_projection'); OUT.mkdir(parents=True, exist_ok=True)

FX_SCALE = 0.9943
PICKS = [('015', 30), ('021', 30), ('023', 20)]  # straight scenes with varied content

def project(scene, fi, fx_scale):
    img = np.array(Image.open(ROOT/scene/'camera'/'front_camera'/f'{fi:02d}.jpg'))
    H, W = img.shape[:2]
    intr = json.load(open(ROOT/scene/'camera'/'front_camera'/'intrinsics.json'))
    fx, fy = intr['fx']*fx_scale, intr['fy']*fx_scale
    K = np.array([[fx,0,intr['cx']],[0,fy,intr['cy']],[0,0,1]])
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
    return img, uv[0][ok], uv[1][ok], z[ok], (W, H)

# zooms: left edge, right edge (skip center — nothing to see there)
BANDS = [
    (  50, 450, 200, 850, 'LEFT edge'),
    (1470,1870, 200, 850, 'RIGHT edge'),
]

for scene, fi in PICKS:
    img, u0_base, v0_base, z0_base, (W,H) = project(scene, fi, 1.0)
    img2, u0_fix,  v0_fix,  z0_fix,  _     = project(scene, fi, FX_SCALE)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=110)
    for col, (u0, u1, v0, v1, name) in enumerate(BANDS):
        for row, (uarr, varr, zarr, tag) in enumerate([
            (u0_base, v0_base, z0_base, f'fx={1970.0:.1f} (baseline)'),
            (u0_fix,  v0_fix,  z0_fix,  f'fx={1970.0*FX_SCALE:.1f} (−0.57%)'),
        ]):
            ax = axes[row, col]
            ax.imshow(img[v0:v1, u0:u1])
            m = (uarr>=u0)&(uarr<u1)&(varr>=v0)&(varr<v1)
            ax.scatter(uarr[m]-u0, varr[m]-v0, c=zarr[m], cmap='turbo',
                       vmin=3, vmax=60, s=14, alpha=0.9, edgecolors='white', linewidths=0.3)
            ax.set_xticks([]); ax.set_yticks([])
            title = f'{name}  —  {tag}'
            ax.set_title(title, fontsize=10, fontweight='bold',
                          color='darkred' if row==0 else 'darkgreen')
    fig.suptitle(f'scene {scene} f{fi:02d} — edge projection, baseline vs fx × {FX_SCALE}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.96])
    out = OUT / f'fx_compare_{scene}_f{fi:02d}.png'
    plt.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')

# one more figure: full-frame baseline vs corrected, stacked
for scene, fi in PICKS[:1]:  # just first scene
    img, u0_base, v0_base, z0_base, (W,H) = project(scene, fi, 1.0)
    _,   u0_fix,  v0_fix,  z0_fix,  _     = project(scene, fi, FX_SCALE)
    fig, axes = plt.subplots(2, 1, figsize=(18, 11), dpi=100)
    for ax, (u_, v_, z_, tag) in zip(axes, [
        (u0_base, v0_base, z0_base, f'BASELINE  fx={1970.0:.2f}'),
        (u0_fix,  v0_fix,  z0_fix,  f'CORRECTED fx={1970.0*FX_SCALE:.2f}  (−0.57%)'),
    ]):
        ax.imshow(img)
        ax.scatter(u_, v_, c=z_, cmap='turbo', vmin=3, vmax=60, s=2, alpha=0.85, edgecolors='none')
        ax.set_xlim(0,W); ax.set_ylim(H,0); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(tag, fontsize=13, fontweight='bold')
    fig.suptitle(f'Full-frame projection comparison  scene {scene} f{fi:02d}',
                  fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.97])
    out = OUT / f'fx_compare_full_{scene}_f{fi:02d}.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')
