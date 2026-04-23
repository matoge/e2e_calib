"""One streetlight pole, scene 040 f55, 22.3°/s. Big figure, pixel-scale dots,
P64 vs PandarGT comparison to isolate motion-comp residual."""
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
SCENE, FI = '006', 2

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
pts_w = lidar[['x','y','z']].values; dev = lidar['d'].values; tp = lidar['t'].values
def proj(mask):
    pw = pts_w[mask]; t = tp[mask]
    pc = Tinv[:3,:3] @ pw.T + Tinv[:3,3:]
    z = pc[2]; uv = (K @ pc)[:2] / z
    ok = (z>0.5) & (uv[0]>=0) & (uv[0]<W) & (uv[1]>=0) & (uv[1]<H)
    return uv.T[ok], z[ok], t[ok]
uv0, z0, t0 = proj(dev==0)
uv1, z1, t1 = proj(dev==1)

# Hero: right streetlight pole
U0,U1,V0,V1 = 1470, 1620, 20, 720
# Expected rotation-induced offset on pole
# forward-FOV sweep time ≈ (140/360)*100ms = 39ms; yaw rate 22.3°/s → 0.87° over sweep
# at ~15m: 0.87° * π/180 * 15 ≈ 0.23m → 0.23 * 1970/15 ≈ 30 px lateral
yaw_during_sweep_deg = 22.3 * 0.039
expected_px_at_15m = np.deg2rad(yaw_during_sweep_deg) * 15 * intr['fx'] / 15

fig, axes = plt.subplots(1, 3, figsize=(18, 14), dpi=150,
                          gridspec_kw={'wspace': 0.03})
sub = img[V0:V1, U0:U1]
for ax, title in zip(axes, ['image only', 'Pandar64 (sweep ~39 ms)', 'PandarGT (~instant)']):
    ax.imshow(sub)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight='bold')

vmin, vmax = 8, 50

m64 = (uv0[:,0]>=U0)&(uv0[:,0]<U1)&(uv0[:,1]>=V0)&(uv0[:,1]<V1)
mGT = (uv1[:,0]>=U0)&(uv1[:,0]<U1)&(uv1[:,1]>=V0)&(uv1[:,1]<V1)

axes[1].scatter(uv0[m64,0]-U0, uv0[m64,1]-V0, c=z0[m64], cmap='turbo',
                vmin=vmin, vmax=vmax, s=28, alpha=0.95, edgecolors='white', linewidths=0.3)
axes[2].scatter(uv1[mGT,0]-U0, uv1[mGT,1]-V0, c=z1[mGT], cmap='turbo',
                vmin=vmin, vmax=vmax, s=28, alpha=0.95, edgecolors='white', linewidths=0.3)

fig.suptitle(f'scene {SCENE}  f{FI}  yaw 12.9°/s  —  right streetlight pole\n'
             f'If comp imperfect, P64 vs PandarGT pole-column centroids should differ '
             f'by ~{expected_px_at_15m:.0f} px at 15 m',
             fontsize=13, fontweight='bold')

# numeric check: find points within narrow pole column and compare centroids
# heuristic: pole is near blue depth band (15–20m), narrow u range
pole_d = (z0[m64] > 12) & (z0[m64] < 22)
p64_u = uv0[m64][pole_d, 0]
# pick dominant column
if len(p64_u) > 20:
    hist, bins = np.histogram(p64_u, bins=30)
    peak_u = (bins[np.argmax(hist)] + bins[np.argmax(hist)+1]) / 2
    win = 25
    P64_on_pole = (np.abs(uv0[m64][:,0] - peak_u) < win) & (z0[m64] > 12) & (z0[m64] < 22)
    GT_on_pole  = (np.abs(uv1[mGT][:,0] - peak_u) < win) & (z1[mGT] > 12) & (z1[mGT] < 22)
    n64 = P64_on_pole.sum(); nGT = GT_on_pole.sum()
    mean64 = uv0[m64][P64_on_pole,0].mean() if n64 else np.nan
    meanGT = uv1[mGT][GT_on_pole,0].mean() if nGT else np.nan
    print(f'pole column u ≈ {peak_u:.1f}  P64 n={n64} mean_u={mean64:.2f}  '
          f'GT n={nGT} mean_u={meanGT:.2f}  Δ={meanGT-mean64:+.2f} px')

out = Path('experiments/turn_projection/pole_006_f02.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'wrote {out}')
