"""Static-object consistency: accumulate LiDAR from N frames (world frame), project
into a single reference frame. If ego pose + cam-LiDAR extrinsic are consistent,
static features (buildings, poles) should appear sharp; drift = blur."""
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

def accumulate(scene, ref_frame, frame_span, stride=1):
    """Load LiDAR from frames in [ref_frame - span//2, ref_frame + span//2], all in world.
    Project into ref_frame's camera."""
    intr = json.load(open(ROOT/scene/'camera'/'front_camera'/'intrinsics.json'))
    K = np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]])
    poses = json.load(open(ROOT/scene/'camera'/'front_camera'/'poses.json'))

    p_ref = poses[ref_frame]
    R_ref = Rotation.from_quat([p_ref['heading'][k] for k in 'xyzw']).as_matrix()
    cam_ref = np.eye(4); cam_ref[:3,:3]=R_ref; cam_ref[:3,3]=[p_ref['position'][k] for k in 'xyz']
    Tinv_ref = np.linalg.inv(cam_ref)

    img = np.array(Image.open(ROOT/scene/'camera'/'front_camera'/f'{ref_frame:02d}.jpg'))
    H, W = img.shape[:2]

    frames = list(range(max(0, ref_frame - frame_span//2),
                         min(80, ref_frame + frame_span//2 + 1), stride))
    all_u, all_v, all_z, all_frame = [], [], [], []
    for fi in frames:
        lidar = pickle.load(open(ROOT/scene/'lidar'/f'{fi:02d}.pkl','rb'))
        pts_w = lidar[['x','y','z']].values
        dev = lidar['d'].values
        m = dev == 0  # Pandar64 only
        pts_w = pts_w[m]
        pc = Tinv_ref[:3,:3] @ pts_w.T + Tinv_ref[:3,3:]
        z = pc[2]; uv = (K @ pc)[:2] / z
        ok = (z>0.5)&(uv[0]>=0)&(uv[0]<W)&(uv[1]>=0)&(uv[1]<H)
        all_u.append(uv[0][ok]); all_v.append(uv[1][ok]); all_z.append(z[ok])
        all_frame.append(np.full(ok.sum(), fi))
    u = np.concatenate(all_u); v = np.concatenate(all_v); z = np.concatenate(all_z)
    fr = np.concatenate(all_frame)
    return img, u, v, z, fr, frames


# Compare 3 scenes: straight / mild yaw / strong turn
PICKS = [
    ('015', 30, 20, 1, 'straight 27 km/h'),     # ref f30, ±10 frames
    ('021', 30, 20, 1, 'fast straight 52 km/h'),
    ('006',  2, 20, 1, 'LEFT TURN 12.8°/s'),
]

fig, axes = plt.subplots(3, 2, figsize=(20, 16), dpi=100)
for row, (scene, ref, span, stride, label) in enumerate(PICKS):
    img, u, v, z, fr, frames = accumulate(scene, ref, span, stride)
    H, W = img.shape[:2]

    # LEFT: ref-frame only
    lidar = pickle.load(open(ROOT/scene/'lidar'/f'{ref:02d}.pkl','rb'))
    pts_w = lidar[lidar['d']==0][['x','y','z']].values
    intr = json.load(open(ROOT/scene/'camera'/'front_camera'/'intrinsics.json'))
    K = np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]])
    poses = json.load(open(ROOT/scene/'camera'/'front_camera'/'poses.json'))
    p_ref = poses[ref]
    R_ref = Rotation.from_quat([p_ref['heading'][k] for k in 'xyzw']).as_matrix()
    cam_ref = np.eye(4); cam_ref[:3,:3]=R_ref; cam_ref[:3,3]=[p_ref['position'][k] for k in 'xyz']
    Tinv = np.linalg.inv(cam_ref)
    pc = Tinv[:3,:3] @ pts_w.T + Tinv[:3,3:]
    zr = pc[2]; uvr = (K @ pc)[:2] / zr
    okr = (zr>0.5)&(uvr[0]>=0)&(uvr[0]<W)&(uvr[1]>=0)&(uvr[1]<H)
    ax = axes[row, 0]
    ax.imshow(img)
    ax.scatter(uvr[0][okr], uvr[1][okr], c=zr[okr], cmap='turbo', vmin=3, vmax=60,
                s=1.5, alpha=0.8, edgecolors='none')
    ax.set_xlim(0,W); ax.set_ylim(H,0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{scene} f{ref:02d}  single frame  |  {label}', fontsize=11, fontweight='bold')

    # RIGHT: accumulated (color = frame index, to see alignment)
    ax = axes[row, 1]
    ax.imshow(img)
    fr_norm = (fr - fr.min()) / max(fr.max() - fr.min(), 1)
    sc = ax.scatter(u, v, c=fr_norm, cmap='plasma', s=0.7, alpha=0.45, edgecolors='none')
    ax.set_xlim(0,W); ax.set_ylim(H,0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'accumulated {len(frames)} frames (f{frames[0]}-f{frames[-1]}) '
                 f'in f{ref:02d} camera  |  color=frame idx', fontsize=11, fontweight='bold')

fig.suptitle('Static-object consistency: project multi-frame LiDAR into single reference camera.\n'
             'Sharp static features = good pose consistency; blur/smear = drift or extrinsic error.',
             fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.97])
out = OUT / 'static_consistency.png'
plt.savefig(out, dpi=100, bbox_inches='tight')
plt.close(fig)
print(f'wrote {out}')


# Quantify: for each scene, compute the median "blur" by measuring point density
# along vertical columns (poles should stay thin if consistent)
print('\n=== quantitative: % of pixels within 3px of any LiDAR point, per frame set ===')
for scene, ref, span, stride, label in PICKS:
    img, u, v, z, fr, frames = accumulate(scene, ref, span, stride)
    H, W = img.shape[:2]
    # for each frame, how spread-out is the set of points?
    # pick points within depth band 10-30m (likely poles/buildings), histogram u positions
    m = (z > 10) & (z < 30)
    if m.sum() == 0: continue
    u_m = u[m]; fr_m = fr[m]
    # std of u per frame (accumulated vs single)
    all_std = u_m.std()
    # per-frame std then averaged (what single-frame spread looks like)
    per_frame_std = np.mean([u_m[fr_m==f].std() if (fr_m==f).sum()>1 else 0
                              for f in np.unique(fr_m)])
    print(f'  {scene} ({label}):  accumulated u-std={all_std:.1f}  '
          f'per-frame u-std={per_frame_std:.1f}  ratio={all_std/max(per_frame_std,1e-6):.2f}')
