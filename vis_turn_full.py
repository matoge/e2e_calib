"""Full-frame projection overlay — clean 1:1 pixel output, no padding."""
import json, pickle
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/mnt/mininas/datasets/pandaset')

def render(scene, fi, tag):
    img = np.array(Image.open(ROOT/scene/'camera'/'front_camera'/f'{fi:02d}.jpg'))
    H, W = img.shape[:2]
    intr = json.load(open(ROOT/scene/'camera'/'front_camera'/'intrinsics.json'))
    K = np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]])
    poses = json.load(open(ROOT/scene/'camera'/'front_camera'/'poses.json'))
    p = poses[fi]
    R = Rotation.from_quat([p['heading'][k] for k in 'xyzw']).as_matrix()
    cam_pose = np.eye(4); cam_pose[:3,:3]=R; cam_pose[:3,3]=[p['position'][k] for k in 'xyz']
    Tinv = np.linalg.inv(cam_pose)

    lidar = pickle.load(open(ROOT/scene/'lidar'/f'{fi:02d}.pkl','rb'))
    pts_w = lidar[['x','y','z']].values; dev = lidar['d'].values
    pc = Tinv[:3,:3] @ pts_w.T + Tinv[:3,3:]
    z = pc[2]; uv = (K @ pc)[:2] / z
    ok = (z>0.5) & (uv[0]>=0) & (uv[0]<W) & (uv[1]>=0) & (uv[1]<H) & (dev==0)
    u_, v_, z_ = uv[0][ok], uv[1][ok], z[ok]

    # pixel-exact figure — 1 px of data = 1 px of png
    dpi = 100
    fig = plt.figure(figsize=(W/dpi, H/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img); ax.set_axis_off()
    ax.scatter(u_, v_, c=z_, cmap='turbo', vmin=3, vmax=60, s=3, alpha=0.85, edgecolors='none')
    ax.set_xlim(0,W); ax.set_ylim(H,0)
    out = Path(f'experiments/turn_projection/raw_{tag}.png')
    plt.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f'wrote {out}  n_pts={len(u_)}')

# top-yaw-rate scene
render('006', 2,  '006_f02_yaw12.9')
# also a mid and a straight control
render('040', 62, '040_f62_yaw5.0')
render('015', 30, '015_f30_yaw1.5')
