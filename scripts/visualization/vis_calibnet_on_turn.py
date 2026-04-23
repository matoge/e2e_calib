"""Run CalibNetDepth on scene 006 f2 (+12.8°/s LEFT turn) with ZERO perturbation.
If PandaSet motion comp is imperfect, LiDAR points have a systematic offset
from their true image location. The model was trained to predict μ pointing
from observed LiDAR point → true image feature → it should show a RIGHTWARD
systematic bias (pointing back from the right-shifted LiDAR to the true pole)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import json, pickle
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.model_depth import CalibNetDepth

ROOT = Path('/mnt/mininas/datasets/pandaset')
DEVICE = torch.device('cpu')  # GPU busy; model is tiny, CPU is fine
CKPT = 'experiments/all_v3_mc/best_model.pt'

model = CalibNetDepth(img_size=64, in_channels=3, n_layers=4,
                       use_convnext=True, use_frustum=True).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
model.eval()


def run_frame(scene, fi, crop_windows, out_tag):
    img = np.array(Image.open(ROOT/scene/'camera'/'front_camera'/f'{fi:02d}.jpg'))
    H, W = img.shape[:2]
    intr = json.load(open(ROOT/scene/'camera'/'front_camera'/'intrinsics.json'))
    K = np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1]])
    poses = json.load(open(ROOT/scene/'camera'/'front_camera'/'poses.json'))
    p = poses[fi]
    Rm = Rotation.from_quat([p['heading'][k] for k in 'xyzw']).as_matrix()
    cam_pose = np.eye(4); cam_pose[:3,:3]=Rm; cam_pose[:3,3]=[p['position'][k] for k in 'xyz']
    Tinv = np.linalg.inv(cam_pose)

    lidar = pickle.load(open(ROOT/scene/'lidar'/f'{fi:02d}.pkl','rb'))
    pts_w = lidar[['x','y','z']].values
    pc = Tinv[:3,:3] @ pts_w.T + Tinv[:3,3:]
    z = pc[2]; uv = (K @ pc)[:2] / z
    # training-convention depth: euclidean range / 100 (not raw z)
    rng_norm = np.linalg.norm(pc, axis=0) / 100.0
    ok = (z > 0.5) & (uv[0] >= 0) & (uv[0] < W) & (uv[1] >= 0) & (uv[1] < H)
    uv, z, rng_norm = uv.T[ok], z[ok], rng_norm[ok]

    n = len(crop_windows)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5.5, rows*5.5), dpi=120)
    axes = np.array(axes).reshape(rows, cols)

    all_mu_x = []
    for k, (u0, u1, v0, v1, name) in enumerate(crop_windows):
        r, c = k // cols, k % cols
        ax = axes[r, c]
        crop = img[v0:v1, u0:u1]
        cH, cW = crop.shape[:2]
        img64 = np.array(Image.fromarray(crop).resize((64, 64), Image.BILINEAR))

        m = (uv[:,0] >= u0) & (uv[:,0] < u1) & (uv[:,1] >= v0) & (uv[:,1] < v1)
        if m.sum() < 8:
            ax.imshow(crop); ax.set_title(f'{name} — no pts', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([]); continue

        u_crop = (uv[m,0] - u0) / cW * 64.0
        v_crop = (uv[m,1] - v0) / cH * 64.0
        d_pt   = rng_norm[m]   # training convention: euclidean range / 100
        uvd = np.stack([u_crop, v_crop, d_pt], axis=1)

        # keep at most 256 pts (model budget)
        if len(uvd) > 256:
            idx = np.random.default_rng(0).permutation(len(uvd))[:256]
            uvd = uvd[idx]

        with torch.no_grad():
            im_b = torch.from_numpy(img64).permute(2,0,1).float().unsqueeze(0).to(DEVICE) / 255.0
            uvd_b = torch.from_numpy(uvd).float().unsqueeze(0).to(DEVICE)
            params = model(im_b, uvd_b).squeeze(0).cpu().numpy()

        mu_x, mu_y = params[:,0], params[:,1]
        log_sx, log_sy = params[:,2], params[:,3]
        sigma_x, sigma_y = np.exp(log_sx), np.exp(log_sy)
        all_mu_x.extend(mu_x.tolist())
        print(f'  {name:28s}  n={len(mu_x):3d}  '
              f'μx={mu_x.mean():+.3f}±{mu_x.std():.3f}  '
              f'μy={mu_y.mean():+.3f}±{mu_y.std():.3f}  '
              f'σx={sigma_x.mean():.3f}  σy={sigma_y.mean():.3f}')

        ax.imshow(img64)
        ax.set_xticks([]); ax.set_yticks([])
        u_c, v_c = uvd[:,0], uvd[:,1]
        # arrow from point to point+mu
        for i in range(len(u_c)):
            ax.plot([u_c[i], u_c[i]+mu_x[i]], [v_c[i], v_c[i]+mu_y[i]],
                    c='#2266ee', lw=0.4, alpha=0.7)
        ax.scatter(u_c, v_c, c='#c13c14', s=8, alpha=0.9, edgecolors='white', lw=0.3)
        mean_mx, mean_my = mu_x.mean(), mu_y.mean()
        ax.set_title(f'{name}  n={len(u_c)}\nmean μ = ({mean_mx:+.2f}, {mean_my:+.2f}) px',
                      fontsize=9, fontweight='bold',
                      color='red' if abs(mean_mx) > 0.5 else 'black')
        ax.arrow(32, 32, mean_mx*6, mean_my*6, width=0.6, color='yellow',
                  length_includes_head=True, head_width=2, alpha=0.85)

    for k in range(n, rows*cols):
        axes[k//cols, k%cols].axis('off')

    yaw_sign = '+' if out_tag.startswith('006') else ''
    fig.suptitle(f'CalibNetDepth on scene {scene} f{fi} — '
                 f'yaw rate {"+12.8" if scene=="006" else "+1.5"}°/s '
                 f'{"(LEFT turn)" if scene=="006" else "(near-straight)"}\n'
                 f'Mean μ x across all crops: {np.mean(all_mu_x):+.3f} px  '
                 f'(yellow arrow = mean μ × 6)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(f'experiments/turn_projection/calibnet_{out_tag}.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[{scene} f{fi}] mean_mu_x={np.mean(all_mu_x):+.3f} px across {len(all_mu_x)} pts  →  {out}')


# scene 006 f2 crops (left turn): pick boxes around static features
# full image is 1920x1080
crops_006 = [
    (  50,  350, 200, 600, 'far-left building'),
    ( 600,  900, 200, 650, 'center buildings'),
    ( 900, 1200, 200, 650, 'center-right bldg'),
    (1350, 1680, 150, 700, 'right building'),
    ( 800, 1100, 400, 800, 'traffic lights/person'),
    ( 300,  750, 350, 800, 'parked cars left'),
]
run_frame('006', 2, crops_006, '006_f02_left_turn')

# scene 015 f30 control: near-straight
crops_015 = [
    (  50,  350, 200, 600, 'far-left'),
    ( 600,  900, 200, 650, 'center'),
    (1350, 1680, 200, 650, 'right'),
]
run_frame('015', 30, crops_015, '015_f30_straight')
