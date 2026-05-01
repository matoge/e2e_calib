"""Vis v504 with FIXED frame alignment — bypass the buggy cache.

For each picked inst (which has correct img + K + pose for frame F),
we re-load the lidar + cuboids directly from PandaSet at scene/cam/frame=F.
Then project with T_gt onto img. No model — just GT proof of alignment.
"""
import sys, pathlib, json, pickle
sys.path.insert(0, str(pathlib.Path('/home/hiro/git/e2e_calib').resolve()))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datasets.pandaset_full import _is_obj_per_point

CACHE  = '/dev/shm/pandaset_v3_full'
SRC    = '/mnt/nvme6t/pandaset'
OUT    = Path('/tmp/v3_fixed_vis'); OUT.mkdir(exist_ok=True)
N_VIS  = 12

manifest = json.load(open(f'experiments/ps_v504_convnext_deform_l4/vis/manifest.json'))

picked = []
for k, v in manifest.items():
    inst = torch.load(Path(CACHE) / 'inst' / v['inst_file'], weights_only=False)
    scene = inst['scene']; cam = inst['cam']; frame = int(inst['frame'])
    # load CORRECT lidar+cuboids for that frame
    ld_path = Path(SRC) / scene / 'lidar' / f'{frame:02d}.pkl'
    cb_path = Path(SRC) / scene / 'annotations' / 'cuboids' / f'{frame:02d}.pkl'
    if not ld_path.exists() or not cb_path.exists():
        continue
    df_l = pickle.load(open(ld_path, 'rb'))
    if 'd' in df_l.columns:
        df_l = df_l[df_l['d'] == 0]
    pts_world = df_l[['x','y','z']].values.astype(np.float32)
    df_c = pickle.load(open(cb_path, 'rb'))
    from datasets.pandaset import USEFUL_LABELS
    if 'label' in df_c.columns:
        df_c = df_c[df_c['label'].isin(USEFUL_LABELS)]
    cubs = []
    for _, obj in df_c.iterrows():
        cubs.append(dict(
            pos=np.array([obj['position.x'], obj['position.y'], obj['position.z']], dtype=np.float32),
            dims=np.array([obj['dimensions.x'], obj['dimensions.y'], obj['dimensions.z']], dtype=np.float32),
            yaw=float(obj['yaw']),
        ))

    # project with stored (correct-frame) pose
    T_gt = inst['T_gt'].numpy()
    K = inst['K_full'].numpy()
    img = inst['img'].permute(1,2,0).numpy()
    IH, IW = img.shape[:2]
    homo = np.column_stack([pts_world, np.ones(len(pts_world))])
    pts_cam = (T_gt @ homo.T)[:3].T
    z = pts_cam[:,2]
    uv = ((K @ pts_cam.T)[:2] / np.maximum(pts_cam[:,2:].T, 1e-6)).T
    vis = (z > 0.5) & (uv[:,0] >= 0) & (uv[:,0] < IW) & (uv[:,1] >= 0) & (uv[:,1] < IH)
    is_obj = _is_obj_per_point(pts_world, cubs).astype(bool)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=110)
    ax.imshow(img)
    m = vis & ~is_obj
    ax.scatter(uv[m,0], uv[m,1], c='yellow', s=2, marker='x', linewidths=0.4, alpha=0.5, label=f'GT bg ({m.sum()})')
    m = vis & is_obj
    ax.scatter(uv[m,0], uv[m,1], c='lime', s=10, marker='x', linewidths=0.9, label=f'GT obj ({m.sum()})')
    ax.set_title(f'{k}: scene={scene} frame={frame} (FIXED-ALIGNED)  cuboids={len(cubs)}', fontsize=10)
    ax.legend(loc='upper right', fontsize=9); ax.axis('off')
    plt.tight_layout()
    out = OUT / f'{k}.png'
    plt.savefig(out, dpi=110, bbox_inches='tight'); plt.close()
    picked.append(k)
    print(f'{k}: {scene}/frame={frame}  vis={vis.sum()}  obj={int((vis&is_obj).sum())}  → {out}')
    if len(picked) >= N_VIS: break

print(f'\nsaved {len(picked)} → {OUT}')
