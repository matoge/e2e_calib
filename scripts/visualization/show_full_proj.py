"""Render full-image GT lidar projection for a V3 inst, with frame-aligned
lidar+cuboids loaded directly from PandaSet source (bypassing the cache's
known frame-shift bug)."""
import sys, pathlib, pickle
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datasets.pandaset_full import _is_obj_per_point, decode_inst_img
from datasets.pandaset import USEFUL_LABELS

PANDASET_SRC = '/mnt/nvme6t/pandaset'


def main(inst_path: str, out_path: str):
    inst = torch.load(inst_path, weights_only=False)
    scene = inst['scene']; cam = inst['cam']; frame = int(inst['frame'])
    img = decode_inst_img(inst).permute(1, 2, 0).numpy()
    IH, IW = img.shape[:2]
    K = inst['K_full'].numpy()
    T_gt = inst['T_gt'].numpy()

    ld = Path(PANDASET_SRC) / scene / 'lidar' / f'{frame:02d}.pkl'
    cb = Path(PANDASET_SRC) / scene / 'annotations' / 'cuboids' / f'{frame:02d}.pkl'
    df_l = pickle.load(open(ld, 'rb'))
    if 'd' in df_l.columns:
        df_l = df_l[df_l['d'] == 0]
    pts_world = df_l[['x', 'y', 'z']].values.astype(np.float32)

    df_c = pickle.load(open(cb, 'rb'))
    if 'label' in df_c.columns:
        df_c = df_c[df_c['label'].isin(USEFUL_LABELS)]
    cubs = []
    for _, obj in df_c.iterrows():
        cubs.append(dict(
            pos=np.array([obj['position.x'], obj['position.y'], obj['position.z']], dtype=np.float32),
            dims=np.array([obj['dimensions.x'], obj['dimensions.y'], obj['dimensions.z']], dtype=np.float32),
            yaw=float(obj['yaw']),
        ))

    homo = np.column_stack([pts_world, np.ones(len(pts_world))])
    pcam = (T_gt @ homo.T)[:3].T
    z = pcam[:, 2]
    uv = ((K @ pcam.T)[:2] / np.maximum(pcam[:, 2:].T, 1e-6)).T
    vis = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < IW) & (uv[:, 1] >= 0) & (uv[:, 1] < IH)
    is_obj = _is_obj_per_point(pts_world, cubs).astype(bool)

    n_obj = int((vis & is_obj).sum())
    n_bg = int((vis & ~is_obj).sum())
    print(f'scene={scene} cam={cam} frame={frame}  vis={vis.sum()}  obj={n_obj}  bg={n_bg}  cuboids={len(cubs)}')

    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    ax.imshow(img)
    m = vis & ~is_obj
    ax.scatter(uv[m, 0], uv[m, 1], c='yellow', s=2, marker='x', linewidths=0.4, alpha=0.5, label=f'GT bg ({n_bg})')
    m = vis & is_obj
    ax.scatter(uv[m, 0], uv[m, 1], c='lime', s=10, marker='x', linewidths=0.9, label=f'GT obj ({n_obj})')
    ax.set_title(f'{scene}/{cam}/frame={frame}  IW={IW} IH={IH}  cuboids={len(cubs)}  (FRAME-ALIGNED)', fontsize=10)
    ax.legend(loc='upper right', fontsize=9)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'saved → {out_path}')


if __name__ == '__main__':
    inst = sys.argv[1]
    out  = sys.argv[2] if len(sys.argv) > 2 else '/tmp/val_07_full.png'
    main(inst, out)
