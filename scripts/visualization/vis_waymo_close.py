"""Closest Waymo vehicles — __getitem__ density check after cache + getitem fix."""
import torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataset_waymo import WaymoCalibDataset

CACHE = '/tmp/waymo_small_cache.pt'
N_SHOW = 8

ds = WaymoCalibDataset(CACHE, split='train')

cands = []
for i, inst in enumerate(ds.instances):
    if inst.get('obj_pos') is None: continue
    d = inst['obj_dims']
    if not (d[0] > 3.0 and d[0] < 6.5): continue
    dist = float(np.linalg.norm(inst['obj_pos']))
    cands.append((dist, i))
cands.sort()
picks = [i for _, i in cands[:N_SHOW]]
print(f"closest dists={[round(cands[k][0],1) for k in range(N_SHOW)]}")

rows, cols = 2, 4
fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5), dpi=130)

for ax, idx in zip(axes.ravel(), picks):
    inst = ds.instances[idx]
    cached_n = inst['pts_vis'].shape[0]
    img, true_uvd, dist_uvd = ds[int(idx)]
    img_np = img.permute(1,2,0).numpy()
    ax.imshow(img_np)
    is_obj = true_uvd[:, 3].numpy() > 0.5
    tu = true_uvd[:, :2].numpy(); du = dist_uvd[:, :2].numpy()
    ax.scatter(du[:,0], du[:,1], c='red',  s=14, alpha=0.7)
    ax.scatter(tu[:,0], tu[:,1], c='lime', s=14, alpha=0.7)
    if is_obj.any():
        ax.scatter(tu[is_obj,0], tu[is_obj,1], facecolors='none',
                   edgecolors='yellow', s=70, lw=1.0)
    ax.set_xlim(0,64); ax.set_ylim(64,0)
    ax.set_xticks([]); ax.set_yticks([])
    dist = float(np.linalg.norm(inst['obj_pos']))
    ax.set_title(f"idx={idx} dist={dist:.1f}m  cache={cached_n}  N={len(tu)}  obj={int(is_obj.sum())}",
                 fontsize=9)

fig.suptitle("Waymo closest vehicles — post-fix density", fontsize=12)
plt.tight_layout()
out = 'vis_waymo_close.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"saved → {out}")
