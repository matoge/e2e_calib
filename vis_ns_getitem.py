"""Visualize NuScenesCalibDataset.__getitem__ outputs for car samples.

Grid of crops with:
  green = true_uvd (GT projection)
  red   = dist_uvd (perturbed projection)
  yellow outline = is_obj == 1 points
"""
import torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataset_nuscenes import NuScenesCalibDataset

CACHE = '/tmp/nuscenes_static_cache.pt'
N_SHOW = 12

ds = NuScenesCalibDataset(CACHE, split='train')

# Find car-like samples: length > 3m in obj_dims
car_idxs = []
for i, inst in enumerate(ds.instances):
    d = inst['obj_dims']
    if d.norm() > 0 and d[1].item() > 3.0 and d[1].item() < 6.0:  # length 3-6m ≈ car
        car_idxs.append(i)
        if len(car_idxs) >= N_SHOW * 3: break

print(f"found {len(car_idxs)} car candidates, sampling {N_SHOW}")
rng = np.random.default_rng(0)
picks = rng.choice(car_idxs, size=N_SHOW, replace=False)

rows, cols = 3, 4
fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5), dpi=140)

for ax, idx in zip(axes.ravel(), picks):
    img, true_uvd, dist_uvd = ds[int(idx)]
    img_np = img.permute(1,2,0).numpy()
    ax.imshow(img_np)
    is_obj = true_uvd[:, 3].numpy() > 0.5
    tu = true_uvd[:, :2].numpy(); du = dist_uvd[:, :2].numpy()
    # all points (bg + obj)
    ax.scatter(du[:,0], du[:,1], c='red',   s=18, alpha=0.7, label='dist')
    ax.scatter(tu[:,0], tu[:,1], c='lime',  s=18, alpha=0.7, label='true')
    # connecting lines to show per-point shift
    for i in range(len(tu)):
        ax.plot([du[i,0],tu[i,0]], [du[i,1],tu[i,1]],
                c='white', lw=0.3, alpha=0.4)
    # obj points highlighted
    if is_obj.any():
        ax.scatter(tu[is_obj,0], tu[is_obj,1], facecolors='none',
                   edgecolors='yellow', s=80, lw=1.2)
    ax.set_xlim(0, 64); ax.set_ylim(64, 0)
    ax.set_xticks([]); ax.set_yticks([])
    shift = (tu - du).__pow__(2).sum(axis=1) ** 0.5
    ax.set_title(f"idx={idx}  N={len(tu)}  obj={is_obj.sum()}  "
                 f"shift={shift.mean():.1f}px", fontsize=11)

axes[0,0].legend(fontsize=7, loc='lower right')
fig.suptitle("NuScenesCalibDataset __getitem__ — vehicle.car samples",
             fontsize=11)
plt.tight_layout()
out = 'vis_ns_getitem.png'
plt.savefig(out, dpi=110, bbox_inches='tight')
print(f"saved → {out}")
