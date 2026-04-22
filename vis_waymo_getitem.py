"""Visualize WaymoCalibDataset.__getitem__ raw output (no cache filtering).

green × = true_uv (GT projection, getitem output)
red • = dist_uv (perturbed projection, getitem output)
yellow ○ = is_obj (getitem true_uvd[:,3] > 0.5)
"""
import torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataset_waymo import WaymoCalibDataset

CACHE  = '/tmp/waymo_v2_cache.pt'
N_SHOW = 12

ds = WaymoCalibDataset(CACHE, split='train')

rng = np.random.default_rng(1)
# Just pick 12 with a Vehicle GT (obj_pos present). No cache-density filter.
vehi = [i for i, inst in enumerate(ds.instances) if inst.get('obj_pos') is not None]
picks = rng.choice(vehi, size=N_SHOW, replace=False)

rows, cols = 3, 4
fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5), dpi=130)

for ax, idx in zip(axes.ravel(), picks):
    inst = ds.instances[int(idx)]
    cache_N  = inst['pts_vis'].shape[0]
    cache_obj = int(inst['is_obj_full'].sum())
    img, true_uvd, dist_uvd = ds[int(idx)]
    img_np = img.permute(1,2,0).numpy()
    ax.imshow(img_np)
    is_obj = true_uvd[:, 3].numpy() > 0.5
    tu = true_uvd[:, :2].numpy(); du = dist_uvd[:, :2].numpy()
    ax.scatter(du[:,0], du[:,1], c='red',  s=14, alpha=0.7, label='dist (getitem)')
    ax.scatter(tu[:,0], tu[:,1], c='lime', s=14, marker='x', alpha=0.7, label='true (getitem)')
    if is_obj.any():
        ax.scatter(tu[is_obj,0], tu[is_obj,1], facecolors='none',
                   edgecolors='yellow', s=70, lw=1.0, label='is_obj')
    ax.set_xlim(0,64); ax.set_ylim(64,0)
    ax.set_xticks([]); ax.set_yticks([])
    dist = float(np.linalg.norm(inst['obj_pos']))
    ax.set_title(f"idx={idx}  dist={dist:.1f}m\n"
                 f"cache: N={cache_N} obj={cache_obj}\n"
                 f"getitem: N={len(tu)} obj={int(is_obj.sum())}",
                 fontsize=9)

axes[0,0].legend(fontsize=7, loc='lower right')
fig.suptitle("Waymo __getitem__ output — cache N/obj vs getitem N/obj", fontsize=12)
plt.tight_layout()
out = 'vis_waymo_getitem.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"saved → {out}")
