"""Closest NuScenes cars — show __getitem__ point density.

Picks the N closest vehicle.car samples and visualizes the current
__getitem__ output. Before cache rebuild, density is capped by the
32×32 dedup grid over a 4× ROI (~64 in-crop pts). After rebuild
(margin=0.5, no dedup), close cars should approach 16×16=256.
"""
import torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataset_nuscenes import NuScenesCalibDataset

CACHE = '/tmp/nuscenes_small_cache.pt'
N_SHOW = 8

ds = NuScenesCalibDataset(CACHE, split='train')

# car length (obj_dims[1]) 3–6m, closest first
cands = []
for i, inst in enumerate(ds.instances):
    d = inst['obj_dims']
    if not (d.norm() > 0 and 3.0 < d[1].item() < 6.0): continue
    dist = float((inst['obj_pos'] - inst['cam_pos']).norm())
    cands.append((dist, i))
cands.sort()
picks = [i for _, i in cands[:N_SHOW]]
print(f"closest {N_SHOW} cars  dists={[round(cands[k][0],1) for k in range(N_SHOW)]}")

rows, cols = 2, 4
fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5), dpi=130)

for ax, idx in zip(axes.ravel(), picks):
    inst = ds.instances[idx]
    cached_n = inst['pts'].shape[0]
    img, true_uvd, dist_uvd = ds[int(idx)]
    img_np = img.permute(1,2,0).numpy()
    ax.imshow(img_np)
    is_obj = true_uvd[:, 3].numpy() > 0.5
    tu = true_uvd[:, :2].numpy(); du = dist_uvd[:, :2].numpy()
    ax.scatter(du[:,0], du[:,1], c='red',  s=16, alpha=0.7)
    ax.scatter(tu[:,0], tu[:,1], c='lime', s=16, alpha=0.7)
    if is_obj.any():
        ax.scatter(tu[is_obj,0], tu[is_obj,1], facecolors='none',
                   edgecolors='yellow', s=80, lw=1.1)
    ax.set_xlim(0, 64); ax.set_ylim(64, 0)
    ax.set_xticks([]); ax.set_yticks([])
    dist = float((inst['obj_pos'] - inst['cam_pos']).norm())
    ax.set_title(f"idx={idx} dist={dist:.1f}m  "
                 f"cache pts={cached_n}  getitem N={len(tu)}  obj={int(is_obj.sum())}",
                 fontsize=9)

fig.suptitle("NuScenes closest cars — current density (pre-rebuild)", fontsize=12)
plt.tight_layout()
out = 'vis_ns_close.png'
plt.savefig(out, dpi=130, bbox_inches='tight')
print(f"saved → {out}")
