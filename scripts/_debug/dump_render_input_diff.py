"""Sanity check: are δ-dependent renders actually different at the byte level?
Dump 3 renders + 2 diffs + per-pixel L1, and a side-by-side PNG."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import importlib.util, sys as _s
spec = importlib.util.spec_from_file_location(
    'rd', str(REPO / 'scripts' / '_debug' / 'overfit_1image_random_delta.py'))
rd = importlib.util.module_from_spec(spec); _s.modules['rd'] = rd
spec.loader.exec_module(rd)

from scripts.data.adapters.kamikado import load_frame
from scripts.ba.ba_kb_jac import project_kb

OUT = REPO / 'docs' / 'assets' / '2026-05-19_diffba'

cf = rd.load_frame(rd.m.SCENE, rd.m.FRAME)
img_full = cf.img.astype(np.float32) / 255.0
H_full, W_full = img_full.shape[:2]
K = cf.K.astype(np.float64); dist = np.asarray(cf.dist, np.float64)
pts_cam = cf.pts_cam.astype(np.float64)
cu, cv = W_full // 2, H_full // 2 + rd.m.DV
u0 = cu - rd.m.TILE // 2; v0 = cv - rd.m.TILE // 2
u1 = u0 + rd.m.TILE;       v1 = v0 + rd.m.TILE
img_tile = img_full[v0:v1, u0:u1].copy()

DELTAS = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 1.0)]
imgs = []
for omx, omy in DELTAS:
    d = rd.make_delta6(omx, omy)
    uv_p, z_p, _ = rd.perturb_pool(pts_cam, K, dist, d, u0, v0, u1, v1)
    uv_l = uv_p.copy(); uv_l[:, 0] -= u0; uv_l[:, 1] -= v0
    img_t = rd.render_input(img_tile, uv_l, z_p).cpu().numpy()[0]   # (3,H,W)
    imgs.append((omx, omy, img_t, len(uv_p)))

base = imgs[0][2]
print(f'{"δ (ωx,ωy)":<14} {"N":>6} {"|diff vs δ=0| L1":>20} {"|diff| Linf":>14}')
for omx, omy, im, n in imgs:
    d = im - base
    print(f'({omx:+.1f},{omy:+.1f}) {n:>6d} {np.abs(d).mean():>20.6f} {np.abs(d).max():>14.4f}')

# side-by-side dump.
fig, axes = plt.subplots(2, len(imgs), figsize=(4*len(imgs), 8))
for i, (omx, omy, im, n) in enumerate(imgs):
    axes[0, i].imshow(im.transpose(1, 2, 0))
    axes[0, i].set_title(f'render δ=({omx:+.1f},{omy:+.1f}), N={n}')
    axes[0, i].axis('off')
    diff = im - base
    axes[1, i].imshow(np.linalg.norm(diff, axis=0), cmap='hot')
    axes[1, i].set_title(f'|im - im(0,0)| L2 over chan')
    axes[1, i].axis('off')
fig.tight_layout()
out = OUT / 'render_input_diff.png'
fig.savefig(out, dpi=120, bbox_inches='tight')
print(f'wrote {out}')
