"""Crop left & right edge bands from t19 flow png for side-by-side view."""
from pathlib import Path
import numpy as np
from PIL import Image

src = Path('/home/hfunaya/git/e2e_calib/scripts/_debug/_outputs/'
           'tss4_t19_stats_km_wv_wm_n4_img128_ml_dense_pe_dgx2_200ep_v2_'
           'cs256_iter1_22dof_stride64_cell32.png')
out = src.with_name(src.stem + '_edgeLR.png')

im = np.array(Image.open(src))
H, W = im.shape[:2]
# 3 panels horizontally: each W//3 wide. inside each, image is 16 cells wide
# crop cu=0..3 (left band) and cu=12..15 (right band)
panel_w = W // 3
# rough: image area inside panel ~ panel_w * 0.92 (some axis labels)
# use 4/16 = 25% width crop from left and right edges of each panel
edge_frac = 4 / 16
edge_w = int(panel_w * edge_frac * 0.92)  # leave a margin

panels = []
for p in range(3):
    x0 = p * panel_w
    x1 = (p + 1) * panel_w
    panel = im[:, x0:x1]
    L = panel[:, :edge_w + 60]   # extra margin for axis ticks
    R = panel[:, -edge_w - 30:]
    panels.append((L, R))

gap = 8
col_w = max(p[0].shape[1] for p in panels)
row_h = panels[0][0].shape[0]
canvas = np.full((row_h * 3 + gap * 2, col_w * 2 + gap, im.shape[2]), 255, dtype=im.dtype)
for r, (L, R) in enumerate(panels):
    y0 = r * (row_h + gap)
    canvas[y0:y0 + row_h, :L.shape[1]] = L
    canvas[y0:y0 + row_h, col_w + gap:col_w + gap + R.shape[1]] = R

Image.fromarray(canvas).save(out)
print(out)
