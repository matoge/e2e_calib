"""Crop left & right edges of the (a)/(b)/(c) resid map for close inspection."""
from pathlib import Path
import numpy as np
from PIL import Image

src = Path('/home/hfunaya/git/e2e_calib/scripts/_debug/_outputs/'
           'tss4_full_frame_KB22dof_iter2fix_edge100_resid_redraw.png')
out = src.with_name(src.stem + '_edgeLR.png')

im = np.array(Image.open(src))
H, W = im.shape[:2]
# 3-panel figure: each panel is W//3 wide. inside each panel, IW=3840,
# IH=1944 -> aspect 1.976.  edge_w = nW/8 = 15 cells = 15*32 = 480 px in
# image space.  panel pixel-per-image-px ratio = (W/3) / 3840.
ratio = (W / 3) / 3840
edge_px_panel = int(round(480 * ratio))  # left edge band width in panel px

panels = []
for p in range(3):
    x0 = int(round(p * W / 3))
    x1 = int(round((p + 1) * W / 3))
    panel = im[:, x0:x1]
    L = panel[:, :edge_px_panel + 30]
    R = panel[:, -edge_px_panel - 30:]
    panels.append((L, R))

# stack: rows = panels (a,b,c), cols = (Left, Right)
gap = 6
col_w = max(p[0].shape[1] for p in panels) + 30
row_h = panels[0][0].shape[0]
canvas = np.full((row_h * 3 + gap * 2, col_w * 2 + gap, im.shape[2]), 255, dtype=im.dtype)
for r, (L, R) in enumerate(panels):
    y0 = r * (row_h + gap)
    canvas[y0:y0 + row_h, :L.shape[1]] = L
    canvas[y0:y0 + row_h, col_w:col_w + R.shape[1]] = R

Image.fromarray(canvas).save(out)
print(out)
