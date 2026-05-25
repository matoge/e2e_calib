"""TSS4 periphery tile sweep — 256x256 grid.

For a handful of frames, dump:
  1. the full overlay (LiDAR projected, no tile lines)
  2. a 15x7 grid of EXACT 256x256 tiles cropped from that overlay,
     each labelled (col,row) so we can eyeball which tiles show
     the periphery drift.

3840 / 256 = 15 (cols), 1952 / 256 = 7.625 -> bottom 160 px (hood) dropped.

Run:
  /home/hfunaya/.pyenv/versions/3.10.4/bin/python scripts/_debug/_tss4_periphery_tiles_256.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    '_tss4_calib_overlay', str(ROOT / 'scripts/_debug/_tss4_calib_overlay.py'),
)
ovr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ovr)  # type: ignore[union-attr]

SEQ_NAME = 'sequence=248_20230612_001946_1686529656324-1686529661227'
FRAME_IDXS = [0, 12, 24, 36, 48]
TILE = 256
OUT_DIR = ROOT / 'docs/assets/2026-05-24_tss4_overlay/tiles256'
OUT_DIR.mkdir(parents=True, exist_ok=True)
DOT_SIZE = 1


def render_overlay(img, lid_path, K, D, R, t, IW, IH):
    d = np.load(lid_path)
    pts = np.stack([d['xs'], d['ys'], d['zs']], axis=-1).astype(np.float64)
    pre = pts[:, 0] > -10.0
    pts = pts[pre]
    uv, z, _ = ovr.project(pts, R, t, K, D)
    in_b = (uv[:, 0] >= 0) & (uv[:, 0] < IW) & (uv[:, 1] >= 0) & (uv[:, 1] < IH)
    arr = img.copy()
    ovr.draw_dots(arr, uv[in_b], z[in_b], size=DOT_SIZE)
    return arr, int(in_b.sum()), int(len(pts))


def make_tile_grid_256(arr, tile=TILE):
    H, W, _ = arr.shape
    n_x = W // tile
    n_y = H // tile
    pad = 8
    canvas_w = n_x * tile + (n_x - 1) * pad
    canvas_h = n_y * tile + (n_y - 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), 32, dtype=np.uint8)
    for j in range(n_y):
        for i in range(n_x):
            y0 = j * tile
            x0 = i * tile
            tile_im = arr[y0:y0 + tile, x0:x0 + tile].copy()
            ovr.stamp_text(tile_im, [f'({i},{j})'], anchor=(4, 4), font_size=14)
            cy = j * (tile + pad)
            cx = i * (tile + pad)
            canvas[cy:cy + tile, cx:cx + tile] = tile_im
    return canvas, n_x, n_y


def main():
    calib = ovr.load_recalib()
    K, D, R, t, IW, IH = ovr.build_K_D_RT(calib)
    seq = ovr.SEQ_ROOT / SEQ_NAME
    cam_files = sorted((seq / 'tss4_fcm').glob('*.jpg'))
    lid_files = sorted((seq / 'vls128_rear_axle').glob('*.npz'))

    for fi in FRAME_IDXS:
        img = np.asarray(Image.open(cam_files[fi]).convert('RGB')).copy()
        arr, n_in, n_pre = render_overlay(img, lid_files[fi], K, D, R, t, IW, IH)
        Image.fromarray(arr).save(OUT_DIR / f'frame_{fi:03d}_full.jpg', quality=88)
        grid, nx, ny = make_tile_grid_256(arr)
        out = OUT_DIR / f'frame_{fi:03d}_tiles256_{nx}x{ny}.jpg'
        Image.fromarray(grid).save(out, quality=88)
        print(f'[frame {fi}] in_image={n_in}/{n_pre}  grid={nx}x{ny}={nx*ny} tiles  -> {out.name}')


if __name__ == '__main__':
    main()
