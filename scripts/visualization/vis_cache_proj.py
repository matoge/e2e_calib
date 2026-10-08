"""Full-image LiDAR projection of any calibration cache (PandaSet / nuScenes / kamikado / woven / TSS4).

Reads frames through the training dataset (PandaSetCalibDatasetFull._load_inst, LMDB or inst/*.pt) and
draws the cached projection `uv_full` (the GT projection the builder computed) on the image, coloured
by depth. Nothing is re-projected here, so what you see is exactly what training uses.

    python scripts/visualization/vis_cache_proj.py --cache <cache> --out out.jpg [--split val] [--n 8]
    python scripts/visualization/vis_cache_proj.py --cache <cache> --out out.jpg --per-scene   # frame 0 of every scene
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np

from datasets.pandaset_full import PandaSetCalibDatasetFull


def render(inst, width=1280):
    img = cv2.imdecode(np.frombuffer(inst['jpg_bytes'], np.uint8), cv2.IMREAD_COLOR)
    img = (img * 0.6).astype(np.uint8)
    IH, IW = img.shape[:2]
    uv = inst['uv_full'].numpy(); z = inst['z_cam'].numpy()
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < IW) & (uv[:, 1] >= 0) & (uv[:, 1] < IH)
    col = cv2.applyColorMap(np.clip(z[inside] * 6, 0, 255).astype(np.uint8)[:, None], cv2.COLORMAP_JET)[:, 0]
    r = max(1, IW // 1600)
    for (u, v), c in zip(uv[inside].astype(int), col):
        cv2.circle(img, (int(u), int(v)), r, c.tolist(), -1)
    s = width / IW
    img = cv2.resize(img, (width, int(IH * s)))
    cv2.putText(img, f"{inst['scene'][-40:]}/{inst['frame']}  {IW}x{IH}  pts {len(uv)} (in image {int(inside.sum())})",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--n', type=int, default=8, help='frames evenly spaced over the split')
    ap.add_argument('--per-scene', action='store_true', help='first frame of every scene instead of --n')
    ap.add_argument('--cols', type=int, default=2)
    a = ap.parse_args()
    ds = PandaSetCalibDatasetFull(a.cache, split=a.split, img_size=256, min_crop_px=256, max_crop_px=256,
                                  grid_n=16, max_offset_m=0.2, max_rot_deg=0.5, oversample=1)
    if a.per_scene:
        seen, idx = set(), []
        for i in range(len(ds)):
            s = ds._load_inst(i)['scene']
            if s not in seen:
                seen.add(s); idx.append(i)
    else:
        idx = np.linspace(0, len(ds) - 1, min(a.n, len(ds))).astype(int).tolist()
    tiles = [render(ds._load_inst(int(i)), width=1920 // a.cols) for i in idx]
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % a.cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[k:k + a.cols], 1) for k in range(0, len(tiles), a.cols)], 0)
    cv2.imwrite(a.out, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f'{len(idx)} frames ({a.split}) -> {a.out}  {grid.shape[1]}x{grid.shape[0]}')


if __name__ == '__main__':
    main()
