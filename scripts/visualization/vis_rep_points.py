"""Draw how each cell's representative point (query) is chosen: training vs val/inference.

Training with --rep-strategy random_train picks a random point in each cell; val and inference
pick the point closest to the cell centre. This loads the dataset in debug mode (debug_dir),
which writes one PNG per window, and puts the first windows side by side.

    python scripts/visualization/vis_rep_points.py --cache <cache> --out <dir> [--frames 0,10,20]

Each window (enlarged 3x):
  grey       = all points in the window (projection under the perturbed pose)
  red        = the representative actually chosen
  cyan ring  = the point closest to the cell centre (what val/inference picks)
  green      = true position of the chosen point (red → green is the training target)
The header shows "chosen==centre k/n": in training mode k is small, in inference mode k == n.

The same debug output is available from any run: set E2E_DATASET_DEBUG=<dir> (or pass
debug_dir= to PandaSetCalibDatasetFull) and the first debug_max windows are written.
"""
import argparse, glob, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import numpy as np

from datasets.pandaset_full import PandaSetCalibDatasetFull


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frames', default='0,10,20', help='frame indices within the split')
    ap.add_argument('--split', default='val', help='which frames to use (both modes use the same frames)')
    ap.add_argument('--seed', type=int, default=20261008)
    ap.add_argument('--crop', type=int, default=256, help='crop size in the original image (resized to 256)')
    a = ap.parse_args()
    frames = [int(x) for x in a.frames.split(',') if x.strip()]
    kw = dict(img_size=256, min_crop_px=a.crop, max_crop_px=a.crop, grid_n=16, max_offset_m=0.2,
              max_rot_deg=0.5, oversample=1, k_per_cell=24, eval_seed=a.seed)
    rows = []
    # 両方とも split='train' の経路で同じ窓・同じずれを作り、代表点の選び方だけ変える
    # (split を変えると窓とずれの引き方も変わり、左右が別の窓になる)。
    # cell_center は val・推論で random_train が使う選び方と同じ (セル中心に一番近い点)。
    for mode, rep, split_flag in [('train', 'random_train', 'train'),
                                  ('inference', 'cell_center', 'train')]:
        d = os.path.join(a.out, mode)
        for f in glob.glob(os.path.join(d, 'rep_*.png')):
            os.remove(f)
        ds = PandaSetCalibDatasetFull(a.cache, split=a.split, rep_strategy=rep,
                                      debug_dir=d, debug_max=len(frames), **kw)
        ds.split = split_flag
        for i in frames:
            ds[i]
        files = sorted(glob.glob(os.path.join(d, 'rep_*.png')))
        if len(files) != len(frames):
            raise RuntimeError(f'{mode}: expected {len(frames)} debug images, got {len(files)} in {d}')
        rows.append([cv2.imread(f) for f in files])
        print(f'{mode}: {len(files)} images in {d}')
    # left column = training (random), right column = inference (centre); one row per frame
    grid = np.concatenate([np.concatenate([tr, inf], 1) for tr, inf in zip(*rows)], 0)
    out = os.path.join(a.out, 'rep_train_vs_inference.jpg')
    cv2.imwrite(out, grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print('montage:', out, grid.shape)


if __name__ == '__main__':
    main()
