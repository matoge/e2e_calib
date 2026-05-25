"""Render the 16 BA-eval candidate idxs as a 4×4 parent-tile grid so we can
eyeball the variety (driving / urban / lighting / parked-car) before locking
them in."""
from __future__ import annotations
import io, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from datasets.pandaset_full import PandaSetCalibDatasetFull

CACHE = '/home/hfunaya/cache_v4/kamikado_v3_tiled'
OUT = REPO / 'scripts' / '_debug' / '_outputs' / 'ba16_preview.png'
IDXS = [0, 17, 279, 559, 839, 1119, 1399, 1679,
        1959, 2239, 2519, 2799, 3079, 3359, 3639, 3919]

def main():
    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE, split='val',
        img_size=256, min_crop_px=128, max_crop_px=512,
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=32, center_band=0.0, preload=False,
    )
    print(f'val total: {len(ds.fnames)}')
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), dpi=110)
    for ax, idx in zip(axes.flat, IDXS):
        inst = ds._load_inst(int(idx))
        if 'jpg_bytes' in inst:
            img = np.array(Image.open(io.BytesIO(inst['jpg_bytes'])).convert('RGB'))
        else:
            img = np.asarray(inst['img'])
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img = np.transpose(img, (1, 2, 0))
        H, W = img.shape[:2]
        is_fish = inst.get('is_fisheye', False)
        fname = ds.fnames[int(idx)]
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'idx={idx}  {W}×{H}  fisheye={is_fish}\n{fname}',
                     fontsize=8)
    fig.suptitle(f'BA-eval candidates (16 fisheye idxs from kamikado val, '
                 f'cache={CACHE})', fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote: {OUT}  ({OUT.stat().st_size/1024:.1f} KB)')

if __name__ == '__main__':
    main()
