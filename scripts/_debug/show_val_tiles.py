"""Show val tiles 0..N-1 in a grid, so the user can pick a non-boring
anchor for the Phase 1.b smoke. Outputs a single PNG with tile index
captions."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from datasets.pandaset_full import PandaSetCalibDatasetFull

CACHE = Path.home() / 'cache' / 'kamikado_v3_tiled'
OUT   = REPO / 'scripts' / '_debug' / '_outputs' / 'val_tiles_grid.png'
N_TILES = 20
COLS = 5

EXP_CFG_PATH = REPO / 'experiments' / 'km_wv_wm_dgx2_n4_img128_8gpu_HEAD' / 'config.py'


def _load_cfg() -> dict:
    src = EXP_CFG_PATH.read_text()
    ns: dict = {}
    exec(src, ns, ns)
    return ns['CFG']


def main():
    cfg = _load_cfg()
    ds = PandaSetCalibDatasetFull(
        cache_dir=CACHE,
        split='val',
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'],
        max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0,
        max_rot_deg=0.0,
        oversample=1,
        grid_n=cfg.get('grid_n', 16),
        center_band=0.0,
        preload=False,
    )
    print(f"[show] {len(ds.fnames)} val instances, showing first {N_TILES}")

    rows = (N_TILES + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 3.0, rows * 3.0))
    axes = axes.ravel()

    shown = 0
    cand = 0
    while shown < N_TILES and cand < len(ds.fnames):
        win = ds.apply_perturbation_explicit(cand, np.zeros(3), np.zeros(3))
        if win is None:
            cand += 1
            continue
        # win = (img_u8, true_uvd, dist_uvd, pad_mask, vfp, bucket_uvd, bucket_valid, pert)
        img_u8 = win[0]
        # img_u8 shape: (3, H, W) uint8
        img = img_u8.permute(1, 2, 0).cpu().numpy().astype('float32') / 255.0
        img = np.clip(img, 0, 1)
        ax = axes[shown]
        ax.imshow(img)
        ax.set_title(f"idx={cand}\n{ds.fnames[cand]}", fontsize=8)
        ax.axis('off')
        shown += 1
        cand += 1

    for k in range(shown, len(axes)):
        axes[k].axis('off')

    fig.suptitle(f"kamikado val first {N_TILES} usable tiles\n"
                  f"pick one — that becomes the anchor for overfit_2dof_ba_stream.py",
                  y=0.995, fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"[show] wrote → {OUT}")


if __name__ == '__main__':
    main()
