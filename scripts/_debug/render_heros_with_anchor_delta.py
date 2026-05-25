"""Re-render hero overlays using the SAVED δ̂ from the anchor solve.
No re-solving — load delta.npy / target.npz, project on the chosen 4 idxs.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import scripts.eval.eval_shared_256x800 as ess
import torch
from datasets.pandaset_full import PandaSetCalibDatasetFull


def main():
    cfg = ess._load_cfg()
    src = Path('scripts/_debug/score_all_anchor17')
    delta = torch.from_numpy(np.load(src / 'delta.npy')).to(ess.DEVICE)
    tgt   = np.load(src / 'target.npz')
    ypr_t, t_t = tgt['ypr'], tgt['t']

    # Picks: 3 sub-pixel from spread-out val idx + 1 mediocre.
    HEROS = [
        ('idx17',   17,   'sub-pixel'),    # anchor itself, low corr
        ('idx1770', 1770, 'sub-pixel'),
        ('idx2725', 2725, 'sub-pixel'),
        ('idx3182', 3182, 'mediocre'),
    ]
    out_dir = Path('docs/assets/2026-05-22_subpixel_calib')
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = PandaSetCalibDatasetFull(
        cache_dir=ess.CACHE, split='val',
        img_size=cfg['img_size'],
        min_crop_px=cfg['min_crop_px'], max_crop_px=cfg['max_crop_px'],
        max_offset_m=0.0, max_rot_deg=0.0,
        oversample=1, grid_n=cfg.get('grid_n', 16),
        center_band=0.0, preload=False)

    print(f'[delta]  ω={delta[:3].cpu().numpy()}  t={delta[3:].cpu().numpy()}')
    print(f'[target] ypr={ypr_t}  t={t_t}')
    print()
    for tag, idx, kind in HEROS:
        inst = ds._load_inst(int(idx))
        out_p = out_dir / f'hero_{tag}.png'
        info = ess.render_3panel_overlay(
            inst, ypr_t, t_t, delta,
            out_path=out_p,
            suptitle=f'val idx={idx} ({kind})  —  shared δ̂ from B=800 anchored on idx=17',
            panel_label='BA-corrected (B=800)')
        print(f'  hero_{tag} idx={idx:5d}  pert={info["reproj_pert_mean"]:6.2f}  '
              f'corr={info["reproj_corr_mean"]:.3f}  → {out_p}')


if __name__ == '__main__':
    main()
