"""Baseline-sweep evaluation: how does pair-net val accuracy degrade with frame interval?

Design: build a fresh val dataset with `baseline_range = (B, B)` for each target B,
then run the trained model over N samples and collect mean val_err / val_nll.
Plot val_err and val_nll as a function of B.

Usage:
  python scripts/eval/baseline_sweep.py --ckpt experiments/cross_frame_v41_strat_panda103
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets.pandaset_pair import PandaSetCrossFrameDataset
from models.cross_frame import CalibNetCrossFrame

import importlib
_train_cross = importlib.import_module('train_cross_frame')


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--scenes-root', default='/mnt/nvme6t/pandaset')
    ap.add_argument('--cameras', default='front_camera')
    ap.add_argument('--baselines', default='3,5,10,15,20,30,40,50,60,70')
    ap.add_argument('--n-samples', type=int, default=400, help='per baseline')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if ckpt_path.is_dir():
        ckpt_dir = ckpt_path
        ckpt_pt  = ckpt_dir / 'best_model.pt'
    else:
        ckpt_dir = ckpt_path.parent
        ckpt_pt  = ckpt_path
    print(f'loading {ckpt_pt}')
    cfg = json.loads((ckpt_dir / 'config.txt').read_text())

    out_dir = Path(args.out_dir or f'experiments/baseline_sweep_{ckpt_dir.name}')
    out_dir.mkdir(parents=True, exist_ok=True)

    uvd = cfg.get('uvd', False)

    # build model matching ckpt
    model = CalibNetCrossFrame(
        img_size=cfg['img_size'],
        n_cross_layers=cfg['n_cross_layers'],
        n_intra_layers=cfg['n_intra_layers'],
        deform_mode=cfg['deform_mode'],
        out_dim=7 if uvd else 5,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_pt, map_location=DEVICE), strict=False)
    model.eval()

    baselines = [int(b) for b in args.baselines.split(',')]
    rows = []
    for B in baselines:
        ds = PandaSetCrossFrameDataset(
            split='val',
            scenes_root=args.scenes_root,
            train_frac=0.8,
            cameras=args.cameras,
            img_size=cfg['img_size'],
            max_points=cfg['max_points'],
            baseline_range=(B, B),
            sigma_ypr=cfg['sigma_ypr'], sigma_t=cfg['sigma_t'],
            crop_range=(cfg['crop_min'], cfg['crop_max']),
            virtual_epoch_len=args.n_samples,
            seed=B + 100,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
        errs, nlls, base_errs = [], [], []
        with torch.no_grad():
            for batch in loader:
                _, m = _train_cross.step(model, batch, uvd_mode=uvd,
                                          frustum_full=cfg.get('frustum_full', True))
                errs.append(0.5 * (m['err_AB'] + m['err_BA']))
                base_errs.append(0.5 * (m['base_AB'] + m['base_BA']))
                nlls.append(m['loss'])
        errs    = np.array(errs)
        nlls    = np.array(nlls)
        bases   = np.array(base_errs)
        rows.append({
            'baseline_frames': B,
            'val_err_px':      float(np.mean(errs)),
            'val_nll':         float(np.mean(nlls)),
            'base_err_px':     float(np.mean(bases)),
            'n_batches':       len(errs),
        })
        print(f'  bl={B:3d}  err={rows[-1]["val_err_px"]:.2f}px  '
              f'nll={rows[-1]["val_nll"]:.2f}  base={rows[-1]["base_err_px"]:.2f}px',
              flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    csv_path = out_dir / 'metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f'wrote {csv_path}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=120)
    fig.patch.set_facecolor('#f6f4ed')
    ax = axes[0]
    ax.plot(df.baseline_frames, df.val_err_px,  '-o', color='#c13c14', label='val_err (corrected)')
    ax.plot(df.baseline_frames, df.base_err_px, '--', color='#6b6a63', label='base_err (no correction)')
    ax.set_xlabel('frame interval (frames)'); ax.set_ylabel('px err')
    ax.set_title('pair-net degradation with frame interval', loc='left')
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    ax2 = axes[1]
    ax2.plot(df.baseline_frames, df.val_nll, '-o', color='#174734')
    ax2.set_xlabel('frame interval (frames)'); ax2.set_ylabel('val_nll')
    ax2.set_title('NLL vs interval', loc='left')
    ax2.grid(alpha=0.25)
    for a in axes:
        for sp in ('top', 'right'): a.spines[sp].set_visible(False)
    plt.tight_layout()
    png_path = out_dir / 'degradation.png'
    plt.savefig(png_path, dpi=140, bbox_inches='tight', facecolor='#f6f4ed')
    print(f'wrote {png_path}')


if __name__ == '__main__':
    main()
