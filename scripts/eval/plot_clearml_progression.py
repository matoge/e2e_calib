"""Pull val_err / val_nll curves for selected ClearML tasks and plot them
side-by-side. Used to generate hero figures for write-ups.

Usage:
    python scripts/eval/plot_clearml_progression.py --out /tmp/leaderboard.png
"""
import argparse
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from clearml import Task

# (task_name, color, style, label) — order = chronological progression
PROGRESSION = [
    ('v70_unified_pair',                '#888', '-',  'pair c2 (unified)'),
    ('v75_unified_multi_c3',            '#39a', '-',  'multi c3'),
    ('v92_unified_multi_c4',            '#0a4', '-',  'multi c4'),
    ('v100_unified_multi_c4_quad',      '#d22', '-',  'multi c4 quad (N=4)'),
]


def get_series(task_name, project='e2e_calib/cross-frame'):
    t = Task.get_task(project_name=project, task_name=task_name)
    m = t.get_reported_scalars()
    # Try new title scheme first, then legacy
    for tag, sub in [('err', 'val'), ('val', 'err')]:
        if tag in m and sub in m[tag]:
            err = m[tag][sub]
            break
    else:
        return None
    for tag, sub in [('nll', 'val'), ('val', 'nll')]:
        if tag in m and sub in m[tag]:
            nll = m[tag][sub]
            break
    else:
        nll = None
    return dict(err_x=err['x'], err_y=err['y'],
                nll_x=(nll['x'] if nll else None),
                nll_y=(nll['y'] if nll else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/leaderboard.png')
    ap.add_argument('--project', default='e2e_calib/cross-frame')
    args = ap.parse_args()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharex=True)
    for name, color, ls, label in PROGRESSION:
        try:
            s = get_series(name, args.project)
        except Exception as e:
            print(f'  skip {name}: {e}')
            continue
        if s is None:
            print(f'  no data: {name}')
            continue
        best_err = min(s['err_y'])
        best_nll = min(s['nll_y']) if s['nll_y'] else float('nan')
        full_label = f'{label}: err={best_err:.2f} / nll={best_nll:.2f}'
        ax[0].plot(s['err_x'], s['err_y'], color=color, ls=ls, lw=2, label=full_label)
        if s['nll_y']:
            ax[1].plot(s['nll_x'], s['nll_y'], color=color, ls=ls, lw=2)

    ax[0].set_xlabel('epoch'); ax[0].set_ylabel('val_err (px)')
    ax[0].set_title('val Δuv error')
    ax[0].legend(loc='upper right', fontsize=9, framealpha=0.85)
    ax[0].grid(alpha=0.3)
    ax[0].set_ylim(1.5, 7.5)

    ax[1].set_xlabel('epoch'); ax[1].set_ylabel('val_nll')
    ax[1].set_title('val NLL (uncertainty calibration)')
    ax[1].grid(alpha=0.3)
    ax[1].set_ylim(1.4, 4.5)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f'saved → {args.out}')


if __name__ == '__main__':
    main()
