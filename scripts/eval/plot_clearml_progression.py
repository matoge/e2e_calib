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
    ('v70_unified_pair',                '#888', '-',  'pair c=2 (baseline)'),
    ('v74_unified_multi_c2',            '#aaa', '-',  'multi c=2'),
    ('v75_unified_multi_c3',            '#39a', '-',  'multi c=3'),
    ('v92_unified_multi_c4',            '#0a4', '-',  'multi c=4'),
    ('v100_unified_multi_c4_quad',      '#d22', '-',  'multi c=4 + N=4 (quad)'),
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

    # Pull all series first
    rows = []
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
        rows.append((name, color, ls, label, s, best_err, best_nll))

    # 2-row figure: top = chronological bar of best val_err per experiment,
    # bottom = per-epoch curves (val_err + val_nll).
    fig = plt.figure(figsize=(13, 8))
    gs  = fig.add_gridspec(2, 2, height_ratios=[1, 1.6], hspace=0.36, wspace=0.22)
    ax_bar = fig.add_subplot(gs[0, :])
    ax_err = fig.add_subplot(gs[1, 0])
    ax_nll = fig.add_subplot(gs[1, 1])

    # Top bar chart — chronological experiment progression
    xs = np.arange(len(rows))
    errs = [r[5] for r in rows]
    nlls = [r[6] for r in rows]
    bar_colors = [r[1] for r in rows]
    bars = ax_bar.bar(xs - 0.2, errs, width=0.4, color=bar_colors,
                       edgecolor='#222', label='val_err (px)')
    ax_bar.set_ylabel('val_err (px)')
    ax_bar.set_xticks(xs)
    ax_bar.set_xticklabels([r[3] for r in rows], rotation=12, ha='right', fontsize=10)
    for x, e in zip(xs, errs):
        ax_bar.text(x - 0.2, e + 0.02, f'{e:.2f}', ha='center', fontsize=9)
    ax_bar2 = ax_bar.twinx()
    ax_bar2.bar(xs + 0.2, nlls, width=0.4, color=bar_colors, alpha=0.45,
                 edgecolor='#666', hatch='///', label='val_nll')
    ax_bar2.set_ylabel('val_nll')
    for x, n in zip(xs, nlls):
        ax_bar2.text(x + 0.2, n + 0.02, f'{n:.2f}', ha='center', fontsize=9)
    ax_bar.set_title('Experimental progression (chronological → right)',
                      fontsize=12, fontweight='bold')
    ax_bar.set_ylim(1.4, 2.6); ax_bar2.set_ylim(1.4, 2.6)
    # highlight last bar
    bars[-1].set_edgecolor('#d22')
    bars[-1].set_linewidth(3)
    ax_bar.annotate('M=4 + c=4 jump',
                    xy=(xs[-1] - 0.2, errs[-1]),
                    xytext=(xs[-1] - 1.2, errs[-1] + 0.4),
                    arrowprops=dict(arrowstyle='->', color='#d22', lw=2),
                    color='#d22', fontsize=11, fontweight='bold')

    # Bottom — per-epoch curves
    for name, color, ls, label, s, best_err, best_nll in rows:
        full_label = f'{label}  (best {best_err:.2f}/{best_nll:.2f})'
        ax_err.plot(s['err_x'], s['err_y'], color=color, ls=ls, lw=2, label=full_label)
        if s['nll_y']:
            ax_nll.plot(s['nll_x'], s['nll_y'], color=color, ls=ls, lw=2)
    ax_err.set_xlabel('epoch'); ax_err.set_ylabel('val_err (px)')
    ax_err.set_title('val Δuv error per epoch'); ax_err.grid(alpha=0.3)
    ax_err.legend(loc='upper right', fontsize=8, framealpha=0.85)
    ax_err.set_ylim(1.5, 7.5)
    ax_nll.set_xlabel('epoch'); ax_nll.set_ylabel('val_nll')
    ax_nll.set_title('val NLL per epoch'); ax_nll.grid(alpha=0.3)
    ax_nll.set_ylim(1.4, 4.5)

    fig.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f'saved → {args.out}')


if __name__ == '__main__':
    main()
