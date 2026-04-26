"""Plot val curves from v50 (pair) vs v51 (multi-frame) runs side by side."""
import re, sys, pathlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EP_RX = re.compile(r'ep\s+(\d+)\s+loss=([0-9.]+).*val_err=([0-9.]+)px.*val_nll=([0-9.]+)')

def parse(log_path):
    eps, loss, val_err, val_nll = [], [], [], []
    for line in Path(log_path).read_text().splitlines():
        m = EP_RX.search(line)
        if m:
            eps.append(int(m.group(1)))
            loss.append(float(m.group(2)))
            val_err.append(float(m.group(3)))
            val_nll.append(float(m.group(4)))
    return eps, loss, val_err, val_nll

def main():
    pair = parse('/tmp/v50_ps39.log')
    multi = parse('/tmp/v51_ps39.log')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=120)
    fig.patch.set_facecolor('#f6f4ed')

    ax = axes[0]
    ax.plot(pair[0],  pair[2],  '-o', color='#6b6a63', label='v50 pair', markersize=3)
    ax.plot(multi[0], multi[2], '-o', color='#c13c14', label='v51 multi-frame', markersize=3)
    ax.set_xlabel('epoch'); ax.set_ylabel('val_err (px)')
    ax.set_title('val pixel error', loc='left')
    ax.grid(alpha=0.25); ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(pair[0],  pair[3],  '-o', color='#6b6a63', label='v50 pair', markersize=3)
    ax.plot(multi[0], multi[3], '-o', color='#c13c14', label='v51 multi-frame', markersize=3)
    ax.set_xlabel('epoch'); ax.set_ylabel('val_nll')
    ax.set_title('val NLL', loc='left')
    ax.grid(alpha=0.25); ax.legend(frameon=False)

    for a in axes:
        for sp in ('top', 'right'): a.spines[sp].set_visible(False)
    plt.tight_layout()

    out = Path('experiments/v50_vs_v51.png')
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor='#f6f4ed')

    # tabular summary
    def best(x, y): return (min(zip(y, x)) if y else (None, None))
    print(f'v50 pair       :  best val_err = {min(pair[2]):.3f} px  best val_nll = {min(pair[3]):.3f}  ({len(pair[0])} eps)')
    print(f'v51 multi-frame:  best val_err = {min(multi[2]):.3f} px  best val_nll = {min(multi[3]):.3f}  ({len(multi[0])} eps)')
    print(f'wrote {out}')

if __name__ == '__main__':
    main()
