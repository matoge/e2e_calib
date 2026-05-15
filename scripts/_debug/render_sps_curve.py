"""Render SPS / val_nll curves from train.log files for the blog."""
import re, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r'\[\s*(\d+)/\d+\]\s+train nll=([+-]?\d+\.\d+).*?val nll=([+-]?\d+\.\d+).*?sps\(global\)=(\d+)'
)

def parse(p: Path):
    eps, tr_nll, va_nll, sps = [], [], [], []
    for ln in p.read_text().splitlines():
        m = LINE_RE.search(ln)
        if m:
            eps.append(int(m.group(1)))
            tr_nll.append(float(m.group(2)))
            va_nll.append(float(m.group(3)))
            sps.append(int(m.group(4)))
    return eps, tr_nll, va_nll, sps

runs = [
    ('os=1, bs=384', 'experiments/km_wv_8gpu_50ep_os1_bs384_v8/train.log'),
    ('os=4, bs=384 (200ep)', 'experiments/km_wv_8gpu_200ep_os4/train.log'),
]

fig, (ax_sps, ax_nll) = plt.subplots(1, 2, figsize=(11, 4))
for label, log in runs:
    p = Path('/workspace') / log
    if not p.is_file():
        continue
    eps, tr, va, sps = parse(p)
    if not eps:
        continue
    ax_sps.plot(eps, sps, marker='o', ms=3, label=label)
    ax_nll.plot(eps, va, marker='o', ms=3, label=f'{label} (val)')
ax_sps.axhline(3000, color='gray', ls='--', lw=0.8, label='target 3000')
ax_sps.set_xlabel('epoch'); ax_sps.set_ylabel('sps(global)')
ax_sps.set_title('Throughput on DGX2 V100×8 (kamikado+woven joint)')
ax_sps.set_ylim(0, 6000); ax_sps.grid(alpha=0.3); ax_sps.legend()
ax_nll.set_xlabel('epoch'); ax_nll.set_ylabel('val NLL')
ax_nll.set_title('Convergence — val_nll');
ax_nll.grid(alpha=0.3); ax_nll.legend()
plt.tight_layout()
out = Path('/workspace/docs/assets/dgx2_sps')
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'sps_curve.png', dpi=110, bbox_inches='tight')
print(f'wrote {out / "sps_curve.png"}')
