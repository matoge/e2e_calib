"""Render hero NLL comparison: all_v2 (fixed 1.5x crop) vs all_v3_mc (multicrop aug)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG      = "#f6f4ed"
CARD    = "#ffffff"
INK     = "#0f0f0e"
INK_DIM = "#6b6a63"
LINE    = "#d9d6cd"
ACCENT  = "#c13c14"   # red  (v3_mc = new)
ACCENT2 = "#174734"   # green (v2 = baseline)

LINE_RE = re.compile(
    r"\[\s*(\d+)/\s*\d+\]\s+"
    r"train nll=([\-+\d.]+)\(obj=([\-+\d.]+) bg=([\-+\d.]+)\) "
    r"mse=([\d.]+)\(obj=([\d.]+) bg=([\d.]+)\)\s+"
    r"val nll=([\-+\d.]+)\(obj=([\-+\d.]+) bg=([\-+\d.]+)\) "
    r"mse=([\d.]+)\(obj=([\d.]+) bg=([\d.]+)\)"
)


def parse(path):
    hist = []
    for line in Path(path).read_text().splitlines():
        m = LINE_RE.search(line)
        if not m: continue
        hist.append(dict(
            ep=int(m.group(1)),
            tr_nll=float(m.group(2)), tr_obj=float(m.group(3)), tr_bg=float(m.group(4)),
            tr_mse=float(m.group(5)),
            va_nll=float(m.group(8)), va_obj=float(m.group(9)), va_bg=float(m.group(10)),
            va_mse=float(m.group(11)),
        ))
    return hist


v2 = parse("experiments/all_v2/train.log")
v3 = parse("experiments/all_v3_mc/train.log")


def best(h, key='va_nll'):
    i = min(range(len(h)), key=lambda k: h[k][key])
    return h[i]

bv2, bv3 = best(v2), best(v3)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), dpi=130,
                          gridspec_kw=dict(width_ratios=[1.0, 1.0]))
fig.patch.set_facecolor(BG)

# ── Panel 1: val NLL curves ───────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(CARD)
ax.plot([h['ep'] for h in v2], [h['va_nll'] for h in v2],
        color=ACCENT2, lw=1.8, alpha=0.55, label='v2 (fixed 1.5× crop)')
ax.plot([h['ep'] for h in v3], [h['va_nll'] for h in v3],
        color=ACCENT,  lw=2.2,              label='v3_mc (multicrop 64–192 px)')
ax.scatter([bv2['ep']], [bv2['va_nll']], s=60, color=ACCENT2, zorder=5,
            edgecolor='white', linewidth=1.2)
ax.scatter([bv3['ep']], [bv3['va_nll']], s=60, color=ACCENT,  zorder=5,
            edgecolor='white', linewidth=1.2)
ax.annotate(f"best {bv2['va_nll']:.3f}\n@ ep{bv2['ep']}",
             xy=(bv2['ep'], bv2['va_nll']), xytext=(8, 14),
             textcoords='offset points', color=ACCENT2, fontsize=9,
             family='monospace')
ax.annotate(f"best {bv3['va_nll']:.3f}\n@ ep{bv3['ep']}",
             xy=(bv3['ep'], bv3['va_nll']), xytext=(-110, -22),
             textcoords='offset points', color=ACCENT, fontsize=9,
             family='monospace', fontweight='600')
ax.set_title('Validation NLL (Gaussian, joint NS+PS+WM)',
              color=INK, fontsize=12, pad=10, loc='left', fontweight='600')
ax.set_xlabel('epoch', color=INK_DIM, fontsize=9, family='monospace')
ax.set_ylabel('−log p(Δu,Δv | μ, Σ)', color=INK_DIM, fontsize=9, family='monospace')
ax.tick_params(colors=INK_DIM, labelsize=8)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
for s in ('bottom', 'left'): ax.spines[s].set_color(LINE)
ax.grid(True, color=LINE, lw=0.5, alpha=0.7)
leg = ax.legend(facecolor=CARD, edgecolor=LINE, labelcolor=INK,
                fontsize=9, frameon=True, loc='upper right')
leg.get_frame().set_linewidth(0.5)

# ── Panel 2: bar chart, best val NLL (total/obj/bg) ───────────────────────
ax = axes[1]
ax.set_facecolor(CARD)
labels = ['total', 'obj', 'bg']
v2_vals = [bv2['va_nll'], bv2['va_obj'], bv2['va_bg']]
v3_vals = [bv3['va_nll'], bv3['va_obj'], bv3['va_bg']]
x = range(len(labels))
w = 0.36
b1 = ax.bar([i - w/2 for i in x], v2_vals, width=w,
             color=ACCENT2, alpha=0.70, label='v2', edgecolor='white', linewidth=0.5)
b2 = ax.bar([i + w/2 for i in x], v3_vals, width=w,
             color=ACCENT,  alpha=0.92, label='v3_mc', edgecolor='white', linewidth=0.5)
for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.2f}",
             ha='center', color=ACCENT2, fontsize=9, family='monospace')
for bar in b2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, f"{h:.2f}",
             ha='center', color=ACCENT, fontsize=9, family='monospace',
             fontweight='600')
for i, (a, b) in enumerate(zip(v2_vals, v3_vals)):
    delta = (b - a) / a * 100
    ax.text(i, max(a, b) + 0.35, f"{delta:+.1f}%",
             ha='center', color=INK_DIM, fontsize=9, family='monospace')
ax.set_xticks(list(x)); ax.set_xticklabels(labels, color=INK_DIM, fontsize=10,
                                             family='monospace')
ax.set_title('Best-epoch val NLL breakdown',
              color=INK, fontsize=12, pad=10, loc='left', fontweight='600')
ax.set_ylabel('NLL', color=INK_DIM, fontsize=9, family='monospace')
ax.tick_params(colors=INK_DIM, labelsize=8)
ax.set_ylim(0, max(max(v2_vals), max(v3_vals)) * 1.2)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
for s in ('bottom', 'left'): ax.spines[s].set_color(LINE)
ax.grid(True, axis='y', color=LINE, lw=0.5, alpha=0.7)
leg = ax.legend(facecolor=CARD, edgecolor=LINE, labelcolor=INK,
                fontsize=9, frameon=True, loc='upper right')
leg.get_frame().set_linewidth(0.5)

# ── Suptitle ──────────────────────────────────────────────────────────────
fig.suptitle(
    'Multicrop augmentation — NLL proper-scoring improvement',
    color=INK, fontsize=14, fontweight='700', y=0.99, x=0.02, ha='left')
fig.text(0.02, 0.935,
          'Same model (1.62 M params), data, seed, schedule. '
          'Only difference: cache bbox_scale 1.5→3.0, train-time sub-window s∈[64,192].',
          color=INK_DIM, fontsize=9, family='monospace')

plt.tight_layout(rect=[0, 0, 1, 0.91])
out = Path('hero_multicrop_nll.png')
plt.savefig(out, facecolor=BG, dpi=130, bbox_inches='tight')
plt.close(fig)
print(f"wrote {out}")
