"""Explain the crop pipeline visually: v2 (fixed 1.5x → 64x64) vs v3_mc (3x → 192x192 cache, random sub-window → 64x64)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import torch.nn.functional as F

BG      = "#f6f4ed"
CARD    = "#ffffff"
INK     = "#0f0f0e"
INK_DIM = "#6b6a63"
LINE    = "#d9d6cd"
ACCENT  = "#c13c14"
ACCENT2 = "#174734"

cache = torch.load('/mnt/nvme6t/e2e_calib_cache/nuscenes_mc_cache.pt', weights_only=False)
tr = cache['train']

rng = np.random.default_rng(7)
picks = []
for i in rng.permutation(len(tr))[:2000]:
    inst = tr[int(i)]
    d = inst['obj_dims']
    if d.norm() > 0 and 3.5 < float(d[1]) < 5.5:
        picks.append(int(i))
        if len(picks) == 1: break

inst = tr[picks[0]]
img_cache = inst['img_cache']
C = img_cache.shape[-1]
S = 64

def to_img(t):
    return t.permute(1,2,0).numpy() / 255.0

def crop_resize(img_cache, u0, v0, s, S):
    sub = img_cache[:, v0:v0+s, u0:u0+s].float().unsqueeze(0)
    out = F.interpolate(sub, size=(S, S), mode='bilinear', align_corners=False).squeeze(0)
    return (out / 255.0).permute(1,2,0).numpy()

cache_img = to_img(img_cache)

v2_crop_px = C // 3
v2_u0 = (C - v2_crop_px) // 2
v2_v0 = (C - v2_crop_px) // 2
v2_out = crop_resize(img_cache, v2_u0, v2_v0, v2_crop_px, S)

sub_sizes = [72, 120, 180]
sub_samples = []
_rng = np.random.default_rng(3)
for s in sub_sizes:
    lo_v = max(0, 2*C//3 - s); hi_v = min(C - s, C//3)
    lo_u = max(0, 2*C//3 - s); hi_u = min(C - s, C//3)
    if hi_v < lo_v: hi_v = lo_v
    if hi_u < lo_u: hi_u = lo_u
    u0 = int(_rng.integers(lo_u, hi_u+1))
    v0 = int(_rng.integers(lo_v, hi_v+1))
    sub_samples.append((u0, v0, s, crop_resize(img_cache, u0, v0, s, S)))


fig = plt.figure(figsize=(14, 10), dpi=130)
fig.patch.set_facecolor(BG)
gs = fig.add_gridspec(3, 6, height_ratios=[0.35, 1.2, 1.2], hspace=0.45, wspace=0.35,
                       left=0.04, right=0.97, top=0.93, bottom=0.05)

ax_title = fig.add_subplot(gs[0, :]); ax_title.axis('off')
ax_title.text(0.0, 0.82, 'Crop pipeline — v2 vs v3_mc',
               color=INK, fontsize=18, fontweight='800', family='sans-serif', transform=ax_title.transAxes)
ax_title.text(0.0, 0.40,
    'Same raw camera image.  v2 crops tight (bbox × 1.5) → fixed 64×64.  v3_mc caches large (bbox × 3.0 → 192×192), '
    'then draws a random sub-window s ∈ [64,192] each training iteration and resizes it to 64×64.',
    color=INK_DIM, fontsize=11, family='sans-serif', transform=ax_title.transAxes)

ax_v2_cache = fig.add_subplot(gs[1, 0:2])
ax_v2_cache.set_facecolor(CARD)
ax_v2_cache.imshow(cache_img)
rect = mpatches.Rectangle((v2_u0, v2_v0), v2_crop_px, v2_crop_px,
                            linewidth=3, edgecolor=ACCENT2, facecolor='none')
ax_v2_cache.add_patch(rect)
ax_v2_cache.text(v2_u0 + v2_crop_px/2, v2_v0 - 8, 'bbox × 1.5  ≈  central 1/3',
                  color=ACCENT2, fontsize=11, ha='center', fontweight='700', family='monospace')
ax_v2_cache.set_title('v2 baseline — fixed crop',
                       color=ACCENT2, fontsize=13, fontweight='700', loc='left', pad=10)
ax_v2_cache.text(C/2, C + 18, f'full camera frame ({C}×{C} shown as ref)',
                  color=INK_DIM, fontsize=9, ha='center', family='monospace')
ax_v2_cache.set_xticks([]); ax_v2_cache.set_yticks([])
for s in ax_v2_cache.spines.values(): s.set_color(LINE)

ax_v2_out = fig.add_subplot(gs[1, 2])
ax_v2_out.set_facecolor(CARD)
ax_v2_out.imshow(v2_out)
ax_v2_out.set_title('→ 64×64 (fed to model)\n  every epoch, identical',
                     color=ACCENT2, fontsize=11, fontweight='600', loc='left', pad=8)
ax_v2_out.set_xticks([]); ax_v2_out.set_yticks([])
for sp in ax_v2_out.spines.values(): sp.set_color(ACCENT2); sp.set_linewidth(2.5)

ax_note_v2 = fig.add_subplot(gs[1, 3:6]); ax_note_v2.axis('off')
ax_note_v2.text(0.02, 0.95,
    'v2 behavior:', color=ACCENT2, fontsize=13, fontweight='700',
    family='sans-serif', transform=ax_note_v2.transAxes, va='top')
ax_note_v2.text(0.02, 0.78,
    '• The model sees the SAME 64×64 image every epoch.',
    color=INK, fontsize=11, family='sans-serif', transform=ax_note_v2.transAxes, va='top')
ax_note_v2.text(0.02, 0.62,
    '• 200 epochs × 60k samples → pixel patterns can be memorized.',
    color=INK, fontsize=11, family='sans-serif', transform=ax_note_v2.transAxes, va='top')
ax_note_v2.text(0.02, 0.46,
    '• Once memorized, the σ head is free to claim tight uncertainty.',
    color=INK, fontsize=11, family='sans-serif', transform=ax_note_v2.transAxes, va='top')
ax_note_v2.text(0.02, 0.30,
    '• Val hits non-memorized residuals → (Δ/σ)² explodes → NLL rises.',
    color=ACCENT, fontsize=11, family='sans-serif', transform=ax_note_v2.transAxes, va='top',
    fontweight='600')

ax_v3_cache = fig.add_subplot(gs[2, 0:2])
ax_v3_cache.set_facecolor(CARD)
ax_v3_cache.imshow(cache_img)
for (u0, v0, s, _), color, lw in zip(sub_samples,
                                       [ACCENT, '#e57733', '#f2a94f'],
                                       [3.0, 2.2, 1.6]):
    rect = mpatches.Rectangle((u0, v0), s, s,
                               linewidth=lw, edgecolor=color, facecolor='none',
                               linestyle='--')
    ax_v3_cache.add_patch(rect)
    ax_v3_cache.text(u0 + s/2, v0 + 14, f's={s}',
                      color=color, fontsize=10, ha='center', fontweight='700',
                      family='monospace',
                      bbox=dict(boxstyle='round,pad=0.2', fc=CARD, ec=color, lw=0.8))
ax_v3_cache.set_title('v3_mc — 192×192 cache + random sub-window  s ∈ [64, 192]',
                       color=ACCENT, fontsize=13, fontweight='700', loc='left', pad=10)
ax_v3_cache.text(C/2, C + 18, f'cached at bbox × 3.0  →  {C}×{C}',
                  color=INK_DIM, fontsize=9, ha='center', family='monospace')
ax_v3_cache.set_xticks([]); ax_v3_cache.set_yticks([])
for s in ax_v3_cache.spines.values(): s.set_color(LINE)

for j, ((u0, v0, s, img), color) in enumerate(zip(sub_samples,
                                                    [ACCENT, '#e57733', '#f2a94f'])):
    ax = fig.add_subplot(gs[2, 2 + j])
    ax.set_facecolor(CARD)
    ax.imshow(img)
    ax.set_title(f'sub s={s} → 64×64',
                  color=color, fontsize=10, fontweight='600', loc='left', pad=6,
                  family='monospace')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(color); sp.set_linewidth(2.0)

ax_note_v3 = fig.add_subplot(gs[2, 5]); ax_note_v3.axis('off')
ax_note_v3.text(0.02, 0.95,
    'v3_mc behavior:', color=ACCENT, fontsize=11, fontweight='700',
    family='sans-serif', transform=ax_note_v3.transAxes, va='top')
ax_note_v3.text(0.02, 0.78,
    '• Each iter: fresh (u,v,s) → new effective image.',
    color=INK, fontsize=9, family='sans-serif', transform=ax_note_v3.transAxes, va='top')
ax_note_v3.text(0.02, 0.62,
    '• Pixel memorization becomes infeasible.',
    color=INK, fontsize=9, family='sans-serif', transform=ax_note_v3.transAxes, va='top')
ax_note_v3.text(0.02, 0.46,
    '• Model must learn scale-invariant features.',
    color=INK, fontsize=9, family='sans-serif', transform=ax_note_v3.transAxes, va='top')
ax_note_v3.text(0.02, 0.30,
    '• σ can only minimize NLL by tracking the true residual distribution.',
    color=ACCENT, fontsize=9, family='sans-serif', transform=ax_note_v3.transAxes, va='top',
    fontweight='600')

out = 'experiments/all_v3_mc/crop_explainer.png'
plt.savefig(out, facecolor=BG, dpi=130, bbox_inches='tight')
plt.close(fig)
print(f'wrote {out}')
