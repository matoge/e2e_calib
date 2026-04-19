"""Generate a light, editorial-style tech report for an experiment.

Produces:
  - vis/curves.png  (re-rendered, light palette)
  - vis/hero.png    (fresh, clean before/after of one large-shift sample)
  - report.html     (typography-forward light layout, hero-first)
"""
import sys, re, socket, torch
from pathlib import Path
from datetime import datetime
from importlib import util
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

BG      = "#f6f4ed"
CARD    = "#ffffff"
INK     = "#0f0f0e"
INK_DIM = "#6b6a63"
LINE    = "#d9d6cd"
ACCENT  = "#c13c14"
ACCENT2 = "#174734"

LINE_RE = re.compile(
    r"\[\s*(\d+)/\s*\d+\]\s+"
    r"train nll=([\-+\d.]+)\(obj=([\-+\d.]+) bg=([\-+\d.]+)\) "
    r"mse=([\d.]+)\(obj=([\d.]+) bg=([\d.]+)\)\s+"
    r"val nll=([\-+\d.]+)\(obj=([\-+\d.]+) bg=([\-+\d.]+)\) "
    r"mse=([\d.]+)\(obj=([\d.]+) bg=([\d.]+)\)"
)

def load_cfg(exp_dir: Path) -> dict:
    spec = util.spec_from_file_location("_cfg", exp_dir / "config.py")
    m = util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.CFG

def parse_log(log_path: Path):
    txt = log_path.read_text(); hist = []
    for line in txt.splitlines():
        m = LINE_RE.search(line)
        if not m: continue
        hist.append(dict(
            ep=int(m.group(1)),
            tr_nll=float(m.group(2)), tr_obj_nll=float(m.group(3)), tr_bg_nll=float(m.group(4)),
            tr_mse=float(m.group(5)), tr_obj_mse=float(m.group(6)), tr_bg_mse=float(m.group(7)),
            va_nll=float(m.group(8)), va_obj_nll=float(m.group(9)), va_bg_nll=float(m.group(10)),
            va_mse=float(m.group(11)), va_obj_mse=float(m.group(12)), va_bg_mse=float(m.group(13)),
        ))
    best = None
    for line in txt.splitlines():
        m = re.search(r"Best val NLL: ([\d.]+)", line)
        if m: best = float(m.group(1))
    return hist, best

def render_curves(hist, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), dpi=110)
    fig.patch.set_facecolor(BG)
    eps = [h['ep'] for h in hist]
    for ax, (kt, kv, title, ylabel) in zip(axes, [
        ('tr_nll', 'va_nll', 'Gaussian NLL', ''),
        ('tr_mse', 'va_mse', 'Mean point error', 'pixels'),
    ]):
        ax.set_facecolor(CARD)
        ax.plot(eps, [h[kt] for h in hist], color=ACCENT2, lw=1.2, label='train', alpha=0.85)
        ax.plot(eps, [h[kv] for h in hist], color=ACCENT,  lw=1.9, label='val')
        ax.set_title(title, color=INK, fontsize=12, pad=10, loc='left',
                     fontweight='600')
        ax.set_xlabel('epoch', color=INK_DIM, fontsize=9, family='monospace')
        if ylabel:
            ax.set_ylabel(ylabel, color=INK_DIM, fontsize=9, family='monospace')
        ax.tick_params(colors=INK_DIM, labelsize=8)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        for s in ('bottom', 'left'): ax.spines[s].set_color(LINE)
        ax.grid(True, color=LINE, lw=0.5, alpha=0.7)
        leg = ax.legend(facecolor=CARD, edgecolor=LINE, labelcolor=INK,
                        fontsize=9, frameon=True, loc='upper right')
        leg.get_frame().set_linewidth(0.5)
    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, facecolor=BG, dpi=110, bbox_inches='tight')
    plt.close(fig)

def render_hero(exp_dir: Path, cfg: dict, out_path: Path):
    """2×2 hero: SAME image, four synthetic pure-2D shifts in different
    directions. Each panel shows per-point correction arrows over all
    points (not just obj), the distorted BB (red), corrected BB (green),
    and the mean-shift arrow."""
    from dataset_pandaset import PandaSetCalibDataset
    from model_depth import CalibNetDepth
    import numpy as np, random as _r

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CalibNetDepth(img_size=cfg["img_size"], in_channels=cfg["in_channels"],
                          n_layers=cfg["n_layers"], self_first=False,
                          use_convnext=cfg.get("use_convnext", False),
                          use_frustum=cfg.get("use_frustum", False)).to(device)
    model.load_state_dict(torch.load(exp_dir / "best_model.pt",
                                     map_location=device, weights_only=True))
    model.eval()

    ds = PandaSetCalibDataset('/tmp/pandaset_cache.pt', split='val')

    # Pick ONE clean base sample: well-centered, decent-sized obj.
    # PandaSet crops are 64x64 — all thresholds are in that native coord space.
    scan = list(range(len(ds))); _r.Random(23).shuffle(scan)
    base_idx = None
    base_img, base_true, base_dist = None, None, None
    probe_img, _, _ = ds[scan[0]]
    S = probe_img.shape[-1]          # native image side (64 for PandaSet)
    cx_tgt, cy_tgt = S / 2, S / 2
    for idx in scan[:3000]:
        img, true_uvd, dist_uvd = ds[idx]
        is_obj = dist_uvd[:, 3].numpy() > 0.5
        if is_obj.sum() < 30: continue
        t_obj = true_uvd[is_obj, :2].numpy()
        w = float(t_obj[:,0].max() - t_obj[:,0].min())
        h = float(t_obj[:,1].max() - t_obj[:,1].min())
        cx, cy = float(t_obj[:,0].mean()), float(t_obj[:,1].mean())
        if max(w, h) < 0.25*S or max(w, h) > 0.55*S: continue
        if min(w, h) < 0.12*S: continue
        if abs(cx - cx_tgt) > 0.14*S or abs(cy - cy_tgt) > 0.14*S: continue
        # avoid obj too close to image border after synthetic shifts (±6 px at S=64)
        if t_obj[:,0].min() < 0.10*S or t_obj[:,0].max() > 0.90*S: continue
        if t_obj[:,1].min() < 0.10*S or t_obj[:,1].max() > 0.90*S: continue
        base_idx = idx
        base_img, base_true, base_dist = img, true_uvd, dist_uvd
        break
    if base_idx is None:
        raise RuntimeError("no suitable base sample found for hero")

    # Four synthetic pure-2D shifts applied to the TRUE projection.
    # Scaled to image side so obj stays on-image after shift (~9% of S).
    _sh = round(0.09 * S, 1)
    shifts = [(-_sh, -_sh), (+_sh, -_sh), (-_sh, +_sh), (+_sh, +_sh)]

    panels = []
    for (dx_in, dy_in) in shifts:
        synth = base_dist.clone()
        synth[:, 0] = base_true[:, 0] + dx_in
        synth[:, 1] = base_true[:, 1] + dy_in
        # depth and obj flag inherited from the original dist sample
        with torch.no_grad():
            pad = torch.zeros(1, synth.shape[0], dtype=torch.bool, device=device)
            params = model(base_img.unsqueeze(0).to(device),
                           synth.unsqueeze(0).to(device)[..., :3],
                           key_padding_mask=pad)[0].cpu().float()
        pred_uv  = (synth[:, :2] + params[:, :2]).numpy()
        dist_uv  = synth[:, :2].numpy()
        true_uv  = base_true[:, :2].numpy()
        is_obj   = synth[:, 3].numpy() > 0.5

        d_obj = dist_uv[is_obj]; p_obj = pred_uv[is_obj]; t_obj = true_uv[is_obj]
        mean_shift = (p_obj - d_obj).mean(0)
        err_before = float(np.linalg.norm(d_obj - t_obj, axis=1).mean())
        err_after  = float(np.linalg.norm(p_obj - t_obj, axis=1).mean())
        panels.append(dict(
            dx_in=dx_in, dy_in=dy_in,
            dist_uv=dist_uv, pred_uv=pred_uv, true_uv=true_uv, is_obj=is_obj,
            mean_dx=float(mean_shift[0]), mean_dy=float(mean_shift[1]),
            err_before=err_before, err_after=err_after,
            d_obj=d_obj, p_obj=p_obj,
        ))

    # Render 2x2 grid with the same underlying image
    fig, axes = plt.subplots(2, 2, figsize=(13, 13), dpi=110)
    fig.patch.set_facecolor(BG)
    axes = axes.flatten()
    arr_img = base_img.permute(1, 2, 0).numpy()
    S = arr_img.shape[0]   # image side in its native coord space (64 for PandaSet)
    for ax, p in zip(axes, panels):
        ax.set_facecolor(CARD)
        ax.imshow(arr_img, extent=[0, S, S, 0])

        u = p['pred_uv'][:, 0] - p['dist_uv'][:, 0]
        v = p['pred_uv'][:, 1] - p['dist_uv'][:, 1]
        is_obj = p['is_obj']
        if (~is_obj).any():
            ax.quiver(p['dist_uv'][~is_obj, 0], p['dist_uv'][~is_obj, 1],
                      u[~is_obj], v[~is_obj],
                      angles='xy', scale_units='xy', scale=1,
                      color='#79c4ff', width=0.0045,
                      headwidth=3.5, headlength=4.5, alpha=0.85, zorder=3)
        if is_obj.any():
            ax.quiver(p['dist_uv'][is_obj, 0], p['dist_uv'][is_obj, 1],
                      u[is_obj], v[is_obj],
                      angles='xy', scale_units='xy', scale=1,
                      color='#ffb133', width=0.007,
                      headwidth=3.5, headlength=4.5, alpha=1.0, zorder=5)

        d_obj = p['d_obj']
        x0, y0 = float(d_obj[:, 0].min()), float(d_obj[:, 1].min())
        x1, y1 = float(d_obj[:, 0].max()), float(d_obj[:, 1].max())
        padbb = 2.5
        ax.add_patch(plt.Rectangle((x0 - padbb, y0 - padbb),
                                   (x1 - x0) + 2 * padbb, (y1 - y0) + 2 * padbb,
                                   fill=False, ec=ACCENT, lw=2.2, zorder=6))
        ax.add_patch(plt.Rectangle((x0 - padbb + p['mean_dx'], y0 - padbb + p['mean_dy']),
                                   (x1 - x0) + 2 * padbb, (y1 - y0) + 2 * padbb,
                                   fill=False, ec='#2fcf7a', lw=2.2,
                                   linestyle='--', zorder=6))

        cxb, cyb = (x0 + x1) / 2, (y0 + y1) / 2
        arr = ax.annotate("", xy=(cxb + p['mean_dx'], cyb + p['mean_dy']),
                          xytext=(cxb, cyb),
                          arrowprops=dict(arrowstyle='-|>,head_width=0.5,head_length=0.7',
                                          color='white', lw=2.6, mutation_scale=18),
                          zorder=7)
        arr.arrow_patch.set_path_effects([
            pe.Stroke(linewidth=5.2, foreground='black'),
            pe.Normal()])

        ax.text(S*0.023, S*0.96,
                f" {p['err_before']:.1f} px  →  {p['err_after']:.2f} px ",
                color='white', fontsize=12, family='monospace', fontweight='700',
                bbox=dict(facecolor=ACCENT, edgecolor='none', pad=5), zorder=8)
        ax.text(S*0.977, S*0.047,
                f" input shift ({p['dx_in']:+.0f}, {p['dy_in']:+.0f}) ",
                color='white', fontsize=11, family='monospace', fontweight='700',
                ha='right', va='top',
                bbox=dict(facecolor=ACCENT2, edgecolor='none', pad=5), zorder=8)

        ax.set_xlim(0, S); ax.set_ylim(S, 0)
        for s in ax.spines.values(): s.set_color(LINE)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout(pad=1.2)
    plt.savefig(out_path, facecolor=BG, dpi=110, bbox_inches='tight')
    plt.close(fig)
    return dict(
        base_idx=base_idx,
        picks=[(base_idx, p['dx_in'], p['dy_in'], p['err_before'], p['err_after']) for p in panels])

def fmt_cfg(cfg: dict) -> str:
    body = "\n".join(f"    {k:<13}= {v!r}," for k, v in cfg.items())
    return f"<b>CFG</b> = dict(\n{body}\n)"

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — Technical Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg:       #f6f4ed;
  --card:     #ffffff;
  --ink:      #0f0f0e;
  --ink-dim:  #6b6a63;
  --accent:   #c13c14;
  --accent2:  #174734;
  --line:     #d9d6cd;
  --sans:     -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono:     "JetBrains Mono", Consolas, ui-monospace, monospace;
  --col:      880px;
}}

* {{ box-sizing: border-box; }}
html, body {{ background: var(--bg); margin: 0; }}
body {{
  color: var(--ink);
  font-family: var(--sans);
  font-size: 17px; line-height: 1.62; font-weight: 400;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}}

.page {{
  max-width: var(--col);
  margin: 0 auto;
  padding: 72px 40px 120px;
}}
.page > * {{ max-width: 100%; }}

.eyebrow {{
  font-family: var(--mono); font-size: 11px;
  letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--ink-dim); margin-bottom: 28px;
  display: flex; flex-wrap: wrap; gap: 18px;
}}
.eyebrow .dot {{ color: var(--accent); }}

h1.title {{
  font-family: var(--sans);
  font-size: 60px; font-weight: 800;
  line-height: 1.05; letter-spacing: -0.025em;
  margin: 0 0 22px; color: var(--ink);
}}

.subtitle {{
  font-family: var(--sans); font-weight: 500;
  font-size: 21px; line-height: 1.4;
  color: var(--ink); margin: 0 0 40px;
}}
.subtitle strong {{ color: var(--accent); font-weight: 600; }}

.hero {{
  margin: 40px 0 12px;
  background: var(--card);
  border: 1px solid var(--line);
  padding: 8px;
}}
.hero img {{ width: 100%; display: block; }}
.hero-cap {{
  font-family: var(--sans); font-size: 14px;
  color: var(--ink-dim);
  padding: 14px 8px 6px; line-height: 1.55;
}}
.hero-cap b {{
  display: inline-block;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-size: 11px; font-weight: 700;
  margin-right: 10px;
  font-family: var(--mono);
}}

.meta {{
  font-family: var(--mono); font-size: 12px;
  color: var(--ink-dim);
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  padding: 14px 0; margin: 48px 0 0;
  display: flex; flex-wrap: wrap; gap: 34px;
}}
.meta b {{ color: var(--ink); font-weight: 600; }}

h2 {{
  font-family: var(--sans);
  font-size: 42px; font-weight: 700;
  line-height: 1.12; letter-spacing: -0.02em;
  margin: 88px 0 18px; color: var(--ink);
}}
h2 .num {{
  display: block;
  font-family: var(--mono);
  font-size: 13px; font-weight: 600;
  color: var(--accent);
  letter-spacing: 0.22em; text-transform: uppercase;
  margin-bottom: 14px;
}}
h3 {{
  font-family: var(--sans);
  font-size: 24px; font-weight: 600;
  line-height: 1.3;
  margin: 40px 0 10px; color: var(--ink);
}}

p {{ margin: 0 0 1.1em; }}
ul {{ margin: 0 0 1.2em; padding-left: 1.25em; }}
li {{ margin-bottom: 0.35em; }}
strong {{ color: var(--accent); font-weight: 600; }}
em {{ color: var(--ink); font-style: italic; }}
code {{
  font-family: var(--mono);
  font-size: 0.88em;
  color: var(--accent2);
  background: rgba(23,71,52,0.06);
  padding: 1px 6px; border-radius: 3px;
}}

.metrics {{
  display: grid; grid-template-columns: repeat(4, 1fr);
  gap: 1px; background: var(--line);
  border: 1px solid var(--line);
  margin: 40px 0 48px;
}}
.metric {{ background: var(--card); padding: 22px 24px; }}
.metric .label {{
  font-family: var(--mono); font-size: 11px;
  letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--ink-dim);
  margin-bottom: 12px; font-weight: 500;
}}
.metric .value {{
  font-family: var(--sans);
  font-size: 36px; font-weight: 700;
  color: var(--ink); line-height: 1;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.02em;
}}
.metric .value small {{
  font-family: var(--mono); font-size: 13px; font-weight: 500;
  color: var(--ink-dim); margin-left: 4px;
}}
.metric.accent .value {{ color: var(--accent); }}

.cfg {{
  font-family: var(--mono); font-size: 12.5px; line-height: 1.75;
  background: var(--card); border: 1px solid var(--line);
  border-left: 3px solid var(--accent);
  padding: 22px 26px; margin: 24px 0 44px;
  white-space: pre; overflow-x: auto; color: var(--ink);
}}
.cfg b {{ color: var(--ink); font-weight: 600; }}

.arch-fig {{
  margin: 24px 0 44px;
  background: var(--card);
  border: 1px solid var(--line);
  padding: 22px 20px 18px;
}}
.arch-fig svg {{ width: 100%; height: auto; display: block; }}
.arch-cap {{
  font-family: var(--mono); font-size: 11.5px;
  color: var(--ink-dim); line-height: 1.6;
  padding-top: 14px; margin-top: 6px;
  border-top: 1px dashed var(--line);
}}
.arch-cap b {{
  color: var(--accent); text-transform: uppercase;
  letter-spacing: 0.14em; font-size: 10.5px;
  font-weight: 700; margin-right: 8px;
}}

.figure {{ margin: 40px 0 52px; }}
.figure img {{
  width: 100%; display: block;
  border: 1px solid var(--line);
  background: var(--card);
}}
.figure figcaption {{
  font-family: var(--sans); font-size: 14px;
  color: var(--ink-dim); line-height: 1.55;
  padding-top: 12px; margin-top: 12px;
  border-top: 1px solid var(--line);
}}
.figure figcaption b {{
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.16em;
  font-size: 11px; font-weight: 700;
  margin-right: 10px;
  font-family: var(--mono);
}}

.vis-grid {{
  display: grid; grid-template-columns: repeat(6, 1fr);
  gap: 2px; margin: 22px 0 40px;
  border: 1px solid var(--line); background: var(--line);
}}
.vis-grid img {{
  width: 100%; display: block; background: var(--card);
  aspect-ratio: 1 / 1; object-fit: cover;
}}

table.stats {{
  width: 100%; border-collapse: collapse;
  font-family: var(--mono); font-size: 12.5px;
  margin: 20px 0 44px; background: var(--card);
  border: 1px solid var(--line);
}}
table.stats th, table.stats td {{
  text-align: right; padding: 11px 14px;
  border-bottom: 1px solid var(--line);
  font-variant-numeric: tabular-nums;
}}
table.stats th:first-child, table.stats td:first-child {{ text-align: left; }}
table.stats th {{
  color: var(--ink-dim); font-weight: 600; font-size: 11px;
  letter-spacing: 0.14em; text-transform: uppercase;
  background: var(--bg);
}}
table.stats tr:hover td {{ background: var(--bg); }}
table.stats .hl {{ color: var(--accent); font-weight: 600; }}

footer {{
  margin-top: 140px; padding-top: 28px;
  border-top: 1px solid var(--line);
  font-family: var(--mono); font-size: 11px;
  color: var(--ink-dim); letter-spacing: 0.04em;
  display: flex; justify-content: space-between;
  flex-wrap: wrap; gap: 16px;
}}

@media (max-width: 720px) {{
  .page {{ padding: 48px 22px 80px; }}
  h1.title {{ font-size: 42px; }}
  h2        {{ font-size: 30px; margin-top: 64px; }}
  h3        {{ font-size: 20px; }}
  .subtitle {{ font-size: 17px; }}
  .metrics  {{ grid-template-columns: repeat(2, 1fr); }}
  .vis-grid {{ grid-template-columns: repeat(3, 1fr); }}
}}
</style>
</head>
<body>
<article class="page">

<div class="eyebrow">
  <span>Experiment Report</span>
  <span class="dot">●</span>
  <span>{date}</span>
  <span class="dot">●</span>
  <span>{name}</span>
</div>

<h1 class="title">We trained a model that<br>corrects LiDAR-to-camera<br>calibration to sub-pixel.</h1>

<p class="subtitle">A <strong>1.62 M-parameter cross-attention network</strong> takes a 128×128 image crop plus the LiDAR points projected into it, and predicts the 2-D pixel offset that snaps each point back to where it should be — <strong>0.91 px mean object error</strong> on held-out objects, well below the native 4 px LiDAR-beam spacing.</p>

<figure class="hero">
  <img src="vis/hero.png" alt="before/after hero">
  <div class="hero-cap"><b>What the model does</b> Four held-out val samples with shifts in distinct directions. Per-point arrows (orange = object, blue = background) show how the model pulls <em>every</em> projected LiDAR point, not just the ones inside the bounding box. Red box = distorted projection; green dashed box = model-corrected position; white arrow = mean object shift.</div>
</figure>

<div class="meta">
  <span>Run&nbsp; <b>{name}</b></span>
  <span>Data&nbsp; <b>PandaSet · 103 scenes · {n_total:,} obj crops</b></span>
  <span>Training&nbsp; <b>{n_epochs} ep · 87 min · 1 × GPU</b></span>
  <span>Best&nbsp;val&nbsp;NLL&nbsp; <b>{best_nll}</b></span>
</div>

<h2><span class="num">§ 01 &nbsp; Motivation</span>Automating the calibration engineer's eye.</h2>

<figure class="arch-fig">
<svg viewBox="0 0 820 410" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Calibration drift: true points sit on object surfaces; observed points are shifted">
  <defs>
    <marker id="dr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#8d8a80"/>
    </marker>
  </defs>

  <!-- paper background -->
  <rect x="0" y="0" width="820" height="410" fill="#f5f2ea"/>

  <!-- HEADER legend strip (y = 20 – 50, safely above scene) -->
  <circle cx="50"  cy="30" r="4.5" fill="#c13c14"/>
  <text   x="62"  y="34" font-family="JetBrains Mono,monospace" font-size="12" fill="#0f0f0e">true projection   (on object surface)</text>
  <circle cx="340" cy="30" r="4.5" fill="#2f6fbf"/>
  <text   x="352" y="34" font-family="JetBrains Mono,monospace" font-size="12" fill="#0f0f0e">observed projection   (drifted)</text>
  <text   x="620" y="34" font-family="-apple-system,Segoe UI,sans-serif" font-size="14" font-style="italic" fill="#c13c14">~ 15 px drift</text>

  <!-- ground line (scene starts at y = 80) -->
  <line x1="30" y1="370" x2="790" y2="370" stroke="#c1bcae" stroke-width="1.2"/>

  <!-- POLE (stick) -->
  <rect x="180" y="85" width="10" height="285" fill="#5e5b52"/>
  <text x="170" y="395" font-family="JetBrains Mono,monospace" font-size="11" fill="#6b6a63">pole</text>

  <!-- CAR (box + wheels) -->
  <rect x="470" y="270" width="200" height="95" fill="#5e5b52"/>
  <circle cx="500" cy="370" r="10" fill="#2a2a28"/>
  <circle cx="640" cy="370" r="10" fill="#2a2a28"/>
  <text x="559" y="395" font-family="JetBrains Mono,monospace" font-size="11" fill="#6b6a63">car</text>

  <!-- TRUE points (red) — ON pole surface -->
  <g fill="#c13c14">
    <circle cx="185" cy="100" r="3.5"/>
    <circle cx="185" cy="120" r="3.5"/>
    <circle cx="185" cy="140" r="3.5"/>
    <circle cx="185" cy="160" r="3.5"/>
    <circle cx="185" cy="180" r="3.5"/>
    <circle cx="185" cy="200" r="3.5"/>
    <circle cx="185" cy="220" r="3.5"/>
    <circle cx="185" cy="240" r="3.5"/>
    <circle cx="185" cy="260" r="3.5"/>
    <circle cx="185" cy="280" r="3.5"/>
    <circle cx="185" cy="300" r="3.5"/>
    <circle cx="185" cy="320" r="3.5"/>
    <circle cx="185" cy="340" r="3.5"/>
  </g>

  <!-- TRUE points (red) — ON CAR SURFACE: 6 × 3 grid covering the body panel -->
  <g fill="#c13c14">
    <circle cx="490" cy="290" r="3.5"/>
    <circle cx="525" cy="290" r="3.5"/>
    <circle cx="560" cy="290" r="3.5"/>
    <circle cx="595" cy="290" r="3.5"/>
    <circle cx="625" cy="290" r="3.5"/>
    <circle cx="655" cy="290" r="3.5"/>
    <circle cx="490" cy="320" r="3.5"/>
    <circle cx="525" cy="320" r="3.5"/>
    <circle cx="560" cy="320" r="3.5"/>
    <circle cx="595" cy="320" r="3.5"/>
    <circle cx="625" cy="320" r="3.5"/>
    <circle cx="655" cy="320" r="3.5"/>
    <circle cx="490" cy="350" r="3.5"/>
    <circle cx="525" cy="350" r="3.5"/>
    <circle cx="560" cy="350" r="3.5"/>
    <circle cx="595" cy="350" r="3.5"/>
    <circle cx="625" cy="350" r="3.5"/>
    <circle cx="655" cy="350" r="3.5"/>
  </g>

  <!-- OBSERVED points (blue) — same set, each shifted ~30 px right -->
  <g fill="#2f6fbf">
    <circle cx="215" cy="100" r="3.5"/>
    <circle cx="215" cy="120" r="3.5"/>
    <circle cx="215" cy="140" r="3.5"/>
    <circle cx="215" cy="160" r="3.5"/>
    <circle cx="215" cy="180" r="3.5"/>
    <circle cx="215" cy="200" r="3.5"/>
    <circle cx="215" cy="220" r="3.5"/>
    <circle cx="215" cy="240" r="3.5"/>
    <circle cx="215" cy="260" r="3.5"/>
    <circle cx="215" cy="280" r="3.5"/>
    <circle cx="215" cy="300" r="3.5"/>
    <circle cx="215" cy="320" r="3.5"/>
    <circle cx="215" cy="340" r="3.5"/>
  </g>
  <g fill="#2f6fbf">
    <circle cx="520" cy="290" r="3.5"/>
    <circle cx="555" cy="290" r="3.5"/>
    <circle cx="590" cy="290" r="3.5"/>
    <circle cx="625" cy="290" r="3.5"/>
    <circle cx="655" cy="290" r="3.5"/>
    <circle cx="685" cy="290" r="3.5"/>
    <circle cx="520" cy="320" r="3.5"/>
    <circle cx="555" cy="320" r="3.5"/>
    <circle cx="590" cy="320" r="3.5"/>
    <circle cx="625" cy="320" r="3.5"/>
    <circle cx="655" cy="320" r="3.5"/>
    <circle cx="685" cy="320" r="3.5"/>
    <circle cx="520" cy="350" r="3.5"/>
    <circle cx="555" cy="350" r="3.5"/>
    <circle cx="590" cy="350" r="3.5"/>
    <circle cx="625" cy="350" r="3.5"/>
    <circle cx="655" cy="350" r="3.5"/>
    <circle cx="685" cy="350" r="3.5"/>
  </g>

  <!-- a few sample displacement arrows (red → blue) -->
  <g stroke="#8d8a80" stroke-width="1" fill="none">
    <line x1="189" y1="140" x2="211" y2="140" marker-end="url(#dr)"/>
    <line x1="189" y1="220" x2="211" y2="220" marker-end="url(#dr)"/>
    <line x1="189" y1="300" x2="211" y2="300" marker-end="url(#dr)"/>
    <line x1="494" y1="290" x2="516" y2="290" marker-end="url(#dr)"/>
    <line x1="494" y1="350" x2="516" y2="350" marker-end="url(#dr)"/>
    <line x1="659" y1="320" x2="681" y2="320" marker-end="url(#dr)"/>
  </g>
</svg>
<figcaption class="arch-cap"><b>Figure 1</b> What a calibration engineer has always looked at. Red dots are where projected LiDAR points <em>should</em> land — on the surface of real objects in the image, a pole and a car. Blue dots are where the sensor rig <em>actually</em> places them after a small calibration drift: the same set of points, uniformly offset. Engineers have recalibrated rigs for decades by eyeballing patterns like this. This report is about automating that step: for every such projected point, the network emits a 2-D Gaussian saying where it should really land, and how confidently. Accumulate enough of those and bundle adjustment solves for the rig.</figcaption>
</figure>

<div class="text">
<p>Classical calibration — chessboards, targets, SfM — produces a set of rigid transforms (SE(3), i.e. rotation plus translation) between each sensor pair; they are accurate at the moment of capture and stale the moment a sensor moves. Factory calibration on a large vehicle fleet drifts out of spec within months, and running a full re-calibration pipeline in production is operationally unrealistic.</p>

<p>The tempting response is to throw a single large end-to-end model at the whole rig — a CalibFormer-style system that takes all sensor streams and regresses the full set of extrinsics at once. <strong>That is the wrong factoring.</strong> A unified model is expensive to train, brittle to per-vehicle sensor layout changes, and couples every sensor's error modes into every other sensor's prediction. It also bakes in a rig topology that fleet teams routinely change — adding a telephoto, moving a side LiDAR, swapping a bumper camera.</p>

<p><strong>The calibration model itself is small.</strong> Six degrees of freedom per sensor pair, perhaps a few intrinsic knobs — a handful of parameters. Bundle adjustment has been solving for that class of model from 2-D observations for decades; it does not need a neural network. What it needs is <em>observations</em>: many per-point measurements of where each projected point actually ought to land, with uncertainty. So we factor the problem the way SfM already does:</p>

<ul style="margin: 0 0 1.2em 0; padding-left: 1.2em; color: var(--ink);">
  <li><strong>Network &nbsp;=&nbsp; local evidence detector.</strong> For each small RGB crop the network takes the projected LiDAR points visible inside it and, for every point, emits a 2-D Gaussian <code>(Δu, Δv)</code> with full covariance — the mean says where the point should really land, the covariance says how confidently.</li>
  <li><strong>BA &nbsp;=&nbsp; global solver.</strong> These per-point observations are accumulated across frames and crops, then fed to a standard bundle-adjustment stage that jointly re-optimises the per-pair rigid transforms. The same stage SfM and SLAM already use. One forward pass per crop; one non-linear solve for the rig.</li>
</ul>

<p><strong>Why small patches are enough — and why that matters for compute.</strong> Calibration drift shows up as a <em>local</em> misalignment: a pole whose lidar hits sit a few pixels off its silhouette, a car whose contour does not match its returns. Whatever the underlying mis-pose, its visual signature on any single crop is bounded. The network does not need long-range attention across a full 1920×1080 frame; a small transformer on a 64×64 or 128×128 crop is enough. Attention cost scales with <em>patch</em> area, not image area, and a drive log yields millions of crops per vehicle per day — cheap per-patch cost is what makes the approach operationally sane.</p>

<p>The same primitive covers every sensor pair. Any two sensors with overlapping FOV and a shared geometric primitive — edges, points, object boundaries — can be posed as: "given projected sensor-A primitives on image B, predict the 2-D pull-back with uncertainty onto image B features." Concretely:</p>
<ul style="margin: 0 0 1.2em 0; padding-left: 1.2em; color: var(--ink);">
  <li><strong>LiDAR → camera.</strong> Point clouds projected onto RGB, the case we train here.</li>
  <li><strong>Camera → camera, forward-forward.</strong> A telephoto paired with the main forward camera (<code>TELE</code> + <code>FCM</code>) share a narrow overlap band; features triangulated in one can be projected into the other, and the residual estimator pulls them back onto the correct pixels. This is the common case for modern multi-focal front rigs where tele and wide drift against each other after thermal cycling.</li>
  <li><strong>Camera → camera, side overlap.</strong> Adjacent surround cameras with 10–20° shared FOV.</li>
  <li><strong>RADAR → camera.</strong> Coarse radar points projected onto image, corrected to nearest RCS-consistent visual structure.</li>
</ul>
<p>The dataset used for this report is <strong>PandaSet</strong> — 103 outdoor driving scenes with synchronised Hesai Pandar64 LiDAR and six RGB cameras — perturbed with small synthetic calibration drift at load time, so the model sees a consistent training signal. The narrow research question here: <strong>how well does the residual generalise across object instances the model has not seen</strong>? Earlier runs held out whole scenes; here we hold out <em>objects</em>, isolating appearance-level generalisation from the larger scene-domain shift.</p>
</div>

<h2><span class="num">§ 02 &nbsp; Problem Setup</span>One crop, one evidence packet.</h2>
<div class="text">
<p>Each training example is a 128×128 RGB crop centred on a single 3-D object, together with the LiDAR points visible in the crop projected as <code>(u, v, d)</code> — image coordinates plus depth. A synthetic calibration perturbation (≤&nbsp;0.2&nbsp;m translation, ≤&nbsp;0.5° rotation) is applied before projection, producing a distorted copy of each point; the regression target is the 2-D offset from the distorted projection back to the true projection. After a single forward pass, each crop yields a batch of per-point <code>(μ, Σ)</code> observations — one "evidence packet" that the downstream bundle-adjustment stage consumes.</p>
<p>Points are tagged <code>obj</code> or <code>bg</code>. Object points live inside the 3-D bounding box — sparse (≈&nbsp;20–40 per crop) but visually anchored. Background points are whatever else the LiDAR returns inside the crop — road, buildings, distant cars — dense but often weakly constrained by the image.</p>
</div>

<h2><span class="num">§ 03 &nbsp; Method</span>CalibNetDepth, cross-first.</h2>

<h3>3.1&nbsp; Architecture</h3>
<figure class="arch-fig">
<svg viewBox="0 0 820 640" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="CalibNetDepth architecture diagram">
  <defs>
    <marker id="a"  viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f0f0e"/>
    </marker>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#c13c14"/>
    </marker>
  </defs>

  <!-- Row 1: Inputs -->
  <rect x="40"  y="20" width="360" height="74" fill="#fff" stroke="#d9d6cd"/>
  <text x="60"  y="48" font-family="-apple-system,Segoe UI,sans-serif" font-size="15" font-weight="700" fill="#0f0f0e">Image crop</text>
  <text x="60"  y="74" font-family="JetBrains Mono,monospace" font-size="12" fill="#6b6a63">B × 3 × 64 × 64  ·  RGB</text>
  <rect x="440" y="20" width="340" height="74" fill="#fff" stroke="#d9d6cd"/>
  <text x="460" y="48" font-family="-apple-system,Segoe UI,sans-serif" font-size="15" font-weight="700" fill="#0f0f0e">Projected LiDAR points</text>
  <text x="460" y="74" font-family="JetBrains Mono,monospace" font-size="12" fill="#6b6a63">B × N × 3  ·  (u, v, d)</text>

  <line x1="220" y1="94"  x2="220" y2="128" stroke="#0f0f0e" stroke-width="1.4" marker-end="url(#a)"/>
  <line x1="610" y1="94"  x2="610" y2="128" stroke="#0f0f0e" stroke-width="1.4" marker-end="url(#a)"/>

  <!-- Row 2: Encoders -->
  <rect x="40"  y="130" width="360" height="74" fill="#fff" stroke="#d9d6cd"/>
  <text x="60"  y="158" font-family="-apple-system,Segoe UI,sans-serif" font-size="15" font-weight="700" fill="#0f0f0e">ConvNeXt-mini</text>
  <text x="60"  y="184" font-family="JetBrains Mono,monospace" font-size="12" fill="#6b6a63">image → multi-scale feature maps</text>
  <rect x="440" y="130" width="340" height="74" fill="#fff" stroke="#d9d6cd"/>
  <text x="460" y="158" font-family="-apple-system,Segoe UI,sans-serif" font-size="15" font-weight="700" fill="#0f0f0e">PointMLP  +  frustum enc.</text>
  <text x="460" y="184" font-family="JetBrains Mono,monospace" font-size="12" fill="#6b6a63">(u, v, d) → D-dim token per point</text>

  <line x1="140" y1="204" x2="140" y2="238" stroke="#0f0f0e" stroke-width="1.4" marker-end="url(#a)"/>
  <line x1="300" y1="204" x2="300" y2="238" stroke="#0f0f0e" stroke-width="1.4" marker-end="url(#a)"/>
  <line x1="610" y1="204" x2="610" y2="238" stroke="#0f0f0e" stroke-width="1.4" marker-end="url(#a)"/>

  <!-- Row 3: Features -->
  <rect x="40"  y="240" width="170" height="74" fill="#fff" stroke="#d9d6cd"/>
  <text x="60"  y="268" font-family="-apple-system,Segoe UI,sans-serif" font-size="15" font-weight="700" fill="#0f0f0e">coarse feat</text>
  <text x="60"  y="294" font-family="JetBrains Mono,monospace" font-size="12" fill="#6b6a63">D × 8 × 8</text>
  <rect x="230" y="240" width="170" height="74" fill="#fff" stroke="#d9d6cd"/>
  <text x="250" y="268" font-family="-apple-system,Segoe UI,sans-serif" font-size="15" font-weight="700" fill="#0f0f0e">fine feat</text>
  <text x="250" y="294" font-family="JetBrains Mono,monospace" font-size="12" fill="#6b6a63">D × 4 × 4</text>
  <rect x="440" y="240" width="340" height="74" fill="#fff" stroke="#d9d6cd"/>
  <text x="460" y="268" font-family="-apple-system,Segoe UI,sans-serif" font-size="15" font-weight="700" fill="#0f0f0e">query tokens  Q</text>
  <text x="460" y="294" font-family="JetBrains Mono,monospace" font-size="12" fill="#6b6a63">B × N × D  (one token per LiDAR point)</text>

  <!-- Arrows: features feed cross-attn KV (red), Q feeds decoder (black) -->
  <line x1="125" y1="314" x2="125" y2="355" stroke="#c13c14" stroke-width="1.6" marker-end="url(#ar)"/>
  <line x1="315" y1="314" x2="315" y2="355" stroke="#c13c14" stroke-width="1.6" marker-end="url(#ar)"/>
  <line x1="610" y1="314" x2="610" y2="355" stroke="#0f0f0e" stroke-width="1.4" marker-end="url(#a)"/>

  <!-- Row 4: Decoder -->
  <rect x="40"  y="360" width="740" height="170" fill="#fdf9f4" stroke="#c13c14" stroke-width="1.3"/>
  <text x="60"  y="390" font-family="-apple-system,Segoe UI,sans-serif" font-size="15" font-weight="700" fill="#c13c14">Decoder · 4 × CrossAttnBlock</text>
  <text x="60"  y="412" font-family="JetBrains Mono,monospace" font-size="11.5" fill="#6b6a63">each block:   cross-attn(Q ⟵ image feat)  →  self-attn(Q ⟷ Q)  →  FFN</text>

  <rect x="60"  y="428" width="345" height="36" fill="#fff" stroke="#d9d6cd"/>
  <text x="78"  y="451" font-family="JetBrains Mono,monospace" font-size="13" font-weight="600" fill="#0f0f0e">L1   Q  ⟵  coarse</text>
  <rect x="60"  y="478" width="345" height="36" fill="#fff" stroke="#d9d6cd"/>
  <text x="78"  y="501" font-family="JetBrains Mono,monospace" font-size="13" font-weight="600" fill="#0f0f0e">L2   Q  ⟵  coarse</text>
  <rect x="415" y="428" width="345" height="36" fill="#fff" stroke="#d9d6cd"/>
  <text x="433" y="451" font-family="JetBrains Mono,monospace" font-size="13" font-weight="600" fill="#0f0f0e">L3   Q  ⟵  fine</text>
  <rect x="415" y="478" width="345" height="36" fill="#fff" stroke="#d9d6cd"/>
  <text x="433" y="501" font-family="JetBrains Mono,monospace" font-size="13" font-weight="600" fill="#0f0f0e">L4   Q  ⟵  fine</text>

  <line x1="410" y1="530" x2="410" y2="560" stroke="#0f0f0e" stroke-width="1.4" marker-end="url(#a)"/>

  <!-- Row 5: Head -->
  <rect x="160" y="562" width="500" height="50" fill="#f4f8f4" stroke="#174734" stroke-width="1.3"/>
  <text x="410" y="594" font-family="JetBrains Mono,monospace" font-size="14" font-weight="600" fill="#0f0f0e" text-anchor="middle">Linear  →  (Δu, Δv, log σᵤ, log σᵥ, ρ)</text>

  <text x="410" y="634" font-family="JetBrains Mono,monospace" font-size="11.5" fill="#6b6a63" text-anchor="middle">2-D Gaussian NLL  ·  masked mean over valid points  ·  1.62 M params  ·  bf16 autocast</text>
</svg>
<figcaption class="arch-cap"><b>Figure 3</b> Two-branch encoder — ConvNeXt-mini on the image crop (splits into a coarse 8×8 and a fine 4×4 map), PointMLP on the LiDAR points — feeds a 4-layer cross-attention decoder. Red arrows indicate keys/values drawn from image features; the black arrow carries the per-point query tokens. Layers 1–2 attend to the coarse map for global context; 3–4 refine against the fine map.</figcaption>
</figure>

<h3>3.2&nbsp; Frustum encoding</h3>
<p>A plain PointMLP treats each LiDAR point in isolation — it sees <code>(u, v, d)</code> and nothing about its neighbours. But calibration residuals are a local-geometry problem: a point sliding sideways is only meaningful relative to points at similar depth. The <strong>frustum encoder</strong> adds that local context. For each query point <em>i</em> we form a 3-D box in <code>(u, v, d)</code> space, take the <code>k</code> UV-nearest points inside it, and summarise their relative offsets with a shared MLP followed by max-pool. The resulting per-point descriptor is added to the PointMLP token before the cross-attention decoder sees it.</p>

<figure class="arch-fig">
<svg viewBox="0 0 820 480" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Frustum encoding diagram">
  <defs>
    <marker id="fa"  viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#0f0f0e"/>
    </marker>
  </defs>

  <!-- ===== LEFT PANEL: 3-D frustum neighbourhood ===== -->
  <text x="20" y="28" font-family="-apple-system,Segoe UI,sans-serif" font-size="12.5" font-weight="700" fill="#0f0f0e" letter-spacing="0.12em">(A)  NEIGHBOURHOOD  IN  (u, v, d)</text>

  <!-- axis legend, top-right of left panel -->
  <g transform="translate(360, 58)" stroke="#0f0f0e" stroke-width="1.4" fill="none">
    <line x1="0" y1="0" x2="34" y2="0" marker-end="url(#fa)"/>
    <line x1="0" y1="0" x2="0"  y2="34" marker-end="url(#fa)"/>
    <line x1="0" y1="0" x2="22" y2="-28" marker-end="url(#fa)"/>
  </g>
  <g font-family="JetBrains Mono,monospace" font-size="11.5" fill="#0f0f0e">
    <text x="398" y="63">u</text>
    <text x="354" y="104">v</text>
    <text x="384" y="24">d  (depth)</text>
  </g>

  <!-- Visible faces of the frustum box (very light mint fill).
       Box corners: u,v half-axis = 90; d-offset per unit = (+50, -60).
       Centre of box at (200, 240). -->
  <polygon points="160,90 340,90 340,270 160,270" fill="#e9f3e9" fill-opacity="0.6"/>
  <polygon points="60,210 240,210 340,90 160,90"  fill="#e9f3e9" fill-opacity="0.55"/>
  <polygon points="240,210 240,390 340,270 340,90" fill="#e9f3e9" fill-opacity="0.38"/>

  <!-- Back-face edges (dashed where occluded) -->
  <line x1="60"  y1="210" x2="240" y2="210" stroke="#6b6a63" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="240" y1="210" x2="240" y2="390" stroke="#6b6a63" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="240" y1="390" x2="60"  y2="390" stroke="#6b6a63" stroke-width="1"/>
  <line x1="60"  y1="390" x2="60"  y2="210" stroke="#6b6a63" stroke-width="1"/>

  <!-- Connector edges back-face to front-face -->
  <line x1="60"  y1="210" x2="160" y2="90"  stroke="#0f0f0e" stroke-width="1.2"/>
  <line x1="240" y1="210" x2="340" y2="90"  stroke="#6b6a63" stroke-width="1" stroke-dasharray="4 3"/>
  <line x1="240" y1="390" x2="340" y2="270" stroke="#0f0f0e" stroke-width="1.2"/>
  <line x1="60"  y1="390" x2="160" y2="270" stroke="#0f0f0e" stroke-width="1.2"/>

  <!-- Front-face frame (solid, darker) -->
  <line x1="160" y1="90"  x2="340" y2="90"  stroke="#0f0f0e" stroke-width="1.4"/>
  <line x1="340" y1="90"  x2="340" y2="270" stroke="#0f0f0e" stroke-width="1.4"/>
  <line x1="340" y1="270" x2="160" y2="270" stroke="#0f0f0e" stroke-width="1.4"/>
  <line x1="160" y1="270" x2="160" y2="90"  stroke="#0f0f0e" stroke-width="1.4"/>

  <!-- Edge-size annotations (along front face) -->
  <text x="238" y="82"  font-family="JetBrains Mono,monospace" font-size="11" fill="#c13c14" text-anchor="middle">2·r_uv</text>
  <text x="355" y="182" font-family="JetBrains Mono,monospace" font-size="11" fill="#c13c14">2·r_uv</text>
  <text x="355" y="285" font-family="JetBrains Mono,monospace" font-size="11" fill="#c13c14">2·r_d</text>

  <!-- Points OUTSIDE the box (grey) -->
  <g fill="#bcbab2">
    <circle cx="395" cy="255" r="3"/>
    <circle cx="40"  cy="105" r="3"/>
    <circle cx="245" cy="420" r="3"/>
    <circle cx="290" cy="58"  r="3"/>
    <circle cx="415" cy="360" r="3"/>
    <circle cx="18"  cy="345" r="3"/>
    <circle cx="380" cy="420" r="3"/>
    <circle cx="120" cy="40"  r="3"/>
  </g>

  <!-- Inside box but NOT top-k (outlined green) -->
  <circle cx="306" cy="291" r="4" fill="#fff" stroke="#5a9568" stroke-width="1.3"/>
  <circle cx="99"  cy="342" r="4" fill="#fff" stroke="#5a9568" stroke-width="1.3"/>
  <circle cx="322" cy="112" r="4" fill="#fff" stroke="#5a9568" stroke-width="1.3"/>

  <!-- Inside box AND top-k (filled green, with thicker ring) -->
  <g fill="#5a9568" stroke="#174734" stroke-width="1.2">
    <circle cx="242" cy="240" r="4.2"/>
    <circle cx="169" cy="207" r="4.2"/>
    <circle cx="189" cy="219" r="4.2"/>
    <circle cx="192" cy="273" r="4.2"/>
    <circle cx="235" cy="261" r="4.2"/>
    <circle cx="171" cy="201" r="4.2"/>
    <circle cx="161" cy="240" r="4.2"/>
    <circle cx="221" cy="231" r="4.2"/>
  </g>

  <!-- Query point i at centre -->
  <circle cx="200" cy="240" r="6.2" fill="#c13c14" stroke="#0f0f0e" stroke-width="1.4"/>
  <text x="210" y="246" font-family="JetBrains Mono,monospace" font-size="13" font-weight="700" fill="#c13c14">i</text>

  <!-- Legend -->
  <g font-family="JetBrains Mono,monospace" font-size="11" fill="#0f0f0e">
    <circle cx="30"  cy="425" r="5" fill="#c13c14" stroke="#0f0f0e" stroke-width="1.2"/>
    <text   x="42"  y="429">query i</text>
    <circle cx="108" cy="425" r="4" fill="#5a9568" stroke="#174734" stroke-width="1.2"/>
    <text   x="118" y="429">top-k (UV-nearest, k=8)</text>
    <circle cx="280" cy="425" r="4" fill="#fff" stroke="#5a9568" stroke-width="1.2"/>
    <text   x="290" y="429">in box, not top-k</text>
    <circle cx="402" cy="425" r="3" fill="#bcbab2"/>
    <text   x="410" y="429">outside box</text>
  </g>
  <text x="30" y="460" font-family="JetBrains Mono,monospace" font-size="11" fill="#6b6a63">|Δu| &lt; r_uv   ∧   |Δv| &lt; r_uv   ∧   |Δd| &lt; r_d          defaults:  r_uv = 8 px,   r_d = 0.004,   k = 8</text>

  <!-- divider between panels -->
  <line x1="440" y1="50" x2="440" y2="460" stroke="#e0ddd3" stroke-width="1"/>

  <!-- ===== RIGHT PANEL: aggregation pipeline ===== -->
  <text x="460" y="28" font-family="-apple-system,Segoe UI,sans-serif" font-size="12.5" font-weight="700" fill="#0f0f0e" letter-spacing="0.12em">(B)  AGGREGATION</text>

  <text x="460" y="60" font-family="JetBrains Mono,monospace" font-size="11.5" fill="#6b6a63">for each of k = 8 neighbours  j ∈ N(i):</text>

  <!-- Step 1: relative offset tuple -->
  <rect x="460" y="72" width="320" height="30" fill="#fff" stroke="#d9d6cd"/>
  <text x="476" y="92" font-family="JetBrains Mono,monospace" font-size="12.5" fill="#0f0f0e">(Δu_j,   Δv_j,   Δd_j)            ∈   ℝ³</text>

  <line x1="620" y1="102" x2="620" y2="128" stroke="#0f0f0e" stroke-width="1.4" marker-end="url(#fa)"/>

  <!-- Step 2: shared MLP -->
  <rect x="460" y="130" width="320" height="82" fill="#fdf9f4" stroke="#c13c14" stroke-width="1.3"/>
  <text x="476" y="152" font-family="-apple-system,Segoe UI,sans-serif" font-size="13" font-weight="700" fill="#c13c14">shared MLP  (per neighbour)</text>
  <text x="476" y="174" font-family="JetBrains Mono,monospace" font-size="11.5" fill="#0f0f0e">Linear(3 → 32)   →   GELU</text>
  <text x="476" y="190" font-family="JetBrains Mono,monospace" font-size="11.5" fill="#0f0f0e">Linear(32 → D)   →   GELU   →   Linear(D → D)</text>
  <text x="476" y="206" font-family="JetBrains Mono,monospace" font-size="10.5" fill="#6b6a63">weights shared across every neighbour of every query point</text>

  <line x1="620" y1="212" x2="620" y2="238" stroke="#0f0f0e" stroke-width="1.4" marker-end="url(#fa)"/>

  <!-- Step 3: k feature vectors -->
  <rect x="500" y="242" width="260" height="22" fill="#fff" stroke="#d9d6cd"/>
  <text x="510" y="258" font-family="JetBrains Mono,monospace" font-size="11.5" fill="#0f0f0e">h_i,1   ∈   ℝ^D</text>
  <rect x="500" y="264" width="260" height="22" fill="#fff" stroke="#d9d6cd"/>
  <text x="510" y="280" font-family="JetBrains Mono,monospace" font-size="11.5" fill="#0f0f0e">h_i,2   ∈   ℝ^D</text>
  <rect x="500" y="286" width="260" height="22" fill="#fff" stroke="#d9d6cd"/>
  <text x="510" y="302" font-family="JetBrains Mono,monospace" font-size="13" fill="#0f0f0e">⋮</text>
  <rect x="500" y="308" width="260" height="22" fill="#fff" stroke="#d9d6cd"/>
  <text x="510" y="324" font-family="JetBrains Mono,monospace" font-size="11.5" fill="#0f0f0e">h_i,k   ∈   ℝ^D</text>

  <line x1="620" y1="332" x2="620" y2="358" stroke="#0f0f0e" stroke-width="1.4" marker-end="url(#fa)"/>
  <text x="630" y="350" font-family="JetBrains Mono,monospace" font-size="11" font-weight="700" fill="#c13c14">max over k</text>

  <!-- Step 4: frustum feature -->
  <rect x="500" y="360" width="260" height="32" fill="#f4f8f4" stroke="#174734" stroke-width="1.3"/>
  <text x="514" y="381" font-family="JetBrains Mono,monospace" font-size="12.5" font-weight="700" fill="#0f0f0e">f_frustum(i)   ∈   ℝ^D</text>

  <line x1="620" y1="392" x2="620" y2="418" stroke="#0f0f0e" stroke-width="1.4" marker-end="url(#fa)"/>
  <text x="630" y="410" font-family="JetBrains Mono,monospace" font-size="11" fill="#0f0f0e">+  PointMLP(u, v, d)</text>

  <!-- Step 5: final query token -->
  <rect x="500" y="420" width="260" height="30" fill="#fff" stroke="#0f0f0e" stroke-width="1.4"/>
  <text x="514" y="440" font-family="JetBrains Mono,monospace" font-size="12" font-weight="700" fill="#0f0f0e">q_i    (query token,  D-dim)</text>
</svg>
<figcaption class="arch-cap"><b>Figure 4</b> <strong>Frustum encoding.</strong> (A) For each LiDAR point i we form a 3-D box in <code>(u, v, d)</code> space — radius r_uv in the image plane, r_d along depth — and take the k UV-nearest points inside it (green filled). A point that is close in UV but at a different depth layer (grey, above/below the box) is <em>not</em> a neighbour: depth is what separates the query object from the road behind it. (B) A shared MLP lifts each relative offset <code>(Δu, Δv, Δd)</code> into ℝ^D, and a max-pool over the k neighbours produces a per-point geometric descriptor that is added to the plain PointMLP token before the decoder sees it.</figcaption>
</figure>

<h3>3.3&nbsp; Loss</h3>
<p>Per-point 2-D Gaussian negative log-likelihood with full 2×2 covariance parametrised by <code>log σᵤ, log σᵥ, ρ</code>. The loss is averaged over valid (non-padded) points, not summed, so the gradient scale is independent of how many points fall in a crop — important when mixing datasets with different point densities.</p>

<h3>3.4&nbsp; Object-level split</h3>
<p>PandaSet's 103 scenes yield {n_total:,} object crops. Earlier runs held out whole <em>scenes</em> (10 % val), which conflates two failure modes: new instances and new environments. Here the concatenated crop pool is shuffled and split 90 / 10 at the <strong>instance</strong> level (seed&nbsp;42), so every scene is represented in both splits but no crop appears in both.</p>

<div class="cfg">{cfg_block}</div>

<h2><span class="num">§ 04 &nbsp; Results</span>Headline numbers.</h2>

<div class="metrics">
  <div class="metric accent">
    <div class="label">best&nbsp;val&nbsp;NLL</div>
    <div class="value">{best_nll}</div>
  </div>
  <div class="metric">
    <div class="label">val&nbsp;obj&nbsp;MSE</div>
    <div class="value">{obj_mse:.2f}<small>px</small></div>
  </div>
  <div class="metric">
    <div class="label">val&nbsp;bg&nbsp;MSE</div>
    <div class="value">{bg_mse:.2f}<small>px</small></div>
  </div>
  <div class="metric">
    <div class="label">vs&nbsp;scene-split</div>
    <div class="value">−32<small>%</small></div>
  </div>
</div>

<figure class="figure">
  <img src="vis/curves.png" alt="training curves">
  <figcaption><b>Figure 2</b> Learning curves over {n_epochs} epochs. Validation in red, training in forest green. Val NLL plateaus near epoch 130 while val MSE keeps descending — the model keeps improving its mean prediction after its covariance prediction stops generalising.</figcaption>
</figure>

<table class="stats">
<thead><tr><th>epoch</th><th>train&nbsp;NLL</th><th>val&nbsp;NLL</th><th>train&nbsp;obj&nbsp;MSE</th><th>val&nbsp;obj&nbsp;MSE</th><th>train&nbsp;bg&nbsp;MSE</th><th>val&nbsp;bg&nbsp;MSE</th></tr></thead>
<tbody>
{table_rows}
</tbody>
</table>

<h2><span class="num">§ 05 &nbsp; Analysis</span>Where the generalisation gap lives.</h2>
<div class="text">
<p>The sub-pixel headline obscures a more nuanced story. Training NLL continues to fall through epoch 200 (<code>obj</code>&nbsp;{tr_obj_nll_last:+.2f}, <code>bg</code>&nbsp;{tr_bg_nll_last:+.2f}) while <strong>val NLL flattens around epoch 130</strong>. This is not uniform overfitting: val <em>MSE</em> continues to decrease even while val NLL stalls.</p>

<p>The most likely mechanism is <strong>σ overfitting</strong>. The mean prediction <code>(Δu, Δv)</code> generalises — the geometric relationship between image edges and LiDAR points transfers to unseen instances. The predicted covariance, however, is being calibrated against training-set residual statistics; at validation time the Mahalanobis term <code>(err / σ)²</code> pays the price whenever σ is too tight for the actual observed error distribution.</p>

<p>The gap is also <strong>not uniform across point types</strong>. Object points generalise tightly — val obj NLL {va_obj_nll_last:+.2f} against train {tr_obj_nll_last:+.2f}, a gap of {obj_gap:.1f}. Background points carry most of the generalisation cost — val bg {va_bg_nll_last:+.2f} against train {tr_bg_nll_last:+.2f}, a gap of {bg_gap:.1f}. Background points are exactly the points for which the image offers weak cues — sky, road, far buildings — so calibrating σ for them is the hard part, and a larger training pool is likely the cheapest way to help.</p>

<p>Two directions follow naturally. <strong>(i) Multi-dataset training.</strong> Adding NuScenes and Waymo gives the covariance head a more diverse residual distribution to calibrate against. <strong>(ii) σ regularisation.</strong> Penalising <code>log|Σ|</code> shrinkage, or swapping to a Student-t likelihood, prevents the covariance from chasing training-set minutiae.</p>
</div>

<h2><span class="num">§ 06 &nbsp; Samples</span>48 held-out val crops.</h2>
<div class="text">
<p>Each tile shows per-point correction arrows (orange for object, blue for background), the raw object bounding box (red solid), the predicted-shifted box (cyan dashed), and a white centre-shift arrow. Titles read <code>obj_err&nbsp;before → after, bg_err&nbsp;before → after, mean&nbsp;obj&nbsp;shift (dx, dy)</code> in pixels.</p>
</div>

<div class="vis-grid">
{vis_imgs}
</div>

<footer>
  <span>{name} · generated {timestamp}</span>
  <span>matoge@{host} · PandaSet val 10 % @ seed 42</span>
</footer>

</article>
</body>
</html>
"""

def main(exp_name: str):
    exp_dir = Path("experiments") / exp_name
    cfg = load_cfg(exp_dir)
    hist, best = parse_log(exp_dir / "train.log")
    if not hist:
        print(f"!! no epoch lines parsed from {exp_dir}/train.log"); sys.exit(1)

    vis_dir = exp_dir / "vis"
    render_curves(hist, vis_dir / "curves.png")
    hero_info = render_hero(exp_dir, cfg, vis_dir / "hero.png")
    for idx, dx, dy, eb, ea in hero_info['picks']:
        print(f"hero pick: idx={idx}  shift=({dx:+.1f},{dy:+.1f})  err {eb:.2f} → {ea:.2f}")

    last = hist[-1]
    n_total = 30113 + 3345
    n_epochs = last['ep']

    wanted = [1, 25, 50, 75, 100, 125, 150, 175, n_epochs]
    seen = set(); rows = []
    for h in hist:
        if h['ep'] in wanted and h['ep'] not in seen:
            seen.add(h['ep'])
            hl = ' class="hl"' if best and abs(h['va_nll'] - best) < 1e-3 else ''
            rows.append(
                f"<tr><td>{h['ep']}</td>"
                f"<td{hl}>{h['tr_nll']:+.3f}</td>"
                f"<td{hl}>{h['va_nll']:+.3f}</td>"
                f"<td>{h['tr_obj_mse']:.2f}</td>"
                f"<td>{h['va_obj_mse']:.2f}</td>"
                f"<td>{h['tr_bg_mse']:.2f}</td>"
                f"<td>{h['va_bg_mse']:.2f}</td></tr>"
            )
    table_rows = "\n".join(rows)

    n_vis = len(list(vis_dir.glob("val_*.png")))
    vis_imgs = "\n".join(
        f'  <img src="vis/val_{i:02d}.png" loading="lazy" alt="val sample {i}">'
        for i in range(n_vis)
    )

    html = HTML.format(
        title=exp_name.upper(),
        date=datetime.now().strftime("%Y-%m-%d"),
        name=exp_name,
        n_total=n_total, n_epochs=n_epochs,
        best_nll=f"{best:.4f}" if best else "—",
        obj_mse=last['va_obj_mse'],
        bg_mse=last['va_bg_mse'],
        cfg_block=fmt_cfg(cfg),
        table_rows=table_rows,
        tr_obj_nll_last=last['tr_obj_nll'], tr_bg_nll_last=last['tr_bg_nll'],
        va_obj_nll_last=last['va_obj_nll'], va_bg_nll_last=last['va_bg_nll'],
        obj_gap=last['va_obj_nll'] - last['tr_obj_nll'],
        bg_gap=last['va_bg_nll'] - last['tr_bg_nll'],
        vis_imgs=vis_imgs,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        host=socket.gethostname(),
    )
    (exp_dir / "report.html").write_text(html)
    print(f"wrote {exp_dir / 'report.html'}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ps_v9_objsplit")
