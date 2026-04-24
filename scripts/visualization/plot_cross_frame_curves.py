"""Interactive Plotly comparison curves for the cross-frame runs.

Reads train.log from each experiment, extracts (epoch, train err, val err,
loss), and writes a single HTML with multi-trace plots so you can toggle
runs in the legend.

Output: docs/cross_frame_curves.html (Plotly via CDN, ~10KB self-contained
except for the JS lib).
"""
import argparse, re
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots


LOG_RE = re.compile(
    r'ep\s+(\d+)\s+loss=(-?[\d.]+)\s+err_AB=([\d.]+)px\s+err_BA=([\d.]+)px\s+'
    r'\(base=([\d.]+)px\)(?:\s+val_err=([\d.]+)px\s+\(base\s+([\d.]+)\))?'
)


def parse_log(path: Path):
    """Yield dicts of per-epoch metrics."""
    epochs, train, val, base, loss = [], [], [], [], []
    for line in path.read_text().splitlines():
        m = LOG_RE.search(line)
        if not m:
            continue
        ep, ls, eA, eB, b = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        v_str, v_base = m.group(6), m.group(7)
        epochs.append(int(ep))
        loss.append(float(ls))
        train.append(0.5 * (float(eA) + float(eB)))
        base.append(float(b))
        val.append(float(v_str) if v_str else None)
    return dict(epochs=epochs, train=train, val=val, base=base, loss=loss)


# ── runs to plot ─────────────────────────────────────────────────────────────

RUNS = [
    ('v04_multi39',       'v04: std attn, 1 cross-layer (baseline)',          '#888'),
    ('v07_bs64',          'v07: std attn, 1 cross-layer, batch 64',           '#bb8866'),
    ('v09_std_2layer',    'v09: std attn, 2 cross-layers',                     '#446688'),
    ('v08_deform_sl',     'v08: deformable SL, 2 cross-layers',               '#cc7a22'),
    ('v10_padclean_deform', 'v10: deform SL + padded crop (pivot centered) ⭐', '#c13c14'),
]


def main(args):
    root = Path(args.experiments_root)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Validation reproj err (px)', 'Train reproj err (px)',
                         'Train NLL loss', 'Train vs Val gap (px)'),
        shared_xaxes=True, vertical_spacing=0.10,
    )

    legend_seen = set()
    for name, label, color in RUNS:
        log_path = root / f'cross_frame_{name}' / 'train.log'
        if not log_path.exists():
            print(f'skip (no log): {log_path}')
            continue
        d = parse_log(log_path)
        eps = d['epochs']
        if not eps:
            print(f'skip (no epochs parsed): {log_path}')
            continue

        # val (top-left)
        v_eps = [e for e, vv in zip(eps, d['val']) if vv is not None]
        v_vals = [vv for vv in d['val'] if vv is not None]
        fig.add_trace(go.Scatter(
            x=v_eps, y=v_vals, mode='lines+markers', name=label,
            line=dict(color=color, width=2.5), marker=dict(size=5),
            legendgroup=label, showlegend=label not in legend_seen,
            hovertemplate='ep %{x}<br>val %{y:.2f} px<extra></extra>',
        ), row=1, col=1)
        legend_seen.add(label)

        # train (top-right)
        fig.add_trace(go.Scatter(
            x=eps, y=d['train'], mode='lines', name=label,
            line=dict(color=color, width=1.5),
            legendgroup=label, showlegend=False,
            hovertemplate='ep %{x}<br>train %{y:.2f} px<extra></extra>',
        ), row=1, col=2)

        # NLL loss (bottom-left)
        fig.add_trace(go.Scatter(
            x=eps, y=d['loss'], mode='lines', name=label,
            line=dict(color=color, width=1.5),
            legendgroup=label, showlegend=False,
            hovertemplate='ep %{x}<br>loss %{y:.2f}<extra></extra>',
        ), row=2, col=1)

        # gap (bottom-right) — only at val sample epochs
        gap_eps, gap_vals = [], []
        for ee, vv in zip(eps, d['val']):
            if vv is None: continue
            tr = d['train'][eps.index(ee)]
            gap_eps.append(ee); gap_vals.append(vv - tr)
        fig.add_trace(go.Scatter(
            x=gap_eps, y=gap_vals, mode='lines+markers', name=label,
            line=dict(color=color, width=1.5),
            legendgroup=label, showlegend=False,
            hovertemplate='ep %{x}<br>val−train %{y:.2f} px<extra></extra>',
        ), row=2, col=2)

    # base reference line in val plot
    base_first = next((parse_log(root / f'cross_frame_{n}/train.log')['base']
                       for n, _, _ in RUNS
                       if (root / f'cross_frame_{n}/train.log').exists()), None)
    if base_first:
        fig.add_hline(y=sum(base_first) / len(base_first), line_dash='dash',
                      line_color='#999', annotation_text='base (no correction)',
                      annotation_position='top right',
                      row=1, col=1)
        fig.add_hline(y=sum(base_first) / len(base_first), line_dash='dash',
                      line_color='#999', row=1, col=2)

    fig.update_xaxes(title_text='epoch', row=2, col=1)
    fig.update_xaxes(title_text='epoch', row=2, col=2)
    fig.update_yaxes(title_text='val err (px)', row=1, col=1)
    fig.update_yaxes(title_text='train err (px)', row=1, col=2)
    fig.update_yaxes(title_text='NLL loss', row=2, col=1)
    fig.update_yaxes(title_text='val − train (px)', row=2, col=2)

    fig.update_layout(
        title=dict(text='Cross-frame net — Phase A1 ablation comparison<br>'
                         '<span style="font-size:13px;color:#666">'
                         'PandaSet 31 train scenes / 8 val scenes (scene-level split). '
                         'Hover for per-epoch values, click legend to toggle runs.'
                         '</span>',
                    x=0.5, xanchor='center'),
        height=820, width=1200, hovermode='x unified',
        legend=dict(orientation='h', yanchor='top', y=-0.10, xanchor='center', x=0.5),
        plot_bgcolor='#fafaf6', paper_bgcolor='#f6f4ed',
        font=dict(family='Noto Sans JP, sans-serif', size=12, color='#0f0f0e'),
        margin=dict(t=110, b=110, l=70, r=40),
    )
    for ax in ('xaxis', 'yaxis', 'xaxis2', 'yaxis2',
               'xaxis3', 'yaxis3', 'xaxis4', 'yaxis4'):
        fig.layout[ax].gridcolor = '#e5e3dc'
        fig.layout[ax].zerolinecolor = '#cccccc'

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs='cdn', full_html=True)
    print(f'saved → {out}')

    # also emit a fragment (div + script, no <html>/<head>) for inline embedding
    frag = out.with_suffix('.frag.html')
    fig.write_html(frag, include_plotlyjs=False, full_html=False)
    print(f'saved fragment → {frag}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiments-root', default='experiments')
    ap.add_argument('--out', default='docs/cross_frame_curves.html')
    args = ap.parse_args()
    main(args)
