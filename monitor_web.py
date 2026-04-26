"""monitor_web.py — live training-curve viewer for cross-frame experiments.

Point a browser at http://<host>:5002/<exp_name>  (e.g. /v20_mine_poc).

Parses experiments/cross_frame_{name}/train.log on every poll and serves:
  • train NLL / val NLL curves
  • train err_AB/BA / val err_AB/BA curves (px)
  • depth err curves (uvd runs)
  • vertical dashed lines at each MIGRATE#N event

Start:   python monitor_web.py
"""
import json, re, time
from pathlib import Path
from flask import Flask, jsonify, request

EXP_ROOT = Path(__file__).resolve().parent / 'experiments'
app = Flask(__name__)

# ─── parser ───────────────────────────────────────────────────────────────────

_EP_RE = re.compile(
    r"ep\s+(?P<ep>\d+)\s+loss=(?P<loss>-?\d+\.\d+)"
    r"\s+err_AB=(?P<err_AB>-?\d+\.\d+)px"
    r"\s+err_BA=(?P<err_BA>-?\d+\.\d+)px"
    r"\s+\(base=(?P<base_AB>-?\d+\.\d+)px\)"
    r"(?:\s+val_err=(?P<val_err>-?\d+\.\d+)px\s+\(base\s+(?P<val_base>-?\d+\.\d+)\)"
    r"(?:\s+val_nll=(?P<val_nll>-?\d+\.\d+))?"
    r")?"
    r"(?:\s+d=(?P<err_d>-?\d+\.\d+)m\(base\s+(?P<base_d>-?\d+\.\d+)\))?"
    r"(?:\s+val_d=(?P<val_err_d>-?\d+\.\d+)m\(base\s+(?P<val_base_d>-?\d+\.\d+)\))?"
    r".*?lr=(?P<lr>\d+\.\d+e-?\d+).*?t=(?P<t_s>\d+)s"
)
_MIG_RE = re.compile(r"\[MIGRATE#(?P<n>\d+)\]\s+moved\s+(?P<k>\d+)"
                     r"\s+val.*?total\s+migrated\s+(?P<total>\d+).*?remaining\s+(?P<rem>\d+)")


def parse_log(log_path: Path) -> dict:
    curves = dict(epoch=[], loss=[], err_AB=[], err_BA=[], base_AB=[],
                  val_err=[], val_nll=[], val_base=[], val_ep=[],
                  err_d=[], val_err_d=[], lr=[])
    migrations = []      # {ep, n, k, total, rem, approx_val_err}
    status = 'unknown'
    last_epoch = -1
    args = None
    for line in log_path.read_text().splitlines():
        if 'args =' in line and args is None:
            try:
                j = line.split('args =', 1)[1].strip()
                args = eval(j, {}, {})
            except Exception:
                args = None
        m = _EP_RE.search(line)
        if m:
            ep = int(m.group('ep'))
            last_epoch = ep
            curves['epoch'].append(ep)
            curves['loss'].append(float(m.group('loss')))
            curves['err_AB'].append(float(m.group('err_AB')))
            curves['err_BA'].append(float(m.group('err_BA')))
            curves['base_AB'].append(float(m.group('base_AB')))
            if m.group('lr') is not None:
                curves['lr'].append(float(m.group('lr')))
            if m.group('val_err') is not None:
                curves['val_ep'].append(ep)
                curves['val_err'].append(float(m.group('val_err')))
                curves['val_base'].append(float(m.group('val_base')))
                curves['val_nll'].append(float(m.group('val_nll')) if m.group('val_nll') else None)
            if m.group('err_d') is not None:
                curves['err_d'].append((ep, float(m.group('err_d'))))
            if m.group('val_err_d') is not None:
                curves['val_err_d'].append((ep, float(m.group('val_err_d'))))
            continue
        mm = _MIG_RE.search(line)
        if mm:
            migrations.append(dict(
                ep=last_epoch, n=int(mm.group('n')), k=int(mm.group('k')),
                total=int(mm.group('total')), rem=int(mm.group('rem')),
                approx_val_err=curves['val_err'][-1] if curves['val_err'] else None,
            ))
    if 'saved curve' in log_path.read_text() or 'best val err' in log_path.read_text():
        status = 'done'
    else:
        status = 'running'
    return dict(args=args, status=status, last_epoch=last_epoch,
                curves=curves, migrations=migrations)


# ─── routes ───────────────────────────────────────────────────────────────────

INDEX_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8"><title>cross-frame monitor — {name}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; background:#f6f4ed; color:#1a1a1a;
        margin: 24px; }}
  h1 {{ font-weight: 600; font-size: 18px; margin: 0 0 6px; }}
  .sub {{ color:#6b6a63; font-size: 12px; margin-bottom: 20px; }}
  .plot {{ background: white; border: 1px solid #e0dcc8; padding: 8px; margin-bottom: 16px;
         border-radius: 4px; }}
  table {{ border-collapse: collapse; font-size: 12px; }}
  th, td {{ padding: 4px 10px; text-align: left; border-bottom: 1px solid #e0dcc8; }}
  th {{ background: #eee6d2; }}
  .status-running {{ color: #c13c14; }}
  .status-done {{ color: #174734; }}
</style>
</head><body>
<h1 id="title">cross-frame monitor — <span style="color:#c13c14">{name}</span>
  <span id="status" class="status-running"></span>
</h1>
<div class="sub" id="subline"></div>
<div class="plot" id="err_plot"></div>
<div class="plot" id="nll_plot"></div>
<div class="plot" id="lr_plot"></div>
<div class="plot" id="d_plot"></div>
<h3 style="margin: 8px 0">migration events</h3>
<table id="mig_tab"><thead><tr>
<th>#</th><th>epoch</th><th>moved</th><th>total migrated</th><th>val pool remaining</th>
<th>val_err at trigger</th>
</tr></thead><tbody></tbody></table>

<script>
const NAME = "{name}";
const CLR_TR_AB = '#c13c14', CLR_TR_BA = '#174734', CLR_VAL_AB = '#d77a1b', CLR_VAL_BA = '#2a7d5f',
      CLR_LOSS = '#174734', CLR_VLOSS = '#c13c14', CLR_MIG = '#8a1a6b', CLR_BASE = '#aaa';

async function refresh() {{
  const r = await fetch(`/api/progress/${{NAME}}?_=${{Date.now()}}`);
  if (!r.ok) return;
  const j = await r.json();
  const c = j.curves;
  // status
  const el = document.getElementById('status');
  el.textContent = j.status === 'done' ? '  ✓ done' : '  ⟳ running';
  el.className = j.status === 'done' ? 'status-done' : 'status-running';
  document.getElementById('subline').textContent =
      `last ep ${{j.last_epoch}}  |  migrations so far: ${{j.migrations.length}}`;

  const shapes = j.migrations.map((m, i) => ({{
    type: 'line', x0: m.ep, x1: m.ep, yref: 'paper', y0: 0, y1: 1,
    line: {{ color: CLR_MIG, width: 1.5, dash: 'dash' }},
  }}));
  const ann = j.migrations.map((m, i) => ({{
    x: m.ep, yref:'paper', y: 0.95, text: `M${{m.n}}`, showarrow: false,
    font: {{ color: CLR_MIG, size: 10 }}, bgcolor:'#f6f4ed'
  }}));

  const err_traces = [
    {{ x: c.epoch, y: c.err_AB, name:'train A→B', line:{{color: CLR_TR_AB, width: 2}} }},
    {{ x: c.epoch, y: c.err_BA, name:'train B→A', line:{{color: CLR_TR_BA, width: 2}} }},
    {{ x: c.val_ep, y: c.val_err, name:'val mean', line:{{color: CLR_VAL_AB, width: 2, dash:'dot'}},
      mode:'lines+markers', marker:{{size:6}} }},
    {{ x: c.epoch, y: c.base_AB, name:'baseline (no corr)', line:{{color: CLR_BASE, width:1, dash:'dash'}} }},
  ];
  Plotly.react('err_plot', err_traces, {{
    title:{{text:'reproj err (px)', font:{{size: 14}}, x:0.02, xanchor:'left'}},
    shapes, annotations: ann, height: 300, margin:{{t:40,r:20,b:40,l:50}},
    xaxis:{{title:'epoch'}}, yaxis:{{title:'px'}}, plot_bgcolor:'#fafaf4',
  }}, {{displaylogo:false, responsive:true}});

  const nll_traces = [
    {{ x: c.epoch, y: c.loss,    name:'train NLL', line:{{color: CLR_LOSS, width: 2}} }},
    {{ x: c.val_ep, y: c.val_nll, name:'val NLL', line:{{color: CLR_VLOSS, width: 2, dash:'dot'}},
      mode:'lines+markers', marker:{{size:6}} }},
  ];
  Plotly.react('nll_plot', nll_traces, {{
    title:{{text:'NLL', font:{{size: 14}}, x:0.02, xanchor:'left'}},
    shapes, annotations: ann, height: 260, margin:{{t:40,r:20,b:40,l:50}},
    xaxis:{{title:'epoch'}}, yaxis:{{title:'nll'}}, plot_bgcolor:'#fafaf4',
  }}, {{displaylogo:false, responsive:true}});

  const lr_traces = [
    {{ x: c.epoch, y: c.lr, name:'lr', line:{{color: '#6b4c8a', width: 2}}, mode:'lines' }}
  ];
  Plotly.react('lr_plot', lr_traces, {{
    title:{{text:'learning rate', font:{{size: 14}}, x:0.02, xanchor:'left'}},
    shapes, annotations: ann, height: 200, margin:{{t:40,r:20,b:40,l:60}},
    xaxis:{{title:'epoch'}}, yaxis:{{title:'lr', type:'log'}}, plot_bgcolor:'#fafaf4',
  }}, {{displaylogo:false, responsive:true}});

  const dTr = c.err_d.length > 0 ? {{
    x: c.err_d.map(p=>p[0]), y: c.err_d.map(p=>p[1]),
    name:'train d err', line:{{color: CLR_TR_AB, width:2}}
  }} : null;
  const dVal = c.val_err_d.length > 0 ? {{
    x: c.val_err_d.map(p=>p[0]), y: c.val_err_d.map(p=>p[1]),
    name:'val d err', line:{{color: CLR_VAL_AB, width:2, dash:'dot'}},
    mode:'lines+markers', marker:{{size:6}}
  }} : null;
  if (dTr || dVal) {{
    Plotly.react('d_plot', [dTr, dVal].filter(Boolean), {{
      title:{{text:'depth err (m)', font:{{size: 14}}, x:0.02, xanchor:'left'}},
      shapes, annotations: ann, height: 260, margin:{{t:40,r:20,b:40,l:50}},
      xaxis:{{title:'epoch'}}, yaxis:{{title:'m'}}, plot_bgcolor:'#fafaf4',
    }}, {{displaylogo:false, responsive:true}});
    document.getElementById('d_plot').style.display = 'block';
  }} else {{
    document.getElementById('d_plot').style.display = 'none';
  }}

  // migration table
  const tb = document.querySelector('#mig_tab tbody');
  tb.innerHTML = j.migrations.map(m =>
    `<tr><td>${{m.n}}</td><td>${{m.ep}}</td><td>${{m.k}}</td><td>${{m.total}}</td><td>${{m.rem}}</td>` +
    `<td>${{m.approx_val_err !== null ? m.approx_val_err.toFixed(2) + ' px' : '—'}}</td></tr>`
  ).join('');
}}
refresh();
setInterval(refresh, 3000);
</script>
</body></html>
"""

LIST_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>cross-frame experiments</title>
<style>
  body {font-family: system-ui; margin: 24px; background: #f6f4ed;}
  h1 {font-size: 16px; margin: 0 0 16px;}
  ul {list-style: none; padding: 0;}
  li {padding: 4px 0;}
  a {color: #174734; text-decoration: none;}
  a:hover {text-decoration: underline;}
</style></head><body>
<h1>cross-frame experiments</h1>
<ul>{items}</ul>
</body></html>
"""


@app.route('/')
def index():
    exps = sorted([p.name.replace('cross_frame_', '') for p in EXP_ROOT.iterdir()
                   if p.is_dir() and p.name.startswith('cross_frame_') and (p / 'train.log').exists()])
    items = '\n'.join(f'<li><a href="/{n}">{n}</a></li>' for n in exps)
    return LIST_HTML.replace('{items}', items)


@app.route('/<name>')
def view(name):
    return INDEX_HTML.format(name=name)


@app.route('/api/progress/<name>')
def api_progress(name):
    log = EXP_ROOT / f'cross_frame_{name}' / 'train.log'
    if not log.exists():
        return jsonify(error=f'no log at {log}'), 404
    return jsonify(parse_log(log))


if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5002
    print(f'[monitor_web] serving experiments from {EXP_ROOT} on port {port}')
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
