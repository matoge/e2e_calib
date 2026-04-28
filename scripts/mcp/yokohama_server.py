"""MCP server on yokohama, exposing project state for cross-machine Claude.

Tools:
  - get_status()                — git HEAD, GPU/CPU snapshot, list of in-flight training procs
  - tail_log(experiment, n=40)  — tail experiments/{exp}/train.log
  - query_clearml(task_id, n=40)— task status + console tail from ClearML server
  - list_caches()                — list e2e_calib_cache subdirs with sizes
  - prepare_download(path, ttl)  — start a one-off HTTP file server, return URL

Run with HTTP transport (cloudflared tunnels it):
    python scripts/mcp/yokohama_server.py --port 5050

To tunnel:
    cloudflared tunnel --url http://localhost:5050
    # → URL printed to stdout; register in office Claude's MCP settings.

Designed for the e2e_calib repo. Read-only by default; only side effect is
prepare_download spawning a short-lived python -m http.server.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = Path('/mnt/nvme6t/e2e_calib_cache')

mcp = FastMCP('yokohama-e2e-calib')


def _run(cmd: list[str], cwd: Optional[Path] = None, timeout: int = 30) -> str:
    try:
        out = subprocess.check_output(
            cmd, cwd=cwd or REPO_ROOT, stderr=subprocess.STDOUT, timeout=timeout
        )
        return out.decode('utf-8', errors='replace').strip()
    except subprocess.CalledProcessError as e:
        return f'[error: {e.output.decode("utf-8", errors="replace")[-500:]}]'
    except subprocess.TimeoutExpired:
        return f'[timeout: {" ".join(cmd)}]'


@mcp.tool()
def get_status() -> str:
    """Return git HEAD, GPU/CPU usage, and in-flight training processes on yokohama."""
    head = _run(['git', 'log', '-1', '--format=%h %s'])
    branch = _run(['git', 'symbolic-ref', '--short', 'HEAD'])
    dirty = _run(['git', 'status', '--short']).count('\n') + 1 if _run(['git', 'status', '--short']) else 0
    gpu = _run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader'])
    procs = _run(['pgrep', '-af', 'python.*train_'])
    procs = '\n  '.join((procs.splitlines() or ['(none)'])[:8])
    return (
        f'host       : {socket.gethostname()}\n'
        f'branch     : {branch}\n'
        f'HEAD       : {head}\n'
        f'dirty files: {dirty}\n'
        f'GPU        : {gpu}\n'
        f'training   : {procs}'
    )


@mcp.tool()
def tail_log(experiment: str, n: int = 40) -> str:
    """Tail experiments/{experiment}/train.log on yokohama."""
    p = REPO_ROOT / 'experiments' / experiment / 'train.log'
    if not p.exists():
        return f'[not found: {p}]'
    lines = p.read_text(errors='replace').splitlines()[-n:]
    return '\n'.join(lines)


@mcp.tool()
def query_clearml(task_id: str, n: int = 40) -> str:
    """Return ClearML task status + console tail. task_id is the long hex id."""
    try:
        from clearml import Task
        t = Task.get_task(task_id=task_id)
        ev = t.get_reported_console_output(number_of_reports=10)
        if isinstance(ev, list):
            tail = '\n'.join(ev[-n:])
        else:
            tail = '\n'.join(str(ev).splitlines()[-n:])
        return f'status: {t.status}\nname:   {t.name}\nproject:{t.get_project_name()}\n--- console ---\n{tail}'
    except Exception as e:
        return f'[clearml query error: {e}]'


@mcp.tool()
def clearml_metrics(task_id: str, last_n: int = 20) -> str:
    """Return the last N scalar values for every metric reported on a task.

    Returns a text table; one block per (title, series). Useful when the
    office Claude wants 'how is task X doing' without a browser.
    """
    try:
        from clearml import Task
        t = Task.get_task(task_id=task_id)
        scalars = t.get_reported_scalars(max_samples=last_n)
        # shape: {title: {series: {x:[...], y:[...]}}}
        out = [f'task: {t.name} ({t.status})']
        for title, series in scalars.items():
            out.append(f'\n[{title}]')
            for sname, xy in series.items():
                xs, ys = xy.get('x', []), xy.get('y', [])
                if not ys: continue
                last = ys[-1]
                first = ys[0]
                trend = '↓' if last < first else ('↑' if last > first else '·')
                # ASCII sparkline
                if len(ys) >= 2:
                    lo, hi = min(ys), max(ys)
                    rng = hi - lo if hi > lo else 1.0
                    bars = ' ▁▂▃▄▅▆▇█'
                    spark = ''.join(bars[min(8, max(0, int((y - lo) / rng * 8)))] for y in ys)
                else:
                    spark = ''
                out.append(f'  {sname:<10s} {trend} last={last:+.3f}  {spark}  '
                           f'(N={len(ys)}, ep={int(xs[-1]) if xs else "?"})')
        return '\n'.join(out)
    except Exception as e:
        return f'[clearml metrics error: {e}]'


@mcp.tool()
def clearml_list_recent(project: str = 'e2e_calib/calib', n: int = 10) -> str:
    """List the most recent ClearML tasks in a project (default 'e2e_calib/calib').

    Returns id / name / status / last update time. Use the id to query further.
    """
    try:
        from clearml import Task
        tasks = Task.get_tasks(project_name=project, task_filter={'order_by': ['-last_update']})
        rows = []
        for t in tasks[:n]:
            tid = t.id
            name = t.name
            status = t.status
            rows.append(f'  {tid[:8]}  {status:<10}  {name}')
        return f'[{project}] last {len(rows)} task(s):\n' + '\n'.join(rows)
    except Exception as e:
        return f'[clearml list error: {e}]'


@mcp.tool()
def clearml_progress_summary(task_id: str) -> str:
    """One-line summary: ep N/M, latest val_nll, ETA based on epoch rate.

    Convenience for office Claude polling 'how far is task X'.
    """
    try:
        from clearml import Task
        t = Task.get_task(task_id=task_id)
        scalars = t.get_reported_scalars(max_samples=200)
        cfg = t.get_parameters_as_dict().get('cfg') or t.get_parameters_as_dict().get('General') or {}
        max_ep = cfg.get('epochs') or cfg.get('cfg/epochs') or '?'
        last_x = last_y = None
        for title, series in scalars.items():
            for sname, xy in series.items():
                xs, ys = xy.get('x', []), xy.get('y', [])
                if xs and (last_x is None or xs[-1] > last_x):
                    last_x = xs[-1]
                    last_y_dict = {'title': title, 'series': sname, 'value': ys[-1]}
                    last_y = last_y_dict
        if last_x is None:
            return f'{t.name} ({t.status}): no scalars yet'
        return (f'{t.name} ({t.status})\n'
                f'  ep {int(last_x)}/{max_ep}\n'
                f'  latest: {last_y["title"]}/{last_y["series"]} = {last_y["value"]:+.3f}')
    except Exception as e:
        return f'[clearml progress error: {e}]'


@mcp.tool()
def list_caches() -> str:
    """List /mnt/nvme6t/e2e_calib_cache subdirs with sizes."""
    if not CACHE_ROOT.exists():
        return f'[cache root missing: {CACHE_ROOT}]'
    rows = []
    for child in sorted(CACHE_ROOT.iterdir()):
        try:
            if child.is_dir():
                # use du for fast aggregate
                size = _run(['du', '-sh', str(child)], timeout=20).split('\t', 1)[0]
                rows.append(f'  {size:>8}  dir   {child.name}')
            else:
                size = child.stat().st_size
                rows.append(f'  {size//(1024**3):>4} GB  file  {child.name}')
        except Exception as e:
            rows.append(f'  [err]  {child.name}: {e}')
    return f'{CACHE_ROOT}:\n' + '\n'.join(rows)


# Note: prepare_download intentionally removed — the previous implementation
# could serve arbitrary local paths over a CF tunnel, which is too dangerous
# even with the random URL. If file transfer is needed in the future, scope
# it to a hardcoded whitelist (e.g., only /mnt/nvme6t/e2e_calib_cache/*).


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=5050)
    ap.add_argument('--host', default='127.0.0.1',
                    help='bind addr; cloudflared tunnels from 127.0.0.1 by default')
    ap.add_argument('--allow-cf-ip', action='append', default=[],
                    help='Allowlist a Cf-Connecting-Ip value. Repeatable. If set,'
                         ' requests without a matching header are 403\'d. Allows'
                         ' bare IPs and exact host headers.')
    args = ap.parse_args()
    mcp.settings.host = args.host
    mcp.settings.port = args.port

    # Build SSE app, then wrap with IP allowlist middleware if requested.
    app = mcp.sse_app()
    if args.allow_cf_ip:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import PlainTextResponse
        allowed = set(args.allow_cf_ip)

        class CfIpFilter(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                cip = request.headers.get('cf-connecting-ip', '')
                if cip not in allowed:
                    return PlainTextResponse(
                        f'forbidden (cf-ip={cip!r})', status_code=403)
                return await call_next(request)
        app.add_middleware(CfIpFilter)
        print(f'[mcp] cf-ip allowlist active: {sorted(allowed)}')

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level='info')


if __name__ == '__main__':
    main()
