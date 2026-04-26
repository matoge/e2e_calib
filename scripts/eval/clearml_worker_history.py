"""Aggregate past ClearML task system metrics by worker / hostname.

Pulls all tasks in a project, groups by hostname (set in task runtime), and
reports per-host: total wall time, mean / peak GPU util, mean / peak memory,
total tasks. Useful when tasks were launched directly (not through agents)
so the built-in Workloads page is empty.

Usage:
    python scripts/eval/clearml_worker_history.py [--project e2e_calib/cross-frame] [--days 7]
"""
import argparse
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta

from clearml import Task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='e2e_calib/cross-frame')
    ap.add_argument('--days', type=float, default=14, help='lookback window')
    ap.add_argument('--names', nargs='*', help='filter task names (substring match)')
    args = ap.parse_args()

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

    print(f'querying project={args.project!r} since {cutoff.isoformat()}')
    tasks = Task.get_tasks(project_name=args.project)

    by_host = defaultdict(lambda: {
        'tasks': [], 'total_seconds': 0.0,
        'gpu_util_mean': [], 'gpu_util_peak': 0.0,
        'gpu_mem_peak_gb': 0.0, 'cpu_mean': [],
        'mem_peak_gb': 0.0,
    })

    for t in tasks:
        try:
            started = t.data.started
            if started is None:
                continue
            if started.replace(tzinfo=timezone.utc) < cutoff:
                continue
            host = (t.data.runtime or {}).get('hostname', '<unknown>')
            ended = t.data.completed or t.data.last_update or started
            secs = (ended - started).total_seconds()
            metrics = t.get_reported_scalars()
            gpu = metrics.get(':monitor:gpu', {})
            mac = metrics.get(':monitor:machine', {})

            def _ys(d, k):
                return d.get(k, {}).get('y', []) or []

            util_y = _ys(gpu, 'gpu_0_utilization')
            mem_y  = _ys(gpu, 'gpu_0_mem_used_gb')
            cpu_y  = _ys(mac, 'cpu_usage')
            ram_y  = _ys(mac, 'memory_used_gb')

            d = by_host[host]
            d['tasks'].append((t.name, t.id, secs, t.get_status()))
            d['total_seconds'] += secs
            if util_y:
                d['gpu_util_mean'].append(sum(util_y) / len(util_y))
                d['gpu_util_peak'] = max(d['gpu_util_peak'], max(util_y))
            if mem_y:
                d['gpu_mem_peak_gb'] = max(d['gpu_mem_peak_gb'], max(mem_y))
            if cpu_y:
                d['cpu_mean'].append(sum(cpu_y) / len(cpu_y))
            if ram_y:
                d['mem_peak_gb'] = max(d['mem_peak_gb'], max(ram_y))
        except Exception as e:
            print(f'  skip task {t.id}: {e}')

    # Print summary
    print()
    print('=' * 110)
    print(f'{"host":<35s} {"tasks":>5s} {"GPU-h":>7s} {"util mean":>10s} {"util peak":>10s} {"mem peak":>10s} {"cpu mean":>10s}')
    print('=' * 110)
    for host in sorted(by_host):
        d = by_host[host]
        h = d['total_seconds'] / 3600
        umean = sum(d['gpu_util_mean']) / max(1, len(d['gpu_util_mean']))
        cmean = sum(d['cpu_mean']) / max(1, len(d['cpu_mean']))
        print(f'{host:<35s} {len(d["tasks"]):>5d} {h:>7.2f} {umean:>9.1f}% {d["gpu_util_peak"]:>9.1f}% {d["gpu_mem_peak_gb"]:>8.2f}GB {cmean:>9.1f}%')
    print('=' * 110)

    # Per-host task list
    print('\nPer-host task breakdown:')
    for host in sorted(by_host):
        d = by_host[host]
        print(f'\n  [{host}] — {len(d["tasks"])} tasks, total {d["total_seconds"]/3600:.2f} GPU-h')
        for name, tid, secs, status in sorted(d['tasks'], key=lambda x: -x[2])[:20]:
            print(f'    {secs/60:>6.1f}min  [{status:<11s}]  {name}')


if __name__ == '__main__':
    main()
