"""Retroactive upload: parse experiments/{exp}/train.log → create a ClearML
task with per-epoch scalars + why/retrospective context.

Useful when a training script ran without --clearml (e.g. local quick run)
and we want to make it visible in ClearML for cross-machine sharing.

Usage:
    python scripts/util/upload_local_run_to_clearml.py \
        --exp ps_v11_lidar_kv \
        --project e2e_calib/calib \
        --why "Step 2 of unified arch: KV lidar-bank concat + pose_emb (vfp anchor); should not regress vs ps_v9_lazy=1.81." \
        --baseline ps_v9_lazy=1.8141

The script:
- Parses train.log for ep/train_nll/val_nll/...
- Calls Task.init() and reports the scalars retroactively (timestamps == log timestamps)
- Writes the why + retrospective comment via clearml_context helpers
- Uploads best_model.pt and curves.png as artifacts (if present)
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


_LINE = re.compile(
    r'^(?P<ts>\d{2}:\d{2}:\d{2})\s+\[\s*(?P<ep>\d+)/\d+\]\s+'
    r'train nll=(?P<tr_nll>[+-][\d.]+)\(obj=(?P<tr_obj_nll>[+-][\d.]+)\s+bg=(?P<tr_bg_nll>[+-][\d.]+)\)\s+'
    r'mse=(?P<tr_mse>[\d.]+)\(obj=(?P<tr_obj_mse>[\d.]+)\s+bg=(?P<tr_bg_mse>[\d.]+)\)\s+'
    r'val nll=(?P<va_nll>[+-][\d.]+)\(obj=(?P<va_obj_nll>[+-][\d.]+)\s+bg=(?P<va_bg_nll>[+-][\d.]+)\)\s+'
    r'mse=(?P<va_mse>[\d.]+)\(obj=(?P<va_obj_mse>[\d.]+)\s+bg=(?P<va_bg_mse>[\d.]+)\)'
)
_BEST = re.compile(r'Best val NLL: (?P<best>[\d.]+)\s+\|\s+time:\s+(?P<min>[\d.]+)min')


def parse_log(log_path: Path):
    rows = []
    best = None
    elapsed = None
    for line in log_path.read_text(errors='replace').splitlines():
        m = _LINE.match(line)
        if m:
            rows.append({k: (int(v) if k == 'ep' else float(v) if k != 'ts' else v)
                         for k, v in m.groupdict().items()})
        b = _BEST.search(line)
        if b:
            best = float(b.group('best'))
            elapsed = float(b.group('min'))
    return rows, best, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, help='experiment name (= dir under experiments/)')
    ap.add_argument('--project', default='e2e_calib/calib')
    ap.add_argument('--why', default='', help='rationale text (## Why block)')
    ap.add_argument('--baseline', default=None,
                    help='baseline reference, format "name=value" e.g. ps_v9_lazy=1.8141')
    ap.add_argument('--conclusion', default='',
                    help='hand-written conclusion appended to retrospective')
    args = ap.parse_args()

    exp_dir = REPO_ROOT / 'experiments' / args.exp
    log_path = exp_dir / 'train.log'
    if not log_path.exists():
        sys.exit(f'no train.log at {log_path}')

    rows, best, elapsed_min = parse_log(log_path)
    if not rows:
        sys.exit(f'no parsable epoch lines in {log_path}')

    cfg_path = exp_dir / 'config.py'
    cfg = {}
    if cfg_path.exists():
        ns = {}
        exec(cfg_path.read_text(), ns)
        cfg = ns.get('CFG', {})

    baseline = None
    if args.baseline:
        nm, val = args.baseline.split('=', 1)
        baseline = {'name': nm.strip(), 'value': float(val), 'metric': 'val_nll'}

    from scripts.util.clearml_context import init_with_context, write_retrospective
    cfg = dict(cfg); cfg['name'] = args.exp; cfg['_retroactive'] = True
    task = init_with_context(args.project, args.exp, cfg, why=args.why, baseline=baseline)
    log = task.get_logger()
    print(f'[upload] task created, replaying {len(rows)} epochs ...')
    for r in rows:
        ep = r['ep']
        log.report_scalar('nll', 'train', r['tr_nll'], ep)
        log.report_scalar('nll', 'val',   r['va_nll'], ep)
        log.report_scalar('mse', 'train', r['tr_mse'], ep)
        log.report_scalar('mse', 'val',   r['va_mse'], ep)
        log.report_scalar('obj_nll', 'train', r['tr_obj_nll'], ep)
        log.report_scalar('obj_nll', 'val',   r['va_obj_nll'], ep)
        log.report_scalar('obj_mse', 'train', r['tr_obj_mse'], ep)
        log.report_scalar('obj_mse', 'val',   r['va_obj_mse'], ep)
        log.report_scalar('bg_nll',  'val',   r['va_bg_nll'],  ep)
        log.report_scalar('bg_mse',  'val',   r['va_bg_mse'],  ep)

    # artifacts
    for name in ('best_model.pt', 'config.py'):
        p = exp_dir / name
        if p.exists():
            task.upload_artifact(name, artifact_object=str(p))
    curves = exp_dir / 'vis' / 'curves.png'
    if curves.exists():
        task.upload_artifact('curves.png', artifact_object=str(curves))
    log_artifact = log_path
    task.upload_artifact('train.log', artifact_object=str(log_artifact))

    write_retrospective(task, dict(
        best_val_nll=best if best is not None else min(r['va_nll'] for r in rows),
        final_val_nll=rows[-1]['va_nll'],
        best_obj_mse=min(r['va_obj_mse'] for r in rows),
        time_min=elapsed_min if elapsed_min else None,
        epochs=rows[-1]['ep'],
    ), baseline=baseline, conclusion=args.conclusion)

    print(f'[upload] done — task id: {task.id}')
    print(f'  view: {task.get_output_log_web_page()}')


if __name__ == '__main__':
    main()
