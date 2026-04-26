"""Parse an existing train.log (from a non-ClearML run) and re-upload its
per-epoch metrics to a new ClearML task. Useful for back-filling history.

Usage:
    python scripts/eval/import_train_log.py \
        --log experiments/cross_frame_v63.../train.log \
        --name v63_aug_sigma2_lidardrop
"""
import argparse
import json
import re
from pathlib import Path

from clearml import Task

EP_RE = re.compile(
    r'ep\s+(\d+)\s+loss=([\d.]+)\s+'
    r'err_AB=([\d.]+)px\s+err_BA=([\d.]+)px\s+\(base=([\d.]+)px\)'
    r'(?:\s+val_err=([\d.]+)px[^v]*val_nll=([\d.]+))?'
    r'.*?lr=([\d.eE+-]+)'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True, help='path to train.log')
    ap.add_argument('--name', required=True, help='task name in ClearML')
    ap.add_argument('--project', default='e2e_calib/cross-frame')
    args = ap.parse_args()

    log_path = Path(args.log)
    txt = log_path.read_text()

    task = Task.init(project_name=args.project, task_name=args.name,
                     auto_connect_frameworks={'pytorch': False})
    # Capture the args = {...} line if present
    m = re.search(r"args = (\{.*?\})", txt, re.DOTALL)
    if m:
        try:
            cfg = eval(m.group(1), {'__builtins__': {}})  # safe-ish for our log
            task.connect(cfg, name='args')
        except Exception:
            pass

    logger = task.get_logger()
    n = 0
    for line in txt.splitlines():
        m = EP_RE.search(line)
        if not m:
            continue
        ep = int(m.group(1))
        loss = float(m.group(2)); err_AB = float(m.group(3))
        err_BA = float(m.group(4)); base = float(m.group(5))
        val_err = float(m.group(6)) if m.group(6) else None
        val_nll = float(m.group(7)) if m.group(7) else None
        lr = float(m.group(8))
        logger.report_scalar('train', 'loss',   value=loss,   iteration=ep)
        logger.report_scalar('train', 'err_AB', value=err_AB, iteration=ep)
        logger.report_scalar('train', 'err_BA', value=err_BA, iteration=ep)
        logger.report_scalar('train', 'base',   value=base,   iteration=ep)
        logger.report_scalar('lr',    'lr',     value=lr,     iteration=ep)
        if val_err is not None:
            logger.report_scalar('val', 'err', value=val_err, iteration=ep)
            logger.report_scalar('val', 'nll', value=val_nll, iteration=ep)
            mean_train_err = 0.5 * (err_AB + err_BA)
            logger.report_scalar('overfit', 'val/train_err',
                                 value=val_err / max(mean_train_err, 1e-6),
                                 iteration=ep)
            logger.report_scalar('overfit', 'val_nll - train_loss',
                                 value=val_nll - loss,
                                 iteration=ep)
        n += 1
    print(f'imported {n} epochs from {log_path}')
    print(f'task url: {task.get_output_log_web_page()}')
    task.close()


if __name__ == '__main__':
    main()
