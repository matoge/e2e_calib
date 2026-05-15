"""Import on-disk training results (experiments/<name>/) into ClearML as
real Task objects, after the fact. Useful for ministar blackout recovery
where good runs never made it to a registry.

For each experiment dir we create one ClearML Task with:
  - hyperparameters from config.py (as Task.connect dict)
  - per-epoch scalars parsed out of train.log (train_nll, val_nll, lr, mse...)
  - best_model.pt as an output_model artifact
  - vis_ep***/*.png as report_image (latest epoch only — keeps Task slim)

Usage:
  CLEARML_API_HOST=... python -m scripts.serving.import_local_experiments \
      --project 'e2e_calib/calib' \
      --pattern 'experiments/tss4_*'
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

from clearml import Task, OutputModel


# `00:53:53  [  1/100]  train nll=+5.935(pt=+5.935 fr=+nan obj=+0.000 bg=+5.936)
#  mse=15.23(obj=...) val nll=+5.620(...) mse=12.73(...) lr=1.20e-04 tot=2.4min`
LOG_RE = re.compile(
    r'^\d+:\d+:\d+\s+\[\s*(?P<ep>\d+)/\d+\]\s+'
    r'train\s+nll=(?P<tr_nll>[+-]?[\d.]+|nan|inf|[+-]nan)'
    r'.*?mse=(?P<tr_mse>[+-]?[\d.]+|nan)'
    r'.*?val\s+nll=(?P<va_nll>[+-]?[\d.]+|nan|inf|[+-]nan)'
    r'.*?mse=(?P<va_mse>[+-]?[\d.]+|nan)'
    r'.*?lr=(?P<lr>[\d.eE+-]+)'
    r'.*?tot=(?P<tot>[\d.]+)min'
)


def _safe_float(s: str) -> float:
    try:
        return float(s.lstrip('+'))
    except ValueError:
        return float('nan')


def parse_train_log(path: Path) -> list[dict]:
    rows = []
    for ln in path.read_text(errors='replace').splitlines():
        m = LOG_RE.match(ln)
        if m:
            rows.append({k: _safe_float(v) if k != 'ep' else int(v)
                          for k, v in m.groupdict().items()})
    return rows


def read_cfg(path: Path) -> dict:
    src = path.read_text()
    # config.py contains `CFG = dict(name=..., ...)`; eval after replacing
    # numpy-printed reprs that wouldn't parse.
    m = re.search(r'CFG\s*=\s*dict\((.*)\)\s*$', src, flags=re.S)
    if not m:
        return {}
    body = m.group(1)
    # Parse k=v pairs naively (config.py is hand-written from train_ps_v3).
    out: dict = {}
    for line in body.split('\n'):
        line = line.strip().rstrip(',')
        if '=' not in line:
            continue
        k, v = line.split('=', 1)
        k = k.strip(); v = v.strip()
        try:
            out[k] = ast.literal_eval(v)
        except Exception:
            out[k] = v
    return out


def import_one(exp_dir: Path, project: str) -> str | None:
    name = exp_dir.name
    train_log = exp_dir / 'train.log'
    cfg_py = exp_dir / 'config.py'
    if not train_log.exists():
        print(f'  skip {name}: no train.log')
        return None
    rows = parse_train_log(train_log)
    if not rows:
        print(f'  skip {name}: no epoch rows parsed')
        return None
    cfg = read_cfg(cfg_py) if cfg_py.exists() else {}

    task = Task.create(project_name=project, task_name=name)
    task.set_comment(
        f'Imported from local disk on {Path.cwd()} — original run.\n'
        f'Source: {exp_dir}\n'
        f'Parsed {len(rows)} epoch rows from train.log.'
    )
    if cfg:
        task.connect(dict(cfg), name='CFG')
    logger = task.get_logger()
    for r in rows:
        ep = r['ep']
        logger.report_scalar('nll/train', 'pt', value=r['tr_nll'], iteration=ep)
        logger.report_scalar('nll/val',   'pt', value=r['va_nll'], iteration=ep)
        logger.report_scalar('mse/train', 'all', value=r['tr_mse'], iteration=ep)
        logger.report_scalar('mse/val',   'all', value=r['va_mse'], iteration=ep)
        logger.report_scalar('lr', 'lr', value=r['lr'], iteration=ep)
        logger.report_scalar('elapsed_min', 't', value=r['tot'], iteration=ep)

    # Upload checkpoints + report.html as artifacts.
    bm = exp_dir / 'best_model.pt'
    if bm.exists():
        om = OutputModel(task=task, name=f'{name}_best',
                          framework='PyTorch')
        try:
            om.update_weights(str(bm), upload_uri=None)
        except Exception as e:
            print(f'    warn: failed to upload {bm}: {e}')
    rh = exp_dir / 'report.html'
    if rh.exists():
        try:
            task.upload_artifact('report.html', artifact_object=str(rh))
        except Exception as e:
            print(f'    warn: failed to upload report.html: {e}')
    # Upload every epoch's vis PNGs with iteration=epoch so the web UI can
    # scrub through training as a flipbook. ClearML packs all iterations of
    # the same (title, series) into a single tile.
    #   vis_pretrain  → iteration 0 (initial)
    #   vis_epNNN     → iteration NNN
    #   vis           → iteration final_ep (last entry from train.log)
    final_ep = max((r['ep'] for r in rows), default=0)
    vis_groups: list[tuple[int, Path]] = []
    if (exp_dir / 'vis_pretrain').is_dir():
        vis_groups.append((0, exp_dir / 'vis_pretrain'))
    for d in sorted(exp_dir.glob('vis_ep*')):
        m_ep = re.search(r'\d+', d.name)
        if m_ep and d.is_dir():
            vis_groups.append((int(m_ep.group()), d))
    if (exp_dir / 'vis').is_dir():
        vis_groups.append((final_ep, exp_dir / 'vis'))
    if vis_groups:
        n_uploaded = 0
        for ep_idx, d in vis_groups:
            for png in sorted(d.glob('*.png'))[:30]:
                # series = filename stem stripped of any epoch tag so the same
                # tile in different epochs lines up under one series name.
                series = re.sub(r'_idx\d+|_ep\d+', '', png.stem) or png.stem
                try:
                    logger.report_image('vis', series, iteration=ep_idx,
                                         local_path=str(png))
                    n_uploaded += 1
                except Exception as e:
                    print(f'    warn vis {png}: {e}')
        print(f'    + {n_uploaded} vis images across {len(vis_groups)} epochs')
    # Task.create() leaves status='created' (draft). Promote through the
    # state machine via the raw API so the WebUI lists the imported run
    # alongside live ones.
    from clearml.backend_api.session.client import APIClient
    api = APIClient()
    try:
        api.tasks.started(task=task.id, force=True)
        api.tasks.completed(task=task.id, force=True)
    except Exception as e:
        print(f'    warn: state promotion failed: {e}')
    print(f'  ok  {name}: {len(rows)} epochs, task_id={task.id}')
    return task.id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='e2e_calib/calib')
    ap.add_argument('--pattern', default='experiments/tss4_*',
                     help='glob pattern (relative to repo root) for experiment dirs')
    args = ap.parse_args()
    root = Path('/home/hiro/git/e2e_calib')
    dirs = sorted(d for d in root.glob(args.pattern) if d.is_dir())
    print(f'found {len(dirs)} dirs matching {args.pattern}')
    for d in dirs:
        try:
            import_one(d, args.project)
        except Exception as e:
            print(f'  ERR {d.name}: {e}')


if __name__ == '__main__':
    main()
