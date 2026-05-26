"""Backfill: attach experiments/<name>/best_model.pt to its existing ClearML
task as an OutputModel.

Differs from import_local_experiments.py in that we DO NOT create a new task —
we look up the existing task by name and just register the weights against it.
Use this for runs that already exist in ClearML (scalars, comments, vis) but
were trained before we wired OutputModel uploads into the loop.

Usage:
  /home/hfunaya/.pyenv/versions/3.10.4/bin/python \
      scripts/serving/backfill_clearml_outputmodel.py \
        --project 'e2e_calib/calib' \
        --pattern 'km_wv_wm_n4_img128_ml_dense_pe_dgx2_200ep_v2'

  # all experiments matching pattern (glob applied to dir name):
  /home/hfunaya/.pyenv/versions/3.10.4/bin/python \
      scripts/serving/backfill_clearml_outputmodel.py \
        --project 'e2e_calib/calib' --pattern '*'

If --create-if-missing is passed, creates a new Task for runs that don't
already have one (delegating to import_local_experiments.import_one).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from clearml import OutputModel, Task


def attach_one(exp_dir: Path, project: str, create_if_missing: bool = False,
               force: bool = False) -> str | None:
    name = exp_dir.name
    bm = exp_dir / 'best_model.pt'
    if not bm.exists():
        print(f'  skip {name}: no best_model.pt')
        return None

    task = Task.get_task(project_name=project, task_name=name)
    if task is None:
        if create_if_missing:
            from scripts.serving.import_local_experiments import import_one
            return import_one(exp_dir, project)
        print(f'  skip {name}: no ClearML task (use --create-if-missing)')
        return None

    # Idempotency: skip if a usable best_model.pt is already attached, both as
    # OutputModel (non-uploading_file URL) and as Artifact.
    if not force:
        out_models = task.get_models().get('output', [])
        has_model = any(m.name == f'{name}_best'
                         and m.url and 'uploading_file' not in m.url
                         for m in out_models)
        has_artifact = 'best_model.pt' in (task.artifacts or {})
        if has_model and has_artifact:
            print(f'  skip {name}: already attached (use --force to re-upload)')
            return task.id

    # Closed tasks (completed/stopped/failed) reject artifact edits via the
    # API. Reopen for the upload, then restore the status. We cap reopen at
    # these three terminal states — running tasks must stay running, and
    # in_progress tasks already accept edits.
    prev_status = task.status
    needs_reopen = prev_status in ('completed', 'stopped', 'failed')
    try:
        if needs_reopen:
            task.mark_started(force=True)
        # Archive any pre-existing dangling _best models (e.g. left over from
        # an earlier run that failed mid-upload, leaving url='uploading_file').
        for m in task.get_models().get('output', []):
            if m.name == f'{name}_best' and (
                    not m.url or 'uploading_file' in m.url):
                try:
                    from clearml import Model as _Model
                    _Model(model_id=m.id).archive()
                except Exception:
                    pass
        om = OutputModel(task=task, name=f'{name}_best', framework='PyTorch')
        om.update_weights(str(bm), upload_uri=None, auto_delete_file=False)
        task.upload_artifact('best_model.pt', artifact_object=str(bm),
                              delete_after_upload=False)
        print(f'  ok   {name}: attached {bm.stat().st_size / 1e6:.1f} MB')
        return task.id
    except Exception as e:
        print(f'  fail {name}: {e}')
        return None
    finally:
        if needs_reopen:
            try:
                if prev_status == 'failed':
                    task.mark_failed(force=True)
                elif prev_status == 'stopped':
                    task.mark_stopped()
                else:
                    task.mark_completed()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='e2e_calib/calib')
    ap.add_argument('--pattern', default='*')
    ap.add_argument('--exp-root', type=Path,
                    default=Path(__file__).resolve().parents[2] / 'experiments')
    ap.add_argument('--create-if-missing', action='store_true',
                    help='if no ClearML task matches the exp name, create one '
                         'via import_local_experiments.import_one (parses '
                         'train.log scalars too).')
    ap.add_argument('--force', action='store_true',
                    help='re-upload even if best_model.pt is already attached.')
    args = ap.parse_args()

    dirs = sorted(p for p in args.exp_root.glob(args.pattern) if p.is_dir())
    print(f'[backfill] {len(dirs)} dirs match {args.exp_root}/{args.pattern}')
    n_ok = n_skip = 0
    for d in dirs:
        r = attach_one(d, args.project, create_if_missing=args.create_if_missing,
                        force=args.force)
        if r is None:
            n_skip += 1
        else:
            n_ok += 1
    print(f'[backfill] done — attached {n_ok}, skipped {n_skip}')


if __name__ == '__main__':
    main()
