"""Publish a trained best_model.pt to the ClearML Model Registry with tags.

Used as the training→serving handoff: when an experiment finishes and you
want the calibration server to pick it up, run:

    python -m scripts.serving.tag_model \
        --ckpt experiments/<name>/best_model.pt \
        --name calib_best \
        --project e2e_calib/calib \
        --tag latest --tag prod

The server polls for `tag=latest` (configurable). Adding `prod` flips
production traffic; remove `prod` from the previous model first to keep
exactly one production candidate (the registry doesn't enforce
uniqueness — that's a policy on us).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from clearml import OutputModel, Task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, type=Path,
                    help='path to best_model.pt')
    ap.add_argument('--project', default='e2e_calib/calib')
    ap.add_argument('--name',    default='calib_best',
                    help='Model name; server resolves via this')
    ap.add_argument('--tag',     action='append', default=[],
                    help='repeat for multiple tags, e.g. --tag latest --tag prod')
    ap.add_argument('--comment', default='',
                    help='free-text note attached to the model')
    ap.add_argument('--demote', action='append', default=[],
                    help='other Model names to strip the same tags from '
                         '(promote-after-demote pattern; pass --tag prod '
                         '--demote old_calib_best to swap prod cleanly).')
    args = ap.parse_args()

    if not args.ckpt.is_file():
        sys.exit(f'ckpt not found: {args.ckpt}')

    # Demote step (strip overlapping tags from other models).
    if args.demote and args.tag:
        from clearml import Model as _M
        for old_name in args.demote:
            for m in _M.query_models(project_name=args.project,
                                      model_name=old_name, max_results=50):
                cur = set(m.tags or [])
                kept = sorted(cur - set(args.tag))
                if kept != sorted(cur):
                    m.tags = kept
                    print(f'demoted {old_name} id={m.id}: tags {sorted(cur)} → {kept}')

    # Lightweight Task just to host the OutputModel registration. Use a
    # 'utility' tag so it doesn't pollute the experiment dashboard.
    task = Task.init(project_name=args.project, task_name=f'tag/{args.name}',
                      task_type=Task.TaskTypes.custom, output_uri=True,
                      auto_connect_frameworks=False, auto_resource_monitoring=False,
                      reuse_last_task_id=False)
    om = OutputModel(task=task, name=args.name,
                      framework='PyTorch',
                      tags=args.tag,
                      comment=args.comment or f'tagged from {args.ckpt}')
    om.update_weights(weights_filename=str(args.ckpt), upload_uri=True)
    print(f'registered model id={om.id}  name={args.name}  tags={args.tag}')
    print(f'URL: {task.get_output_log_web_page()}')
    task.close()


if __name__ == '__main__':
    main()
