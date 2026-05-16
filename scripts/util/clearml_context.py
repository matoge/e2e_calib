"""ClearML context helpers — annotate tasks with WHY + RETROSPECTIVE.

Usage from a training script:

    from scripts.util.clearml_context import init_with_context, write_retrospective

    cml_task = init_with_context(
        project='e2e_calib/calib', name=cfg['name'], cfg=cfg,
        why=args.why,
        baseline={'name': 'ps_v9_lazy', 'metric': 'val_nll', 'value': 1.8141},
    )
    # ... training loop ...
    write_retrospective(cml_task, dict(
        best_val_nll=best_val,
        final_val_nll=history['va_nll'][-1],
        time_min=(time.time() - t0) / 60,
        epochs=len(history['ep']),
    ), baseline={'name': 'ps_v9_lazy', 'value': 1.8141})

Result on ClearML web UI: task.comment carries the rationale + recent commits
+ baseline reference up front, plus an observable retrospective block at the
end. Retrospective stays factual (numbers, deltas); add a 'conclusion'
keyword to inject hand-written interpretation.
"""
from __future__ import annotations

import subprocess
from typing import Optional


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=10
                                        ).decode('utf-8', errors='replace').strip()
    except Exception as e:
        return f'[git error: {e}]'


def init_with_context(project: str, name: str, cfg: dict,
                       why: str = '', baseline: Optional[dict] = None,
                       reuse_last_task_id: bool = False):
    """Create a ClearML Task with a rich comment header.

    cfg is connected as hyperparams (under 'cfg/...'). The comment captures
    rationale, baseline reference, and recent git context — everything the
    'why this run exists' question needs at glance, anchored at submit time.
    """
    from clearml import Task
    # Disable matplotlib auto-track: ClearML wraps plt.savefig and uploads each
    # figure as a Plot entry, using the figure title as the storage path. Our
    # vis titles contain ":" "→" and spaces → URL-encoded as 0x3a / %E2%86%92,
    # which the files_server returns 401 for. Result: every Plot in the UI
    # shows as a broken image. We already report vis explicitly via
    # cml_logger.report_image (clean paths, lands in Debug Samples), so the
    # auto-track is pure duplication that adds dead links.
    # Also disable pytorch auto-binding: torch.load on inst/*.pt during the
    # preflight val pass made ClearML treat every .pt as a model artifact and
    # spam "Connecting multiple input models" warnings (one per call). The
    # warnings serialize through a single-thread reporter and stalled the
    # val pass for several minutes.
    task = Task.init(project_name=project, task_name=name,
                     reuse_last_task_id=reuse_last_task_id, output_uri=True,
                     auto_connect_frameworks={'matplotlib': False,
                                              'pytorch': False})
    # Important: connect MUTATES cfg in place — on remote runs ClearML re-fills
    # the dict's keys with stored hyperparam values. Don't shallow-copy here,
    # or the caller's dict misses the re-fill.
    task.connect(cfg, name='cfg')

    lines: list[str] = []
    lines.append(f'## Why\n{why or "(no rationale provided)"}')
    if baseline:
        lines.append(f'\n## Baseline\n- **{baseline["name"]}** → '
                     f'{baseline.get("metric","val_nll")}={baseline["value"]}')
    lines.append('\n## Recent commits')
    lines.append('```\n' + _run(['git', 'log', '--oneline', '-7']) + '\n```')
    dirty = _run(['git', 'status', '--short'])
    if dirty:
        lines.append('\n## Dirty working tree at submit')
        lines.append('```\n' + dirty[:1500] + '\n```')
    task.set_comment('\n'.join(lines))
    return task


def write_retrospective(task, history: dict,
                         baseline: Optional[dict] = None,
                         conclusion: str = '') -> None:
    """Append a retrospective block to task.comment after training finishes.

    history keys (all optional):
      best_val_nll, final_val_nll, best_obj_mse, time_min, epochs
    """
    if task is None:
        return
    lines = ['', '---', '## Retrospective (auto)']
    if 'best_val_nll' in history:
        lines.append(f'- best val_nll: **{history["best_val_nll"]:.4f}**')
    if 'final_val_nll' in history and 'best_val_nll' in history:
        lines.append(f'- final val_nll: {history["final_val_nll"]:.4f}')
    if history.get('best_obj_mse') is not None:
        lines.append(f'- best obj_mse: {history["best_obj_mse"]:.3f} px')
    if 'time_min' in history:
        lines.append(f'- training time: {history["time_min"]:.1f} min')
    if 'epochs' in history:
        lines.append(f'- epochs: {history["epochs"]}')
    if baseline and 'best_val_nll' in history:
        delta = history['best_val_nll'] - baseline['value']
        sign = '↓ better' if delta < 0 else ('↑ worse' if delta > 0 else '·')
        lines.append(f'- vs {baseline["name"]} ({baseline["value"]:.4f}): '
                     f'{delta:+.4f}  {sign}')
    if conclusion:
        lines.append(f'\n## Conclusion\n{conclusion}')
    try:
        existing = getattr(task, 'comment', None) or ''
        task.set_comment(existing + '\n' + '\n'.join(lines))
    except Exception as e:
        print(f'[clearml retrospective write failed: {e}]')
