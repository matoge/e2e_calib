"""ClearML → accelerate launch shim for PandaSet DDP training.

Why this exists:
  clearml-task runs `python <script>` directly, so the `accelerate launch
  --num_processes=N` wrapper in submit_clearml_task.sh is ignored. This file
  is what clearml-task actually invokes; it reads env vars + forwarded CLI
  args and execs `accelerate launch ... train_ps_v3_ddp.py ...`.

Expected usage (from submit_clearml_task.sh):
  python scripts/training/launch_ddp_ps.py --num-gpus 4 \
      --cache /home/hfunaya/cache/pandaset_v3_full \
      --epochs 2 --batch-size 16 ...  (any train_ps_v3_ddp.py flag)

Env overrides:
  LDP_NUM_GPUS            (default from --num-gpus; final fallback=4)
  LDP_MIXED_PRECISION     (default auto: sm_80+ → bf16, sm_70 → fp16, else no.
                           Pass 'auto' explicitly, or set to fp16/bf16/no to pin.)
  LDP_TARGET_SCRIPT       (default scripts/training/train_ps_v3_ddp.py)

Design notes:
  - We DO NOT modify train_ps_v3_ddp.py; all flags it already accepts are
    passed straight through.
  - Exit code = child accelerate launch exit code; SIGTERM is forwarded so
    ClearML agent can kill the task cleanly.
  - Prints a banner with the resolved command so the ClearML log makes the
    dispatch visible.
"""
import os
import sys
import signal
import shutil
import subprocess
from pathlib import Path


def _auto_mixed_precision():
    """Pick fp16/bf16/no by querying the visible GPU's compute capability.

    Why: V100 (sm_70) has no bf16 tensor cores → bf16 falls back to a slow
    software path (measured ~6× slower than fp16). Ampere+ (sm_80+) has
    native bf16, which has a wider exponent range than fp16 and is
    preferred for training stability. CPU-only / no CUDA → 'no'.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "no"
        major, _ = torch.cuda.get_device_capability(0)
    except Exception:
        return "bf16"  # conservative default for modern clusters
    if major >= 8:
        return "bf16"   # Ampere, Ada, Hopper, Blackwell
    if major == 7:
        return "fp16"   # Volta / Turing — bf16 tensor cores absent
    return "no"         # Pascal and older: no AMP tensor cores


def _parse_known(argv):
    """Split off our launcher-specific flags; pass everything else through."""
    num_gpus = None
    env_mp = os.environ.get("LDP_MIXED_PRECISION")
    mp = env_mp  # None means "not user-specified yet" → will auto-pick below
    target = os.environ.get(
        "LDP_TARGET_SCRIPT", "scripts/training/train_ps_v3_ddp.py"
    )
    passthrough = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--num-gpus":
            num_gpus = int(argv[i + 1]); i += 2
        elif a.startswith("--num-gpus="):
            num_gpus = int(a.split("=", 1)[1]); i += 1
        elif a == "--mixed-precision":
            mp = argv[i + 1]; i += 2
        elif a.startswith("--mixed-precision="):
            mp = a.split("=", 1)[1]; i += 1
        elif a == "--target-script":
            target = argv[i + 1]; i += 2
        elif a.startswith("--target-script="):
            target = a.split("=", 1)[1]; i += 1
        else:
            passthrough.append(a); i += 1
    if num_gpus is None:
        num_gpus = int(os.environ.get("LDP_NUM_GPUS", "4"))
    if mp is None or mp == "auto":
        mp = _auto_mixed_precision()
    return num_gpus, mp, target, passthrough


def _require_cache(passthrough):
    """`--cache` must be given explicitly. No silent host-specific fallback.

    過去: /mnt/nvme6t/... や /home/hfunaya/cache/... を自動 inject していたが、
    「ノードによって path が違うのに勝手に fallback すると別データで学習が回り
    気付けない」問題があり撤廃した (2026-05-04)。
    --cache 無しで呼ばれたら即 fail-fast。submit_clearml_task.sh 側の
    auto-inject (queue 別に default path) はここより前で走るので通常は
    hit しない。
    """
    if any(a == "--cache" or a.startswith("--cache=") for a in passthrough):
        return passthrough
    print(
        "[launch_ddp_ps] FATAL: --cache was not provided. No host-specific "
        "fallback will be applied. Pass --cache /path/to/cache explicitly "
        "(or via submit_clearml_task.sh --args '--cache ...').",
        flush=True,
    )
    sys.exit(3)


def _restore_argv_from_clearml(passthrough):
    """Restore Args/* hyperparams that clearml-task registered but the agent
    did NOT inject into sys.argv because this shim never calls Task.init().

    clearml-task --args key=value → Task.hyperparams['Args'][key] = value.
    At agent-execution time, clearml SDK will overwrite argparse values ONLY
    if the script calls Task.init() (auto_connect_arg_parser). This launcher
    is a plain Python script — no Task.init — so by default argv is empty.

    Fix: detect we're running under a clearml-agent venv-builds dir and
    reconstruct --key value (or --flag for True) for every Args/* param.
    Launcher-consumed flags (num-gpus, mixed-precision, target-script) are
    skipped; they have already been parsed via _parse_known on the CLI
    path. Everything else is forwarded to the downstream train script.
    """
    # Only attempt restoration when there are no passthrough args at all
    # (i.e. agent invoked us with just `python launch_ddp_ps.py`).
    if passthrough:
        return passthrough
    try:
        from clearml import Task  # type: ignore
    except Exception as e:
        print(f"[launch_ddp_ps] clearml SDK import failed ({e}); cannot "
              "restore argv from hyperparams", flush=True)
        return passthrough
    try:
        task = Task.init(
            project_name="_tmp_launch_ddp_ps",
            task_name="_tmp",
            continue_last_task=True,
            auto_connect_arg_parser=False,
            auto_connect_frameworks=False,
            auto_resource_monitoring=False,
            reuse_last_task_id=True,
        )
    except Exception as e:
        print(f"[launch_ddp_ps] Task.init failed ({e}); cannot restore argv",
              flush=True)
        return passthrough
    try:
        params = task.get_parameters() or {}
    except Exception as e:
        print(f"[launch_ddp_ps] get_parameters failed: {e}", flush=True)
        return passthrough
    # Launcher flags are already handled on the direct-CLI path.  If they
    # appear in hyperparams they are consumed here too via re-parse later.
    restored = []
    for full_key, val in params.items():
        if not full_key.startswith("Args/"):
            continue
        key = full_key[len("Args/"):]
        sval = "" if val is None else str(val)
        if sval == "True":
            restored.append(f"--{key}")
        elif sval == "False":
            # argparse store_true default-is-False semantics; skip.
            continue
        else:
            restored += [f"--{key}", sval]
    if restored:
        print(f"[launch_ddp_ps] restored {len(restored)} argv tokens from "
              f"ClearML Args/* hyperparams", flush=True)
    return restored


def main():
    argv = sys.argv[1:]
    num_gpus, mp, target, passthrough = _parse_known(argv)
    # If invoked by clearml-agent with no CLI args, rebuild argv from the
    # Task's Args/* hyperparams (agent 3.0.0 does not auto-inject them into
    # sys.argv for a plain-python script that lacks Task.init()).
    if not passthrough:
        restored = _restore_argv_from_clearml(passthrough)
        if restored:
            # Re-parse in case launcher-consumed flags (num-gpus,
            # target-script, mixed-precision) came through hyperparams too.
            num_gpus, mp, target, passthrough = _parse_known(restored)
    passthrough = _require_cache(passthrough)

    # Resolve repo root = parent of scripts/ (same layout on dgx1 and dgx2).
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    target_abs = (repo_root / target).resolve()
    if not target_abs.exists():
        print(f"[launch_ddp_ps] ERR: target script not found: {target_abs}",
              flush=True)
        sys.exit(2)

    accelerate_bin = shutil.which("accelerate")
    if accelerate_bin is None:
        # Fall back to python -m accelerate.commands.launch (accelerate pkg path).
        cmd = [sys.executable, "-m", "accelerate.commands.launch"]
    else:
        cmd = [accelerate_bin, "launch"]
    cmd += [
        f"--num_processes={num_gpus}",
        f"--mixed_precision={mp}",
        str(target_abs),
    ]
    cmd += passthrough

    print("[launch_ddp_ps] cwd    :", os.getcwd(), flush=True)
    print("[launch_ddp_ps] repo   :", repo_root, flush=True)
    print("[launch_ddp_ps] target :", target_abs, flush=True)
    print("[launch_ddp_ps] GPUs   :", num_gpus, " mp:", mp, flush=True)
    print("[launch_ddp_ps] cmd    :", " ".join(cmd), flush=True)
    sys.stdout.flush()

    # Run under the repo root so relative imports in train_ps_v3_ddp.py work.
    proc = subprocess.Popen(cmd, cwd=str(repo_root))

    def _fwd(signum, _frame):
        try:
            proc.send_signal(signum)
        except Exception:
            pass

    signal.signal(signal.SIGTERM, _fwd)
    signal.signal(signal.SIGINT, _fwd)
    rc = proc.wait()
    print(f"[launch_ddp_ps] child exited rc={rc}", flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
