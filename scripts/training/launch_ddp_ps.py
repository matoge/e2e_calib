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
  LDP_MIXED_PRECISION     (default bf16; A100 ok, V100 should set fp16)
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


def _parse_known(argv):
    """Split off our launcher-specific flags; pass everything else through."""
    num_gpus = None
    mp = os.environ.get("LDP_MIXED_PRECISION", "bf16")
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
        else:
            passthrough.append(a); i += 1
    if num_gpus is None:
        num_gpus = int(os.environ.get("LDP_NUM_GPUS", "4"))
    return num_gpus, mp, target, passthrough


def _default_cache_if_missing(passthrough):
    """If caller did not supply --cache, inject the dgx1 default cache path.

    Rationale: the ClearML UI should be able to re-run this task on dgx1
    without remembering to pass --cache. If you submit from dgx2 the default
    dataset path is different, but the submit script injects the right one
    per-host before this shim is hit, so we only step in when it's empty.
    """
    if any(a == "--cache" or a.startswith("--cache=") for a in passthrough):
        return passthrough
    default = os.environ.get(
        "LDP_DEFAULT_CACHE", "/home/hfunaya/cache/pandaset_v3_full"
    )
    if Path(default).exists():
        print(f"[launch_ddp_ps] --cache not given; injecting {default}", flush=True)
        return passthrough + ["--cache", default]
    print(
        f"[launch_ddp_ps] WARN: no --cache given and default {default} does not "
        "exist. Training script will fall back to its own default and likely "
        "FileNotFoundError.",
        flush=True,
    )
    return passthrough


def main():
    argv = sys.argv[1:]
    num_gpus, mp, target, passthrough = _parse_known(argv)
    passthrough = _default_cache_if_missing(passthrough)

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
