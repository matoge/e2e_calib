"""Top-level watcher loop.

Runs the once-checker as a subprocess every 60 s for up to ``MAX_MINUTES``.
Exits 0 on success, 1 on failed task, 2 on timeout.

Driven by parent ``python3 -c "import subprocess; ..."`` so output is streamed
back via stdout.
"""
import subprocess
import time
import sys

ONCE = "/home/hfunaya/git/e2e_calib/scripts/_debug/_watch_loop_once.py"
PYBIN = "/home/hfunaya/.pyenv/versions/3.10.4/bin/python3"
MAX_MINUTES = 88  # leave 2 min headroom
INTERVAL_S = 60

t_start = time.time()
i = 0
while True:
    i += 1
    elapsed = time.time() - t_start
    elapsed_min = elapsed / 60.0
    print(f"\n========= iter {i}  elapsed={elapsed_min:.1f}min =========")
    sys.stdout.flush()
    try:
        r = subprocess.run([PYBIN, ONCE], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as e:
        print(f"once-check TIMEOUT after 120s: {e}")
        time.sleep(INTERVAL_S)
        if elapsed_min > MAX_MINUTES:
            print("WATCHER_RESULT=GIVEUP_TIMEOUT")
            sys.exit(2)
        continue
    print(r.stdout)
    if r.stderr.strip():
        print("STDERR_TAIL:", r.stderr[-600:])
    rc = r.returncode
    if rc == 0:
        print("WATCHER_RESULT=SUCCESS")
        sys.exit(0)
    if rc == 1:
        print("WATCHER_RESULT=TASK_FAILED")
        sys.exit(1)
    if elapsed_min > MAX_MINUTES:
        print("WATCHER_RESULT=GIVEUP_TIMEOUT")
        sys.exit(2)
    time.sleep(INTERVAL_S)
