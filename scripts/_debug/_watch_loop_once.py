"""One iteration of the watch loop: print status + tail and detect target lines.

Exit codes:
  0 = found both target lines (epoch summary + BA[r0.5_t0.05])
  1 = task failed
  2 = still running, target not yet seen
"""
import sys
from clearml.backend_api.session.client import APIClient
from clearml import Task

TASK_ID = "e53cfe80ee1e47e9b59288b490572a87"

c = APIClient()
t_meta = c.tasks.get_by_id(task=TASK_ID)
status = t_meta.status
print(f"status={status}")

t = Task.get_task(task_id=TASK_ID)
log = t.get_reported_console_output(number_of_reports=600)
print(f"total_lines={len(log)}")

# Search for target lines.
ep1_line = None
ba_lines = []
for line in log:
    s = line.rstrip("\n")
    if "[  1/50]" in s and "train nll" in s:
        ep1_line = s
    if "BA[r0.5_t0.05]" in s or "BA[r1.0_t0.1]" in s or "BA[r1.5_t0.2]" in s:
        ba_lines.append(s)

# Print last 25 short tail lines (filtering out the noisy docker setup blob).
def _short(s):
    return s if len(s) < 400 else s[:380] + "...[trunc]"
print("--- tail ---")
relevant = [l for l in log if not l.startswith("Executing:") and "apt-get" not in l[:40] and "deb.deb" not in l]
for line in relevant[-25:]:
    print(_short(line.rstrip("\n")))
print("--- /tail ---")

if ep1_line:
    print("FOUND_EP1: " + ep1_line)
for b in ba_lines:
    print("FOUND_BA: " + b)

if status in ("failed", "stopped", "closed"):
    print("RESULT=FAILED")
    sys.exit(1)
if ep1_line and any("BA[r0.5_t0.05]" in b for b in ba_lines):
    print("RESULT=DONE")
    sys.exit(0)
print("RESULT=WAIT")
sys.exit(2)
