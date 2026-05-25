"""Quick read-only ClearML console tail (used by watcher loop)."""
import sys
from clearml import Task

TASK_ID = "e53cfe80ee1e47e9b59288b490572a87"
N_TAIL = int(sys.argv[1]) if len(sys.argv) > 1 else 80

t = Task.get_task(task_id=TASK_ID)
log = t.get_reported_console_output(number_of_reports=400)
print(f"total_lines={len(log)}")
for line in log[-N_TAIL:]:
    print(line, end="" if line.endswith("\n") else "\n")
