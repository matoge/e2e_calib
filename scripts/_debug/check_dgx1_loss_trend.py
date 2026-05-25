"""Honest loss trend for the running DGX1 task — bucket by step ranges."""
from clearml import Task
import re
import statistics

t = Task.get_task(task_id="ef81ab9ff02a4be2a0fe2c8b03d1f47b")
events = t.get_reported_console_output(number_of_reports=500)
out = "".join(events)

losses = [(int(m.group(1)), float(m.group(2)))
          for m in re.finditer(r"step\s+(\d+)\s+loss=\+([0-9.]+)", out)]
print(f"samples: {len(losses)}")
buckets = {}
for s, l in losses:
    b = (s // 500) * 500
    buckets.setdefault(b, []).append(l)
for b in sorted(buckets.keys()):
    vals = buckets[b]
    print(f"step {b:5d}-{b+499}: n={len(vals):2d}  "
          f"median={statistics.median(vals):7.0f}  "
          f"min={min(vals):7.0f}  max={max(vals):9.0f}")
print("--- last 8 raw lines ---")
for l in out.split("\n")[-8:]:
    print(l)
