"""Quick read-only ClearML status check (used by watcher loop)."""
from clearml.backend_api.session.client import APIClient

TASK_ID = "e53cfe80ee1e47e9b59288b490572a87"
c = APIClient()
t = c.tasks.get_by_id(task=TASK_ID)
print("status:", t.status)
print("started:", t.started)
print("last_update:", t.last_update)
