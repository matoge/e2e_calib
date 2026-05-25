"""Check whether BA overlay debug-images were uploaded for the watched task."""
from clearml import Task

TASK_ID = "e53cfe80ee1e47e9b59288b490572a87"
t = Task.get_task(task_id=TASK_ID)

# Try get_reported_plots first.
try:
    plots = t.get_reported_plots() or []
    print(f"get_reported_plots: {len(plots)} entries")
    ba_plots = [p for p in plots if "ba_overlay" in str(p)]
    print(f"  ba_overlay plot count: {len(ba_plots)}")
    if ba_plots:
        print("  first plot keys:", list(ba_plots[0].keys()))
        # Print some metadata
        for i, p in enumerate(ba_plots[:5]):
            try:
                print(f"  [{i}] metric={p.get('metric')} variant={p.get('variant')} iter={p.get('iter')}")
            except Exception as e:
                print(f"  [{i}] (no metric attrs): {e}")
except Exception as e:
    print(f"get_reported_plots: ERROR {e!r}")

# Try debug images via the task's reporter API.
try:
    debug_imgs = t.get_reported_debug_images_metadata() if hasattr(t, "get_reported_debug_images_metadata") else None
    print(f"debug images metadata: {type(debug_imgs).__name__ if debug_imgs is not None else 'method missing'}")
except Exception as e:
    print(f"debug images metadata ERROR: {e!r}")

# Use the events API to count debug-image events directly.
# Need to first list available debug-image metrics, then query.
try:
    from clearml.backend_api.session.client import APIClient
    c = APIClient()
    # First, find what metrics are reporting debug images.
    try:
        meta = c.events.get_task_metrics(tasks=[TASK_ID], event_type="training_debug_image")
        print("training_debug_image metrics:", meta)
    except Exception as e:
        print("get_task_metrics error:", repr(e))
    # Now query actual ba_overlay images
    try:
        res = c.events.debug_images(
            metrics=[{"task": TASK_ID, "metric": "ba_overlay"}],
            iters=10,
        )
        print("ba_overlay debug_images response type:", type(res).__name__)
        # Pull metric+variant+url info
        metrics_block = getattr(res, "metrics", None) or (res.get("metrics") if isinstance(res, dict) else None)
        if metrics_block:
            for mb in metrics_block:
                m = mb.get("metric") if isinstance(mb, dict) else getattr(mb, "metric", None)
                iters_list = mb.get("iterations") if isinstance(mb, dict) else getattr(mb, "iterations", None) or []
                print(f"  metric={m}  iterations_blocks={len(iters_list)}")
                for it in iters_list[:3]:
                    iter_n = it.get("iter") if isinstance(it, dict) else getattr(it, "iter", "?")
                    events = it.get("events") if isinstance(it, dict) else getattr(it, "events", []) or []
                    print(f"    iter={iter_n}  events={len(events)}")
                    for ev in events[:5]:
                        v = ev.get("variant") if isinstance(ev, dict) else getattr(ev, "variant", None)
                        url = ev.get("url") if isinstance(ev, dict) else getattr(ev, "url", None)
                        print(f"      variant={v}  url={(url or '')[:120]}")
    except Exception as e:
        print("ba_overlay events.debug_images ERROR:", repr(e))
except Exception as e:
    print(f"events listing ERROR: {e!r}")
