"""Curate ZOD Frames by ego motion: keep straight + slow + low-accel snapshots.

ZOD has 100K single-frame snapshots, but most include turns/intersections/accel.
For calib BA validation we want clean kinematic conditions where cam-LiDAR time
offset wouldn't induce projection artifacts (constant heading + speed → no time
offset effect on uv residuals).

For each frame, ego_motion.json provides ~22 velocity/accel/angular_rate samples
spanning ~2s around the core shutter timestamp.

Note: `angular_rates` field is high-freq IMU output with vibration noise; even
the per-frame mean over 22 samples returns implausibly large values (median ~17
deg/s, max 3000+ deg/s on z-axis). Heading-change-via-velocity is the reliable
signal: median 0.04 deg/s, p95 3 deg/s — physically plausible.

Computed signals (per frame):
- speed_kmh   : mean ||velocity[xy]|| × 3.6
- yaw_dps     : |Δheading| / dt_total  where heading = atan2(vy, vx) (unwrapped)
- accel_long  : |d(speed)/dt| mean (along-track accel via velocity derivative)

Filter (defaults, ~76% of ZOD passes):
  10 ≤ speed ≤ 80 km/h   (skip parked + extreme highway)
  |yaw_rate| < 1.0 deg/s (straight)
  |accel|    < 2.0 m/s²  (no aggressive accel/brake)

Output: tab-separated file (frame_id, speed, yaw, accel) for feeding to
build_zod_v3.py as a `frame_filter` set.
"""
import argparse, json, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT_DEFAULT = Path("/mnt/nvme6t/zod/frames/single_frames")


def stats(frame_dir: Path):
    """Per-frame (frame_id, speed_kmh, yaw_dps, accel_along_ms2) or None."""
    em_path = frame_dir / "ego_motion.json"
    if not em_path.exists():
        return None
    try:
        em = json.load(open(em_path))
        vel = np.array(em["velocities"])
        ts = np.array(em["timestamps"])
        if len(vel) < 3:
            return None
        sp_xy = np.linalg.norm(vel[:, :2], axis=1)
        speed_kmh = float(sp_xy.mean() * 3.6)
        # yaw rate from heading change
        headings = np.arctan2(vel[:, 1], vel[:, 0])
        headings = np.unwrap(headings)
        dt_total = float(ts[-1] - ts[0])
        if dt_total <= 0:
            return None
        yaw_dps = float(abs(np.degrees(headings[-1] - headings[0]) / dt_total))
        # along-track accel from velocity derivative
        accel_along = float(abs(np.gradient(sp_xy, ts).mean()))
        return (frame_dir.name, speed_kmh, yaw_dps, accel_along)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT_DEFAULT),
                    help="ZOD single_frames root")
    ap.add_argument("--out", default="/mnt/nvme6t/zod/frames/curated.txt",
                    help="output tsv: <frame_id>\t<speed>\t<yaw>\t<accel>")
    ap.add_argument("--speed-min", type=float, default=10.0)
    ap.add_argument("--speed-max", type=float, default=80.0)
    ap.add_argument("--yaw-max", type=float, default=1.0,
                    help="max |yaw rate| in deg/s")
    ap.add_argument("--accel-max", type=float, default=2.0,
                    help="max |along-track accel| in m/s²")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.root)
    frame_dirs = sorted(root.iterdir())
    print(f"total frames: {len(frame_dirs)}")

    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for s in ex.map(stats, frame_dirs, chunksize=200):
            if s is not None:
                results.append(s)
    print(f"done {time.time() - t0:.1f}s, valid={len(results)}")

    arr = np.array([(r[1], r[2], r[3]) for r in results])
    sp, yaw, ac = arr[:, 0], arr[:, 1], arr[:, 2]
    print(f"speed (km/h): med={np.median(sp):.1f}  p95={np.percentile(sp,95):.1f}")
    print(f"yaw (deg/s): med={np.median(yaw):.3f}  p95={np.percentile(yaw,95):.3f}")
    print(f"accel (m/s²): med={np.median(ac):.3f}  p95={np.percentile(ac,95):.3f}")

    mask = ((sp >= args.speed_min) & (sp <= args.speed_max) &
            (yaw < args.yaw_max) & (ac < args.accel_max))
    keep = [r for r, m in zip(results, mask) if m]
    print(f"\nfilter: speed [{args.speed_min},{args.speed_max}] "
          f"yaw<{args.yaw_max} accel<{args.accel_max} → "
          f"{len(keep)}/{len(results)} ({100*len(keep)/len(results):.1f}%)")

    out = Path(args.out)
    with open(out, "w") as f:
        for r in keep:
            f.write(f"{r[0]}\t{r[1]:.2f}\t{r[2]:.3f}\t{r[3]:.3f}\n")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
