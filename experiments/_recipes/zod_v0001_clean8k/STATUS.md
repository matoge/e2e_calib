# `zod_v0001_clean8k` — ZOD curation + cache + training recipe

> Naming: `{dataset}_v{NNNN}_{tag}` so directories sort chronologically by version
> and the tag conveys the data-flow intent (here: ZOD curated ≈ 8k frames).

## Status

| Stage | Status | Artifact |
|---|---|---|
| 1. Curate frame-id list | ✅ done | `/mnt/nvme6t/zod/frames/curated_slow_y1_a2_s5_20.txt` (8975 frame ids) |
| 2. Build V3 tile cache | ✅ done | `/mnt/nvme6t/e2e_calib_cache/zod_v3_tiled_clean/` (295k tiles) |
| 3. Train 50 ep | ✅ done | `experiments/zod_tile_clean_50ep/best_model.pt` (val_nll **+1.085** @ ep50) |

Date snapshot: 2026-05-13 (curation + cache built 2026-05-11; training completed 2026-05-12 01:15).

## Why this recipe exists

ZOD has 100K single-frame snapshots. For calibration-bias detection via BA, we
want **kinematically clean** frames: slow + straight + low accel. Otherwise the
LiDAR scan-time projection error (proportional to |v · Δt|) drowns the static
calibration bias we're trying to measure.

The curation script `scripts/preprocessing/curate_zod_frames.py` filters per
frame using:
  * `speed_kmh ∈ [5, 20]`   (slow → small |v| → small motion artifact)
  * `|yaw_dps| < 1.0`       (straight, world-frame yaw rate via pose unwrap)
  * `|accel|   < 2.0 m/s²`  (no aggressive accel/brake)

Pose-based yaw is mandatory — body-frame `velocities` heading is near-0 for
ANY trajectory because the frame rotates with the vehicle. See
`memory/project_zod_curation.md` for the trap analysis.

## Reproducible chain

```bash
cd /home/hiro/git/e2e_calib

# 1) Curate. Outputs frame-id list to curated_slow_y1_a2_s5_20.txt.
python scripts/preprocessing/curate_zod_frames.py \
    --root /mnt/nvme6t/zod/frames/single_frames \
    --out  /mnt/nvme6t/zod/frames/curated_slow_y1_a2_s5_20.txt \
    --speed-min 5  --speed-max 20 \
    --yaw-max   1.0 \
    --accel-max 2.0
# → 8975 frames retained (~9% of 100K snapshots)

# 2) Build V3 tile cache using the curated list as `frame_filter`.
python scripts/preprocessing/build_zod_v3.py \
    --root /mnt/nvme6t/zod/frames/single_frames \
    --frame-filter /mnt/nvme6t/zod/frames/curated_slow_y1_a2_s5_20.txt \
    --out /mnt/nvme6t/e2e_calib_cache/zod_v3_tiled_clean \
    --workers 8
# → ~295k tiles (33 tiles × 8975 frames)
# Note: MC currently disabled (scanwise projection only); see
# project_zod_curation.md "MC bug in build_zod_v3.py" warning.

# 3) (Optional) Pack to LMDB for faster training.
python scripts/preprocessing/convert_tile_cache_to_lmdb.py \
    --cache-dir /mnt/nvme6t/e2e_calib_cache/zod_v3_tiled_clean \
    --map-size-gb 60 --workers 8

# 4) Train pixel-only CalibNet on this cache (50 ep was enough; bump for full).
python scripts/training/train_ps_v3.py \
    --name zod_tile_clean_50ep \
    --cache /mnt/nvme6t/e2e_calib_cache/zod_v3_tiled_clean \
    --epochs 50 \
    --val-every 5
# → experiments/zod_tile_clean_50ep/best_model.pt (val_nll +1.085 @ ep50)
```

## Filter retention table (pose-based yaw, see memory project_zod_curation):

| filter | retained | listfile |
|---|---:|---|
| speed [5-30] yaw<1.0 accel<2.0 | 15.1% (15K) | — |
| **speed [5-20] yaw<1.0 accel<2.0** | **9.0% (8975)** | **`curated_slow_y1_a2_s5_20.txt`** ← used here |
| speed [5-30] yaw<0.5 accel<2.0 | 18613 | `curated_v4_pose_y0.5_a2_s0_30.txt` |
| speed [5-30] yaw<0.1 accel<1.0 | 3.3% (1.5K) | `curated_v3_pose_y0.1_a1_s5_20.txt` |
| speed [5-20] yaw<0.1 accel<1.0 | 1.3% (591) | — |
| speed [5-30] yaw<0.1 accel<1.0 | 12560 | `curated_strict_y0.1_a1_s5_30.txt` |
| (body-frame yaw, BROKEN — do not use) | 76579 | `curated_v2_y1_a2_s10_80.txt` |

## Known issues

- **MC bug**: `build_zod_v3.py` `_ego_motion_apply` produces visible 5-20 px
  projection drift on poles/signs. The clean cache was built with MC disabled
  (`zod_v3_tiled_clean` is scanwise projection only). When MC is fixed the
  cache should be rebuilt; for now scanwise is acceptable for the
  narrow-FOV ZOD front DNAT.

- `zod_v3_tiled_clean_v2` (839k tiles) uses the broken body-frame yaw — DO
  NOT train on it.

## Successor recipes (when written)

- `zod_v0002_strict_y0.1` — tighter yaw filter (1.3K frames); for
  fine-grained calib bias detection where motion artifact must be < 1 px.
- `zod_v0003_mc_fixed` — same `speed-5-20 yaw-1` cut but with MC bug fixed
  in `build_zod_v3.py`; expected to lift val_nll baseline below +1.0.
