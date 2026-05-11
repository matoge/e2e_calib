# ZOD setup — download → cache → train

Reproduces the home setup on a fresh workstation. End-to-end gets you a
curated tile cache of ZOD Frames + a CalibNet training run.

Estimated wall-clock (on 2 Gbps fiber + 12-core box):
- Download lidar tarballs (~360 GB): **30-60 min** (7 parallel × Dropbox cap)
- Download images (~50 GB):           **5-10 min**
- Extract tarballs (~410 GB → ~410 GB): **20-40 min** (parallel tar)
- Cache build (100K frames, tile mode, motion filter): **30 min**
- Train 50 ep (zod_tile_clean_50ep, single GPU): **2-3 hr**

## 0. Repo sync

```bash
cd /path/to/e2e_calib
git pull origin main
# Latest critical fixes: 611c02d (SDK MC), f875bbc (pickle fix in builder)
```

## 1. Dropbox credentials

```bash
# Get a short-lived token from https://www.dropbox.com/developers/apps
# (4 hour validity — generate fresh before each session if expired)
export DBX_TOKEN=sl.u.AGc...
# Persist in ~/.bashrc if convenient:
echo "export DBX_TOKEN=$DBX_TOKEN" >> ~/.bashrc
```

## 2. Download

```bash
# Choose a 500GB+ disk location
ZOD_ROOT=/mnt/your_ssd/zod
mkdir -p $ZOD_ROOT

# 10 core lidar tarballs (~37 GB each, ~360 GB total)
bash scripts/preprocessing/download_zod_dropbox.sh $ZOD_ROOT

# Also need image tarballs (separate Dropbox path - manual or below):
# images_dnat_000000_049999.tar.gz (25 GB)
# images_dnat_050000_099999.tar.gz (23 GB)
# annotations.tar.gz, infos.tar.gz, mini.tar.gz
# These came from the same Dropbox folder; if your account has access:
for name in \
    annotations.tar.gz infos.tar.gz mini.tar.gz \
    images_dnat_000000_049999.tar.gz \
    images_dnat_050000_099999.tar.gz; do
    out="$ZOD_ROOT/$name"
    [ -f "$out" ] && [ $(stat -c%s "$out") -gt 100000000 ] && continue
    api_arg=$(printf '{"url":"https://www.dropbox.com/scl/fo/q81qqpiqygaeys7mppgoe/ABMW5G9RLSH6wqncsW8zY34/single_frames?rlkey=ocr9n0gq3u083zj8sn1yo1ak6","path":"/%s"}' "$name")
    curl -sX POST https://content.dropboxapi.com/2/sharing/get_shared_link_file \
        -H "Authorization: Bearer $DBX_TOKEN" \
        -H "Dropbox-API-Arg: $api_arg" --output "$out"
done
```

## 3. Extract

```bash
# Tarballs unpack into $ZOD_ROOT/frames/single_frames/{fid}/...
mkdir -p $ZOD_ROOT/frames
for tf in $ZOD_ROOT/lidar_velodyne_core_*.tar.gz \
          $ZOD_ROOT/images_dnat_*.tar.gz \
          $ZOD_ROOT/annotations.tar.gz \
          $ZOD_ROOT/infos.tar.gz; do
    setsid bash -c "tar xzf '$tf' -C $ZOD_ROOT/frames/ && touch '$tf.done'" </dev/null &>/dev/null &
done
wait  # for parallel tar
ls $ZOD_ROOT/frames/single_frames/099000/lidar_velodyne/ | head  # sanity
```

## 4. Cache build (curated)

```bash
# Filter: yaw_rate < 0.1°/s (pose-based, world-frame) + speed < 20 km/h
# → ~8K clean frames, SDK motion compensation applied per-frame.
python scripts/preprocessing/build_zod_v3.py \
    --src $ZOD_ROOT/frames \
    --out /mnt/your_ssd/cache/zod_v3_tiled_clean \
    --tile \
    --tile-w 512 --tile-h 512 --tile-stride 384 \
    --max-yaw-rate 0.1 \
    --max-speed-kmh 20 \
    --workers 12
```

Tunable filters (`--max-yaw-rate`, `--max-speed-kmh`):

| yaw threshold | speed threshold | frames retained |
|---|---|---|
| 0.1°/s | 20 km/h | ~8K   (saturated, recommended) |
| 0.1°/s | 50 km/h | ~17K  (needs MC, MC works) |
| 0.5°/s | 30 km/h | ~16K  (more turning allowed) |
| 1.0°/s | 80 km/h | ~50K  (much looser, includes mild turns) |

## 5. Train

```bash
# Single GPU. For dual-GPU, see scripts/training/launch_ddp_ps.py.
CUDA_VISIBLE_DEVICES=0 setsid nohup python scripts/training/train_ps_v3.py \
    --name zod_tile_clean_50ep \
    --cache /mnt/your_ssd/cache/zod_v3_tiled_clean \
    --epochs 50 \
    --rot-deg 1.5 --t-m 0.6 \
    --workers 12 \
    --min-crop-px 128 --max-crop-px 384 \
    --batch-size 128 \
    --n-layers 4 --img-size 128 \
    --convnext --deform-mode sl \
    --val-size 8000 \
    --oversample 1 \
    --lr 0.0003 --lr-min 1e-7 \
    > /tmp/zod_clean_train.out 2>&1 < /dev/null &
disown
```

Monitor: `tail -F experiments/zod_tile_clean_50ep/train.log`

## Notes

**Dropbox token refresh.** The `sl.` prefix tokens expire after 4 hours.
For longer-running pipelines, regenerate before each session or implement
the refresh-token OAuth flow (separate DBX_APP_KEY/DBX_APP_SECRET needed).

**Disk space.** Tarballs (~410 GB) + extracted (~410 GB) + cache (~10 GB)
= 830 GB if you keep tarballs. Delete tarballs after extract verified.

**Yaw filter.** Use pose-based (world-frame) yaw rate. The `velocities`
field is body-frame, so `atan2(vy,vx)`-based yaw is broken (always near 0,
filters in roundabout turners). The `angular_rates` IMU field is too
noisy (median ~17°/s on dataset, max 3000°/s). `_frame_motion_metrics`
in `build_zod_v3.py` uses the correct pose-based method.

**Motion compensation.** Uses ZOD SDK `motion_compensate_pointwise`.
Output is in LIDAR frame (compensated to target_ts); the cache builder
applies `T_vl` externally to get vehicle frame. Don't trust the older
in-house `_ego_motion_apply` (kept for fallback only — has a frame
convention bug).
