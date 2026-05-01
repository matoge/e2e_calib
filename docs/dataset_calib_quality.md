# Lidar–Camera Calibration Quality across Four Public AD Datasets

A practitioner's note on lidar→camera projection quality for the four public datasets we use as training material for the calibration-residual network: **Waymo Open Dataset v2, nuScenes, ZOD (Zenseact Open Dataset), and PandaSet**. Every example below is rendered at the dataset's native resolution from publicly available data using each project's official toolkit.

## Summary

| Dataset | Cameras | LiDAR | Resolution / FOV | Calib Quality | Quirks |
|---|---|---|---|---|---|
| **Waymo** v2 | 5 | 64-beam custom | 1.9 MP / ~50° HFOV | **★★★★★ near-perfect** | LCP per-shot compensation + image-lidar BA. Narrow FOV → "close-up" feel. |
| **nuScenes** | 6 | HDL-32E | 1.4 MP / ~70° | **★★★★★ clean across all 6 cams** | Sparse 32-beam returns. Best multi-cam reference. |
| **ZOD** Frames | 1 (front) | **VLS-128** | **8.3 MP / 121°** | ★★★★ center clean / **edges & high-speed residuals** | Drop outer 8% pivots; filter <50 km/h. Dense long-range returns (250 m). |
| **PandaSet** | 6 | Pandar64 | 2.0 MP / ~60° | front ★★★★ / **side cams ★★** | 30–75 ms cam–lidar time offset on the 5 non-front cams. |

**Suitability for the company sensor (8.3 MP front cam + VLS-128) — pretrain order:**
1. **Waymo** — gold-standard signal quality. Pretrain backbone here.
2. **nuScenes** — multi-cam robustness blend.
3. **ZOD** — exact sensor-resolution match, residual-aware fine-tune.
4. **PandaSet** (front-only) — extra domain diversity.

---

## Waymo Open Dataset v2

5 cameras (front / front-left / front-right / side-left / side-right) plus a 64-beam custom rooftop lidar. Waymo distributes a precomputed `lidar_camera_projection.parquet` (LCP) per segment that encodes, for every range-image pixel, the target camera plus pixel-accurate (u, v). The LCP is the output of Waymo's internal pipeline that bakes in:

- Per-shot lidar timestamp ego-pose interpolation
- Mechanical-rotation rolling-shutter correction across the spinning lidar
- Joint image–lidar tie-point bundle adjustment

Replicating this quality with only the raw range image and `vehicle_pose.parquet` is **not** achievable — the LCP is the data the user-facing toolkit is meant to consume.

**Highway scene (102 km/h):**

![Waymo 102 km/h highway](images/dataset_calib/01_waymo_highway_102kmh.png)

Streetlight poles, sign edges, and lane markings line up to within a pixel even at the highest ego speed in the dataset. The trade-off is the moderate horizontal FOV (~50°) and resolution (1.9 MP), which makes scenes feel "zoomed in" — far objects fall off quickly compared to wide-FOV sensors.

**Verdict:** The reference baseline. There is no better public source for clean lidar–camera supervision signal.

---

## nuScenes

6 cameras × HDL-32E lidar (32 laser channels, ~30 k points per frame). Lower lidar density than the others, but the calibration is uniformly stable across cameras.

**Front camera:**
![nuScenes front cam](images/dataset_calib/04_ns_front.png)

**Back camera:**
![nuScenes back cam](images/dataset_calib/05_ns_back.png)

**Front-left camera:**
![nuScenes front-left cam](images/dataset_calib/06_ns_front_left.png)

Synchronization and extrinsics are well-fitted on every camera. Closest to Waymo in calib hygiene; the only catch is the relatively sparse return density at long range (32-beam vs. 64+ on Waymo / 128 on ZOD).

**Verdict:** Multi-camera baseline. Use it to teach the network that the residual prediction is camera-position-invariant.

---

## ZOD (Zenseact Open Dataset)

The **only public dataset that ships with VLS-128**, the same Velodyne lidar as the company's production stack. One front camera (8.3 MP, 121° HFOV) + the VLS-128 with its full 250 m return range. Sensor-resolution and lidar-density parity with company data are unique to ZOD.

### Important toolkit gotcha

The ZOD SDK exposes both `frame.compensate_lidar()` and `motion_compensate_scanwise()`. **Despite the "scanwise" name, both apply only `core_timestamp` block compensation** — they do *not* interpolate ego pose at the per-shot timestamps stored in `lidar_data.timestamps`. With a 115 ms VLS-128 sweep, block compensation leaves several pixels of residual at 50 km/h.

The right helper is `motion_compensate_pointwise`, which interpolates ego pose at every shot's timestamp:

```python
from zod.utils.compensation import motion_compensate_pointwise
cam_ts = frame.info.camera_frames['front_dnat'][0].time.timestamp()
pc = motion_compensate_pointwise(
    frame.get_lidar()[0],
    frame.ego_motion,
    frame.calibration.lidars[Lidar.VELODYNE],
    target_timestamp=cam_ts,
)
```

After this, ZOD projection quality is competitive with Waymo's LCP across the central image region.

**City driving (~30 km/h):**

![ZOD city, pointwise compensated](images/dataset_calib/02_zod_pointwise_clean.png)

Streetlights, signs, and fences sit cleanly on the lidar return. The VLS-128's 250 m range puts dense points on the horizon — far past where Waymo's lidar (~75 m practical) drops off.

**Arterial road, ~60 km/h, night, wet pavement:**

![ZOD 60 km/h arterial residual](images/dataset_calib/03_zod_arterial_residual.png)

The center of the frame is fine, but the outermost left/right edges drift by 2–3 px. The likely root cause is **insufficient precision in the published intrinsics** (KB four-coefficient fit done with checkerboards skewed toward image center), rather than a Kannala-Brandt model class limitation. Real lenses are well-fit by KB *if the coefficients are tight*; ZOD's are loose at extreme off-axis angles.

### Practical filters for ZOD

- **Drop the outer 8% of pivot positions** (`edge_margin_frac=0.08`) so training never anchors crops in the residual-heavy ring.
- **Filter to ≤50 km/h** during training. The full 100 k frames split roughly 65/35 between city/arterial (where the filter passes) and highway (where it drops).

**Verdict:** Domain matcher. With pointwise compensation + edge-margin + speed filter, ZOD pulls its weight as the closest public proxy to the company sensor.

---

## PandaSet

6 cameras × Pandar64 lidar. Front camera quality is in line with the others, but a **30–75 ms cam–lidar capture-pipeline offset on the five non-front cameras** — confirmed during V3 cache construction — produces visible disagreement between lidar returns and image content.

**front_camera (reference):**
![PandaSet front cam](images/dataset_calib/07_pandaset_front.png)

Ground plane, signs, and building edges line up. Comparable to the other "good" cameras above.

**back_camera (time-offset visible):**
![PandaSet back cam](images/dataset_calib/08_pandaset_back.png)

Lidar points sit a few pixels off the image content — most visible on the road surface markings. The cause is the capture box's cam–lidar synchronization pipeline, not a bad extrinsic.

**right_camera:**
![PandaSet right cam](images/dataset_calib/09_pandaset_right.png)

Same pattern in the vertical direction.

**Verdict:** Use **front_camera only** for calib-residual learning. The five side cams can pollute the training signal with a fixed time-offset bias that the network learns to "compensate" toward, hurting transfer.

---

## What this means for training

```
[Waymo, 800k LCP-clean cam-frames]
       ↓ pretrain — calibration sense, all the clean signal we get
[ZOD, 100k @ <50 km/h, edge 8% dropped]
       ↓ fine-tune — sensor-resolution match, long-range residuals
[nuScenes, 240k across 6 cams]
       ↓ blend — multi-cam robustness
[PandaSet, ~100k front_camera only]
       ↓ blend — extra domain diversity
[Company "good" frames]
       ↓ final fine-tune
                ↓
       Zero-shot eval (TSS4 etc.)
```

The first two stages alone are projected to give us ~90 % of the achievable calib-residual performance; the company-data stage is for the last 10 % and for capturing the production-vehicle-specific calibration drift.

---

## Reproduction

All images above are reproducible from the public datasets using the official toolkits:

- **Waymo** projection uses the LCP parquet (`gs://waymo_open_dataset_v_2_0_0/training/lidar_camera_projection/`). See `datasets/waymo_lcp.py` and `scripts/preprocessing/build_waymo_v3.py`.
- **ZOD** projection uses `zod.utils.compensation.motion_compensate_pointwise` + `zod.utils.geometry.project_3d_to_2d_kannala`. See `datasets/zod_full.py::ZODCalibDataset.__getitem__`.
- **nuScenes** projection uses the standard `T_world_from_cam`/`K` pinhole model. Preview via the V3 cache (`scripts/preprocessing/build_nuscenes_v3.py`).
- **PandaSet** projection uses per-frame `poses.json` × `intrinsics.json`. Preview via `scripts/preprocessing/build_pandaset_full_v3.py`.

Dataset access:

| Dataset | Source | Approx. size |
|---|---|---|
| Waymo Open Dataset v2 | `gs://waymo_open_dataset_v_2_0_0` | ~7 TB full / training-only ~5 TB |
| ZOD Frames | dropbox/zod.zenseact.com (462 GB calib subset) | 462 GB |
| nuScenes | nuscenes.org (registration required) | ~300 GB |
| PandaSet | HuggingFace `georghess/pandaset` | ~44 GB |
