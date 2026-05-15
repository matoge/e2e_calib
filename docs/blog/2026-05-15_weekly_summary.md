# Weekly 3P (2026-05-09 → 2026-05-15)

E2E_CALIB σ-net. Progress / Problem / Plan.

---

## Progress

### ★ σ-net on **TMPOC production data** (kamikado unlock)

Until this week σ-net was trained only on public datasets (ZOD, Waymo,
PandaSet, nuScenes, DDAD, AV2). **This week it ran end-to-end on TMPOC
production-vehicle data** (front cam + VLS-128, kamikado provided). The data
is the gate that turns σ-net from a paper trick into something deployable on
the cars we actually ship.

| | |
|---|---|
| cache | `tss4_v3_tiled` (V3 tile cache, 4-ch intensity), built from TMPOC raw |
| frames | 795 → train 716 / val 79 (frame-level split, seed=42, no scene leak) |
| instances | train 229,120 / val 25,280 (post tile-expansion) |
| model | ConvNeXt + frustum + deform_sl + 4-layer cross-attn |
| training | 100 ep, batch=256, RTX 5080, bf16, 198 min |
| **best val NLL** | **2.1348** (ep 100, monotonic descent) |

`obj NLL = 0` because TMPOC cache has no obj annotations → bg-only NLL = pt
NLL. ZOD baseline on the same arch sits ~1.7 with 10× more frames; with a
TMPOC drive of equivalent frame budget the gap should close.

Sample viz (left = tile location on full frame, red box = crop; right = tile
crop with LiDAR projection):

![val 00](../assets/weekly_2026-05-15/val_00_idx023682.png)
![val 01](../assets/weekly_2026-05-15/val_01_idx019684.png)
![val 02](../assets/weekly_2026-05-15/val_02_idx001577.png)
![val 03](../assets/weekly_2026-05-15/val_03_idx031377.png)

(σ-ellipse overlay needs a separate viz pass on `best_model.pt` — the
training-time viz preset only shows tile + point projection.)

[full config](../../experiments/tss4_20260514_intensity_4ch_100ep_framesplit/config.py)

### Joint training across public + internal (3 caches)

Now training a single σ-net on **ZOD + TMPOC + PandaSet** simultaneously
(1.56M train instances, `rep_strategy=nearest_cam`, os=4 per cache,
`val_every=5`). Run `zod_tss4_ps_20260515_joint_repnear_os4_50ep_lrmin1e6`:

| ep | train | val | save |
|---:|---:|---:|---|
| 1 | 5.462 | 5.079 | ★ |
| 5 | 4.287 | 4.152 | ★ |
| 7 | 3.940 | (val every 5) | (in flight) |

`lr_min` raised 1e-7 → 1e-6 vs last run (the old run plateaued at val 3.211
with the schedule freezing for 9 ep before any movement). 50ep ETA ~14h.

### Infra wins (enabled the above)

- **ClearML server move + fileserver patch.** Legacy ministar died →
  server now lives on the new LAN IP, and the `POST /<path>` fileserver patch
  (CF Tunnel forwards `/files` as a path prefix → default container 405's)
  is persisted via docker-compose bind-mount, surviving container recreate.
- **Cache → ClearML Datasets backup.** PandaSet finalized (22 GB); ZOD
  in-flight (50 GB landed of ~128 GB). CF Tunnel 100 MB body cap was killing
  uploads with `413 Payload Too Large`; bypass via
  `CLEARML_FILES_HOST=http://192.168.1.9:18081` to hit fileserver port direct
  on LAN.
- **Waymo V3 tile cache (4-ch intensity).** 595 / 798 segs done (75%),
  ~20 GB written, finishes tonight. Joins joint training as the 4th cache.

<div style="page-break-after: always;"></div>

---

## Problem

### My dev PC is broken

Primary dev machine is down. All this week's training/infra ran on backup
hardware. Until repaired/replaced, throughput on dev tasks (model
iteration, viz pipeline, paper-prep work) is bottlenecked on whatever I can
spin up on other boxes.

### HEATRUN box is failing

The HEATRUN cache-build/long-run host is showing signs of imminent failure
(intermittent freezes / disk errors / unclear root cause yet). It is
currently running the Waymo V3 build — if it dies before ~02:00 tonight, the
build resumes from `_done_<seg>.flag` markers on a different host, but we'll
lose a half-day. Need to triage HEATRUN this weekend before queueing the
WOVEN rebuild.

### LAUNCHBOX cannot be exposed as a service

LAUNCHBOX's IP is not exposable externally (network policy), so the
σ-net demo / inference-as-a-service path we'd planned to host on it is
**blocked for general access**. Options:
1. Move the demo to a host with a public IP (yokohama1 is already serving
   `clearml.budda.site` via CF Tunnel — same trick can host the demo).
2. Run LAUNCHBOX behind an internal-only URL and document the VPN/tunnel
   step for users (degrades UX).
3. Drop the LAUNCHBOX path entirely and consolidate on yokohama1 +
   Cloudflare Tunnel.

Leaning toward option 1: yokohama1 has the compute and the existing TLS/CF
config; moving the demo there avoids the IP-exposure issue without
introducing a new VPN hop. Confirm with networking before committing.

<div style="page-break-after: always;"></div>

---

## Plan (next week)

1. **WOVEN cache rebuild from Loom raw.** ClearML lost the data zip chunks
   (only state.json metadata survived the ministar death). Source data is on
   Loom-team PC; rebuild from raw frames + LiDAR using V3 tile builder.
   Adds 4th internal sensor stack to joint training.
2. **Waymo cache → 5-cache joint training.** Public + internal (ZOD + Waymo
   + PandaSet on the public side, TMPOC + WOVEN on the internal side).
   Tests whether public scale generalizes the internal-stack σ predictions.
3. **TMPOC σ-ellipse viz.** Run a separate viz pass on `best_model.pt` to
   render covariance ellipses — needed to ship the result to non-numeric
   stakeholders.
4. **Cross-frame σ-net (start).** First milestone: VCPE + KV concat with
   UV-emb-only query path, smoke-trained on ZOD pair frames. This is the
   foundation for learning-based VO + BA on TMPOC data.
5. **HEATRUN triage + LAUNCHBOX → yokohama1 demo migration** (infra debt).
6. **Finish current 50ep joint** (`os=4, lr_min=1e-6`); compare end-of-tail
   val vs the `os=1` baseline; decide schedule for the next run.
