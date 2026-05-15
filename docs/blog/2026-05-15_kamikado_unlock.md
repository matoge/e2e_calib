# σ-net on our own car — the TMPOC unlock

*Posted 2026-05-15. Weekly engineering log.*

## TL;DR

For the past 6 months our calibration-uncertainty network (σ-net) has been
chasing val NLL numbers on public datasets — ZOD, Waymo, PandaSet, nuScenes,
DDAD, AV2. Every one of those was a stepping stone, but none of them were the
sensor stack we ship. This week we crossed that line: **σ-net trained
end-to-end on TMPOC production-vehicle data** (cam + VLS-128, handed over by
kamikado-san) reaches **val NLL 2.135 in 3.3 h** on a single RTX 5080.

That's not a marginal datapoint. That's the network learning to predict
per-point calibration uncertainty on the exact cameras and exact LiDAR our
vehicles actually drive with — including the same intensity statistics, the
same lens distortion, the same mount geometry. It means the BA → odometry →
map-quality pipeline we've been building has a real entry point on company
hardware, not just on someone else's open-source car.

## Why TMPOC matters more than another public number

Public datasets train σ-nets that **look right** on public-dataset frames. They
don't transfer cleanly to our vehicles because:

- different intensity calibration (lens vignette + LiDAR reflectance handling)
- different mount drift envelope (production cars vibrate differently than
  research mules)
- different scene statistics (Tokyo vs Phoenix vs San Francisco)
- different camera intrinsics — even "the same" sensor model has per-unit
  variation we don't see in academic data

So every step toward the production pipeline has needed a "does this work on
TMPOC" gate. Until this week we couldn't even *try* — we didn't have a clean
4-ch intensity cache from a TMPOC drive. kamikado-san closed that gap.

## The setup

| | |
|---|---|
| **Data** | TMPOC production drive, front cam + VLS-128, 795 frames |
| **Cache format** | V3 tile cache (512×512 px tiles, 4-ch intensity stacked under RGB) |
| **Cache size** | 229 k train instances + 25 k val instances (post tile-expansion) |
| **Split** | frame-level, seed=42 (716 train / 79 val — no scene leak) |
| **Model** | ConvNeXt backbone → frustum point pool → deform_sl → 4-layer cross-attn → (Δuv, Σ) head |
| **Training** | 100 epochs, batch=256, RTX 5080, bf16, cosine 3e-4 → 1e-7 |
| **Time** | 198 min (3.3 h) wall clock |

Architecturally nothing changed from the public-dataset σ-net. The only thing
this week proved is that the *data* generalizes when we feed it our own. The
network shape didn't have to bend; the data was the gate.

## The result

**Best val NLL = 2.1348** at epoch 100, monotonically descending all the way:

| ep | train NLL | val NLL |
|---:|---:|---:|
| 10 | 3.1 | 2.8 |
| 50 | 2.5 | 2.3 |
| 100 | 2.33 | **2.135** |

For context: ZOD baseline on the same architecture sits around val NLL 1.7
with ~10× more frames. TMPOC reaches 2.135 with only 795 frames — once we
have a TMPOC drive with the same frame count budget as ZOD, we expect the
numbers to converge.

Sample frames live at
`experiments/tss4_20260514_intensity_4ch_100ep_framesplit/vis_ep100/`.
Each panel pair shows the tile location on the full camera frame (left,
red rectangle = the crop) and the tile crop itself with projected LiDAR
points overlaid (right). σ-ellipse overlay is not in this vis preset —
verifying that needs the next-cycle viz pass on best_model.pt.

## What this enables next

This isn't the end of the story — it's the entry point for everything we
actually want to ship:

1. **Σ-weighted bundle adjustment on TMPOC drives.** With σ-net giving us
   per-pt covariance, BA can downweight noisy measurements automatically. We
   already showed this works on PandaSet (1-step linearized BA on model
   residuals drops 5–10 px miscalibration to ≤ 1 px); now we can run the same
   loop on our own data.
2. **Cross-frame σ-net.** The single-frame σ-net is the building block for a
   2-frame variant that predicts (Δuv, Σ) for *temporal* mismatch (the
   "pose-residual net"). That's the foundation for a learning-based VO + loop
   closure. This week's TMPOC unlock means cross-frame work can start on
   production data, not on academic poses with unknown noise floors.
3. **Joint training across public + internal.** We're now running a 3-cache
   joint (ZOD + TMPOC + PandaSet, 1.56 M instances) to see whether mixing
   public scale with internal sensor specifics gives the best of both. Run
   `zod_tss4_ps_20260515_joint_repnear_os4_50ep_lrmin1e6` is in flight; first
   5 epochs already cross val NLL 4.15 with `val_every=5`.
4. **WOVEN integration.** Once Loom-team WOVEN cache is rebuilt (the previous
   upload to ClearML lost its data zip chunks during the server move and only
   the state.json metadata survived), we'll have 4 caches in the joint
   training, including a second internal stack.

## Boring infra wins that made it possible

Three things shipped in parallel under the hood, none individually exciting
but all load-bearing for the unlock:

- **ClearML server move + fileserver patch.** The legacy ministar host
  finally died. Server recovered to a new LAN IP and the `clearml-fileserver`
  `POST /<path>` route patch (CF Tunnel forwards `/files` as a path prefix →
  the container's default `POST /` returns 405) is now persisted via
  docker-compose bind-mount, so the next container recreate doesn't lose it.
- **Cloudflare 100 MB body-cap workaround for cache backups.** When uploading
  PandaSet / ZOD caches as ClearML Datasets, the default 512 MB chunks hit a
  CF Tunnel `413 Payload Too Large`. Override `CLEARML_FILES_HOST=
  http://192.168.1.9:18081` on the upload client to bypass CF and hit the
  fileserver port directly on the LAN. PS finished at 22 GB, ZOD is mid-flight.
- **Waymo V3 tile cache (4-ch intensity).** Building over 798 segments with
  4 parallel workers + `max_tasks_per_child=1` (glibc heap fragmentation
  defense) + per-seg `_done_<seg>.flag` resume markers. 75% complete as of
  this writing — expected to finish tonight and slot in as the 4th joint-
  training cache.

## What's next

| ETA | item |
|---|---|
| this week | WOVEN cache rebuild from Loom raw → 4-cache joint |
| this week | Waymo cache done → 5-cache joint (3 public + 2 internal) |
| next week | Cross-frame σ-net (VCPE + KV concat, UV-emb-only query) |
| 4 weeks | TMPOC σ-weighted BA loop closed end-to-end |

The thesis stays: one residual network — Δuv + Σ — covers calibration,
odometry, BA, and map-quality scoring. This week we got it pointed at our own
sensors for the first time. Everything downstream gets easier from here.
