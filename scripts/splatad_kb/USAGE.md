# SplatAD on a Woven Sequence — end-to-end usage

Two repos, one pipeline. `loom` produces a GS-ready sequence directory
(mask + pinhole + GICP pose), `e2e_calib` kicks SplatAD in Docker on that
directory.

```
raw woven_sequence/
        │
        ▼  loom/tools/woven_sequence_gs/  (stages 00–06)
sequence with _sam3/, _pinhole/, gicped_poses in metadata.json
        │
        ▼  e2e_calib/scripts/splatad_kb/_kick_splatad_woven.sh
SplatAD training run under /raid/home/hfunaya/splatad_woven/<NAME>/
```

## 0. Prereqs

- Docker with `--runtime=nvidia`.
- `splatad-v100:neurad` image on the host (neurad-studio + splatad fork
  of gsplat baked in). Built once outside this repo.
- `/home/hfunaya/clearml-dgx2.conf` bind-mounted at `/root/clearml.conf`
  so scalars go to `e2e_calib/splat_ad` on the DGX2 ClearML server
  (http://172.16.200.185:8082).
- A GPU with SM 7.0+ (V100 OK — the kick script pins
  `TORCH_CUDA_ARCH_LIST=7.0`).

## 1. Prepare the sequence (loom side)

The `loom/tools/woven_sequence_gs/` pipeline turns a raw
`woven_sequence/` directory into a GS-ready one. Stages:

| stage | script | output |
|---|---|---|
| 00 | `00_manifest.py` | sanity manifest of frames / labels |
| 01 | `01_sam3.py` | SAM3 dynamic-object masks (car / pedestrian / …) under `_sam3/` |
| 02 | `02_bbox_and_mask.py` | AND of SAM3 ∩ projected 3D bbox → "definitely dynamic" |
| **02b** | `02b_gicp.py` | frame-to-frame GICP → `metadata.json::gicped_poses` (movers culled) |
| 03 | `03_projection_compensated.py` | camera-delay-compensated LiDAR projection + overlay sanity |
| **04** | `04_undistort_pinhole.py` | KB fisheye → pinhole (+optional crop) under `_pinhole/` |
| 05 | `05_clearml_submit.py` | (optional) submit `simple_trainer` GS via ClearML |
| 06 | `06_visualize.py` | render sanity overlays |

Top-level launcher:

```bash
cd ~/git/loom/tools/woven_sequence_gs
./run.sh <SEQ_DIR> <VEHICLE>
# e.g.: ./run.sh /raid/home/hfunaya/woven_canary_local/canary_unilab/test01/sequence=ip708_... ip708
```

After stages 01–04 the sequence directory has:

```
<seq>/
  metadata.json               # +gicped_poses (from 02b)
  saved_annotations/*.json    # actor cuboids
  vls128_rear_axle/*.npz
  _sam3/masks_and_bbox/*.png  # per-frame keep-mask
  _pinhole/
    camera/front_camera/*.jpg # undistorted pinhole
    camera/front_camera/intrinsics.json
    recalib_pinhole.json
```

This layout is what `woven_dataparser.py` (below) consumes.

## 2. Kick SplatAD (e2e_calib side)

```bash
cd ~/git/e2e_calib
./scripts/splatad_kb/_kick_splatad_woven.sh <SEQ_DIR> <VEHICLE> [GPU=4] [MAX_STEPS=20000]
# e.g.: ./scripts/splatad_kb/_kick_splatad_woven.sh \
#         /raid/home/hfunaya/woven_canary_local/canary_unilab/test01/sequence=ip708_... \
#         ip708 4 30000
```

What the script does inside the container:

1. `pip install --no-deps clearml` (image doesn't ship it).
2. Runs `_register_woven_dataparser.py` — injects `WovenDataParserConfig`
   into `nerfstudio.configs.dataparser_configs.dataparsers` so tyro
   exposes `woven-data` as an ns-train subcommand.
3. Wraps `ns-train` with a tiny `Task.init(...)` shim so tensorboard
   scalars auto-bind to ClearML (`e2e_calib/splat_ad`).
4. `ns-train splatad --output-dir /out_parent/$NAME --experiment-name
   $NAME --max-num-iterations $MAX_STEPS --vis tensorboard woven-data
   --data /seq --vehicle $VEHICLE`.

Outputs land at `/raid/home/hfunaya/splatad_woven/$NAME/splatad/<datetime>/`:

```
config.yml
events.out.tfevents.*         # tensorboard (also mirrored to ClearML)
nerfstudio_models/step-*.ckpt
```

### Turning on pose refinement (SO3xR3)

To reproduce the DoD-2 pose-rectification result (see
`docs/splatad_ps001_pose_verification_en.md`), add
`--pipeline.model.camera-optimizer.mode SO3xR3` inside the `ns-train`
line at the end of `_kick_splatad_woven.sh`. On PandaSet the same flag
recovers ~cm translation + 0.1° rotation of GT slop and gives +3.65 dB
far-range PSNR; on Woven you get an analogous refinement on top of
GICP-init.

### If you OOM in eval

Some VLS128 frames trip the batched-repeat path in
`full_images_lidar_datamanager.py`. Run `_patch_force_n_batches_1.py`
inside the container once (it's idempotent and pins
`max_points_per_tile *= 1024`), then retry.

## 3. Existing runs (reference)

Under `/raid/home/hfunaya/splatad_woven/`:

- `splatad_ip708_30k_camopt_1420/` — 30k step, SO3xR3 ON (main
  pose-refinement run, 1.7 GB, ckpt survived).
- `splatad_ip708_0641/` — 20k step, pose frozen (baseline).
- `woven_001_smoke_*/`, `woven_dbg*/`, `woven_pylon_maskinit_smoke_*/` —
  earlier smoke / debug runs.

`config.yml` + tensorboard events + ckpt are all preserved, so
render / re-eval can be re-run without retraining.

## 4. Related files

- `_kick_splatad_woven.sh` — the Docker kick script above.
- `woven_dataparser.py` — SplatAD ADDataParserConfig for one Woven
  sequence (pinhole input, `gicped_poses` preferred, per-point LiDAR
  timestamps, actor cuboids from `label_20000` / `20010` / `20027`).
- `_register_woven_dataparser.py` — patches
  `nerfstudio.configs.dataparser_configs` to add `woven-data`.
- `_patch_force_n_batches_1.py` — OOM guard for the eval lidar
  rasterization path.
- `bake_masks.py` — bakes the `_sam3/masks_and_bbox/` PNGs into the
  training data (used by the SAM3 stage upstream).
- `docs/splatad_ps001_pose_verification_en.md` (e2e_calib) — PandaSet
  001 pose-refinement verification.
- `loom/tools/woven_sequence_gs/README.md` — the upstream pipeline.

## 5. PandaSet path (for reference)

`_kick_splatad_woven.sh` is Woven-only. For PandaSet:

- Skip `_register_woven_dataparser.py` — neurad-studio ships
  `pandaset-data` natively.
- Gzip every `.pkl` under the dataset first (the `pandaset` Python
  package hard-codes `.pkl.gz`).
- Swap the last `ns-train` line for
  `pandaset-data --data /data/pandaset --sequence 001 --cameras front`.
- Turn on `--pipeline.model.camera-optimizer.mode SO3xR3` to reproduce
  the +3.65 dB result.

Full recipe: `docs/splatad_ps001_pose_verification_en.md`.
