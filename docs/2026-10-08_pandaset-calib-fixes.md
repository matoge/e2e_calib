# PandaSet: training, inference and the API on one code path (fixes and results, 2026-10-07 to 08)

## Summary

- PandaSet stage 1 (no BA) → stage 2 (with BA) now runs through the same code path for training, inference (`infer_calib.py`) and the API (`/api/eval_frame`).
- With a pose perturbed from outside (rotation ±0.5° per axis, translation ±0.2 m per axis), the median error after correction over 40 val frames is:
  - yaw 0.027°, pitch 0.029°, roll 0.065° (about 1 px on the PandaSet front camera, f = 1970 px)
  - x 0.010 m, y 0.013 m, z 0.020 m
- Given the correct pose, a similar error remains after correction (yaw 0.026°, pitch 0.039°, roll 0.063°).
- The largest errors are roll and z (forward). Val p90: roll 0.148°, z 0.075 m.

## 1. Results (ps_s2_own24_rt, real inference path)

40 frames each; absolute error after correction. Camera frame: yaw = about the y axis (down), pitch = about the x axis (right), roll = about the optical axis; x/y/z = camera-centre offset.

| | yaw | pitch | roll | x | y | z |
|---|---|---|---|---|---|---|
| val, before correction, median | 0.189° | 0.323° | 0.276° | 0.102 m | 0.082 m | 0.114 m |
| **val, after, median** | **0.027°** | **0.029°** | **0.065°** | **0.010 m** | **0.013 m** | **0.020 m** |
| val, after, p90 | 0.065° | 0.060° | 0.148° | 0.026 m | 0.030 m | 0.075 m |
| val, no perturbation, after, median | 0.026° | 0.039° | 0.063° | 0.009 m | 0.016 m | 0.018 m |
| val, no perturbation, after, p90 | 0.078° | 0.100° | 0.140° | 0.020 m | 0.033 m | 0.046 m |
| train, after, median | 0.021° | 0.020° | 0.050° | 0.010 m | 0.013 m | 0.010 m |

One val frame through the API (`/api/eval_frame`) with three perturbations (stage-1 weights):

| perturbation | before (geodesic / distance) | after |
|---|---|---|
| none | 0.000° / 0.000 m | 0.143° / 0.043 m |
| (0.3, −0.2, 0.25)° / (0.1, −0.05, 0.15) m | 0.438° / 0.187 m | 0.159° / 0.048 m |
| (−0.45, 0.4, −0.1)° / (−0.18, 0.12, 0.05) m | 0.611° / 0.222 m | 0.104° / 0.037 m |

Val during training (stage 2; "POSE rot" in the training log = mean of the absolute values of the 3 rotation axes, not the geodesic angle):

| ep | rotation (3-axis mean) | translation | chi2r | σ |
|---|---|---|---|---|
| 1 | 0.098° | 0.045 m | 22719 | 4.5 px |
| 10 | 0.056 | 0.021 | 0.9 | 5.0 |
| 30 | 0.045 | 0.014 | 2.6 | 3.1 |

The train log (same definition) gives 0.025° at ep30. With the same evaluation code, train scenes give 0.036° and val scenes 0.046°.

## 2. What was fixed

| problem | fix | file |
|---|---|---|
| The per-cell representative point (query) was chosen from the GT projection (since 2026-05-31), so the model could read the answer from the coordinates | choose it from the perturbed projection (`uv_off_c`) | `datasets/pandaset_full.py` |
| CalibNet2's Deformable reference point was `sigmoid(Linear(q))` and did not look at the point's own position (after training it was still 42–100 px away from the point) | `--ref-mode query`: reference point = the point's own uv. The offset is learned only by the sampling locations inside DA | `models/calibnet2.py` |
| The local encoder took 16 random points from the surrounding 3×3 cells, at most 8 per cell | `--frustum-nb own --k-per-cell 24` (default): all points of the own cell, up to 24 | `models/model_depth.py`, `train_cnd2_ddp.py` |
| In both training and val the query was always the point closest to the cell centre | `--rep-strategy random_train`: random within the cell during training only; val and inference use the point closest to the centre | `datasets/pandaset_full.py` |
| If any grid window had too few points, the whole frame was re-drawn (silently a different frame in training; in inference it crashed with "no valid window after 1024 re-rolls") | fill missing windows with a copy of a valid window and set `w_active=0` (excluded from BA) | `datasets/pandaset_full.py` |
| Val had `center_band=0.5` (only the vertical middle 50% of rows) with no recorded reason | removed; val takes windows from the whole image | `train_cnd2_ddp.py` |
| Val windows and perturbations changed every epoch (stage 1 used numpy's global RNG) | `--val-seed` (default 20261008) fixes the RNG in val `__getitem__` | `pandaset_full.py`, `train_cnd2_ddp.py` |
| ClearML's task list showed seconds (e.g. 180) as Iterations | report lr at iteration 1 right after start-up | `train_cnd2_ddp.py`, `train_grid_depth.py` |
| ClearML's reporter subprocess (fork) sometimes hung in `Task.init` and no scalars arrived | `sdk.development.report_use_subprocess: false` in `~/clearml.conf` (set on each machine) | — |
| "Converting mask without torch.bool dtype" printed in bulk on every val | a torch 2.0.0 bug (fires even for bool masks); only this warning is suppressed | `train_cnd2_ddp.py` |
| eval every 10 epochs | default changed to every 5 | `train_cnd2_ddp.py` |
| confusion between geodesic angle and 3-axis mean | `pose_error_axes` reports yaw, pitch, roll, x, y, z; the API response also has `error_before_axes` / `error_after_axes` | `scripts/inference/infer_calib.py`, `services/calib_api/server.py` |

## 3. How to train

```bash
# Stage 1 (no BA). --frustum-nb own --k-per-cell 24 is the default
python -u datasets/train_cnd2_ddp.py --cache <pandaset_v3_full> \
  --min-crop-px 256 --max-crop-px 256 --img-size 256 --grid-n 16 --batch-size 4 --workers 6 \
  --val-fraction 0.1 --scene-split --n-iter 4 --lr 3e-4 --rot-deg 0.5 --t-m 0.2 \
  --clearml --clearml-project e2e_calib/calib --no-with-ba --oversample 40 --epochs 20 \
  --cross-attn deform --ref-mode query --rep-strategy random_train --name <stage1 name>

# Stage 2 (with BA), from the best stage-1 weights
python -u datasets/train_cnd2_ddp.py <stage-1 args without --no-with-ba and --oversample> \
  --with-ba --resume-ckpt experiments/<stage1 name>/best_model.pt --start-epoch 0 --epochs 30 \
  --ba-iter 4 --ba-damping 1e-3 --ba-weight 0.05 --ba-loss-type nll \
  --ba-warmup-start 0 --ba-warmup-end 10 --name <stage2 name>
```

- Stage-2 example: `_kick_ps_s2_after_rt.sh`.
- Time (one RTX 3090): stage 1 about 3.2 min/epoch, stage 2 about 3.5–4 min/epoch.
- Environment: the last runs used the `sam3` env (torch 2.11). `sam3` has no libturbojpeg, so pass `TURBOJPEG_LIB=<path to libturbojpeg.so.0>`. The API (uvicorn, FastAPI) ran in the `neurad` env (torch 2.0). Weights saved with torch 2.11 load in 2.0.

### nuScenes + PandaSet together

`--cache` takes a comma-separated list. nuScenes (1600×900) gives 7×4 = 28 windows of 256 px and PandaSet (1920×1080) gives 8×5 = 40; the trainer pads to 40 with copies of real windows and sets `w_active=0` on the copies so BA ignores them.

```bash
./_kick_nsps_s1.sh   # ns_850x4 (850 scenes, CAM_FRONT) + pandaset_v3_full, stage 1, otherwise identical to ps_s1_own24_rt
```

Run `nsps_s1_own24_rt` started 2026-10-08 11:51: train 5,779 / val 740 frames (scene split). No results yet.

## 4. Checking inference and the API

```bash
# 40 frames through the real inference path (perturbed pose given from outside). ZERO=1 for no perturbation
CKPT_EXP=<stage2 name> N=40 python tests/test_infer_pandaset.py

# API
E2E_EXP=<stage2 name> python -m uvicorn services.calib_api.server:app --port 5092
curl -X POST localhost:5092/api/eval_frame -F image=@image.jpg -F points=@points.txt \
     -F calib=@calib.json -F rot_deg='[0.3,-0.2,0.25]' -F t_m='[0.1,-0.05,0.15]'
```

### Web page

```bash
E2E_EXP=ps_s2_own24_rt python -m uvicorn services.calib_api.server:app --host 0.0.0.0 --port 5092
# open http://<host>:5092/calibrate
```

- "Pick from PandaSet val": enter a frame index (0–399) or pick one at random, enter the perturbation (rotation ZYX in degrees, translation along camera axes in m), and press "Perturb and correct". The cache is `E2E_PS_CACHE` (default `/mnt/ssd2t/work/e2e_calib/cache/pandaset_v3_full`).
- It shows a table of the errors before/after (yaw, pitch, roll, x, y, z) and the image with LiDAR points overlaid (green = correct pose, red = perturbed pose, cyan = corrected).
- Uploading your own data (image, point cloud, calib JSON) works the same way.

The CLI takes the perturbation per axis with `--rot` and `--t`:

```bash
python scripts/inference/infer_calib.py ps_s2_own24_rt --image image.jpg --points points.txt \
    --calib calib.json --rot '[0.3,-0.2,0.4]' --t '[0.05,-0.1,0.15]'
```

### Before / after (CLI)

Pick a PandaSet val frame by index, perturb, correct and save an overlay. Left panel: before correction (red = perturbed pose, green = correct pose). Right panel: after correction (red = perturbed pose, green = correct pose, cyan = corrected; cyan is drawn on top, so where it overlaps green only cyan is visible).

```bash
python scripts/inference/infer_calib.py ps_s2_own24_rt --pandaset-val 123 \
    --rot '[0.4,-0.3,0.4]' --t '[0.1,-0.1,0.15]' --overlay out.png
```

| frame | | yaw | pitch | roll | x | y | z |
|---|---|---|---|---|---|---|---|
| val 250 (scene 015/10), perturbation rot [−0.45, 0.4, −0.1]°, t [−0.18, 0.12, 0.05] m | before | +0.399° | −0.103° | −0.451° | −0.180 m | +0.120 m | +0.050 m |
| | after | −0.025° | +0.015° | **+0.098°** | +0.007 m | +0.002 m | −0.011 m |
| val 123 (scene 042/43), perturbation rot [0.4, −0.3, 0.4]°, t [0.1, −0.1, 0.15] m | before | −0.303° | +0.398° | +0.398° | +0.100 m | −0.100 m | +0.150 m |
| | after | +0.024° | −0.045° | **+0.138°** | −0.009 m | −0.026 m | +0.011 m |
| val 250, no perturbation | before | 0 | 0 | 0 | 0 | 0 | 0 |
| | after | −0.034° | −0.001° | **+0.087°** | +0.010 m | −0.009 m | −0.020 m |

In all three cases roll is the largest residual. With no perturbation the output still moves by roll 0.087° and z 0.020 m.

val 250, perturbed:

![](_figs/2026-10-08/cli_val250_shift.jpg)

val 123, perturbed (the API's `/api/pandaset/eval_image` draws with the same function, `render_overlay`):

![](_figs/2026-10-08/cli_val123_shift.jpg)

val 250, no perturbation:

![](_figs/2026-10-08/cli_val250_zero.jpg)

### Before / after (API)

```bash
# returns the before/after overlay as PNG; errors in headers X-Error-Before / X-Error-After / X-Frame (JSON)
curl -X POST localhost:5092/api/pandaset/eval_image -F i=123 \
     -F rot_deg='[0.4,-0.3,0.4]' -F t_m='[0.1,-0.1,0.15]' -D - -o out.png

# same as JSON (per-axis errors error_before_axes / error_after_axes and projected points in overlay)
curl -X POST localhost:5092/api/pandaset/eval -F i=123 \
     -F rot_deg='[0.4,-0.3,0.4]' -F t_m='[0.1,-0.1,0.15]'

# before/after PNG for your own data
curl -X POST localhost:5092/api/eval_frame_image -F image=@image.jpg -F points=@points.txt \
     -F calib=@calib.json -F rot_deg='[0.3,-0.2,0.25]' -F t_m='[0.1,-0.05,0.15]' -o out.png
```

## 5. Open issues

- Given the correct pose, an error remains after correction (roll 0.063°, z 0.018 m).
- The dataset's candidate-point prefilter is GT projection ±64 px. Near points whose shift exceeds 64 px (0.2 m translation at 5 m depth ≈ 79 px) are dropped based on GT. The `z > 0.5` test also uses depth under the GT pose.
- Val is 5 scenes (400 frames), train 34 scenes (2,720 frames); the cache holds only 39 scenes.
- Train numbers logged during training are epoch averages of in-progress weights under random queries and shifted grids, not the val conditions.
- ClearML val visualisations now pick frames evenly across the set and the window with the most points (from the next run on).

## Related

- Synthetic-data isolation: [2026-10-08_toy-bench-ladder.md](2026-10-08_toy-bench-ladder.md) (Deformable with the reference point at the point's own position: 7.03 → 1.66 px on grid-point data)
