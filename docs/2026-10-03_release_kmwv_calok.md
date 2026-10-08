# Release — CalibNet2 kmwv_calok 2026-10-03

Checkpoint: `experiments/kmwv_calok_s3_ba40_1003_1107/best_model.pt`

---

## Summary

- Rebuilt the LiDAR→FCM calibrator on **kamikado + 11 CAL OK woven_sequence
  seqs** (2018 fisheye frames, 3 vehicles). This replaces the 2026-09-01
  release which was trained on `woven_v3_full` (uncurated 1400 frames).
- Validation on held-out scene (val_fraction = 0.1, scene-split): **POSE
  rot median = 0.0221° / t = 7.5 mm / χ²_reduced = 1.0**. train/val NLL gap
  is tight (0.88 / 0.50) — no over-fit.
- On a true out-of-training sequence (tss4_calib_raw_01, 10-frame fuse):
  per-frame χ²/6 = 0.63 (vs 1.29 for the 0901 release), lateral residual
  13 mm → 0.5 mm (**28× tighter**).

## Validation metrics (training val, scene-split hold-out)

| metric | ep50 best | notes |
|---|---|---|
| val NLL | **0.4964** | gaussian2d_nll on pts, lower = better |
| train NLL | 0.8787 | train/val gap 0.38 → healthy |
| POSE rot (median) | **0.0221°** | 388 val frames, 40-tile fused BA |
| POSE t (median) | 7.5 mm | |
| χ²_reduced | 1.0 | sigma well-calibrated |
| σ_median | 2.40 px | in cam-FRD px |

Perturbation during val: ±0.5° yaw/pitch/roll, ±0.2 m tx/ty/tz, random each sample.

## Out-of-training sequence test

Sequence: `/mnt/ecp-perception/woven_sequence/tss4_calib_raw_01/20230612_001946/sequence=248_...1686529656324-1686529661227`
(TSS4 ip248, not in CAL OK list, not seen in training). 10 frames joint-BA.

| axis | 0901 release | **1003 release** | ratio |
|---|---|---|---|
| pitch° | — | +0.001 | — |
| yaw° | — | +0.000 | — |
| roll° | — | +0.000 | — |
| t_x mm | +1.64 | +4.57 | ≈ |
| **t_y mm** | **+13.23** | **+0.47** | **28× tighter** |
| t_z mm | −1.42 | −2.72 | ≈ |
| **χ²/6** | **1.29** | **0.63** | **2.0× lower** |
| frac PD | 1.00 | 1.00 | |

The 1003 model recovers an almost-zero residual on this seq, matching the
expectation that its shipped `setting-*.json` has an accurate rig calibration
and that the model is not biased by over-fitting to any specific vehicle.

## Training recipe

Standard single-stage: [`scripts/train_kmwv_calok.sh`](../scripts/train_kmwv_calok.sh)

```
cache          = kamikado_v3_full + woven_v3_cal_ok (frame-level LMDB)
epochs         = 50 (resume from S1 best; original 2-stage path still works)
lr             = 3e-4 → cosine → 1e-6
batch_size     = 4 × 8 GPUs = 32 global
mixed_precision= fp16 (model fwd) + fp64 (BA GN)
rot_deg        = 0.5
t_m            = 0.20
oversample     = 40 (per-cache)
crop_grid      = True (8×5 cell on 3840×2160 → 40 tiles)
share_pert     = True (one δ per frame → one fused pose)
ba_weight      = 0.05, warmup ep0→ep10 (resume path)
ba_w_source    = infohead (InfoHead2x2 drives GN weights)
```

Chain used: S1 (30 ep pts-only) → S3 (50 ep BA) on frame-level cache.
Single-stage recipe (BA warmup 10→30) also validated for reproducibility.

## Dataset

Published at `/mnt/ecp-perception/e2e_calib_dataset/`:

- `kamikado_v3_full/` — 1102 fisheye frames, 11 scenes (3.5 GB LMDB)
- `woven_v3_cal_ok/` — 2018 fisheye frames, 11 CAL OK sequences (7.4 GB LMDB)
- `README.md` — schema + build procedure

ClearML Dataset: `e2e_calib/datasets/kmwv_cal_ok` (uploading in background).

## Deployment / release process (current)

1. ✅ `scripts/train_kmwv_calok.sh` kicks training with `--clearml`
2. ✅ Train task auto-registered in ClearML project `e2e_calib/calib`
3. 🚧 Serving image `e2e-calib-serve:gpu` rebuild with new weight baked
   (currently has `kmwv_s3_ba40_512r256_0901_1344`; update to
   `kmwv_calok_s3_ba40_1003_1107/best_model.pt`)
4. 🚧 ClearML `OutputModel` for the released checkpoint
   (`scripts/serving/import_local_experiments.py` + `tag_model.py` wires this)
5. 🚧 CRON `/DATADISK2/clearml_server/data/fileserver` on heatrun →
   `/mnt/ecp-perception/clearml_backup/` for durability (single-disk risk)

## Follow-ups

- Kick a parallel training with `--rot-deg 1.0` (wider pert) for
  comparison; current 0.5° is sufficient for release.
- Verify ClearML `OutputModel` tag flow end-to-end on this ckpt.
- Add held-out sequence suite (`tss4_calib_raw_01/02/03`) to a nightly eval.
