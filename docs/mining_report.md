# NLL-Mining Curriculum for Cross-Frame Residual Net

**Date**: 2026-04-24 night session
**Status**: v26 running, sentinel-mode auto-stop design validated

## Goal

Cross-frame residual net (`CalibNetCrossFrame`, 1.59M params) predicts
`(Δu, Δv, Δd, Σ_3D)` per LiDAR point reprojected into neighbouring frame.
End-goal: per-point 3D gaussians feed a Σ-weighted BA → high-precision point
cloud map. Training signal is gaussian NLL.

**Problem**: under standard random-sampling training, σ collapses to
train-residual magnitude (~1 px) while val residuals stay ~3 px → val NLL
blows up (z = err/σ ≈ 6-8×). This is a σ-calibration overfit, not μ overfit.

## Hypothesis (user-designed strategy, 2026-04-24)

Train on random pairs. When val NLL plateaus or rises (σ overfit begins),
inject ~K random val samples into the training set, rewind model weights to
a recent snapshot (before overfit took hold), reset the LR schedule, and
resume. Repeat. Stop when M consecutive migrations fail to improve a
"global best" val NLL — the model has extracted all the information it can.

**Why it should work**: σ overfit symptom = "model is overconfident on
unseen distribution". Mining forces unseen samples into train → model's σ
has to widen to cover them → calibration fixes itself. Mined samples ≡ BA
outlier-prone regions, so Σ honesty on them is directly BA-useful.

## Experiment timeline

All runs: PandaSet 39 scenes (31 train / 8 val, deterministic shuffle),
front_camera only, img 64², max_points 256, virtual_epoch 4000,
batch 32, AdamW 1e-3 → 1e-5 cosine, deform-SL, 2 intra + 2 cross layers.
`+Δd` means 7-dim output (added depth residual, 3D gaussian NLL).

| run | config delta | key numbers | notes |
|---|---|---|---|
| v10 | baseline (uv only) | val_err **2.65 px** | reference point |
| v11_base | repro of v10 | val_err 2.61 | reproducibility confirmed |
| v12_uvd | +Δd | val_err 2.61, no d_err gain | depth head didn't improve |
| v13_mix | +PandaSet+Waymo | val_err **2.42** | multi-dataset helps |
| v15_uvd_mix | +Δd+mix | val_err 2.55, val_d 0.21 m | mix + uvd |
| v17_sigmafloor | MIN_SIGMA 0.3→0.7 | val_err 2.64 | manual σ floor patch |
| v18_more_visits | virtual_epoch 20 000 (5×) | val_err **1.85** @ ep48 | OOM-killed ep52 |
| v22 | mine K=100, rewind | val_nll **2.09** | early mining PoC, σ-calib improved |
| v24 | mine K=500, rewind+LR-reset, val_pool 5000 | val_nll **2.09** | bigger migration, same best |
| v25 | `log_every=1` K=500 | val_nll **0.93** @ ep43 | **val leaked — 60% of val_pool migrated into train, NLL underestimates generalization** |
| v26 | sentinel 1000 + ∞ mining pool + auto-stop | running | real held-out eval; see below |

## Architecture of the mining loop

```
for ep in epochs:
    train one epoch (regular virtual_epoch batches)
    + train on migrated_samples (extra batches)
    val eval → v = sentinel NLL

    if v < best_val_for_mining - eps:
        best_val_for_mining = v
        stall_count = 0
    else:
        stall_count += 1

    if v < global_best_val - eps:
        global_best_val = v
        n_mig_since_global_best = 0

    if stall_count >= overfit_patience:
        K fresh samples from mining pool → migrated_samples
        next_mining_idx += K                       # sentinel+∞ mode
        rewind model to snapshot from N val-checks ago
        LR ← max, re-cosine for remaining epochs
        stall_count = 0
        n_mig_since_global_best += 1

    if n_mig_since_global_best >= stop_threshold:
        break   # data saturated
```

### Key design decisions (decided collaboratively)

1. **trigger metric = val NLL, not val err_px**.
   val_err improves long after σ has collapsed; NLL catches σ overfit first.
   User: 「NLLがはねるって話だっただろw」 — confirmed from v20 log.

2. **uniform random sampling, not NLL-weighted**.
   "Hard" by model's own judgment amplifies model bias (same pathology as
   uncertainty-sampling active learning sometimes underperforming random).
   User explicitly rejected weighting: 「難しいことはしない」.

3. **sentinel (v26+) — eval and mining must not share samples**.
   v25 caught us with 60 % of the val pool migrated into train → val NLL
   signal became "memorization NLL" not "generalization NLL". v26 fixes with
   idx `[0..sentinel_size)` = permanent held-out eval, idx
   `[sentinel_size..∞)` = mining pool (virtually infinite since `ds_val[i]`
   is an idx-seeded pure function of `i`).

4. **rewind snapshot is N val-checks BEFORE trigger**, not the best snapshot.
   The best snapshot is already at overfit-edge; rewinding there re-overfits
   immediately. User: 「もう少し前のほうがいい気がする」.

5. **LR reset with cosine restart** on each migrate.
   Without reset, migrated samples arrive at minuscule LR (v21 had
   LR ≈ 2e-4 by the time mining fired) so weights can't move enough to
   absorb them. User: 「LRスケジューラーはリセットするぐらいがいいんじゃないの」.

6. **auto-stop criterion = M migrations without global-best improvement**.
   Cleaner than a fixed epoch budget — stops precisely when the data-info
   well is dry. User: 「stop criterionが最後のオーバーフィットでいいと思う。
   もうその時点でデータ食いつくしましたで」.

## Observations in flight

- **Mining doesn't invent "hard" → it just adds diversity**. Each 500-sample
  migration reliably drops val NLL post-rewind, suggesting val distribution
  is rich enough that random sampling IS the relevant diversity. If val
  were more clustered / redundant, NLL-weighted mining would start to
  matter. User: 「データが思ったより多様で、たんにサンプルするだけで結構
  バリエーションとれるってことかな」 — confirmed.

- **val_err-based trigger (v20) fired too late**. σ had already collapsed
  while val_err was still slowly improving. NLL trigger catches it at the
  correct moment. Direct demonstration of NLL as the calibration-sensitive
  signal.

- **First-400-fixed eval (v22) is biased**. Seeing only first 400 idx
  each check hides generalization on the rest. v24 switched to full-pool
  via DataLoader for unbiased averaging (~1.8 s eval for pool 5000,
  trivial). v26 goes further: parallel DataLoader on sentinel only.

- **val→train leakage in v25 = ~0.6 of pool migrated** by ep43. val NLL
  0.93 there is largely memorization, not generalization. The sentinel
  mode in v26 eliminates this entirely by construction.

## v26 current run (in progress)

- val_pool = 1000 (sentinel, never migrated)
- mining pool = idx [1000..∞) — effectively infinite
- migrate_k = 500
- overfit_patience = 3 (noise-tolerant)
- rewind_back = 3 val-checks
- stop_no_improve_migrations = 3
- log_every = 1 (see every epoch)

### Progression (live)
```
ep0 : val_nll 5.85  (cold)
ep2 : val_nll 3.29  ← global best
ep5 : stall 3 → MIGRATE#1, rewind ep2, LR reset
ep6 : val_nll 3.22  ← new global best
ep8 : val_nll 2.84
ep9 : val_nll 2.71
ep12: val_nll 2.69  ← best after brief stall
ep14: val_nll 2.53
ep15: val_nll 2.41  ← current best
…
```

This is on **true held-out** 1000 samples (never in train). Directly
comparable to v11_base's val_err 2.65 px / estimated val_nll ~3.0.

## Next steps

1. Let v26 run — auto-stop will decide when to terminate (or fill 100 ep).
2. Repeat on **109 PandaSet scenes** (all HF) with same mining settings.
3. Then add 2nd camera (back_camera) → 39×2 cam.
4. Stretch: multi-dataset (PandaSet + Waymo-PS + ZOD when ready).
5. Validation: χ² / NEES test on σ — now that mining fixes calibration, check
   `(err/σ)²` distribution is actually χ²(2) on val. This is the BA-readiness
   acid test.

## Reference: run commands

```bash
# v26: sentinel + infinite + auto-stop, 39 scenes, 1-cam
python train_cross_frame.py \
  --name v26_sentinel_autostop --full \
  --scenes-root /mnt/nvme6t/pandaset_39 \
  --cameras front_camera \
  --epochs 100 --batch-size 32 --lr 1e-3 --log-every 1 \
  --max-points 256 --virtual-epoch 4000 \
  --crop-min 128 --crop-max 256 --baseline-min 1 --baseline-max 20 \
  --deform-mode sl --n-cross-layers 2 --n-intra-layers 2 \
  --num-workers 4 --uvd \
  --mine-val --val-pool-size 1000 --sentinel-size 1000 \
  --migrate-k 500 --overfit-patience 3 --overfit-metric nll \
  --rewind-back 3 --stop-no-improve-migrations 3

# Live web view
python monitor_web.py 5002  # http://localhost:5002/v26_sentinel_autostop
```
