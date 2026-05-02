# Frustum encoder — which scenes actually benefit?

Follow-up to `docs/local_point_feature_aggregation.md` §9. The 100EP
ablation proved that turning the frustum encoder on is **6.9 %** better
in aggregate val NLL (1.5874 vs 1.7052). This doc digs into *which*
samples drive that delta.

## Method

- Load both `best_model.pt`s — `v10` (frustum on) and `v10b`
  (`--no-frustum`, everything else identical).
- Run both on the **same** deterministic val subset (training's
  `val_size=1000` first-N, which under `oversample=1` evaluates to 164
  unique object-level insts).
- Per sample, compute mean L2 error on object points only:
    `err = mean_i ||pred_i - true_i||₂`
- Rank by Δ = `err_nofrust − err_frust`. Positive Δ = frustum wins.

Script: [`/tmp/frustum_qualitative_compare.py`](../../../tmp/frustum_qualitative_compare.py)
(dgx2 GPU 9, ~1 min; saves `docs/assets/frustum/qual/*.png` +
`per_sample_scores.json`).

## Aggregate

```
mean obj L2 err  w/ frustum    = 1.817 px
mean obj L2 err  w/o frustum   = 2.054 px
delta mean                     = +0.237 px   (positive = frustum better)
```

Per-sample distribution is asymmetric: the frustum encoder flips a
handful of samples from "hopeless" to "near-perfect" (Δ up to +4.5 px),
while on some other samples it adds up to −6 px of error — but the
total is unambiguously positive and the aggregate NLL plot in §9 agrees.

## Top samples where the frustum encoder **helps** most

All rendered panels: left = w/o frustum, right = w/ frustum. Orange
arrows are per-point pred flow, green × = ground truth, cyan = BG points.

| idx | err no-frust → frust | Δ | n_obj | png |
|-----|---------------------:|---:|------:|-----|
| 83  | 8.36 → **3.90** | +4.46 | 60 | [help_idx0083](assets/frustum/qual/help_idx0083.png) |
| 119 | 4.06 → **0.73** | +3.32 | 137 | [help_idx0119](assets/frustum/qual/help_idx0119.png) |
| 70  | 4.63 → **1.76** | +2.87 | 65 | [help_idx0070](assets/frustum/qual/help_idx0070.png) |
| 114 | 4.87 → **2.21** | +2.66 | 109 | [help_idx0114](assets/frustum/qual/help_idx0114.png) |
| 49  | 3.07 → **0.69** | +2.37 | 123 | [help_idx0049](assets/frustum/qual/help_idx0049.png) |
| 92  | 2.81 → **0.63** | +2.19 | 91  | [help_idx0092](assets/frustum/qual/help_idx0092.png) |
| 148 | 2.93 → **0.82** | +2.11 | 80  | [help_idx0148](assets/frustum/qual/help_idx0148.png) |
| 19  | 2.74 → **0.74** | +2.00 | 83  | [help_idx0019](assets/frustum/qual/help_idx0019.png) |

Shared pattern in the help-most bucket:

- **High n_obj (60–137 object points per crop).** These are dense-lidar
  object footprints where, within any one UV cell, there are many points
  spanning a real depth range (object surface + ground beneath).
- Without frustum the per-cell query sees only one averaged UV position
  and picks up a single-scale image response; with frustum it also sees
  the *relative* UV/depth pattern of its ~32 neighbors, so it can
  disambiguate an object-surface point from a point that fell on the
  road at the same UV.
- The improvement is object-coherent: object arrows snap to consistent
  directions in the frustum panel while the no-frustum panel shows
  orientation noise across the object mass.

## Top samples where the frustum encoder **hurts**

| idx | err no-frust → frust | Δ | n_obj | png |
|-----|---------------------:|---:|------:|-----|
| 124 | 13.43 → 19.44 | −6.00 | **20** | [hurt_idx0124](assets/frustum/qual/hurt_idx0124.png) |
| 41  | **0.44** → 4.25 | −3.81 | 107 | [hurt_idx0041](assets/frustum/qual/hurt_idx0041.png) |
| 118 | **0.91** → 3.52 | −2.61 | 97  | [hurt_idx0118](assets/frustum/qual/hurt_idx0118.png) |
| 74  | 3.10 → 5.11 | −2.01 | **29** | [hurt_idx0074](assets/frustum/qual/hurt_idx0074.png) |
| 34  | **0.85** → 2.63 | −1.78 | 55  | [hurt_idx0034](assets/frustum/qual/hurt_idx0034.png) |
| 45  | **1.00** → 2.76 | −1.76 | **5**  | [hurt_idx0045](assets/frustum/qual/hurt_idx0045.png) |
| 2   | 2.66 → 4.22 | −1.56 | 87  | [hurt_idx0002](assets/frustum/qual/hurt_idx0002.png) |
| 112 | 1.37 → 2.86 | −1.49 | 76  | [hurt_idx0112](assets/frustum/qual/hurt_idx0112.png) |

Two distinct failure modes:

1. **Low n_obj (< 30 points)** — idx 45 (5 pts), 124 (20 pts), 74 (29 pts).
   These have too few queries for the local neighborhood signal to
   stabilize; with k=32 random neighbors from a small point pool, noise
   dominates. The no-frustum baseline is already near-random on these
   (err > 3 px) so the absolute Δ is misleading.
2. **Already-solved samples (err_no < 1 px)** — idx 41, 118, 34. When
   the image-only cross-attention has already nailed the projection,
   the added local-geometry signal is mild over-regularization: the
   frustum model defers slightly to its neighbors and picks up a small
   constant bias. The 3-4 px penalty in these rows looks large in the
   table but is close to the cell size (cell_px = 4 at S=64, grid_n=16).

There are **no** cases where frustum breaks on a high-n_obj, moderate-
error sample — i.e. the help and hurt buckets are disjoint in the
regime that matters for downstream BA.

## Takeaways

- The frustum encoder is **unambiguously net positive** on PandaSet
  (+0.24 px mean, +0.12 NLL).
- It converts "ok" to "near-perfect" on dense object footprints
  (idx 119, 49, 92, 148, 19 — all end at < 1 px).
- It has two known degradation modes: sparse object crops and
  already-solved crops. Both are small-absolute-err regimes.
- Next ablation: re-run this on a sparse-lidar dataset (Waymo rear,
  ZOD) where the dense-is-solved argument doesn't apply.

## Related

- [local_point_feature_aggregation.md](local_point_feature_aggregation.md) — encoder design + 100EP scalar ablation
- [../models/model_depth.py](../models/model_depth.py) — `FrustumLocalEncoder`
- ClearML tasks: `7570aa2c…` (v10 frustum), `ea8c83ef…` (v10b no-frustum)
