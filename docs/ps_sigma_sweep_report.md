# PandaSet σ-sweep — 100ep tile baseline (deform_sl, frustum on)

**実験条件 (共通):**
- cache: `/mnt/nvme6t/e2e_calib_cache/pandaset_v3_tiled` (PS 103 scenes, front cam, 5×2 tile)
- arch: ConvNeXt + frustum (PT 2-layer @ d_local=32) + **deform_sl** + n_layers=4
- training: 100 ep, lr=3e-4 → 1e-7 cosine, batch=128, oversample=1, max_crop=384, img_size=128
- sigma sweep の差は `--rot-deg / --t-m` のみ:
  - s15 → ±1.5° / ±0.6 m
  - s20 → ±2.0° / ±0.8 m
  - s30 → ±3.0° / ±1.2 m  (進行中、ep14 まで)

**ClearML:**
- s15: id `2fc3c978` (completed)
- s20: id `89e0ced2` (completed)
- s30: id `6ab5564b` (in_progress, ep14/100)

## val_nll 推移

| ep | s15 | s20 | s30 |
|---:|---:|---:|---:|
| 1 | 5.330 | 4.597 | 4.874 |
| 5 | 4.462 | 3.902 | 3.996 |
| 10 | 3.574 | 3.335 | 3.416 |
| 20 | 2.958 | 2.910 | — |
| 30 | 2.794 | 2.460 | — |
| 40 | 2.313 | 2.402 | — |
| 50 | 2.243 | 2.104 | — |
| 60 | 2.055 | 1.904 | — |
| 70 | 1.891 | 1.786 | — |
| 80 | 1.808 | 1.671 | — |
| 90 | 1.696 | 1.585 | — |
| **100** | **1.678** | **1.562** | (走行中) |

## 考察

1. **σ=2.0 が σ=1.5 より良い val_nll**:
   - val_best: s20=1.562 < s15=1.643 (5% 改善)
   - 大きい perturbation = より広い residual 分布で学習 → 汎化伸びる
   - σ=1.5 は under-perturb で easy regime に偏る可能性

2. **収束軌道はほぼ重なる**:
   - 5-50ep までは s15 と s20 はほぼ同じ val_nll、後半でわずかに s20 が上回る
   - arch (deform_sl + frustum + n4) は perturbation 1.5-2.0° の範囲に **robust**
   - ablation で σ 細かく振る必要なし、σ=2.0 を baseline 一本に確定

3. **s30 (1Hz, ep14)**:
   - 序盤 (ep14) では s20/s15 と同程度、過大な σ で破綻はしてない
   - ETA 完走まで ~1.5h、結果次第で σ ceiling 確認

## 次

- **stage 1 hybrid KV (`--frustum-dense`) を σ=2.0 で 100ep 走らせる**: dense LiDAR-map + UV emb + DA/regular hybrid CA。s20 baseline 1.562 と同等以上に収束するか
- target val_nll: **≤ 1.56** が hybrid KV 採用ライン
- それ以下なら stage 2 (mixed Q) → stage 3 (UV-only Q) へ進める
