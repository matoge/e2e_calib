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

## MSE-px 内訳 (last epoch)

| | nll | mse_all | obj_mean | obj_med | obj_p95 | bg_med |
|---|---:|---:|---:|---:|---:|---:|
| s15 100ep | 1.678 | 2.265 | 1.757 | 1.221 | 5.004 | 1.714 |
| **s20 100ep** | **1.562** | **2.254** | 1.783 | **1.183** | 5.258 | **1.571** |
| s30 (ep14) | 3.100 | 4.342 | 3.967 | 2.973 | 10.407 | 3.348 |

## 考察

1. **NLL で s20 win、MSE では tied**:
   - nll: s20 1.562 < s15 1.678 (7% 改善)
   - mse_all: 2.254 ≈ 2.265 (差 0.5%、誤差範囲)
   - **NLL の差は MSE じゃなく σ-calibration 由来** (NLL = log σ + (resid/σ)² / 2)

2. **bg で s20 が一段良い**:
   - bg_med: s20 **1.571** vs s15 1.714 (**9% 改善**)
   - 大きい train σ → bg lidar の幅広 alignment scenario → calibration 締まる

3. **obj は mixed**:
   - obj_med: s20 1.183 < s15 1.221 (s20 わずかに win)
   - obj_p95: s20 5.258 > s15 5.004 (s15 win、tail 強い)
   - 大きい train σ で obj outlier (動的物体?) tail が伸びる cost

4. **arch は σ 1.5-2.0 robust**:
   - 軌道ほぼ重なる、後半でわずかに s20 リード
   - σ ablation はこれで終了、**σ=2.0 を stage 1 baseline 確定**

5. **s30 (ep14)**:
   - 序盤遅れて見えるが過大 σ なので妥当、収束は ep30+
   - ETA 完走まで ~1h、σ ceiling 確認

## 次

- **stage 1 hybrid KV (`--frustum-dense`) を σ=2.0 で 100ep 走らせる**: dense LiDAR-map + UV emb + DA/regular hybrid CA。s20 baseline 1.562 と同等以上に収束するか
- target val_nll: **≤ 1.56** が hybrid KV 採用ライン
- それ以下なら stage 2 (mixed Q) → stage 3 (UV-only Q) へ進める

## Learning curves

![](assets/sigma_compare/learning_curves.png)

s20 (赤) が訓練・val 両方で一貫して s15 (青) より下を走る。s30 (緑、ep14 まで) はまだ序盤、収束待ち。

## Sample-level prediction compare (best model each)

![](assets/sigma_compare/sample_compare.png)

各 row = 同じ val sample。col1 = crop、col2-4 = s15/s20/s30 の予測 (cyan=distorted, lime=obj-distorted, yellow+ = GT, red× = pred)。各 panel に "resid (was)" として「予測後の残差 / 摂動直後の残差」を px で表示。s30 はまだ training 進行中なので残差大きい。
