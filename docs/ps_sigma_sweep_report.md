# PandaSet σ-sweep — 100ep tile baseline (deform_sl, frustum on)

**実験条件 (共通):**
- cache: `/mnt/nvme6t/e2e_calib_cache/pandaset_v3_tiled` (PS 103 scenes, front cam, 5×2 tile)
- arch: ConvNeXt + frustum (PT 2-layer @ d_local=32) + **deform_sl** + n_layers=4
- training: 100 ep, lr=3e-4 → 1e-7 cosine, batch=128, oversample=1, max_crop=384, img_size=128
- sigma sweep の差は `--rot-deg / --t-m` のみ:
  - s15 → ±1.5° / ±0.6 m
  - s20 → ±2.0° / ±0.8 m
  - s30 → ±3.0° / ±1.2 m

**ClearML (3 runs all completed):**
- s15: id `2fc3c978`
- s20: id `89e0ced2`
- s30: id `6ab5564b`

## val_nll 推移 (100ep 完走)

| ep | s15 | s20 | s30 |
|---:|---:|---:|---:|
|  1 | 5.330 | 4.597 | 4.874 |
|  5 | 4.462 | 3.902 | 3.996 |
| 10 | 3.574 | 3.335 | 3.416 |
| 20 | 2.958 | 2.910 | 2.998 |
| 30 | 2.794 | 2.460 | 2.728 |
| 40 | 2.313 | 2.402 | 2.346 |
| 50 | 2.243 | 2.104 | 2.167 |
| 60 | 2.055 | 1.904 | 1.995 |
| 70 | 1.891 | 1.786 | 1.838 |
| 80 | 1.808 | 1.671 | 1.685 |
| 90 | 1.696 | 1.585 | 1.637 |
| **100** | **1.678** | **1.562** | **1.602** |

## MSE-px 内訳 (last epoch)

| | nll | mse_all | obj_mean | obj_med | obj_p95 | bg_med |
|---|---:|---:|---:|---:|---:|---:|
| s15 100ep | 1.643 | 2.265 | 1.757 | 1.221 | 5.004 | 1.714 |
| **s20 100ep** | **1.562** | 2.254 | 1.783 | **1.183** | 5.258 | **1.571** |
| s30 100ep | 1.602 | 2.302 | 1.799 | 1.198 | 5.538 | 1.570 |

best val_nll は s20 で **1.562**、s30 が **1.602** で 2.5% 後退、s15 は **1.643** で最低。

## 学習曲線

![](assets/sigma_compare/learning_curves.png)

- 訓練・val 両方で **s20 (赤) が一貫してリード**、特に ep30+ で他の 2 σ から離れる
- s15 (青) と s30 (緑) は ep70+ でほぼ重なる軌道 — **σ 不足 (s15)** と **σ 過剰 (s30)** が同程度の cost
- val 後半の oscillation は σ=1.5/3.0 の方が大きい (perturbation 範囲の広さで variance 増)

## サンプル比較 (同 val idx に各 σ best モデルを推論)

![](assets/sigma_compare/sample_compare.png)

col1 = crop、col2-4 = s15/s20/s30 の予測。
- cyan = distorted (perturbation 後 LiDAR uv)
- lime = obj-distorted
- yellow+ = GT (perturbation 無し LiDAR uv)
- red× = model 予測

各 panel の "resid (was)" は予測後の平均残差 / 摂動直後の残差 [px]。

## 注意: 各 σ の val 評価条件は別

各 model は **自分の訓練 σ で perturbation された val** で評価されてる。同じテストセットで比較したわけじゃない:
- s15 → ±1.5° / 0.6m 摂動 val
- s20 → ±2.0° / 0.8m 摂動 val
- s30 → ±3.0° / 1.2m 摂動 val

NLL = log σ + (resid/σ)² /2 は Gaussian の entropy 形なので、σ 大 → resid 大 + σ_pred 大 が同程度に伸びれば NLL は同じ。**val_nll の絶対値で sweet spot は決められない**。

## 考察

1. **σ=2 > σ=1.5 は明確** (条件依存しない):
   - bg_med: s20 1.571 vs s15 1.714 (**9%改善**)、obj_med: s20 1.183 vs s15 1.221
   - 大きい摂動範囲で **effective dataset variation 増 → bg calibration 締まる**
   - s15 は under-perturb で汎化が weak、これは σ=2 評価で測っても同じ序列になる
   - σ aug ボーナスは少なくとも σ=2 まで上り坂

2. **σ=3 では tail が暴れる (over-perturb の cost)**:
   - obj_p95: s30 5.538 vs s20 5.258 (5%悪化)
   - 大摂動 + 動的物体は **完全に context 外** に飛ぶサンプル増加 → 予測不能 tail
   - s30 の MSE 全体 (2.302) も s20 (2.254) より 2% 悪い
   - val_nll 1.602 が見かけ上 s15 1.643 より良いのは「自前の σ_pred スケール」の差で **直比較不可**

3. **bg は σ=2 と σ=3 で同質**:
   - bg_med: s20=1.571、s30=1.570 (差 0.06%)
   - σ≥2 で bg calibration は天井、追加摂動は効かない

4. **arch は σ=1.5-3.0 robust**:
   - 訓練曲線は 3 σ で大きく崩れない、grad 流れる範囲
   - 結論: **σ=2.0 が defensible default**。σ=1.5 (under)、σ=3 (over) より良いことは bg/obj median で確認できる

5. **σ aug の本質は real-world robustness**:
   - 推論時の真の perturbation は学習時 σ より大きい/不確定 → 学習 σ 小 (s15) は under-cover
   - 大きめ σ で学習しておくほうが推論時 large perturbation に対応できる
   - **σ=3 の tail が悪化**してるのは img_size=128 の **context 不足** が原因 (大摂動で物体が crop 外に飛ぶ → context 外 → 予測不能)
   - context を img_size=256 等に拡張すれば σ=3 も tail 抑えられて優位性出る可能性大
   - 当面 img_size=128 では σ=2 が現実解、本気 robustness 求めるなら **σ=3 + img_size=256** ablation

## 次

- **stage 1 hybrid KV (`--frustum-dense`) を σ=2.0 で 100ep**: dense LiDAR-map + UV emb + DA/regular hybrid CA を s20 baseline (1.562) と比較、frame token 移行への必須前提
- frame token / cross-frame の curriculum は `project_uvemb_query_curriculum` (memory) 参照
