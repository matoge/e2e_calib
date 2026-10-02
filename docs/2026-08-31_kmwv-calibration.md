# kamikado + WovenSequence LiDAR–カメラ外部キャリブレーション

2026-08-31 / CalibNet2 + 外側 GN ／
[English](2026-08-31_kmwv-calibration.en.md) ／
[nuScenes 版](2026-08-31_nuscenes-calibration.md) の続き ／
[使い方](2026-08-31_kmwv-usage.md)

---

## 要約

前段の [nuScenes 版](2026-08-31_nuscenes-calibration.md) と**同じネットワーク・同じソルバ・同じ段階学習**を、自社データ（kamikado 8 シーン + WovenSequence 9 seqs、どちらも tss4_fcm fisheye、4K 画像）に当てた。学習に一度も出ていないシーンで **1 フレーム 0.0204°／6.3 mm、32 枚合成で 0.0067°／1.45 mm**。

nuScenes 版の 1 フレーム 0.037°／7.7 mm、32 枚合成 0.0068°／1.43 mm と同じか、やや良い。学習は 5 スケール上げ (nuScenes 100ep vs kmwv 30ep→50ep)、データセット固有の調整はしていない。

**スクラッチではない。** 初期値は nuScenes CAM_FRONT で学習した `experiments/front_670x3/best_model.pt`
（`--resume-ckpt`）。つまりこの結果は「17 シーン / 学習 2404 フレームでゼロから学習できる」ではなく、
**nuScenes で事前学習した重みを 2404 フレームで別センサ（4K 魚眼 tss4_fcm）に移せる**という主張。
新しいセンサ構成の立ち上げコストとしては、こちらの方が実務的に意味がある。

| | シーン | 学習フレーム | val フレーム |
|---|---|---|---|
| kamikado | 8 | 1004 | 98 |
| WovenSequence | 9 | 1400 | 389 |
| 合計 | **17** | **2404** | **487** |

scene-split（val のシーンは一度も学習していない）。

| データ | 1 フレーム rot | 1 フレーム trans | F=32 rot | F=32 trans | χ²/6 (F=1) |
|---|---|---|---|---|---|
| nuScenes CAM_FRONT (report) | 0.037°  | 7.7 mm | 0.0068° | 1.43 mm | 1.15 |
| **km + wv (この報告)** | **0.0204°** | **6.3 mm** | **0.0067°** | **1.45 mm** | **1.0** |

---

## データ

| | kamikado | WovenSequence |
|---|---|---|
| シーン数 | 8 | 9 |
| 画像 | 3840 × 2160、tss4_fcm 魚眼 (KB4) | 同左 |
| val | scene-split、held-out (train と共有ゼロ) | 同左 |
| LiDAR↔カメラの時刻差 | 33.4 ms 前後 (metadata に per-seq で入っている) | 32.0-33.8 ms |
| 補正 | ✔️ | ✔️ (`_pose_at_camera_time`) |

**時刻差の扱い。** WovenSequence は camera shutter と LiDAR sweep 中心が 33 ms ずれる。build_woven_sequence_v3.py が `_pose_at_camera_time` で LiDAR 時刻の pose を camera 時刻に補間して LiDAR→camera 変換を焼き込むので、model は同期後の投影しか見ない。9 seq × 5 フレームで検証: v ≈ 3〜10 m/s、期待補正量 `v × delay` = 108-380 mm、実測補正量との linearity 誤差 0%（`delay=0` を渡すと 0 mm、期待通り）。cache に焼き込み済み、学習側は何もしていない。

**入力スケールの合わせ方。** nuScenes は 1600 × 900 native を 256 × 256 crop で 28 窓/フレーム。km/wv は 4K のままだと 15 × 9 = 135 窓になり over-count する。**512 × 512 で crop して 256 × 256 にリサイズ** すれば nuScenes の入力条件と同じ解像度で 8 × 5 = 40 窓/フレームに収まる。`--per-cache-crop-px km:512,wv:512 --img-size 256` を trainer に足した。

---

## 学習

nuScenes 版 §学習をどう進めたか と同じ 2 段。段 2（1 窓 BA、W = σ）は今回は省略（段 1 → 段 3 で問題なく降下した）。

| 段 | 内容 | エポック | held-out 1 フレーム |
|---|---|---|---|
| S1 | 点だけ (`gaussian2d_nll`、BA なし) | 30 | rot 0.266° / t 66.7 mm |
| S3 | 40 窓融合 BA (`W = InfoHead`) | 50 | **rot 0.0204° / t 6.3 mm** |

- **S1 → S3 で rot 13×、t 10× 改善。** BA を入れた 1 エポック目で既に 6× 差が付く。
- **χ²/6 = 1.0（S3 ep50）。** σ 側は `gaussian2d_nll` 専用、`W` は `InfoHead2x2` 経由でしか勾配を受けない（`--ba-w-source infohead`）。nuScenes 版で確認した「σ と W を同じにすると 3113 → 26.4 px に膨れる」現象は km/wv でも同じ。分けているので発生しない。
- **BA warmup。** `--ba-weight 0.05 --ba-warmup-end 10` で ep0-10 に BA 損失を 0 → 0.05 に立ち上げる。ep1 は chi²/6 = 22 で over-shoot、ep30 で 0.9 まで落ち、ep50 で 1.0。

![curves](_figs/2026-08-31_kmwv_curves.png)

*(a)* val NLL の推移。段境界（点線）で発散しない。*(b, c)* rot / trans を log 軸で。ep50 付近で緑点線（nuScenes report の 1 フレーム値）を割る。*(d)* χ²/6。S1 は 0.4 前後（1 窓 GN + W=σ で under-confident）、S3 は BA warmup で ep1-5 に跳ね、以降 ~1 に落ち着く。

---

## held-out val サンプル

摂動 ±0.5°／±0.20 m。赤 × が model への入力（摂動後 uv）、緑が本当の投影位置（GT）、水色が model が戻した位置（`mu`）、青い楕円が 1σ（`sigma`）。

![val grid](_figs/2026-08-31_kmwv_val_grid.png)

| 窓 | 点数 | 摂動 | 残差 | σ 中央値 | 中身 |
|---|---|---|---|---|---|
| woven #0 | 137 | 9.91 px | **1.25 px** | 0.88 px | 壁 + ポール、屋根の縁 |
| woven #2 | 191 | 12.62 px | **3.70 px** | 1.27 px | 別車両の側面 |
| kamikado #0 | 54 | 16.49 px | **9.92 px** | 6.07 px (p90 7.24) | 明るい路面、edge なし |
| kamikado #2 | 110 | 8.89 px | **2.79 px** | 1.14 px | ポール + 空、混在 |

**良い窓と悪い窓を model 自身が σ で見分ける。** kamikado #0 は残差 9.9 px と大きいが、σ もそれに合わせて 6 px に広がる。楕円で目視で「この窓は解けていない」と分かる。外側の χ² ゲートは σ ではなく GN 後の食い違いで判定するので、σ 単体で弾く必要はないが、σ の広がり方は model がその窓の情報量を正しく認識している証拠。

---

## 複数フレーム合成

nuScenes 版 §χ² ゲート §過分散補正 §CI と同じ 3 つの規則を、S3 の best_model.pt に対して post-process で回した。dataset を val 1 エポックだけ走らせ、per-frame `(δ_pred, δ_gt, H)` を dump（`--dump-pose`）→ `scripts/eval/frame_fusion.py` で残差空間で inv-var pool する。

```
r_i = δ_pred_i − δ_gt_i         # 各フレームで観測された「戻し切れなかった量」
H_i = J^T W_i J                 # 各フレームの情報行列
δ̄  = (Σ H_i)⁻¹ Σ H_i r_i        # 残差の inv-var pool  → sum
c_i = (r_i − δ̄)^T H_i (r_i − δ̄) / 6    # フレームごとの χ²/6
     drop c_i > 3、2 回反復     → gate3
W_ci = 1/F 等重み平均             → CI
```

**残差空間で pool する理由。** 学習時は各フレームに独立な ε をサンプリングして val する（"pose を戻す" タスクの汎化評価）。実運用のキャリブレーションは rig の固定 δ_gt を N フレームで推定するが、統計的には `δ_pred_i − δ_gt_i` の期待値 = rig 誤差なので、残差空間で pool すれば同じ結果になる（`r_i` の平均が rig 誤差、`H_i` は各フレーム独立）。nuScenes 版もこの設定。

### 結果

| F | rot_med (sum) | rot_med (gate3) | t_med (sum) | t_med (gate3) | χ²/6 (sum) |
|---|---|---|---|---|---|
| 1 | 0.0252° | 0.0252° | 6.23 mm | 6.23 mm | 1.05 |
| 2 | 0.0159° | 0.0159° | 3.72 mm | 3.72 mm | 1.08 |
| 4 | 0.0107° | 0.0103° | 4.51 mm | 3.78 mm | 2.19 |
| 8 | 0.0091° | 0.0110° | 1.88 mm | 1.60 mm | 1.70 |
| 16 | 0.0095° | 0.0079° | 1.78 mm | 1.36 mm | 2.97 |
| **32** | **0.0067°** | **0.0032°** | **1.45 mm** | 1.67 mm | 4.53 |

（N = 64 valid frames の非重複窓。gate3 は F ≤ 2 では sum と一致、F ≥ 4 で少数フレームを弾き始める。）

![frame fusion](_figs/2026-08-31_kmwv_fusion.png)

- **(a) rot vs F**、**(b) trans vs F**：3 規則とも 1/√F にほぼ乗る。F=32 で **rot は sum 0.0067° / gate3 0.0032°**、trans 1.4-1.7 mm。nuScenes report F=32 の 0.0068° / 1.43 mm と一致。
- **(c) χ²/6 = k**：`sum` は F=1 で 1.05（校正済み）だが F を増やすと 1.7 → 2.2 → 4.5 と上がる。フレーム間の相関を無視して `H` を素直に足すぶんの over-count が nuScenes 版と同じ挙動で現れる。`CI` は逆に F=32 で 0.14 まで下がる（保守側に振れすぎ、nuScenes 版 §CI の "χ²/6 が 3.86 → 0.24" と同構造）。実用は `sum` の推定値に、実測 `k = 4.5` で共分散を割った 1σ を報告する。

---

## 結論

- **1 フレーム 0.020° / 6.3 mm、32 フレーム合成 0.007° / 1.4 mm。**
- **共分散校正 (χ²/6 → 1.0) も同時に達成。** 段 3 の InfoHead 分離が動いている。
- **32 フレームで nuScenes report の飽和点に到達。** 学習データが 1/50 ぐらいなのに、モデル・ソルバ・分解手順が共通なので数値が同じところまで来る。

### 落とし穴（nuScenes 版と同じ）

| | |
|---|---|
| σ と W を同じにしない | 分けないと σ が膨らんで μ が壊れる |
| `k` を固定値にしない | データから求める。F=32 で 4.5、F ≤ 2 で 1 |
| val はシーンで分ける | フレーム単位で切ると隣接フレームが train に入る |
| 4K を直接投げない | crop 512 → resize 256 で nuScenes と入力条件を揃える |
| WovenSequence の時刻差 | 33 ms、cache build で焼き込み済み、学習側は触らない |

### 未実施 / 次にやること

- **rank gather。** 現状 dump は rank 0 の val loader 分だけ (N=64)。8 GPU 全部 gather すれば N ~500、p90/max の推定が安定する。
- **k の scene 依存性。** nuScenes 版でシーンごとに 1.46-4.65 とばらつくことが確認済み。km/wv でも per-scene の k 分布を出す。
- **フレーム間隔 stride 依存。** nuScenes 版で stride 1/4/10 は改善しないと判明済み。km/wv でも同じか確認する価値はある。

---

## 使ったもの

| | |
|---|---|
| ckpt | `experiments/kmwv_s3_ba40_512r256_0831_0325/best_model.pt` (val_nll 0.914 @ ep45) |
| dump | `experiments/kmwv_pose_dump_0831_1249/pose_dump_ep001.pt` (N=64) |
| ClearML | http://172.18.2.49:8085/projects/d72252aa72f94b2192269ba448f3420b/experiments/61f4cd8f70c345abb4a5570ffc274711 |
| fusion 解析 | `scripts/eval/frame_fusion.py` |
| kick scripts | `_kick_kmwv_s1_pts_512.sh`、`_kick_kmwv_s3_ba40_512.sh`、`_kick_kmwv_pose_dump.sh` |
| trainer 変更 | `--per-cache-crop-px`、`--dump-pose` を `datasets/train_cnd2_ddp.py` に追加 |
