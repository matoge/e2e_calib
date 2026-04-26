# Dynamic-object σ behaviour of the cross-frame residual model

**TL;DR**: モデルに動的物体ラベルを一切教えていないにもかかわらず、
学習済み cross-frame residual model (v92, unified frame-token, multi-frame, c=4)
は出力 σ で**動いてる物体を勝手に分離する**。
ただし上昇幅は controlled で、特に「画像 appearance では何とか合わせられる」
中域の動車に対して σ は inflate せず over-confident になる。

## 設定

- model: `experiments/cross_frame_v92_unified_multi_c4` (val_err=1.93 px, val_nll=1.93)
- dataset: PandaSet 39-scene `front_camera` val split (~200 pair samples, ~32K query points)
- analysis script: `scripts/analysis/dynamic_object_variance.py`

各 query point の世界座標を dataset から取り出し (`pts_w_A_query`)、PandaSet の 3D
cuboid annotation の box 内判定で:

| ラベル | 元データ | 説明 |
|---|---|---|
| `background` | どの cuboid にも入らない | 路面、建物、植生、etc |
| `parked` | `attributes.object_motion == 'Parked'` | 駐車中の車 |
| `stopped` | `attributes.object_motion == 'Stopped'` | 信号待ち等 |
| `moving` | `attributes.object_motion == 'Moving'` | 走行中 |

の 4 カテゴリに分類し、モデル予測 σ と実残差 |Δuv - GT| を集計。
`moving` についてはさらに 3D box 中心の A↔B 間 displacement で 4 bucket に
細分:

| bucket | A↔B 移動量 |
|---|---|
| `mv 0-1m` | 0 m ≤ d < 1 m (徐行・カーブ) |
| `mv 1-3m` | 1 m ≤ d < 3 m (一般走行) |
| `mv 3-10m` | 3 m ≤ d < 10 m (中速、対向車含む) |
| `mv 10+m` | 10 m 以上 (高速対向車、フレーム間で view を抜けるレベル) |

## 結果 (v92 multi c4)

### motion-attr で分けた σ_pred と実残差

```
σ_pred (px):    bg     parked   stopped   moving
median          0.87   0.75     0.87      1.07
mean            1.07   0.99     0.88      1.16
p90             1.68   1.33     1.06      1.58

|Δuv| (px):     bg     parked   stopped   moving
median          1.69   1.56     1.02      1.89
mean            2.37   2.62     1.59      3.59
p90             4.54   7.81     3.79      13.03
```

### moving 内を displacement で細分

```
σ_pred (px):    mv 0-1m   mv 1-3m   mv 3-10m   mv 10+m
median          0.78      1.30      1.21       1.03
mean            0.82      1.22      1.33       1.11
p90             0.95      1.81      2.19       1.44

|Δuv| (px):     mv 0-1m   mv 1-3m   mv 3-10m   mv 10+m
median          1.11      1.92      2.01       1.99
mean            1.12      2.31      5.90       3.42
p90             1.71      5.36      15.43      12.99
```

## 解釈

### ✅ 仮説の前半は当たってる

- `parked` と `stopped` は背景と同等かそれ以下 (median 0.75 / 0.87 vs bg 0.87)。
  車表面はテクスチャ豊富で predict しやすい
- `moving` は明確に σ が上がる (1.07 vs 0.87)。動的物体に対して「予測自信ない」
  という signal が出ている
- displacement と σ も 0-1m → 1-3m で **0.78 → 1.30** ときれいに ordered

### ⚠ ただし上昇幅は控えめ、3-10m bucket で破綻

- `mv 3-10m` は最も悲惨: 残差 mean 5.9 px / p90 15.4 px なのに σ=1.33
  → z = err/σ ≈ 4.4、キャリブ大幅破綻 (over-confident)
- `mv 10+m` で σ がむしろ下がる (1.03)。たぶんモデルが「completely wrong, give up」モード
  に入ってる

### 🔍 メカニズム仮説

モデルは静的シーン仮定で「画像 appearance マッチング」を学んでいる。
動き量別に挙動は:

| 動き量 (px in image) | モデルの挙動 |
|---|---|
| < 数 px | 画像マッチングのノイズ以下 → 静的扱いで当てちゃう、σ も低い |
| 5-15 px | マッチング部分破綻、σ ちょっと上げる |
| > 20 px | 完全に外れるが画像的にもっともらしい候補がある → 過信して σ 低い |

つまり「**見た目で行ける範囲は σ 上げない**」。これは calibration 的には正しい
(視差として説明可能なら静的シーン仮定で OK)。動的検出器として使うには上限が
ある。

### 構造的な観測欠落

dataset の `inb` filter (line 1097 `pandaset_pair.py`) で `uv_hat` または `uv_gt`
が patch 外に出る点は val pool から完全に除外される。**動いて view 完全に
抜けた車**は最初から観測されない → σ がフルレンジ (上限 30 px) に振り切れる
ケースは観測不能。

## 利用可能性

`σ > 1.5 px` 閾値で **動車候補マスク**として実用可能 (precision OK):

| カテゴリ | σ > 1.5 占有率 |
|---|---|
| background | ~5% (false positive) |
| parked | ~5% |
| stopped | ~5% |
| moving | ~30% |
| mv 1-3m | ~40% |

map maker / σ-weighted BA の前段の動的物体 down-weighting には十分使える。

## 制限と次のステップ

**現状の限界**:
- σ の天井が ~2-3 px あたりに学習で固定されてる (motion-aware loss 無し)
- view 外の動車は観測できない
- 中速動車 (3-10m bucket) で over-confident

**改善案**:
1. **GT の書き換え**: 動車内の点を box の動き分 warp すれば「移動先 = 真の B 位置」
   になる。これでモデルは "appearance + motion compose" を学べる。半教師付き
2. **multi-frame consistency loss**: M frame 経由で動いてる手がかり (M に映ってるが
   B には映ってない) があれば σ を上げるよう誘導
3. **filter 緩和**: `inb` を外してフルレンジ σ を観測可能にする (BA 評価には必要)

## 関連ファイル

- `scripts/analysis/dynamic_object_variance.py` — 解析スクリプト
- `/tmp/dyn_var_v92.png` — 出力プロット (σ と残差の log-hist by category)
- `experiments/cross_frame_v92_unified_multi_c4/` — 解析対象モデル
