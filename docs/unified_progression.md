# Unified frame-token 架構 — val_err 2.27 → 1.85 px、val_nll 2.27 → 1.59

PandaSet 39-scene front_camera を使った cross-frame residual prediction (= 隣接フレーム間で
LiDAR 点が画像のどこに写るかを補正する小タスク) で、過去のベストだった
**val_err 2.27 px / val_nll 2.27** (v55, legacy multi-frame モデル) を、
**フレームトークン統合 + cross-attn 深さスケーリング + multi-frame KV 拡張** の 3 段階で
**val_err 1.85 px / val_nll 1.59** まで押し下げた。

![progression](images/leaderboard_curves.png)

## 用語の定義 (self-contained)

このページで使う略語:

- **val_err (px)** — validation set 上での per-point 平均ピクセル誤差
  `mean(|pred_uv − gt_uv|)`。低いほど良い。
- **val_nll** — 同じ点群上の Gaussian negative-log-likelihood
  `0.5 z² + log σ + ...` の平均。誤差だけでなく予測 σ の正しさも測る。低いほど良い。
- **C (= n_cross_layers)** — 1 つの forward の中で cross-attention block を
  何段 stack するか。各 block は `cross-attn → self-attn → FFN → 残差出力` の単位。
  C=2 なら 2 段、C=4 なら 4 段。深さが「思考容量」に相当する。
- **N (= 1 + n_KV_frames)** — モデルが見るフレーム総数。
  - N=2: pair (anchor A と target B のみ)
  - N=3: triplet (A, B + 中間フレーム M を 1 つ KV として追加)
  - N=4: quad (A, B + 中間フレーム M1, M2 を 2 つ追加)
- **multi-frame KV** — 中間フレーム M を attention の Key/Value として与え、
  Query (A 由来) が複数フレームから自由にサンプリングして
  「この点は M でも見えてるからこっち」みたいな推論ができる構造。
- **frame token** — 各フレームの画像と LiDAR 点を同じ 8×8 grid 上で 1 個のテンソル
  `(D, 8, 8)` に統合した表現。観測がない grid セルは 0 + マスクチャンネル。

## 旧アーキテクチャ (v55, legacy multi モデル)

- 画像と LiDAR を別系統で処理:
  - 画像: `(D, 8, 8)` の coarse feature grid + MSDeformAttn
  - LiDAR: 疎な点トークン list (~256 個) + plain attention
- 1 つの cross-block 内に q/k/v/o projection が **2 系統**: 画像用と LiDAR 用が別。
- val_err 2.27 px / val_nll 2.27 で、ここ何ヶ月か頭打ちだった。

## 新アーキテクチャ (unified frame-token、v70 以降)

LiDAR の点を画像と同じ 8×8 grid に scatter (= 各セルに該当 grid の点の特徴を平均)。
「観測がない = 0 + has_pt mask」を許容することで、画像と LiDAR を **単一の frame_token
`(D, 8, 8)`** に潰す。これにより:

- 画像と LiDAR の差別がなくなり、cross-attention は MSDeformAttn 1 本で十分
- per-point Q は anchor frame の frame_token を bilinear sample + PointMLP で構築
- multi-frame: 各 KV フレームの grid に anchor からの絶対 pose embedding を broadcast
  → MSDeformAttn の中の 1 個の softmax で「どのフレームのどこからサンプル」が同時に決まる

詳細: `models/cross_frame_unified.py`、設計議論: `docs/multi_frame_attention.md`

## 段階別の効果

| 名前 | 設定 | val_err | val_nll | 何が変わった |
|---|---|---|---|---|
| v55 | legacy multi (C=2, N=3) | 2.27 | 2.27 | 旧アーキの最高 |
| v70 | unified pair (C=2, N=2) | 2.35 | **2.00** | アーキ刷新で NLL が初めて 2.0 台に |
| v75 | unified multi (C=3, N=3) | **2.09** | 2.04 | C=3 にして multi-frame の効果が初めて出た |
| v92 | unified multi (C=4, N=3) | **1.93** | 1.93 | C=4 で更に伸びる |
| v100 | unified multi (C=4, **N=4**) | **1.85** | **1.59** | quad で NLL が劇的改善 |

## 「C を深くすると multi-frame が効く」の解釈

このプロジェクトで一番非自明だった発見:
**cross-attention block 数 C を深くするだけだと val_err はほぼ動かない。
multi-frame KV を入れた上で C を深くすると、初めて深さが効く。**

| | pair (N=2) | multi (N=3) | 差 |
|---|---|---|---|
| C=2 | val_err 2.35 (v70) | val_err 2.38 (v74) | ±0 |
| C=3 | val_err 2.36 (v71) | val_err 2.09 (v75) | **−0.27 px** |
| C=4 | val_err 2.29 (v84) | val_err 1.93 (v92) | **−0.36 px** |

### 何が起きてるか

- **pair モード (N=2)** では KV が 1 フレームしかなく、各ピクセルの帰属候補が単純。
  C=2 で十分その単純な情報を処理しきってしまい、深さを増やしても閾値が変わらない。
- **multi-frame (N=3 以降)** では「target B のここ?」「中間 M のここ?」という
  **複数仮説を比較する思考が必要** になる。
  - C=2: 1 段で「どの仮説に注目するか」と「精緻化」を同時にやらされ、両方妥協。
    結果、multi-frame の余計な KV を捌ききれず、pair と同じ。
  - C=3: 1 段目で「M との合致確認」、2 段目で「B への精緻化」のように分業可能になり、
    M の情報を初めて活かせる → −0.27 px。
  - C=4: 更に分業が細分化、M と B の確信度を比較しながら refine できる → −0.36 px。

つまり **「C は multi-frame 情報を統合するための思考容量」** として機能している。
KV が 1 つだけなら容量は要らない、KV が増えると容量が必要、という綺麗な分業になる。

### N=4 (quad) で NLL が劇的に下がる理由

v100 (N=4) は val_err では v92 (N=3) より −0.08 px だが val_nll は **−0.34** と
不釣り合いに大きく落ちている (1.93 → 1.59)。これは σ-calibration が桁違いに正しく
なったということ:

- 中間フレーム M1 と M2 という独立した観測が 2 つあると、各点について
  **「3 frame 全部で観測一貫している = static」** vs
  **「フレームによって位置が違う = 動的 or オクルージョン」** がモデル内で判別可能になる。
- 結果として static な点には小さい σ、動的・不確定な点には大きい σ、と
  きれいに割り振られる → NLL が下がる。

これは Σ-weighted bundle adjustment や、動的物体を down-weight した
クリーンな点群マップ生成にそのまま使える性質。
σ の挙動を直接観測した結果は `docs/dynamic_object_sigma.md` を参照。

## まだ試してないこと

1. **motion-warped GT で再学習** (進行中) — 動的物体 (PandaSet で `Moving` ラベルが
   付いた cuboid 内の点) について、box の rigid 変換 A→B で uv_gt を書き換えてから訓練。
   現状モデルは「動いてないと仮定して predict」しているが、warped GT で再学習すると
   motion-aware になるか。
2. **6 cameras 同時** — 今は front_camera のみ。PandaSet には 6 カメラある。
   データ 6 倍 + view 多様化で過学習を抑えられるか。
3. **C=5 以上で天井探索** — 一度試したが NLL 学習が壊れた (σ output が clamp 上限に
   貼り付いた)。proj head の zero-init を 5 段重ねるとうまく合算が学べないらしい、
   要再設計。
4. **Σ-weighted BA 試作** — 既存の v100 σ で実マップ生成。clean な静的マップ +
   分離された動的物体の可視化。
5. **Company data (VLS-128 LiDAR) への移行** — Zenseact Open Dataset (= 同じ VLS-128
   sensor) で fine-tune できるか。

## 関連ファイル

- `models/cross_frame_unified.py` — メインアーキテクチャ実装
- `models/cross_frame_multi.py` — 旧 multi モデル (v55、比較用)
- `train_cross_frame.py` — `--model unified` `--multi-frame` `--quad-frame` `--motion-warp-gt` フラグ
- `datasets/pandaset_pair.py` — quad mode (N=4) と motion-warp GT の実装
- `scripts/eval/plot_clearml_progression.py` — このページの図を生成 (ClearML SDK 経由)
- `docs/dynamic_object_sigma.md` — σ で動的物体を勝手に分離する解析
- `docs/multi_frame_attention.md` — multi-frame attention の設計の数理議論
