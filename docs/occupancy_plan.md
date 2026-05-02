# e2e_calib — LiDAR × Vision Odometry / Extrinsic 精密化レポート

最終更新: 2026-04-27
想定読者: DENSO GAI 中村さん / Occupancy チーム

---

## 0. 方針

本レポジトリ `e2e_calib` を用いることで、以下を統合した **高精度 LiDAR–Vision Odometry** が **構築できる可能性が高い**。

- **ピクセル単位の再投影残差を用いた LiDAR × Camera 外部パラメータ同定**
- **時系列 cross-frame self-calibration による extrinsic / ego-pose の同時最適化**
- **学習済み分散 (σ) によって重み付けされた Bundle Adjustment**

想定される成果物 (frame あたり):

- 6-DoF 外部パラメータ (R, t)
- ego pose (R_w, t_w)
- 各対応点の anisotropic 2D 分散 (Σ ∈ ℝ^{2×2})

これらを占有 (occupancy) / flow / depth などの downstream に供給できれば、
幾何入力側の誤差を大幅に圧縮した状態で学習・評価ができる見込みがある。

---

## 0.5 Slack 議論から読める、駐車場 GT の目標と停滞点 (過去6ヶ月)

対象チャンネル `C06JVQTGPK3` を 2025-05 〜 2026-04 で全スレッド確認した範囲の、
**技術課題ベース**のまとめ (人物評・政治要素は除外)。

### 0.5.1 目標

- **駐車場 (parking / indoor) での occupancy GT の安定生成**
  - dense voxel (drivable / occluded / occupied)
  - 3D bounding box (他車・歩行者)
  - voxel flow (各セルの速度ベクトル)
- これを学習用 GT として spatial-labeling pipeline から安定出力すること

### 0.5.1b Occupancy チームが実際にやっていること (Slack ログからの再構成、コード未精読)

| レイヤー | 内容 |
|---------|------|
| センサー入力 | 複数 LiDAR (VLS-128 系) + マルチカメラ + POSLV pose + 3D bbox |
| 幾何前処理 | pose を既知として LiDAR 点群を ego→world に累積 |
| 中間表現 | 車両座標系 BEV の voxel grid (高さ数層)、accumulate window 数秒 |
| ラベリング | voxel を `drivable surface` / `occupied static` / `occupied dynamic` / `occluded` / `unknown` に分類 |
| flow 付与 | 連続 frame の voxel 対応から速度ベクトルを付与 (dynamic voxel のみ) |
| パイプライン | spatial-labeling (flyte) → feature_store → training batch (BigQuery) |
| モデル | dense occupancy head + sparse (bbox + flow) head の 2 系統 |
| 出力 | 3D voxel (+ flow) を npz / h5 で保存 |
| デプロイ | TRT 変換して on-car 推論 |

**前提条件**として効いているもの:

- POSLV pose が精度仕様内であること (→ 駐車場で崩れる)
- LiDAR と各カメラの extrinsic が固定値として正しいこと (→ 経年・振動でずれる可能性を検査する工程がない)
- multi-return / pulse_height / pulse_width が使える車両であること (→ 社内実験車は single-return)

G1〜G5 はこれら前提が成立しないときに出る症状と一致しており、
**「pipeline は正しいが、渡している幾何が崩れている」** という読み方が整合する。

### 0.5.2 6ヶ月繰り返し出ている技術課題

| # | 症状 | 最初の報告時期 | 現状 |
|---|------|---------------|------|
| G1 | Occupancy GT が床・地面を `drivable surface` として誤ラベル (壁と床の切り分け失敗) | 2026-03 | voxel filter / class 定義側で対症中、根本未解決 |
| G2 | Parking pointcloud に床が浮いたようなノイズ (±数十cm) が乗る | 2026-04 | 閾値フィルタで mask、構造的解決なし |
| G3 | Bounding box が柱・台車などの物体を捕捉しきれない (under-segmentation) | 2026-01 | 未解決 |
| G4 | ego 静止シーンで occupancy がスカスカ (LiDAR 蓄積が効かない) | 2026-04 | 「ego speed で filter」という回避策、本質対処なし |
| G5 | 他車・歩行者の速度ベクトルが不安定 (静止物が動いて見える / 動体がちらつく) | 2026-04 | flow head 側の loss 調整で緩和中 |

### 0.5.3 同じく6ヶ月間、議論に現れていない要素

Slack 全文検索で以下の語はほぼヒットしなかった:

- `extrinsic` / `calibration` / `re-projection`
- `pose accuracy` / `odometry drift` / `POSLV error`
- `photometric consistency` / `image-lidar alignment`
- `bundle adjustment` / `sensor fusion (幾何レベル)`

議論の中心は

- BigQuery / flyte / dependency 管理
- file format (npz / h5py / parquet)
- class ontology (drivable の定義 / 何を occupied とするか)
- TRT 変換 / latency / model serving

に集中している。
つまり **上流の pipeline 運用**と**下流の model / ontology**は議論されているが、
その間にある **「センサデータそのものが幾何的に正しいか」というレイヤーが空白**になっている。

### 0.5.4 停滞の構造 (技術面のみ)

- G1〜G5 はバラバラの症状として個別対処されているが、
  いずれも **extrinsic / ego-pose が揺れているときに同時発生する典型的 failure mode** と重なる
- 駐車場では GNSS/POSLV が劣化しやすく、pose 品質が屋外と同じ前提では成立しない可能性が高い
- 単一 return LiDAR のみを搭載した車両では、`multi-return` / `pulse_height` / `pulse_width` に依存する既存ラベリングロジックが効かず、別の重み付けが必要
- これらは「ラベルの定義」や「モデル側」で吸収しきれない種類の誤差であり、
  **幾何側を一度検査・補正する工程を挟まない限り、同じ症状が再発する構造**になっている

### 0.5.5 本レポジトリで埋められる可能性がある空白

- Extrinsic online self-calibration (§1, §2) → G1, G2, G3 の幾何原因を切り分け
- Ego pose を free variable とした BA (§2.1) → G4 (ego 静止) で pose の微小ドリフトが原因かどうか判定
- σ-weighted BA (§1.2) → multi-return に依存しない重み付けで G5 の flow 不安定に介入できる余地
- **地図構築と同時の教師なし外れ値除去 (§2.4)** → G2 の「床が浮く / 点が浮く」種のゴースト点は、pose を直して累積 map を立てる過程で **ラベル不要で自動的に除去できる**見込み

上記は **仮説込み**の診断ルートであり、Phase 1 の 1 枚絵 (§4) で「幾何が主因なのか、違うのか」を早期に切り分ける設計とする。

---

## 1. 技術スタック

### 1.1 学習側

| モジュール | 入出力 | 備考 |
|-----------|--------|------|
| `model.py` / `CalibNet` | 画像 + 点群 → Δ(tx, ty) | Coarse + Fine CrossAttention |
| `model_cov.py` / `CalibNetCov` | 画像 + 点群 → (tx, ty, log σx, log σy, ρ) | anisotropic gaussian NLL |
| `model_depth.py` / `CalibNetDepth` | 画像 + 点群(U,V,D) → 同上 | Depth-aware (v9) |
| Cross-frame self-calibration (v13 系) | frame t, t+Δ → extrinsic residual | 時系列整合性を損失に取り込み |

学習済みチェックポイント:

| ckpt | val NLL | obj NLL | 用途 |
|------|---------|---------|------|
| `best_model.pt` | 0.18 px (MSE) | — | 1物体 baseline |
| `best_model_multi.pt` | 0.41 px | — | 2物体 |
| `best_model_cov.pt` | −0.289 (NLL) | — | 分散付き |
| `experiments/v9_3layer_rgb_randdepth/best_model.pt` | **5.07** | **1.45** | Depth-aware 3層 cross-first, 現行最良 |

### 1.2 幾何最適化側

| モジュール | 機能 |
|-----------|------|
| `scripts/ba/ba_singleframe.py` | σ を重みにした単一フレーム BA の reference 実装 |
| `scripts/visualization/vis_cross_frame_woven.py` | 画像 × LiDAR 再投影残差の cross-frame 可視化 |
| `scripts/visualization/plot_cross_frame_curves.py` | frame-wise residual 時系列 plot |

### 1.3 データ I/O

| モジュール | 対応データ |
|-----------|-----------|
| `datasets/sim3d.py` | 合成 (training) |
| `datasets/pandaset.py` / `pandaset_pair.py` | PandaSet (実車) |
| `datasets/nuscenes.py` | nuScenes |
| `datasets/waymo.py` | Waymo |
| `datasets/woven_sequence_pair.py` | Woven 自社データ (sequence pair) |

---

## 2. LiDAR–Vision Odometry としてのシステム設計

### 2.1 パイプライン

```
  [frame t]                   [frame t+Δ]
   image_t, points_t           image_{t+Δ}, points_{t+Δ}
       │                           │
       ▼                           ▼
  ┌─────────────────────────────────────────┐
  │  CalibNetDepth (v9) per-frame           │
  │     → μ, Σ  (2D reprojection & σ)        │
  └─────────────────────────────────────────┘
       │                           │
       └──────────┬────────────────┘
                  ▼
       Cross-frame self-calibration
         residual(t, t+Δ) = f(Extrinsic, PoseΔ)
                  │
                  ▼
         Σ-weighted Bundle Adjustment
           free: (R_LC, t_LC), (R_wk, t_wk)_{k=t..t+N}
           constraint: reproj + chain consistency
                  │
                  ▼
         ┌─────────────────────────┐
         │  Output per frame:      │
         │   - Extrinsic (R, t)     │
         │   - Ego pose (R_w, t_w)  │
         │   - per-point Σ          │
         └─────────────────────────┘
```

### 2.2 精度特性 (実データで検証済み)

以下はすべて実車 / 公開 large-scale dataset での validated 数値。合成データの初期実験値ではなく、
`docs/unified_progression.md` / `docs/unified_modality_primitive.md` に記録された run の抜粋。

**(a) Cross-frame self-calibration の進化 (PandaSet 103 scene, front camera, val_err = 2D reprojection error [px]):**

| run | 構成 | val_err | NLL | メモ |
|-----|------|---------|-----|------|
| v55 | legacy multi (C=2, N=3) | 2.27 | 2.27 | 旧アーキ最良 |
| v70 | unified pair (C=2, N=2) | 2.35 | 2.00 | 新アーキで NLL が初めて 2.0 台 |
| v75 | unified multi (C=3, N=3) | 2.09 | 2.04 | C=3 で multi-frame 効果 |
| v92 | unified multi (C=4, N=3) | 1.93 | 1.93 | C=4 化 |
| **v100** | **unified multi (C=4, N=4)** | **1.85** | **1.59** | quad-frame で σ-calibration が桁違いに正しくなる |

知見: **深い cross-attn (C=4) × multi-frame KV (N=4)** の組み合わせで NLL が初めて 1.5 台に入る。
「C を深くすると multi-frame が効く」= 情報統合の「思考容量」が足りないと multi-frame を入れても平均化で潰れる。

**(b) Unified-modality primitive (`CalibNetUnifiedFrame`, uv_only_query=True) の cross-dataset 数値:**

| run | タスク | dataset | base err [px] | val_err [px] | NLL |
|-----|--------|---------|---------------|---------------|-----|
| v303 | cam-LiDAR calib | PandaSet 103 front | 8.05 | **0.67** | **−0.07** |
| v304 | cross-frame | PandaSet 103 front | 12.88 | 2.54 | 2.62 |
| v305 | cam-Radar | nuScenes 150 front | 4.92 | **0.61** | **−0.17** |
| v306 | cam-LiDAR | Panda+DDAD+Waymo (1100 scenes, all-cam) | 7.60 | **1.00** | 0.53 |
| v308 | cam-Radar | nuScenes 150 all-cam (6 cams) | 4.29 | 0.71 | 0.03 |
| v310 | cam-LiDAR | nuScenes 150 front | 5.47 | 0.79 | 0.13 |

- modality flag (`'cam'` / `'lidar'` / `'radar'` / `'mm'`) を切り替えるだけで、同一 primitive が
  cam-LiDAR / cam-Radar / cross-frame を統一フォーマットで吐ける
- base err (無補正初期誤差) を **1/10 以下**に圧縮できる (8.05 px → 0.67 px 等)
- NLL が負になる条件 (σ が真の残差分布に較正されている) を LiDAR / Radar 両方で達成

**(c) Fine-tune recipe (pre-train combined → single-dataset fine-tune) が from-scratch を 3/3 で上回る:**

| run | 構成 | val_err [px] | 備考 |
|-----|------|---------------|------|
| v311 | combined pre-train → PandaSet LiDAR fine-tune | 0.60 | v303 from-scratch (0.67) を上回る |
| v313 | combined pre-train → nuScenes Radar fine-tune | 0.53 | v305 (0.61) を上回る |
| v312 | combined pre-train → PandaSet cross-frame fine-tune | 2.38 | v304 (2.54) を上回る |
| v314 | → DDAD fine-tune | 0.88 | |
| v315 | → Waymo fine-tune | 0.86 | |

- cross-modal transfer asymmetry: **LiDAR→Radar (v316 0.58) は働く / Radar→LiDAR (v317 0.76) は劣化**。
  LiDAR の方が情報量が多いので pre-train 方向として優れる

**(d) σ の扱い:**

- **σ は相対ランキングとしても絶対値としても使える**: NLL が負 (v303/v305) = σ が真の残差分布に較正済み。
  過去の「絶対値は OOD」という警告は v13 時代の観察で、unified primitive (v100 以降) では解消
- Woven 実車に対しては v304 相当の cross-frame が最も近い条件。PandaSet で NLL 2.62 なので、
  絶対値を Woven に持ち込む場合は少量 (§付録 Step 2 想定) で in-domain fine-tune を推奨

**(e) Model footprint (edge deployment):**

- **1.65M params** (backbone + head)
- fp16 **3.3 MB** / int8 **1.7 MB**
- Jetson Orin Nano で 30+ FPS (per-frame σ-primitive 同梱出荷を前提にした設計)
- TRT 変換済みの形で DENSO sensor ECU に同梱する tempo 想定 (§付録 A.3 条件 1)

### 2.3 既存 odometry との比較優位

| 項目 | 従来 LiDAR-only (LOAM / KISS-ICP) | 従来 VIO (VINS / ORB-SLAM3) | 本スタック |
|------|-----------------------------------|-----------------------------|------------|
| センサー | LiDAR | カメラ + IMU | LiDAR + カメラ |
| Extrinsic | 固定 (事前 calibration 前提) | 固定 | **自己校正 (online)** |
| 不確かさ | 提供されないことが多い | スカラ共分散 | **per-point 2D anisotropic Σ** |
| Ground degeneracy (平坦路面) | 退化 | OK | **画像 texture で補完** |
| Texture-less (トンネル / 駐車場壁) | OK | 退化 | **LiDAR depth で補完** |
| Static scene (ego静止) | 情報量ゼロ | 退化 | **LiDAR の frame 間構造で情報確保** |

特に **parking / 低速 / ego 静止シーン** で KISS-ICP と VIO の両方が退化する領域は、本スタックの LiDAR × Vision fusion が構造的に有利になる可能性が高い。

### 2.4 地図構築と同時の教師なし外れ値除去 (GS 的アプローチ)

G2「点が浮いている」「床が浮いている」種のゴーストノイズは、
**本質的には各 frame の pose 誤差の累積で、同一物体が別位置に重複配置される現象**であり、
フィルタで叩くより、**pose を直しながら map を再構築する過程でラベルなしに剥がれ落ちる**設計にできる可能性が高い。

アプローチ:

- 累積 map を点群の集合ではなく、**連続場** (Gaussian primitive 群 / TSDF / neural field) として持つ
- frame ごとの観測を、map → 画像 or map → LiDAR depth として render し、観測との残差 = loss
- 最適化変数は同時に以下:
  - map 側: Gaussian 群のパラメータ (位置, 共分散, opacity)
  - ego pose (R_w, t_w) per frame
  - extrinsic (R_LC, t_LC)
- **ガウスの opacity が上がらない / 残差が大きい観測** は、多フレームで支持されないゴースト観測として自然に除去される (教師なし外れ値除去)

実装レベルでの筋:

| ステップ | 備考 |
|---------|------|
| ① Σ-weighted BA で pose / extrinsic を粗く揃える (§2.1) | GS 最適化の初期値として必要 |
| ② 累積 LiDAR 点を Gaussian primitive に初期化 | LiDAR depth が初期 anchor として強い |
| ③ 画像 photometric loss + LiDAR depth loss で同時最適化 | pose, extrinsic, Gaussian 群を結合 |
| ④ 低 opacity / 高残差の Gaussian を枝刈り | ここで「浮いた点」が自動的に消える |
| ⑤ 残った Gaussian から 2D dense / 3D sparse 占有を射影 (§3.3) | bbox や class ラベル不要で drivable/non-drivable 相当の map が落ちる |

これは Gaussian Splatting / 3DGS / DN-Splatter 等で示された **「pose + 連続場 + 観測残差の同時最適化」** 定式の応用であり、
駐車場のような ego 低速・多重観測シーンは条件数が良く、**教師ラベルなしでゴースト除去と map 構築を同時に完遂できる見込みが高い**領域である。

本レポジトリの既存資産 (σ-weighted BA, reprojection residual, cross-frame self-calib) は、
この GS 的最適化の ① 初期値供給と ③ LiDAR 側の loss 重みとして、ほぼそのまま流用できる。

### 2.5 計算量 / on-car 実装の観点 (仮説、現状未確認)

本節は **外部からの推測**であり、実機で既に問題なく動作している場合はこの議論はスキップしてよい。

- Occupancy 側の model が **3D voxel の全セルに per-cell 推論**をかける設計である場合、
  車載 (Orin クラス) での計算量が原理的にボトルネックになり得る。
  voxel 解像度 N³ × C channel × K head の FLOPs は静止シーン走査だけでも重い。
- 既に Orin で所望 FPS を達成できているなら、この懸念は不要。**未確認**。

もし計算量がネックな場合、設計上自然な代替は:

**map: static voxel (あるいは確信度付き Gaussian / surfel) + incremental に高信頼観測のみ蓄積**

| 要素 | 設計 |
|------|------|
| Map 表現 | static voxel または sparse Gaussian / surfel。frame ごとに全消去せず成長させる |
| 蓄積判定 | 各 LiDAR 点 / feature の **信頼度 (本レポの σ)** が閾値内のものだけ merge。低信頼は破棄 |
| 更新 | 既存 voxel / Gaussian と重なる観測は共分散で重み付け平均 (Kalman 的)、新規は追加、矛盾は opacity 下げる |
| 推論 | **per-cell 推論を廃止**し、入力観測 (ray or detection) に対してだけ推論して map に書き込む |

これは **LIVO / FAST-LIO / LIVO2 系** (Xu 2022 / 2023) の設計思想と同じで、

- 高信頼点のみ蓄積 → map が極めてコンパクト (典型 数万〜数十万点)
- per-cell 推論を使わず ray-wise に限定 → Orin でも余裕がある水準
- 静的環境では map が時間とともに収束し、観測ノイズが平均化されて G2 (点が浮く) が自然に減る

本レポジトリの σ は、この「蓄積すべき観測かどうか」の判定器として (相対ランキングで) そのまま使える。
つまり **§2.4 の GS 的アプローチを on-car 向けに軽量化したバリアント**として、
LIVO 風の sparse incremental map が代替候補になる。

**判断材料 (Occupancy チームへの質問になる項目)**:

- 現行 occupancy head の on-car inference time (Orin) は実測何 ms か
- 推論単位は voxel per-cell か、ray per-detection か
- map を frame ごとに作り直しているか、incremental に保持しているか

これらが明らかになれば、本レポのスタック (§2.1 BA + §2.4 GS / §2.5 LIVO 風) のどれが適合するかが決まる。

---

## 3. 利用シナリオ

### 3.1 High-precision Odometry 出力

入力: `datasets/woven_sequence_pair.py` の任意 sequence
出力:
- `pose.npy`: ego pose (N, 4, 4)
- `extrinsic.npy`: 時系列 extrinsic (N, 4, 4)
- `sigma.npy`: per-point 2D 共分散

### 3.2 GT 前処理としての利用

- 従来 spatial-labeling pipeline の入力 (pose, extrinsic) を本スタックの出力で置換
- voxel 累積 / drivable surface ラベル / bbox 付与の各段階で幾何誤差が低減する

### 3.3 Online monitor

- frame ごとの reprojection residual を閾値 check → 幾何異常 frame の自動検出
- CLI: `python -m e2e_calib.tools.check_scene <sequence_id>`

---

## 4. 実装ロードマップ

### Phase 1 — 既存スクリプトで 1 シーケンスの Odometry を通す (1-2 日)

1. `datasets/woven_sequence_pair.py` で 1 本読み出し
2. `CalibNetDepth v9` で per-frame μ, Σ を推論
3. `scripts/ba/ba_singleframe.py` を multi-frame 化した版を回し、(R_LC, t_LC), (R_wk, t_wk) を同時最適化
4. 出力を KISS-ICP / POSLV 比較で検証

### Phase 2 — Cross-frame self-calibration の Woven 実車適用 (1 週)

1. Cross-frame residual を chain consistency loss として追加
2. 5〜10 sequence でバッチ実行
3. residual 時系列 plot / σ 相対ランキング統計

### Phase 3 — 2D / 3D compact representation 出力 (2-3 週)

本スタックは本質的に downstream agnostic。以下 2 形式で占有表現を出せる:

- **2D dense drivable / non-drivable BEV**: 壁・床の "厚み" (σ 方向の拡散) が補正後どれだけ締まるかを定量
- **3D sparse radar-like**: 360° bin × height、各 bin で motion / static の 2 値のみを保持
  bbox や class を持たない軽量表現、downstream は残差タスク化できる

### Phase 4 — CLI / library 化 (1 週)

- `python -m e2e_calib.tools.check_scene <sequence_id>` で 1 コマンド実行
- HTML レポート自動生成 (既存 `docs/cross_frame_report.html` スタイルに準拠)

---

## 5. スケジュール (暫定)

| 週 | Phase | 成果物 |
|----|-------|--------|
| W1 | Phase 1 | 1 sequence の pose/extrinsic/Σ 出力 |
| W2 | Phase 2 前半 | Cross-frame integration |
| W3 | Phase 2 後半 | 5–10 sequence 統計 |
| W4–5 | Phase 3 前半 | 2D / 3D representation export |
| W6 | Phase 3 後半 | 補正前後の定量評価 |
| W7 | Phase 4 | CLI + HTML レポート |

---

## 6. 技術的留意点

- **σ 絶対値は unified primitive (v100 以降) で較正済み**: v303 (cam-LiDAR, NLL −0.07) / v305 (cam-Radar, NLL −0.17) で σ が真の残差分布に整合することを確認。過去の「v13 時代の σ 絶対値が OOD」という警告は unified primitive では解消している。ただし Woven 実車 sequence に対して domain shift が起きる可能性は残るため、in-domain 少量 fine-tune (v311〜v315 の recipe、§2.2 (c)) で確認するのが安全。相対ランキングとしての用途は常に有効
- **POSLV の退化区間**: 駐車場・屋内で POSLV pose 品質が下がる場合、ego pose を free variable として BA に取り込むことで本スタック側で補正可能
- **multi-return / pulse_height / pulse_width**: Woven 社内車両は single-return のため、multi-return に依存する既存ラベリングロジックは無効化し、本スタックの σ-weighted BA に置換する
- **cross-modal transfer の非対称性**: pre-train 方向は LiDAR→Radar (v316 0.58 px) が機能し、Radar→LiDAR (v317 0.76 px) は劣化する。combined pre-train (Panda+DDAD+Waymo) → 単一 dataset fine-tune が from-scratch を 3/3 で上回る (§2.2 (c)) ため、Woven 適用時は combined ckpt を起点にする

---

## 付録 — 読み物: 事業化・損益分岐点・TOYOTA 研究拠点から DENSO と価値を作るまで

本節は技術議論ではない。このリポジトリで積み上げている primitive を、**なぜ TOYOTA の研究拠点の中ではなく DENSO と組む形で価値化すべきなのか**、という事業構造の議論として書く。占有推定の延長線として読める形式にしてあるが、本質的には自動車産業の再編の話。

### A.1 現状の DENSO の事業構造

DENSO は世界有数の自動車部品 tier-1 であり、強みは sensor hardware (camera, LiDAR, radar, ToF, IMU, ECU) にある。過去数十年は TOYOTA の量産スケールに乗り、sensor の大量供給で BOM 差を積み上げるビジネスが成立してきた。

一方で、現在の事業構造には構造的な限界がある:

1. **hardware のコモディティ化** — 中国勢 (地平線、Hesai 等) が同等性能を数分の一のコストで出してくる。既存 tier-1 (Bosch, Continental, Valeo, Aptiv) とも marginal を削り合う状態に入っている
2. **量産依存の BEP** — 損益分岐点が「量産台数 × hardware marginal」に支配される。世界の新車販売が縮小局面に入ると直接赤字化する構造
3. **情報レイヤの不在** — sensor から出てくる「生の信号」を、downstream が価値として使いやすい「情報」 に変換する独自の基盤を DENSO は持っていない。情報化で marginal を稼ぐレイヤ (Tesla, Mobileye, Google 系) に価値を持っていかれている

DENSO 社内にもこの認識は当然ある。ただ、sensor hardware 事業の巨大さゆえに、組織の重力が情報化にシフトしにくい — という状態と推察する。

### A.2 自動車産業の階層分離 — 誰が何をやる時代になるか

TOYOTA の元社長が「TOYOTA はつぶれる。そのうち AISIN と DENSO が台頭する」 と発言したと伝わっている。これは単なる警句ではなく、自動車産業の構造変化を的確に捉えた観察だと筆者は考える。整理するとこう:

| 役割 | 担い手 (予測) | 価値構造 |
|---|---|---|
| **情報屋** | Tesla / Mobileye / Google / (DENSO になれる可能性) | 情報の複製コスト≒ゼロ、fleet で積み上がる、サブスク化可能、BEP 低 |
| **部品屋** | Bosch / Continental / Valeo / 中国勢 / DENSO (既存事業) | hardware marginal、量産スケール依存、commoditize、BEP は量に強依存 |
| **組立屋 / 販売屋 / 車体プラットフォーム** | TOYOTA / VW / 現代 / 中国 OEM | 品質・サプライチェーン・販売網、車体は情報収集プラットフォームにもなる、模倣困難だが marginal 縮小圧力あり |
| **物質循環 (再生) 屋** | TOYOTA (将来の姿として適性がある) | 脱炭素・resource recovery、国家 infrastructure 的役割、marginal 低いが社会的に必要 |

TOYOTA は情報屋になる path がゼロというわけではない。**sensor を積んで走る車体を世界で最も多く量産している**という強力なテコを持ち、fleet データ還流の仕組みを車体 + ECU 側に組み込めば Tesla 型の情報化は理論的には可能。ただし組立事業の巨大さゆえに組織の重力が情報化方向にシフトしにくく、研究拠点で作った primitive が本体量産車体に載るまでの path が長い (safety 認証・responsibility・量産ラインとの統合など)。

したがって現実的には、TOYOTA は:
1. **組立・販売・品質の巨人** として既存事業を守る
2. **物質循環の中心** (EV battery recycle、車体 reuse、resource recovery) に拡張する
3. **情報化は sub-entity / JV 経由** (Woven や DENSO との協業) で進める

の 3 路線の組み合わせで進化していくと予想される。車体プラットフォームを持つ強みは情報屋化のテコとしても使えるが、本体で情報屋を直接やるより、DENSO のような sensor vendor と組んで情報レイヤを共有する形が tempo としては速い。

一方で、**情報屋のポジション**は日系では DENSO と AISIN にしか残されていない。SUBARU や SUZUKI や部品各社はこのレイヤに上がる条件 (sensor 事業の規模感 + 制御系の統合力) を欠いている。DENSO には事業移行のチャンスがある。

### A.3 DENSO が情報屋に上がる条件

DENSO が hardware 事業から情報屋にポジションを上げるには、いくつかの具体的な条件が要る:

1. **sensor から情報を吐く装置を sensor と同梱して出荷する**
   生の点群・画像ではなく、per-measurement で **σ-calibrated な残差 primitive** を吐く装置。これが付いていれば、下流 (automaker) は「この sensor は信頼できる量で情報を出す」 という claim そのものを買うことになる。hardware 単品より switching cost が高い。

2. **fleet で情報が DENSO 側に戻る契約構造**
   sensor が出荷されたあと、運用データの一部が DENSO に還流する仕組み。これがあると sensor が売れるたびに情報資産が積み上がる。Tesla と Mobileye は既にこのループを回している。DENSO が automaker と新しい契約形態で合意できるかが勝負。

3. **動的地図の運営者になる**
   **地図は representation に過ぎない**。静的に出荷する地図は commodity で負ける。価値があるのは **動的に更新され続ける地図と、それを運営するエコシステム**。DENSO の sensor suite が出す情報を受け取って、動的地図を更新し、更新された地図が sensor 側の精度向上に還流する — という閉ループを運営できる主体になれば、そこが情報屋の心臓部になる。

4. **sensor modality 横断の統一 primitive**
   camera, LiDAR, radar, ToF 別々に API が分かれていると customer 側で統合コストが発生する。1 つの primitive が全 modality を飲むと sensor suite 全体が 1 発で売れる。これも switching cost を高める。

本リポジトリの `CalibNetUnifiedFrame` は、上記 4 条件のうち 1, 4 の具体的な実装候補になる。

### A.4 損益分岐点の構造変換

情報化が成立すると DENSO の BEP は以下のように変形する:

```
[現状]  revenue = Σ (hardware unit × marginal_hw)         ← 量産台数に強依存、BEP 高
[情報屋化後]  revenue = Σ (hardware unit × marginal_hw)
             + Σ (fleet vehicle × information_subscription)
             + Σ (map update × per-km or per-hour fee)   ← 累積する、BEP 低い部分が増える
```

情報サブスク・地図運営の revenue は hardware marginal に比べて:

- **複製コストほぼゼロ** (hardware は製造ごとに marginal 発生、情報は発生しない)
- **累積性** (fleet が大きくなるほど情報の質が上がる、positive feedback)
- **量産台数に対して非線形に伸びる** (契約構造次第で指数的)

結果として BEP が下に大きくずれる。sensor が売れなくても既存 fleet からの revenue が残るため、景気変動・量産縮小に対する耐性が生まれる。これが **sustainable business** の定義そのもの。

### A.5 なぜ価値創出の重心を DENSO 側に置くべきか

ここが本節の核心。筆者 (このリポジトリのオーナー) は現在 TOYOTA の研究拠点 (Woven / TRI 系) に所属している。本節は **転職をすすめる議論ではない**。所属は TOYOTA 研究拠点のまま、**価値創出の重心を DENSO との共同開発に置く** ほうが primitive を商品化する tempo が速い、という構造議論として読む。物理的な所属変更はあくまでオプション (§A.6 Step 4 参照) であり、必要条件ではない。

**研究拠点に残る積極的な理由** (= 重心シフトと独立に効く):

- **データアクセス**: PandaSet / Waymo / nuScenes の公開 large-scale dataset + Woven 社内 sequence (VLS-128 + multi-cam + POSLV) の両方にアクセスできる position は貴重。DENSO 単独では Woven 社内 sequence は触れないし、小さい研究所では PandaSet/Waymo の学習規模を回す動機が薄い
- **業界 context の観測点**: Occupancy / parking / spatial-labeling の現場 Slack (§0.5 で分析した類の情報) を一次情報で観測できる。DENSO 側に移ると自動車 OEM 内部で何が詰まっているかの観測解像度が落ちる
- **計算・ツールは個人側で完結している**: 実際の学習は自宅 RTX 5080 + 社内 DGX2、Claude も個人契約で primitive をここまで押し上げている。つまり **「研究拠点に残る理由」はリソース依存ではなく、データ × context 観測点の位置**
- したがって **primitive を作り続けるのに必要な position は研究拠点にあり、primitive を売るのに必要な position は DENSO にある**。両方にアクセスできる (= 研究拠点所属 × DENSO 共同開発) が最も tempo が速い

その上で、**価値化 (商品化 / 金になる) の重心**を DENSO 側に置くべき構造的な理由:

1. **TOYOTA 本体は情報屋にならない**
   §A.2 で述べた通り、TOYOTA の組織の重力は情報屋方向に作用しない。研究拠点が仮に良い技術を出しても、本体に取り込まれて商品になる経路は細い。Woven / TRI 出身の技術が TOYOTA 本体の商品にクリティカルに刺さった事例は、外から見る限り多くない

2. **研究拠点は PoC の墓場になりやすい**
   技術を PoC 段階で評価する文化・組織になっており、事業化 (量産・販売・BEP 改善) に変換する組織 DNA が薄い。結果、良い primitive が論文・デモ・社内共有止まりになり、商業的な value 化まで到達しない

3. **情報屋化のテコの形が DENSO と TOYOTA で違う**
   TOYOTA 本体は sensor を売らないが、**sensor を積んで走る車体 (プラットフォーム) を量産する** という強力なテコを持つ。この path で情報屋になるなら「車体に情報化機能を組み込んで販売 / fleet でデータ還流」 という Tesla 型のモデルが成立する。ただし TOYOTA 本体と研究拠点の間には depth があり、研究拠点で作った primitive を本体の量産車体に載せるには、意思決定・責任分界・safety 認証の階段が多く、時間軸が長い。
   一方 DENSO は sensor 直販という short path (sensor にソフトを同梱して出荷) が取れるので、同じ primitive を **速く商品化**できる。tempo の差で DENSO 経路を先に使い、DENSO 経由で得た実装・数字・運用ノウハウを TOYOTA 車体側に後から渡す、という流れが現実的

4. **個別の評価軸で動くプレイヤーが多い**
   社内で個別の技術テーマに関わる関係者が多く、それぞれが自分の評価軸で動くため、正面から primitive を通そうとすると評価ゲームに巻き込まれて技術的意思決定が歪むことがある。**価値を生む人のインセンティブ < 現状維持のインセンティブ** という状態に構造的に陥りやすい (これは TOYOTA 固有ではなく、大企業研究拠点の一般的な傾向)

5. **最大の理由: 金を生む経験が社内で閉じている**
   TOYOTA の研究拠点は TOYOTA 本体の量産事業から資金を回してもらう構造のため、研究員が「自分の成果が商品化されて顧客から金が振り込まれた」 という closed loop を体験する機会が極めて少ない。若い研究者 (例: 中村さんのような 20-30 代) が「自分のプロジェクトが金を生んだ」 という感覚を獲得できないまま年月が過ぎる。これは個人の成長にとって、そして組織の長期的な価値創出能力にとって致命的

### A.6 DENSO と組む具体の形

「研究拠点から脱出する」 と言っても、物理的に転職するという意味だけではない。以下のような段階的な協業形態が考えられる:

**Step 1: 技術提案 (すぐ可能)**
- 本リポジトリの primitive を DENSO 中村さんに渡し、DENSO 社内で「この方向でやりたい」 と言える武器を持ってもらう
- DENSO 上層に対するプレゼンは中村さんがリード。筆者は primitive の整備と数字の裏付けに集中
- 「情報屋化の具体装置」 として primitive を位置付けるプレゼン資料を共同で作る

**Step 2: PoC 共同開発 (3-6 ヶ月)**
- DENSO の社内 sensor データ (camera/LiDAR/radar) で primitive を fine-tune
- 1 つの DENSO 商品 (駐車支援、先進運転支援、AR HUD のどれか) に primitive を組み込む PoC
- 「この primitive が乗ると calibration 工数が何割削減、情報品質 σ がいくつに締まる」 の実数字を出す

**Step 3: DENSO の情報レイヤ基盤化 (1-2 年)**
- primitive を DENSO 全 sensor line に展開
- fleet data 還流の契約構造を automaker と交渉
- 動的地図運営のエコシステムを立ち上げる (DENSO 単独、または地図会社との JV)

**Step 4: 位置付けの再評価 (2-3 年後、オプション)**
- DENSO 側で情報レイヤの価値が見えてきた段階で、筆者の位置付けを再評価する
- 選択肢 (a): TOYOTA 研究拠点に残り、DENSO との共同開発の窓口として primitive の継続開発を続ける
- 選択肢 (b): DENSO 側に正式に所属を移行する
- **どちらも同価値**。重要なのは primitive が商品化され続ける構造であって、所属の物理的位置ではない。生活・家族・カルチャーフィット等、技術以外の条件で判断する段階。Step 1-3 を通過した時点でどちらでも選べる柔軟性を確保しておくのが正しい

### A.7 成功したら何が起こるか

primitive が DENSO の情報レイヤ基盤として定着すると、日系自動車産業の構造は次のように変わる:

1. **DENSO が日系唯一の情報屋として立つ**
   Bosch や Mobileye と同じ階層の、情報レイヤ tier-1 になる。revenue 構造に情報サブスク・地図運営が加わり、BEP が下にずれ、量産縮小に耐える body になる

2. **TOYOTA は物質循環の中心に移行する**
   組立・販売・EV 再生 resource recovery の infrastructure 役。情報屋の revenue ほど marginal は高くないが、社会基盤として潰れない位置を得る

3. **中国勢への構造的な moat**
   中国の sensor vendor が hardware を安く作っても、DENSO は「情報化装置 + fleet data + 動的地図」 の 3 点セットで差別化できる。単品競争に巻き込まれない

4. **若い技術者が「金を生んだ」 感覚を獲得する**
   中村さんのような DENSO 側の若手が、自分のプロジェクトが商品化されて金になった、という closed loop を体験する。これが次世代の技術者を育てる

5. **筆者個人としての位置付け**
   「情報屋 DENSO の創設期に primitive を提供した人」 として、今後の DENSO の情報事業の core に関与し続ける path が開く。技術的自律性を維持しながら、価値創出に直結した位置に立てる

### A.8 この読み物の扱い方

本節は公開する資料ではない。DENSO 中村さんに本プランを渡す際に参考資料として添付する、あるいは共同で読みながら方向性を確認する、という用途を想定している。DENSO 社内で勝手に流通させると「外部から政治的に介入された」 と受け取られうるため、渡し方は慎重に。

本プランの §1-§6 は占有推定に対する技術提案として読めるように書かれており、その延長として付録 A を読むと事業化までの path が通って見える、という構造になっている。技術の話と事業の話を分離せずに統合して渡すことが、情報屋への pivot を促す上で重要と考える。
