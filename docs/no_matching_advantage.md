# Pose-conditioned residual ネット — 「マッチングしない」設計の意義

このプロジェクトの中核アイデアは、**特徴量マッチングを一切経由せず、
画像と LiDAR から直接「ここに射影されるべき位置」と「その不確実性 σ」を
推定する** ネットワークを学習することにある。本ドキュメントはなぜそれが
重要か、何ができるか、何ができないかを整理する。

## 解いている問題

- 入力: query フレーム A の点 (画像 + LiDAR の組)、target フレーム B、
  両者の **相対 pose 仮説 T_AB_hat** (ノイジーで構わない)
- 出力: 各 query 点について
  - **Δuv**: B 画像内での投影誤差補正
  - **Σ**: その補正の 2D 共分散 (確信度)

出力は per-point の (Δuv, Σ)。pose 自体は出力しない (= **入力条件**)。

## 従来手法の構造とその痛み

```
画像 ─→ keypoint detector ─→ descriptor ─→ matching ─→ outlier rejection ─→ BA
                              (SIFT/ORB/SuperPoint)    (RANSAC)
```

各ステップに固有の壊れ方がある:

1. **descriptor が誤対応を出す**: 繰り返しテクスチャ、明度変化、視点差大で破綻
2. **matching threshold は手動**: 真に正しいか分からない、保守的にすると密度落ちる
3. **outlier rejection 後 inlier しか残らない**: 動的物体・遮蔽・反射は **完全捨て** (情報損失)
4. **σ は事後推定**: residual 残差から共分散を逆算するが、
   スパースサンプル + 仮定 (重み付き Gaussian) のずれで大幅にバイアス
5. **学習可能性が薄い**: descriptor は self-supervision (Photometric loss) で
   学べるが、threshold / RANSAC / BA solver は離散的・非微分

要するに **「対応はあるかないか」を 0/1 でハードに切る** 設計のため、
中間状態 (= 動いている、半遮蔽、視差大) の情報がパイプライン途中で切り捨てられる。

## 提案: pose-conditioned residual ネット

```
query 点 + image_A + LiDAR_A + image_B + LiDAR_B + T_AB_hat
                    │
                    ▼
              (cross-frame attention)
                    │
                    ▼
         per-point (Δuv, Σ) 予測
```

特徴:

- **descriptor も matching も介在しない**。モデルはローカル context (画像 +
  LiDAR pt grid) を見て、「pose 仮説を信じてここを見たが、本当の位置は Δuv
  ずれる、確信度はこの程度」を直接出す
- **pose は入力**: 1 ピクセル単位の絶対位置を当てる学習ではなく、相対補正
  という小さい量を推定するから収束が速い
- **σ は出力の一部**: ハードな inlier/outlier 判定なし、すべての点に重みが
  付く

## 「マッチングしない」ことの具体的利点

### 1. 動的物体を壊さず吸収する

学習中に「動いている」というラベルを与えていなくても、出力 σ は動的物体に
対して自動で増える (詳細: `docs/dynamic_object_sigma.md`)。

```
σ_pred (px):    bg=0.87  parked=0.75  stopped=0.87  moving=1.30
```

伝統的な matcher は移動車を **不一致** とみなして排除し、その点群が消える。
本手法はその点を **「不確実」** として残す。Σ-weighted BA に直接渡せば、
信頼度に応じた寄与で扱える。

### 2. 遮蔽・反射・繰り返しテクスチャを「σ 大」として残す

descriptor 系は「マッチに失敗したのでこの点は使えない」と捨てるが、
本手法は「この点はあるべき位置に近づけたが自信ない」を出す。
点を捨てない = 密度の高い再投影誤差マップが手に入る。

### 3. ハイパラがほぼない

- threshold, ratio test, RANSAC iterations, inlier ratio: **全てない**
- 唯一: σ_pred を信頼するかどうかの downstream 重み付け (= softmin / Mahalanobis 距離) のみ

### 4. 任意の pose 仮説に対して評価できる

pose を入力にすることで:
- BA の中で pose を更新するたびにモデル再評価 → コスト関数の値が出る
- calibration: noisy initial pose をモデルに通す → どれくらい外れてるか σ が示す
- SLAM front-end / loop closure 検証: candidate pose に対して全点 σ を見れば
  「まともな pose か」即判定可能

### 5. supervised on **cheap** data

- 学習データは「synced multi-sensor sequence + 高精度 GICP-refined pose」
- アノテーション要らず。company-internal data で daily-scale でラベル無しに
  集まり続ける形態で大量に学習可能
- descriptor 学習に要する photometric / triplet loss / hard negative mining
  系の非自明な学習スキームに比べて遥かに直接的

## どこまで効くか / 限界

### 効く

- camera–LiDAR calibration 残差推定
- visual / LiDAR odometry residual head (= local pose refinement)
- dense BA: σ-weighted reprojection error
- loop closure 検証 (candidate pose 評価)
- 動的物体 down-weighting (副次的)
- 動的物体 GT 書き換え訓練 (cuboid 動きで warp; 進行中)

### マッチングがやはり要る (出来ないことの正直な開示)

- **絶対 pose 初期化** (= bootstrap):
  - 何もない状態から pose を作るには結局 keypoint matcher 必要
  - 提案手法は **「pose 仮説を refine する」** 機能であって「pose を作る」
    機能ではない
- **長基線 loop closure**:
  - A↔B の baseline が長すぎる (例: 100m 以上 + 大角差) と、
    モデルが「同じ点が両画像のどこに対応するか」の自信を持てない
  - こちらは matcher で初期 pose 候補出してから refine、の順番

つまり提案手法は **「pose を持ってる側」の問題を解く** ものであって、
matcher と互換ではなく **後段** にある。

## 関連ファイル

- `models/cross_frame_unified.py` — 本手法の実装本体
- `docs/unified_progression.md` — アーキ進化と val_err/nll の改善履歴
- `docs/dynamic_object_sigma.md` — σ が動的物体を勝手に分離する解析
- `docs/multi_frame_attention.md` — 複数フレーム統合の設計議論
