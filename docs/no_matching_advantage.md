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

## 既存の E2E 系手法と何が違うのか

過去 5 年で「特徴量マッチングを学習可能にする」「マッチング自体を回避する」
という方向の研究が大量に出ている。代表例:

| 系統 | 代表手法 | 出力 | pose の扱い | 残差/σ |
|---|---|---|---|---|
| 学習型 matcher | LightGlue, RoMa, DKM | 対応点ペア + 信頼度 | 出力 (RANSAC) | 暗黙 |
| Detector-free matcher | LoFTR, ASpanFormer | dense correspondences | 出力 | 暗黙 |
| Differentiable SfM/SLAM | DROID-SLAM, ACE | 密 flow + dense BA | **出力** (推定) | flow 信頼度として |
| Pair → pointmap | DUSt3R, MASt3R | 3D point map (両視点) | **出力** (pointmap から逆算) | per-point depth conf |
| Multi-view → all | VGGT, Fast3R, Spann3R | poses + intrinsics + 3D | **全部出力** | per-token confidence |
| 学習型 PnP | PixLoc, DSAC* | pose 直接回帰 | **出力** | scene-coord 残差 |
| **本手法** | (this) | **per-point (Δuv, Σ)** | **入力** (条件) | 直接出力 |

決定的な違い:

### A. pose を「出力」ではなく「入力 (条件)」にする

DUSt3R/VGGT/DROID 系は最終的に pose を **回帰** する。学習はラベル付き pose
で supervised、推論時は入力画像群から pose を「当てる」。これは強力だが:
- ノイジーな初期 pose があるシナリオで使えない (= 自分で出した pose で
  上書きしてしまう)
- BA / SLAM の **iter ごとの pose 仮説評価** に使えない (毎回別 pose を
  出力しちゃう)

本手法は **pose 仮説を入力**。「この pose で正しいか? どれくらいズレてるか?」
を per-point で answer する。downstream solver (BA, calibration, SLAM front-end)
が pose を更新するたびに評価できる、汎用 residual head として使える。

### B. 出力が「シーン再構成」ではなく「残差 + 不確実性」

DUSt3R / VGGT は 3D 点群 / 深度マップを出す → そこから downstream で BA
する。これは **シーン全体を再構成する** 重い問題を解いている。

本手法は **pose 仮説に対する局所残差** だけ出す。シーン構造の再構築は不要。
代わりに「同じシーンの別 pose 仮説」「同じ点の別フレーム評価」が高速で
回せる、軽量で再利用しやすい layer。

### C. matcher 系全般の前提「両方見えてる」を外せる

LightGlue / LoFTR / DUSt3R も含めて、対応学習型は基本的に
「両画像に同じ点が見えている」前提で設計されている。片方にだけ写って
いる点は noise (= 学習で penalty / 切り捨て)。

本手法は片側 (A) の query 点だけを起点にし、B への射影予測 + σ を出す。
**B に写ってなくても点は捨てない**。σ が大きい状態で残る → 動的物体・
遮蔽・視野外移動を「不確実」として保持する。

### D. 視点変化を invariance で吸収しない

descriptor / 学習型 matcher は「同じ 3D 点は視点が変わっても似た特徴量を
出す」invariance を学習タスクで内製してきた。これは大角度で破綻しやすい
(>45° 視差、近景 → 遠景の急変)。

本手法は **pose を入力にもらう** ので「視点が変わる」を invariance で
解決する必要がない。射影位置を計算で出して、その周辺 context を見る。
角度差が大きくても σ がそれを反映するだけ、descriptor invariance のような
学習困難な不変量を要求しない。

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
