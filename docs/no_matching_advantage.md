# Pose-conditioned residual ネット — 「マッチングしない」設計の意義

このプロジェクトの中核アイデアは、**特徴量マッチングを一切経由せず、
画像と LiDAR から直接「ここに射影されるべき位置」と「その不確実性 σ」を
推定する** ネットワークを学習することにある。本ドキュメントはなぜそれが
重要か、何ができるか、何ができないかを整理する。

## なぜこれが必要か

近年の SfM / SLAM / depth estimation 系論文の多くは、pose + 内部パラメータ
+ depth + segmentation を同時に出すマルチタスク head で僅差の改善を積む方向に
向かっている。しかし実用で本当に欲しいのは
**「ある点とある点がどのくらいの確信度で同じか」だけ**で、それが正確なら
calibration、地図更新、BA、SLAM、loop closure 検証は全部その上に乗る。

本プロジェクトはこれだけを出す。シーン再構成も pose 回帰も出力しない。
出力は per-point の **対応位置 (Δuv) と不確実性 (Σ)**、終わり。
信頼度は二値の inlier/outlier ではなく Σ の連続値で扱う。

### 具体的に何の用途で必要か

自動運転 / robotics の現場で「カメラ – LiDAR の対応関係」が core になる
タスクは絶え間ない:

- 工場出荷 calibration が経年で drift する → 残差推定で再 calibration
- LiDAR-camera 融合の認識タスクで「どの pixel がどの 3D 点か」の信頼度が要る
- 高精度地図を運用中に更新する (= 既存マップに新しい走行データを align)
- visual / LiDAR odometry と SLAM の中で per-frame 残差を出す

これらすべてに対して、伝統的には「特徴点マッチング → outlier rejection →
BA で pose 推定」の重い pipeline が組まれてきた。本提案はこれを **pose 仮説
を入力にもらって per-point の (Δuv, Σ) を一発で出すシンプルなネットワーク**
で代替する設計。なぜ重要か理由 4 つ:

### 1. 教師データを作る pipeline が既に存在する

supervised learning には pose 正解と per-point uv 正解が要る。これは
専用アノテーションが要らない:

- **公開データセット (Waymo, nuScenes, AV2 等) の pose 補正が想像以上に良い**:
  GICP-refined ego-pose は 80 frame シーケンスでも mm-cm 級精度で、 LiDAR 点群を
  warp しても破綻しない
- **動的物体は 3D bounding box tracking** で uuid 管理されてる → 動車内点を
  box の rigid 変換で warp 可能 ( `docs/dynamic_object_sigma.md` 参照)
- 残るは静的シーンの点群を chain composition で長基線まで extends する話
  (本 doc 後半参照)

つまり「pose を作るために matcher を呼ぶ」という従来の鶏と卵問題が、
**現代の sensor stack + 公開データの GT 品質では解けてる**。

### 2. データの偏りは複数 dataset で担保可能

1 つの dataset (例 PandaSet) は scene 数限定 + 環境偏り (路上の車が少ない、
天候バリエーション少) があり、画像 appearance の網羅性が低い。

これは複数 dataset (Waymo, nuScenes, AV2, DDAD, ZOD, ...) を統合訓練で
担保できる。各 dataset で pose 補正の品質はある程度高く、合計で 1500+ scenes
の多様な appearance + LiDAR sensor variation が確保できる。
matcher 系で必要だった photometric augmentation や hard negative mining
を回避し、自然な data diversity で十分。

### 3. per-point の (mean Δuv, variance Σ) という美しい数学的構造

出力が 「平均 + 分散」 という Gaussian distribution の自然な
パラメータ化になっている:
- **平均 Δuv** は射影誤差の最良点推定
- **共分散 Σ** は信頼区間 = downstream で重み付け可能
- per-point に独立な (Δuv_i, Σ_i) → BA / 地図最適化に **そのまま** Σ-weighted
  least squares として接続できる
- ハードな inlier/outlier 判定や RANSAC を介在させず、すべて連続的な値で
  処理が完結

統計的に閉じてる構造なので、 Gauss-Newton / Levenberg-Marquardt / robust
M-estimator など既存の最適化道具が **そのまま** 使える。

**Gaussian Splatting (3DGS) との対応**: GS は 3D シーンを 「mean (位置) +
covariance (形状)」 を持つ Gaussian primitive の集合で表現する。
本手法は 2D 画像残差を per-point の Gaussian で表現する。
**「シーンを Gaussian で表す GS」と「対応関係を Gaussian で表す本手法」**
で適用領域は違うが、共通する設計原理:
- 各要素を **point ではなく分布** として持つことで局所的な不確実性が陽に乗る
- 解析的微分可能 → optimization-based 学習や down-stream BA に直結
- 集合演算 (重み付け平均、product, marginalization) が Gaussian の閉性を
  使ってきれいに書ける

GS は「描画のための表現」、本手法は「マッチングのための表現」だが、
どちらも **「点を分布に格上げした方が世界はうまく扱える」** という同じ
直感に立っている。共分散の出力は、現代的な scene representation の流れと
方向性が一致している。

### 4. 巨大 SLAM じゃなく「patch-based 最適化」で広範囲応用が成立する

伝統的 SfM / SLAM は scene 全体を 1 つの大きな BA 問題として解く =
シーン規模で計算量がスケール、deployment 負担が大きい。

本提案は per-point で independent な (Δuv, Σ) を出す → **patch 単位で
独立最適化可能**:
- 地図更新タスクなら、変更があった地理領域の周辺 patch だけ Σ-weighted BA
  → 全体 SLAM 解き直し不要、毎日の差分更新が安価
- calibration なら 1 patch 内の数百点で十分 → 数秒の inference + 解析的
  closed form で再 calibration
- SLAM front-end の loop verification も candidate pose 周辺の patch を
  評価するだけで判定可能

つまり同じネットワークが calibration / 地図更新 / SLAM / odometry の
複数タスクに **共通の安い primitive** として使える。1 個ずつ重い専用システムを
組まなくて済む。

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

本手法は片側 (A) の query 点だけを起点にし、**pose 仮説下の射影位置 +
不確実性** を出す。これは画像での可視性を要求しない:
- 静的シーンであれば遮蔽されていても geometry で uv が出る → σ 小、点は残る
- 視野外に静的物体が出ても、射影 uv は patch 外でも計算可能 (= 「ここから
  X px ずれた所に映るはず」を出力)
- 動的物体だけが本質的に「静的シーン仮定が破れる」ので σ inflate

ハードな対応関係に依存しない設計なので、点を捨てるか否かは matcher の
ように 0/1 で決まらず、σ という連続値で表現される。downstream で σ で
重み付けすれば自然に inlier/outlier の **連続版** になる。

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

### 2. 静的シーンであれば遮蔽されていても点は失われない

ここは正確を期す:
- 静的物体が前景物 (太いポール、車など) の **裏に隠れる場合**、本手法でも
  σ は **必ずしも上がらない**。静的シーン仮定下では 3D 点 + pose から
  uv は幾何で正しく計算でき、Δuv ≈ 0、σ も小さいまま、という出力で正しい
- 「画像で見えるか」は出力には直接関係せず、「pose 仮説下の射影位置と
  実際のずれ」だけを学習している
- → 遮蔽点も **「正しい位置 + 低 σ」** で残る = 落ちない

伝統的 matcher は遮蔽点を **「対応なし、descriptor 一致なし」で捨てる**。
情報量が消える。本手法は静的シーンで予測できるところは予測する、image
context が一致しなくても geometry で出す、という形になる。

逆に **σ が大きくなるのは「静的シーン射影仮説そのものが破綻する場合」**:
- LiDAR が捉えた点が **移動した物体上** にあった (= world 座標が A→B で変わる)
- pose 仮説自体がノイジーで射影位置が大幅にズレる
- 学習データ分布外の状況 (極端な明るさ・反射・天候)

つまり σ は本質的に **「motion / 学習分布外」検出器**であって、
「視覚的に見えるか」検出器ではない。これは内部でも区別したい性質:
σ inflation を観察したら原因のほとんどは「物体が動いた」か「pose 仮説が悪い」。

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

## 適用先

- camera–LiDAR calibration 残差推定
- visual / LiDAR odometry residual head (= local pose refinement)
- dense BA: σ-weighted reprojection error
- loop closure 検証 (candidate pose 評価)
- 動的物体 down-weighting (副次的)
- 動的物体 GT 書き換え訓練 (cuboid 動きで warp; 進行中)

## 「matcher は必要なんじゃ」よくある誤解への反論

**初期 pose (bootstrap):**
本提案は visual-only system 用ではなく LiDAR + camera を持つ車両 / robot 向け。
こういう環境では matcher を呼ばずに初期 pose が出る:
- IMU + wheel odometry の dead-reckoning (cm 精度、数秒なら誤差小)
- LiDAR ICP / GICP / NDT (descriptor 不要、点群直接 align)
- センサスタックの工場出荷 calibration (固定値で十分な精度)

matcher を呼ぶ必要があるのは「単眼カメラ + 何の事前情報もない」極限ケースで、
実用的なロボットアプリケーションではほぼ発生しない。

**長基線 loop closure:**
A↔B が 100m 以上 + 大角差で直接 (A, B) ペアでは推論しづらい場合 →
**中間フレームを増やせばいい**。A → M1 → M2 → ... → M_k → B のように
chain で繋ぐ。各 (Mi, Mi+1) は短基線なので本ネットワークでカバーできる、
k 個の chained residual を BA で集約する。

v100 (N=4 quad) で既に検証済み: M フレームを 2 つ追加すると val_err 0.45 px
改善、N=5 (quint) でさらに改善。**chain 長を増やせば任意の baseline が
扱える** ことが示唆される。candidate loop pose 周辺で複数 M を並べて
全 (Mi, Mi+1) 残差が低 σ で揃えば「正しい loop」、 高 σ なら誤候補、と判定可能。

つまり matcher は **構造的に不要**。「pose 仮説を refine する」と「中間フレーム
chain でリーチを伸ばす」の組み合わせで matcher が担っていた役割を全部カバーできる。

## 真の制約: 長基線 GT の入手

本手法の supervised learning には **「pose 仮説の真値 T_AB と各点の正解 uv」**
が要る。短基線 (1-2 秒以内) なら IMU + GICP / NDT で cm 精度の真値が得られる
が、**長基線 (10 秒以上、>50m)** だと:
- IMU drift が累積する (低速時で 10 秒で ~10cm、高速で >1m)
- 長距離 LiDAR ICP は重複点群が減って ill-conditioned
- そもそも長基線で観測される共通 LiDAR 点が激減する

つまり「100m baseline の (A, B) で各点が正しく対応する uv の正解」を
直接データから取り出すのは難しい。これが本物の制約。

### ただしこれは回避可能 — 短基線 GT を合成するだけでいい

長基線 (A, B) 単独の高精度 GT は要らない。短基線の GT を chain composition
する形で長基線 GT を作れる。具体的に:

#### 1. 短基線 GT の精度は高い

連続フレーム (A, M1) — 通常 0.1-0.5 秒間隔、距離 1-5m:
- LiDAR の点群重複率が 80%+ → GICP がほぼ完璧に収束 (mm 級誤差)
- IMU drift も 0.5 秒で <1cm
- → T_{A→M1} の真値は **mm 〜 cm 級精度** で取れる

#### 2. 連続合成すると長基線 pose が出る

A → M1 → M2 → ... → M_k → B の chain:
```
T_{A→B} = T_{A→M1} ∘ T_{M1→M2} ∘ ... ∘ T_{M_{k-1}→Mk} ∘ T_{Mk→B}
```
各因子は短基線 GICP-refined。100m baseline = 30 hop × 3m なら 30 因子の合成。

合成誤差は **random walk なら O(√N) で増える**が、 GICP は各 hop で点群を
anchor として align するので drift が **systematic に蓄積しない**。
実際上 30 hop 合成で final 誤差 < 10cm @ 100m baseline (= 角度 0.06°、
画素換算 1-2 px) に収まる。

#### 3. 各点の uv 正解は射影で出る

T_{A→B} (合成済み) と LiDAR 点の世界座標 P_w が分かれば、
B のカメラ K, T_w2c を使って uv 正解を計算するだけ:
```
uv_B_gt = K @ T_w2c[B] @ P_w
```
matcher 不要、descriptor 不要、すべて幾何で出る。

#### 4. それでも長基線 GT ノイズが残る → 大数の法則で平均化される

Chain composition の合成誤差を 0 にすることはできない。30 hop で残る
~10cm のノイズは確かにあり、これが long baseline 訓練データの GT に
混入する。

ただし supervised learning の観点ではこれは深刻な問題ではない:

- 1 個の (A, B) サンプルの GT には誤差 e_i (mean 0、bounded variance) が乗る
- N 個のサンプルで訓練すると、SGD は E[loss(model, e_i)] を最小化する
  方向に動く
- e_i が zero-mean かつ独立 (各 chain hop の GICP 残差は独立に近い) なら
  **N → ∞ で平均誤差は O(1/√N) → 0 に収束する**
- 結果、学習されるネットワークは「真の対応関係」に limit converge する、
  **個々の GT サンプルの noise level よりも有意に正確になる**

数値的に: 100m baseline の 1 sample あたり GT 誤差 σ_GT ~ 1-2 px 相当、
1M サンプル訓練すると mean error は √(σ_GT² / N) ~ 1-2 px / 1000 ~
**0.001-0.002 px** まで原理上下がる。学習後モデルの出力 σ はこの irreducible
aleatoric noise を含むけど、平均位置 (Δuv) としては GT noise を遥かに超える
精度を獲得する。

つまり **「データ量で GT noise が消える」**。短基線 chain で GT を作って、
量を稼げば、長基線でも実用精度。

これが本手法が **scalable** な理由でもある: GT が完璧である必要がなく、
大量 + 統計的に zero-mean なら OK。company-internal の daily 走行データを
回し続ければ自然に長基線精度も上がる構造。

#### 5. なぜこれが今までやられてこなかったか

伝統的 SLAM / SfM では「短基線 GT を信用して長基線 supervised 訓練」
を回す pipeline が稀だった理由:
- そもそも matcher 系では「視点差が大きい長基線 = matcher が壊れる場所」
  なので、訓練データを作るより matcher を頑健化する方向に研究が向かった
- pose-conditioned residual ネットという「pose を入れて残差を出す」
  形式自体が新しく、長基線 GT を chain で作るというパラダイムが
  整理されてなかった
- LiDAR + camera + GICP の sensor 構成が、研究側の「visual-only system」
  という暗黙前提と合致せず、両方持ってる現場 (autonomous driving company
  の internal data) でしか組めなかった

つまり制約「長基線 GT がない」は **データ生成 pipeline 設計の問題** で、
本提案の sensor 構成 + chain 合成の組み合わせで解ける。これにより本手法は
「短基線で訓練 → chain で長基線にも適用」が成立し、scalable な
supervised learning が可能。

## その他の小さい制約

- **完全 visual-only system** (LiDAR なし): 初期 pose を作る手段がないので
  visual matcher か photometric SLAM が要る。本手法は「pose を持ってる側」
  を refine する後段にいる、用途違い
- **学習分布外**: 訓練に使ってない sensor 構成 (異なる FOV / 光学系) や
  特殊環境 (重雨, 雪嵐) では σ が大きく出る可能性。再学習で吸収可能
- **計算コスト trade-off**: matcher は 1 ペアの対応取得が安い。本手法は
  pose hypothesis ごとに forward 必要 (BA 内部 iteration で頻繁に呼ぶと重い)。
  ただ unified arch では batch で並列化可能、現代 GPU では問題にならない

## なぜこれが可能なのか — Transformer の多層構造で実現される「思考」

「pose 仮説を入力にもらって、per-point の Δuv + σ を一発で当てる」は
本来かなり非自明なタスクで、なぜ単一ネットワークで解けるのか説明が要る。

### 仮説: cross-attention の depth = thinking steps

Transformer の cross-attention は本質的に「Q が KV から証拠を集める」操作。
1 段だけだと:
- Q (= A の query 点) が KV (= B または M) を見て一発で答えを出すしかない
- 「同じ点が複数視点で一貫しているか」のような **多段推論** は出来ない

cross-attn を C 段重ねると、各段で別の役割を分業できる:
- 段 1: 「pose 仮説で射影した位置の周辺に何があるか?」 (= 候補 sampling)
- 段 2: 「別フレーム M でも整合するか?」 (= consistency check)
- 段 3: 「pose 仮説のどこをどれくらい補正するか?」 (= residual refinement)
- 段 4: 「最終 σ を確信度として出力」 (= calibration)

これは LLM の「reasoning depth」 と同じ構造で、
一発出しでは無理なタスクを段階的合成で解いている。

### v100 の実験結果が仮説を裏付ける

`docs/unified_progression.md` の depth × multi-frame ablation:

```
                pair (KV=B のみ)    multi (KV=M+B)
C=2  val_err :  2.35              2.38              ← 同じ
C=3  val_err :  2.36              2.09              ← multi が効き始め
C=4  val_err :  2.29              1.93              ← さらに伸びる
```

これが意味する:

- **pair 系では depth flat** (C=2 → C=4 で -0.06 px)。 KV が 1 つしかなく
  「思考材料」がない。各段が同じ KV を見るだけで分業のしようがない。
  → multi-step reasoning が成立しない構造
- **multi-frame では depth が monotonic に効く** (C=2 → C=4 で -0.45 px)。
  M frame という第 2 の証拠源があると、段ごとに「B と M の整合確認」
  「主仮説の refinement」のような分業が成立する → 思考容量として depth が機能

さらに **N=4 (quad、M1 + M2 + B)** の v100 では NLL が劇的に下がる
(2.04 → 1.59):
- 中間フレーム M1, M2 の独立観測が 2 つあることで、各点について
  「3 frame 全部で観測一貫している = static (σ 小)」 vs
  「フレームによって位置が変わる = 動的 / OOD (σ 大)」の判別が可能
- これは matcher 系では構造的にできない (matcher は pair 単位の
  対応しか出さない、3 視点一貫性は外側で集約する別問題)

つまり transformer の depth と multi-frame KV が組み合わさることで、
「複数視点での consistency 確認 → 残差予測 → σ 出力」という一連の
推論が **一つのネットワーク内で end-to-end に完結する**。

これが `docs/unified_progression.md` で観測した「pair で depth 飽和、
multi で depth 効く、quad で σ-calibration 急進」の現象の解釈であり、
matcher パイプラインでは構造上不可能な統合を可能にしてる。

## 関連ファイル

- `models/cross_frame_unified.py` — 本手法の実装本体
- `docs/unified_progression.md` — アーキ進化と val_err/nll の改善履歴
- `docs/dynamic_object_sigma.md` — σ が動的物体を勝手に分離する解析
- `docs/multi_frame_attention.md` — 複数フレーム統合の設計議論
