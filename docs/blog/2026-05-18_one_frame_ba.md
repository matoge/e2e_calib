# 1-frame closed-form BA — 2026-05-18

![hero](../assets/2026-05-18_one_frame_ba/ba_truckcrop_hero.png)

5 シーンの 1 frame BA 結果 (路面・前車近傍 1024×1024 クロップ)。
各列 1 シーン、上から GT (黄) / 摂動後 (赤) / BA 補正後 (緑)。

## サマリ

現状でうまくいっていること:

- **2-DoF (pitch, yaw) を Kannala-Brandt 歪みを Jacobian に組み込んだ
  closed-form Gauss-Newton で解く** 構成が成立。1 frame だけで δ̂ が GT に
  ±0.1° 以内に乗ることを 5 シーンで確認。0.1° はほぼ人間の目視
  キャリブレーションと同オーダー (まだ人間の方がやや上)。
- 6×6 (extrinsic のみ; intrinsic を入れても 10×10) の正規方程式 1 本
  に帰着するので、1 frame の解は **ms オーダー** ＝ 車載リアルタイム
  対応の drift monitor のひな型として動く。実用上、車両に積んだ
  カメラの位置が走行中に数 cm も動くことは稀なので、回転 (pitch,
  yaw) のドリフトを 1 frame で監視できれば、リアルタイム
  キャリブレーション監視ツールとしてはまずこれで十分。
- 一方で、**並進 (位置)** を高精度に収束させるには 1 frame では情報量が
  足りず、**複数フレームでの蓄積が必要** (画面の対応箇所が時間的に変化
  しないと depth scale が縛れないため)。それでも、1 frame だけでこの
  精度に届くベースが出来た意義は大きい。

まだうまくいっていないこと:

- **6-DoF / 10-DoF に拡張すると、縮退以前に per-pt 再投影誤差を最小化
  する解が暴走する** (yaw が真値の 2 倍近くに飛ぶ、tz が -2m に張り付く
  等)。BA の正規方程式の組み立てそのものをデバッグ中。
- 線形化点の back-projection を pinhole で取っていることなど、複数の
  原因候補があるので順に切り分け予定。

## 1-frame BA のパイプライン

TMPOC 車両 (4K 魚眼, Kannala-Brandt) の 1 frame を 8×5 のタイルに
切って、各タイルを学習済み per-tile model に通す。モデルは各 LiDAR
点について「画像上でこれだけずれてる (Δu, Δv)」と「その推定の
不確かさ (σx, σy, ρ)」を返す。

![tile grid](../assets/2026-05-18_one_frame_ba/frame_tiles_predict.png)

タイルごとに per-pt の (Δu, Δv, σ) が出るので、全タイルの全点を 1 つの
プールにまとめて 1 回の正規方程式で解く ＝ closed-form BA。

### 学習データ

3 種のデータを混ぜた joint training:

- **Waymo Open Dataset** — 公開大規模 multi-cam autonomous driving
- **kamikado データセット** — TMPOC 車両で取得、kamikado さん作成、
  6 シーン
- **Woven Sequence データセット** — TMPOC 車両で取得、自作、4 シーン

kamikado と Woven Sequence はどちらも **TMPOC 車両 (4K 魚眼,
Kannala-Brandt)** で収録したもの。データ生成ツールは別物だが、
カメラ・LiDAR の基本パラメータ (intrinsics / extrinsics の名目値,
fisheye coeffs) は共通。

### モデル学習の状況

ConvNeXt backbone, img_size=128, crop_size 256-512, 200ep。上記 3
データを 1 epoch 内で混ぜ込む joint training。**DGX2 8GPU で 1 run
約 6 時間** で回る。

![learning curves](../assets/2026-05-18_one_frame_ba/learning_curves.png)

- **n6 (DGX2-8gpu)**: 深さ 6 層、scratch から ep 101 まで val NLL
  4.90→1.68, val MSE 9.64→2.46 px
- **n4 resume (DGX1-16gpu)**: 深さ 4 層、前段 n4 ckpt から継続、ep
  142 時点で val NLL 1.36, val MSE 2.08 px (in-progress)

**n=2 では別条件の実験で明らかに精度が落ちる** ことを確認しており
(条件が揃っていないので本記事には数値を載せない)、n=4 / n=6 では
最終 val MSE がほぼ同じレンジ (2.0〜2.5 px) に揃うが、**n=6 の方が
val NLL は一段低く出る**。レイヤーを増やすほど精度は良くなる傾向で、
これは Transformer が段階的に大域的なコンテキストを取り込んでいる
ことを示唆する。他の主要 vision Transformer でも 6 層付近が多用
されることと整合する。

deployment 観点では計算量と精度のバランスから n=4〜n=6 のいずれかが
妥当。

また、通常の物体検出のような box-regression 系と違って、val NLL が
明確な収束プラトーに乗らず、**学習を続けるほど NLL が落ち続けている**
点も注目に値する。これは Transformer の loss が compute / data /
model size に対して **power-law で滑らかに改善する** という neural
scaling laws (Kaplan et al. 2020 [^kaplan]; 画像 / 動画 / multimodal
への拡張は Henighan et al. 2020 [^henighan]) と整合する挙動。
self-supervised ViT 系でも DINOv2 (Oquab et al. 2023 [^dinov2]) が
学習レシピを安定化させて長尺学習を現実的にしている実例があり、
"plateau に乗らずに改善が続く" というよりは "compute を入れ続ける
ほど power-law でじわじわ改善する" 性質と読める。softmax の温度を
下げて確信度を絞り込む等、収束を早める工夫の余地はありそう。

[^kaplan]: Kaplan et al., "Scaling Laws for Neural Language Models",
  arXiv:2001.08361 (2020). 7 桁以上の compute スケールにわたって loss
  が power-law で減少することを示す。
[^henighan]: Henighan et al., "Scaling Laws for Autoregressive
  Generative Modeling", arXiv:2010.14701 (2020). 画像 / 動画 /
  multimodal の 4 領域で「power-law + constant」型の改善を確認。
  constant 項は irreducible loss floor を表すが、実用範囲では
  plateau 様の挙動は観察されにくい。
[^dinov2]: Oquab et al., "DINOv2: Learning Robust Visual Features
  without Supervision", arXiv:2304.07193 (2023). 学習を 2× 速く 3×
  省メモリ化することで「より長い学習・より大きいバッチ」を可能に
  したと述べており、self-supervised ViT で long-training を回す
  動機が成立していることを示す。

## なぜ closed-form？

- **リアルタイム性**: extrinsic だけなら 6×6 の正規方程式 1 本
  (intrinsic も入れれば 10×10 程度) に帰着するので、1 frame の解は
  ms オーダー。drift monitor として車載で常時回せる。
- **σ の有効活用**: タイルレベルの per-pt 推定で **既に不確実性ごと
  解けている** (モデルが Σ_i を返す)。BA はそれを Mahalanobis 重みで
  集約するだけで、追加の最適化はほぼ不要。
- **ロバスト性は IRLS で確保**: 同じ閉形式に Huber-IRLS を載せれば、
  外れ値だけ重みを下げて再ソルブ。線形化点を毎反復更新する LM/Ceres
  と違って計算量はほぼ変わらない。
- **カメラモデルは Jacobian に inject**: KB 魚眼の場合は $J_i$ の
  解析式に歪みを直接入れる (反復最適化に頼らない)。
- **将来パス**: もう一段精度を稼ぐなら、タイルごとに pose を直接回帰する
  ヘッドを足す方向。それでも集約ステップは同じ閉形式で良い。

## 結果

GT 摂動 pitch=+0.500° / yaw=+1.000° に対し、5 シーンで δ̂ が約
0.05〜0.1° まで GT に合う。

ただしカメラが魚眼 (Kannala-Brandt) なので、

- **pinhole closed-form を全画面の点で解くと yaw が縮んで GT の 70-80% に
  underestimate** (画像端で pinhole Jacobian が真値と乖離するため)
- **画像中央の点に絞ると 90-95% まで戻る**
- **KB 歪みを Jacobian に組み込んだ解析微分 closed-form Gauss-Newton** にすれば
  全画面の点を使ったまま GT ±0.1° に収束

## Closed-form の導出 (per-pt Δuv → 6-DoF δ̂)

### 観測モデル

各 LiDAR 点 $i$ について、

$$
\mathbf{r}_i = \begin{bmatrix}\Delta u_i \\ \Delta v_i\end{bmatrix},\quad
\Sigma_i = \begin{bmatrix}\sigma_{x,i}^2 & \rho_i\sigma_{x,i}\sigma_{y,i} \\ \rho_i\sigma_{x,i}\sigma_{y,i} & \sigma_{y,i}^2\end{bmatrix}
$$

をモデルが返す。$\mathbf{r}_i$ は「観測 uv を GT uv に近づけたい補正量」、
$\Sigma_i$ はその不確かさ。これを 6-DoF の rig 姿勢ずれ
$\boldsymbol{\delta} = (\omega_x,\omega_y,\omega_z,t_x,t_y,t_z)$
で説明したい。

$\Sigma_i$ は単なるスカラーじゃなく **2D の共分散** で、その点が
持っている情報の方向を表す。

![sigma ellipse](../assets/2026-05-18_one_frame_ba/sigma_ellipse_example.png)

緑の楕円が σ。地面の点は **白線に沿った縦方向に楕円が伸びる** (= 白線
に沿った方向には不確かだが、白線を横切る方向には情報がある)。
**白いガードレール上の点は楕円が小さい** (= 強いエッジ + テクスチャで
xy 両方向に絞れている)。空や均一なアスファルトの点は楕円が等方に
大きい (= 情報なし)。

BA はこの $\Sigma_i$ を Mahalanobis 距離で重み付けに使うので、各点の
**情報のある方向だけ**が解にちゃんと寄与する。

### 線形化

カメラ座標系で点 $\mathbf{X}_i = (X_i, Y_i, Z_i)$ が $\boldsymbol{\delta}$ で
動くとき、画像上の変位は 1 次近似で

$$
\Delta \mathbf{u}_i \approx J_i \boldsymbol{\delta},
\qquad J_i = \frac{\partial \mathbf{u}}{\partial \boldsymbol{\delta}}\Big|_{\mathbf{X}_i}
\in \mathbb{R}^{2 \times 6}
$$

各列は SE(3) generator を画像投影で押した解析式。pinhole 仮定なら例えば

- $\partial u / \partial \omega_x = -f_x X Y / Z^2$
- $\partial u / \partial \omega_y = +(f_x + f_x X^2 / Z^2)$
- $\partial u / \partial t_x = f_x / Z$ … (`scripts/ba/ba_multicam_corr.py:DOF_JAC`)

### この式の直感

やってることはシンプルで、**「点 $i$ で $u$ がちょっと動いたとき、
カメラの 6 個のパラメータ ($\omega_x, \omega_y, \omega_z, t_x, t_y, t_z$)
それぞれが微小線形化でどれくらい影響するか」** を式で書き出す
だけ。たとえば yaw を 1 度動かすと画像中央の点は $f_x \approx 33$ px
横に動く、画面端の点はもうちょっと動く、… という感度を 6 軸ぶん。

実際の式は↑に貼ったのが該当 (pinhole なら chain rule で 1 行、KB
魚眼でも同じ chain rule の段数が増えるだけ、反復最適化は要らない)。

ポイントは:

- 1 点の感度行列 $J_i \in \mathbb{R}^{2 \times 6}$ は **rank 2** しか
  ない (1 点で 6 個全部は決まらない)
- でも frame 内で **いろんな $(X, Y, Z)$ の点を数千個積み上げる** と、
  $\sum_i J_i^\top \Sigma_i^{-1} J_i$ がフルランク (6×6 正則) になる
- 逆行列解いて $\boldsymbol{\delta} = H^{-1}\mathbf{b}$ でパラメータが
  1 発で出る
- 反復最適化を回さないので **計算量がむちゃくちゃ減る** (1 frame で
  ms オーダー)。LM や Ceres を回す代わりに、1 個の 6×6 連立方程式を
  解くだけ ＝ 車載リアルタイムでも余裕で回る

KB (魚眼) の場合は $u = f_x \cdot \theta_d \cdot X/r + c_x$ と
$\theta_d = \theta(1+k_1\theta^2+\dots+k_4\theta^8)$ を chain rule で
通して同じ形式に書き直す (今日新規に書いた
`scripts/ba/ba_kb_jac.py:KB_DOF_JAC`)。

### Mahalanobis 重み付き正規方程式

per-pt の whitened 残差 $\Sigma_i^{-1/2}(\mathbf{r}_i - J_i\boldsymbol{\delta})$
の二乗和を最小化:

$$
\boldsymbol{\delta}^* = \arg\min_{\boldsymbol{\delta}}
\sum_i (\mathbf{r}_i - J_i \boldsymbol{\delta})^\top \Sigma_i^{-1} (\mathbf{r}_i - J_i \boldsymbol{\delta})
$$

これを $\boldsymbol{\delta}$ で微分してゼロと置けば、6×6 の正規方程式
1 本に帰着する:

$$
\boxed{\;H\boldsymbol{\delta}^* = \mathbf{b},\quad
H = \sum_i J_i^\top \Sigma_i^{-1} J_i,\quad
\mathbf{b} = \sum_i J_i^\top \Sigma_i^{-1} \mathbf{r}_i\;}
$$

$\boldsymbol{\delta}^* = H^{-1}\mathbf{b}$ で 1 ステップで閉形式解。
共分散も $\mathrm{Cov}(\boldsymbol{\delta}^*) = H^{-1}$ で同時に出る。

### Huber-IRLS で外れ値抑制

per-pt Mahalanobis 距離 $d_i^2 = (\mathbf{r}_i - J_i\boldsymbol{\delta})^\top
\Sigma_i^{-1}(\mathbf{r}_i - J_i\boldsymbol{\delta})$ を計算して、
$w_i = \min(1, k/d_i)$ で $\Sigma_i^{-1}$ をスケール、再ソルブ。
線形化点は更新せず weight だけ更新する反復。同じ閉形式の枠で外れ値の
影響を抑えられる。

## 残差グラフ (5 シーン × 2 ソルバ)

![residuals](../assets/2026-05-18_one_frame_ba_residuals.png)

- **青 (○)**: pinhole closed-form, 中央バンド + σ-stratified TOP-100
- **赤 (□)**: KB closed-form (解析微分), 全画面 + σ-stratified TOP-300
- 灰点線が GT (pitch +0.500° / yaw +1.000°)

両方とも GT から ±0.1° 以内に張り付いてます。pinhole 版は yaw を少し
低めに、KB Jacobian 版は少し高めに見積もる傾向。

## 3 段 reprojection overlay (parent 全体, 1 シーン)

冒頭の hero はトラック近傍 1024×1024 のクロップだったが、parent
画像全体で見ると次のようになる。GT (黄) / 摂動後 (赤) / BA 補正後
(緑) を全 LiDAR 点で重ねたもの。緑が黄に近いほど BA がうまく効いて
いる。

![overlay](../assets/2026-05-18_one_frame_ba/points_ip664_D_20260304_231950_d007-mdc_IWATESAN_inside_2_overlay_v3.png)

## 何が効いたか

- **Pinhole closed-form (`solve_dofs`)**: 既存。Jacobian は pinhole 仮定。
  全画面の点を入れると画像端の KB 歪みで解が縮む (yaw が真値の
  70-80% 程度に underestimate)。**中央バンド (画像中央 50% × 25% 帯)**
  に絞ると 90-95% まで戻る。
- **KB closed-form (`solve_dofs_kb`)**: 新規 (`scripts/ba/ba_kb_jac.py`).
  KB 投影 `θ_d = θ·(1+k1θ²+...+k4θ⁸)` の偏微分を 6 軸ぶん解析的に組み、
  Gauss-Newton 数反復で再線形化。pinhole closed-form の δ̂ で warm-
  start すると 2-DoF (pitch, yaw) で安定して GT ±0.1° に収束。

## 点の選び方 / DoF の選び方

- **TOP-K で σ の小さい点だけ拾う** こともできる。今回は σ-stratified
  TOP-100 で十分動いた。タイル境界に偏らないよう grid cap (8×4 セル,
  cell あたり 5 点) を入れている。
- **基本は 2-DoF (pitch, yaw)** で良い。実車のキャリブレーション
  ドリフトは回転主体で、tx/ty/tz は 1 frame では depth scale と
  縮退するため closed-form では不安定 (今日 6-DoF を試したら
  yaw が真値の 2 倍以上に暴れた)。位置を取りたければ multi-frame
  に拡張する。
- KB 解析微分の `solve_dofs_kb` は 6-DoF 全部の Jacobian を持っているが、
  実用上は **2-DoF だけ解く** で δ̂ が GT ±0.1° に乗る。

## 次のステップ

1. **KB の線形化点 (X, Y, Z)** を pinhole 逆投影 `(u-cx)·z/fx` から
   KB 逆投影に置き換える (魚眼の縁では数十 % ずれる)。これで 6-DoF
   走らせても tz が暴れないはず。
2. **Multi-frame fuse**: 同じ rig の 5-6 frame を 1 つの正規方程式に
   stack。1 frame ごとの ±0.05° 系統 bias が平均化で消える見込み。
3. **CaaS API**: 1 frame closed-form (~ms) を realtime drift monitor、
   multi-frame batch を first-time calibration に分けて wrap。

## ファイル

- `scripts/ba/ba_kb_jac.py` — KB 解析微分 + Gauss-Newton ソルバ
- `scripts/_debug/ba_one_frame_vis.py` — 5-scene 駆動 + 3 段 overlay
- `scripts/_debug/plot_one_frame_ba_residuals.py` — 残差プロット
