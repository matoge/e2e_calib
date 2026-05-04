# モデル進化の 7 ステージ・ロードマップ

クロスアテンション構造とデータ側の変化を中心に整理。

---

## TL;DR

```
stage 0  baseline calib  ────────────┐
                                      │  ← model 側: KV/Q を進化
stage 1  hybrid KV       ─────┐       │
stage 2  mixed Q          ────┼───────┤
stage 3  UV-only Q        ────┘       │
                                      │  ← データ側: aug + frame pair
stage 4  scale-aug        ─────┐      │
stage 5  real cross-frame ─────┼──────┘
stage 6  long-baseline    ─────┘
```

**0→3**: model 拡張、推論時に LiDAR を要らなくする
**4→6**: データ側で scale + cross-frame に拡張

**Stage 4 の `RelativePoseEmb` が同フレ scale-aug と real cross-frame を統合する key**

---

## ステージ早見表

| stage | 解く問題 | 新規 model 要素 | 新規データ要素 | GT 出所 |
|-------|---------|----------------|---------------|---------|
| 0 | 画像→LiDAR Δuv calib | (baseline) | 1 画像 + LiDAR | LiDAR pixel offset |
| 1 | stage0 を hybrid KV で再現 | uv_emb 追加 + LiDAR-dense | 同上 | 同上 |
| 2 | UV-only でも LiDAR Δuv 引き寄せ | query drop schedule | 同上 | same-cell LiDAR Δuv |
| 3 | LiDAR 完全抜き camera-only ready | Q から LiDAR 完全除去 | 同上 | 同上 |
| 4 | 視点変動 (scale / crop) 吸収 | **RelativePoseEmb on KV** | crop_A → crop_B ペア | crop transform から既知 |
| 5 | 短Δt 実 frame pair | (stage4 流用) | 2 frame 実 pair | 短Δt の信頼 pose GT |
| 6 | 200m+ 合成 + Σ 学習 | scale curriculum | long-Δt + 合成スケール | 合成 Δuv + Σ |

---

## クロスアテンション構造の進化

### Stage 0 — baseline (現行 calib)

```
┌─────────────────────────────────┐
│  KV: image (DA + conv pos)      │
└────────────┬────────────────────┘
             │
             ▼  cross-attn (softmax)
┌─────────────────────────────────┐
│  Q: LiDAR pivot (uvd → MLP)     │
└────────────┬────────────────────┘
             ▼
            Δuv
```

### Stage 1 — hybrid KV (uv_emb 共有 + LiDAR-dense 追加)

```
両 KV バンクを 同じ HxW グリッド に揃える ← この sizing が DA を共通利用可にする鍵

┌─────────────────────────┐ ┌─────────────────────────────┐
│  KV-image  (HxW grid):  │ │  KV-lidar  (HxW grid):      │
│  image + uv_emb         │ │  LiDAR-dense + uv_emb       │   ← uv_emb 共有
└──────────┬──────────────┘ └────────────┬────────────────┘
           │                              │
           └──────────┬───────────────────┘
                      ▼  Deformable-Attention (両バンクとも同じ DA)
          ┌─────────────────────────┐
          │  Q: LiDAR-dense+uv_emb  │
          └────────────┬────────────┘
                       ▼
                      Δuv      ← stage0 と同性能必須
```

> **狙い**: アーキ拡張 (KV 二極化 + uv_emb 共有 + 両バンク同サイズ) で性能を落とさず、後続 stage の足場を作る。
> LiDAR-dense は **わざと画像と同じ HxW にする** = DA が両バンクに同じように効く構造。
> これにより以降の stage で KV の構成を変えても attention 機構を統一できる。

#### 補足: LiDAR-dense の意味と bilinear sampling の懸念

**「LiDAR-dense」とは:** LiDAR を **前処理段階で画像と同じ HxW grid にラスタライズ済み** の状態。
- 投影 (u,v) のセルに feature 入る、当たってないセルは 0 (sparse grid)
- 点同士の衝突は前処理で解決済 (例: 同セル多点は depth-softmax 加重で 1 個に縮約)
- DA は preprocessing 後の grid に対して動く (= もう画像)

**なぜ画像と同じ HxW にするのか:**
- Deformable-Attention は grid を前提に offset prediction + bilinear sampling する
- 画像バンクと同じ grid にすれば **両バンクで同一の DA module が再利用できる**
- attention 機構を統一できるので以降の stage で KV 構成変えても触らなくて済む

#### 中身: bilinear sampling と zero セル

DA の bilinear は本質的に「**fractional offset での値を differentiable にサンプリング** するためだけの仕組み」。単なる微分可能サンプラ。

```
sample at (u + 0.3, v + 0.7)
  = w_00·v_00 + w_01·v_01 + w_10·v_10 + w_11·v_11
    (w は (u,v) との距離重み、合計 1)
```

**user 指摘の正しい点:** 元データが保存されてれば zero セルが混じっても原理的には OK — bilinear は補間してるだけ。

**問題は「ゼロが他を歪める」:**

```
4 セル全部 dense:    sample = 正しい補間値
3 dense + 1 zero:    sample = 真値の 75% (zero に引っ張られる)
2 dense + 2 zero:    sample = 真値の 50%
```

→ クエリ位置が dense 領域内か空セル境界かで信号強度が変動 → 学習が歪む。

**user 提案: 4 近傍が全部 dense なときだけ bilinear、そうでないときはしない**

これ妥当。実装イメージ:

```python
# 通常 bilinear
sample = sum(w_i * v_i for i in 4)

# user 提案 (gated)
if all(m_i == 1 for i in 4):     # 全部 dense
    sample = sum(w_i * v_i)
else:
    sample = nearest_valid_or_skip
```

**もう一段ソフトに: masked-renormalized bilinear (推奨)**

hard gate より soft 化した方が境界の gradient が安定:

```python
mask_i ∈ {0, 1}                                # validity
numer = sum(w_i * v_i * mask_i)
denom = sum(w_i * mask_i) + ε
sample = numer / denom                          # 有効セルだけで重み再正規化
```

- 4 全部 dense → 通常 bilinear 動作
- 3 dense + 1 zero → 残り 3 セルで再正規化、歪み無し
- 0 dense → 出力 0 + validity 0 を伝播

**さらに validity mask を feature 拡張で持つ:**
```
KV-lidar (HxW, channels = D + 1):  feature(D) ⊕ validity(1)
```
attention は「この場所に元から信号あったか」を attribute として参照可能、空セルを自然に避ける。

**結論 (user の直感に沿う):**
- bilinear はそのまま、ただし **mask で重み再正規化**
- 完全空セルは出力 0 + validity フラグを伝播
- これで「zero が他を歪める」問題は消える、user 提案より滑らかで grad 安定

#### bilinear をスパースでも成立させる他の方法

| 方法 | 仕組み | 強み | 弱み |
|------|--------|-----|-----|
| **Pull-Push pyramid** (Gortler '96) | 多解像度 pyramid 上で非ゼロ cell を平均→下層へ伝播し dense 化 | 古典・高速・data 整合性保つ | 1 回の前処理コスト |
| **Multi-scale DA** (Deformable-DETR 流) | 1/8, 1/16, 1/32 の階層 feature pyramid を DA が同時参照 | 既存 DA 構造のまま、粗解像度で欠損吸収 | 構造拡張が要る |
| **kNN-radius bilinear** | 4 grid 近傍じゃなく **連続空間で k 近傍 valid cell** を距離重みで集約 | 局所密度に自動 adapt、欠損で破綻しない | k-NN 検索コスト |
| **Gaussian splat preprocessing** | 各 LiDAR 点を 局所密度 σ の Gaussian で塗る | 結果が滑らか dense、bilinear がそのまま動く | 元の sharp さが少し落ちる |
| **Distance Transform aux** | 「最近 valid cell までの距離」を +1ch の特徴として持つ | attention が信頼度を直接参照 | 補完力は弱い (補助のみ) |
| **学習 "no-data" token** | 空セルに学習可能 vector (≠ 0) | 実装超軽量 (1 vector 追加だけ) | 補完しない、attention に任せる |

**用途別おすすめ:**

```
最小コストで効くやつ:
  1. masked-renormalized bilinear (基本これ)
  2. + 学習 "no-data" token (1 行追加)
  3. + validity mask channel

性能重視で 1 段上げる:
  Pull-Push 前処理 で dense 化 (LiDAR-dense の段階で全セル埋める)
  → bilinear 何も悩まなくて良くなる、grad きれい
  → 古典手法だが現代の attention と相性良い

DA 構造そのものに統合したいなら:
  Multi-scale DA (Deformable-DETR 公式)
  → 1/32 解像度なら VLP-32 でも全セル埋まる
  → 細かい解像度で詳細、粗解像度で欠損吸収
  → 既に画像側もこうしてるはず、LiDAR バンクも同じ pyramid に乗せる
```

**現実解 (推奨スタック):**
```
[preprocessing]  Pull-Push で 1/8 解像度まで dense 化
[KV-lidar 構成]  multi-scale (1/8, 1/16, 1/32) + validity channel
[DA sampling]    masked-renormalized bilinear (空セル ε 残ってる場合の保険)
```

→ 普段は Pull-Push で大体埋まる + DA は multi-scale で粗を吸収 + 万一の隙間は masked bilinear が丸める。3 重保険で「zero が学習を歪める」問題は構造的に発生しない。

---

(以下は scatter 段階での選択肢、すでに preprocessing 済 = 参考情報)

**dense 化の選択肢:**

| 方法 | 仕組み | メリット | 懸念 |
|------|--------|---------|------|
| **zero-fill (sparse scatter)** | 投影 (u,v) のセルだけ feature 入れて他は 0 | 実装最小、情報そのまま | DA が空セル踏んで無駄打ち、勾配sparse |
| **bilinear scatter** | 投影点 feature を 4 近傍セルに重み付き加算 | DA offset がスムーズに勾配通る | 多点が近接するとセル衝突で情報混合 |
| **nearest / Voronoi** | 各セルに最近点 feature コピー | 全セル埋まる | 遠い点まで広がる、距離概念崩れる |
| **MLP scatter (KP-Conv 系)** | 各点が周辺セルに学習重みで散布 | 表現力高い | 計算コスト + 実装重い |

**zero-fill の懸念 (具体的に):**

1. **density-dependent な空率**
   - 16×16 grid + VLS-128 (1.6M pts/scan)  → ほぼ全セル埋まる、zero-fill OK
   - 64×64 grid + VLP-32 (~30K pts/scan)   → 30-50% 空、DA の無駄打ち増
   - 128×128 grid + 遠景 (sparse 領域)        → 70-90% 空、機能不全リスク

2. **DA offset の学習負荷**
   - 「空セルを避ける」を model が暗黙的に学習する必要がある
   - これは DA の本来の仕事 (どこの特徴が有用か) と直交する余計なタスク
   - 学習効率が落ちる

3. **density 情報の喪失**
   - 「LiDAR が密に当たってる ⇔ 近距離 / 反射が強い」というシグナルが消える
   - bilinear なら密領域はセル累積値が大きくなる = density が値の大きさで残る
   - zero-fill だと「あるかないか」の binary 情報のみ

**bilinear の妥当性 (user 指摘):**

- bilinear で広がっても、後段の MLP/attention が **uv_emb を見て逆畳み込み的に元位置を復元** できる
   → 構造的に正しい主張。bilinear は invertible-with-context な操作
- softmax CA や DA の offset prediction は「元の点位置」を再発見できる
- 実質的に「bilinear で blurry にしたものを attention が再 sharp 化」
- ただし **複数点がセル衝突** したケースだけは原理的に分離不能 (sum したら戻せない)
   → 衝突セル内で max-pool / softmax-by-depth で 1 点に縮約しておくのが安全

**結論**:
- 現実的には **bilinear scatter** が筋。zero-fill は grid 解像度 / LiDAR 密度の組み合わせで破綻しうる
- セル内多点衝突に備えて、scatter 前に in-cell aggregation (depth-softmax 加重 1 個に縮約) を入れる
- 衝突回避のために LiDAR-dense の解像度は **画像 feature の解像度より粗くする** ことを許容 (16×16 程度)

---

### Stage 2 — mixed Q (LiDAR+uv / uv-only を 50/50)

```
┌─────────────────────────┐ ┌─────────────────────────────┐
│  KV-image  (HxW)        │ │  KV-lidar  (HxW)            │
│  (stage1 同)            │ │  (stage1 同)                │
└──────────┬──────────────┘ └────────────┬────────────────┘
           └──────────┬───────────────────┘
                      ▼  Deformable-Attention (両バンクとも)
        ┌────────────────────────────────┐
        │  Q: mix                        │
        │   50% [LiDAR-dense + uv_emb]   │   ← drop schedule 20% → 80%
        │   50% [uv_emb only]            │
        └────────────────┬───────────────┘
                         ▼
                        Δuv
                  (uv-only Q でも same-cell LiDAR Δuv へ引き寄せ)
```

> **狙い**: shared uv_emb 経由で **UV-only Q から geometry 情報を引き出す回路** を学習。LiDAR Q が "教師" の役割

---

### Stage 3 — UV-only Q (camera-only ready)

```
┌─────────────────────────┐ ┌─────────────────────────────┐
│  KV-image  (HxW)        │ │  KV-lidar  (HxW)            │
│  (stage1 同)            │ │  (stage1 同)                │   ← KV は LiDAR 残す
└──────────┬──────────────┘ └────────────┬────────────────┘
           └──────────┬───────────────────┘
                      ▼  Deformable-Attention (両バンクとも)
          ┌─────────────────────────┐
          │  Q: uv_emb only         │   ← Q から LiDAR 完全除去
          └────────────┬────────────┘
                       ▼
                      Δuv
```

> **狙い**: 推論時 Q は UV のみで OK = camera-only deployment 準備完了。KV 側 LiDAR は学習用のまま

---

### Stage 4 — scale-aug + RelativePoseEmb ★

```
1 枚の画像から crop_A と crop_B を生成 (位置・スケール変える)
crop_A → crop_B の相対視点変化を RelativePoseEmb として KV に注入

┌──────────────────────────────────┐ ┌──────────────────────────────────┐
│  KV-image (crop_B, HxW):         │ │  KV-lidar (crop_B, HxW):         │
│  image + uv_emb + RelativePoseEmb│ │  LiDAR-dense + uv_emb + RPE      │   ← RPE 注入
└──────────────┬───────────────────┘ └──────────────┬───────────────────┘
               └─────────┬───────────────────────────┘
                         ▼  Deformable-Attention (両バンクとも)
          ┌──────────────────────────────────┐
          │  Q: uv_emb only (crop_A 視点)    │
          └──────────────┬───────────────────┘
                         ▼
                        Δuv (crop_A → crop_B の対応)
                   GT: crop transform から既知 (pose GT 不要)
```

**RelativePoseEmb の中身:**
```
RPE = MLP([
  Δt_xyz,           # 仮想的視点移動量 (scale-aug では z 方向)
  Δscale_log,       # 1/scale_factor の log
  Δyaw, Δpitch,     # 視点回転 (cropping 位置変動)
])
```

> **狙い**:「VFP / crop scale を変える = 物理的視点が動く」を model に明示する。
> RPE なしだと「同じ画像なのに違う Δuv を要求される」という**学習矛盾**が起こる

---

### Stage 5 — real cross-frame (短 Δt)

```
2 frame の実 pair (frame_A, frame_B), 短 Δt (1-3 秒)

┌──────────────────────────────────┐ ┌──────────────────────────────────┐
│  KV-image (frame_B, HxW):        │ │  KV-lidar (frame_B, HxW):        │
│  image + uv_emb + RPE            │ │  LiDAR-dense + uv_emb + RPE      │
└──────────────┬───────────────────┘ └──────────────┬───────────────────┘
               └─────────┬───────────────────────────┘
                         ▼  Deformable-Attention (両バンクとも)
          ┌──────────────────────────────────┐
          │  Q: uv_emb only (frame_A 視点)   │
          └──────────────┬───────────────────┘
                         ▼
                        Δuv (frame_A → frame_B)
                   GT: 実 pose GT から (短 Δt 信頼域)
```

> **stage 4 の枠組みのまま**。RPE の中身が「合成変換」から「実 pose 差分」に変わるだけ。
> model アーキは触らない。

---

### Stage 6 — long-baseline (scale curriculum + Σ)

```
stage 5 + 合成 long-baseline (scale-aug で 200m+ 視点) + Σ 推定

┌──────────────────────────────────┐ ┌──────────────────────────────────┐
│  KV-image  (HxW):                │ │  KV-lidar  (HxW):                │
│  image + uv_emb + RPE (long)     │ │  LiDAR-dense + uv_emb + RPE      │
└──────────────┬───────────────────┘ └──────────────┬───────────────────┘
               └─────────┬───────────────────────────┘
                         ▼  Deformable-Attention (両バンクとも)
          ┌──────────────────────────────────┐
          │  Q: uv_emb only                  │
          └──────────────┬───────────────────┘
                         ▼
                  (Δu, Δv, Σ)        ← Σ も同時推定
                  GT: 合成 + 動的 Σ inflation
```

> **狙い**: scale curriculum (50m → 100m → 200m → 400m) + 動的物体の Σ 学習。
> 信号機/遠ビル = 小Σ、moving car = 大Σ、BA で自動 weighted

---

## データ側の進化

| stage | 学習データ | Pose GT 必要 | aug | 1画像→ペア倍率 |
|-------|-----------|-------------|-----|---------------|
| 0 | 1 画像 + LiDAR | × | none | 1 |
| 1 | 同上 | × | none | 1 |
| 2 | 同上 | × | query drop (20→80%) | 1 |
| 3 | 同上 | × | query drop continued | 1 |
| 4 | 1 画像 + LiDAR | **× (crop transform で既知)** | crop-scale aug | **N²** (爆発的) |
| 5 | 2 frame 実 pair (短Δt) | ○ (短Δt信頼域) | + real pair | N² + pair |
| 6 | 短Δt + 合成長 | ○ (短Δt信頼域のみ) | scale curriculum | N² + 多 scale |

**Stage 4 が data-multiplier の転換点:**
- Pose GT 依存が消える → 公開DS 全部 + TSS4 (無キャリブ) + WOVEN 全部 学習に使える
- 1 画像から N² 個のペアが作れる → データ量爆発
- adapter は学習の前提条件じゃなく deployment 関心事になる

---

## Stage 4 の核心: RelativePoseEmb

### 何が問題か

```
1 枚の画像から crop_A と crop_B を作る
crop_A: 中心 (u₀, v₀), scale 1.0    ← 「近くで見た」
crop_B: 中心 (u₀, v₀), scale 0.5    ← 「2倍遠くから見た」(2× 広FOV)

同じ LiDAR 点が crop_A の (u_a, v_a) と crop_B の (u_b, v_b) に映る
   u_b ≠ u_a, v_b ≠ v_a   (scale で変わる)
```

RPE なしで `Q = uv_emb(u_a, v_a)` だけ与えると:
- crop_A では (u_a, v_a) で見つかる
- crop_B では (u_b, v_b) に行ってほしい
- でも model は「同じ画像」だから「同じ Δuv で良いはず」と学習する
- **学習矛盾 → 性能崩壊**

### 解決策

KV 側に「ここの視点は元から **どう動いたか**」を埋め込む:

```
RPE_B = MLP([
  Δscale = 0.5,         # 2× 広 FOV
  Δt_z   = +simulated,  # 「2倍距離から」を z 方向移動として扱う
  Δyaw   = 0,           # crop 中心が同じなら回転無し
  ...
])
```

KV-image_B = image_B_features + uv_emb + **RPE_B**

これで model は `crop_B = crop_A の視点を Δscale, Δt_z だけ動かしたもの` を理解
→ Q (crop_A 視点の uv) → KV (crop_B + RPE_B) で正しい Δuv を出せる

### なぜこれが unifying key か

```
              ┌── 同フレ scale-aug:    RPE = (Δscale, Δt_z 仮想, ...)
RelativePoseEmb ─┤
              └── real cross-frame:    RPE = (実 Δt, 実 R, ...)

両者の違いは RPE の中身だけ。model アーキ・タスク・loss は完全に同一。
```

→ Stage 4 で RPE を入れた瞬間、stage 5 (実 pair) と stage 6 (合成長) が **同じネットワークで処理可能**。
RPE は「視点変動の universal encoding」。

---

## Stage 0→6 で勝ち取るもの

```
Stage 0 → 1   ┃  KV/Q 拡張、性能維持   (アーキ準備)
Stage 1 → 2   ┃  shared uv_emb で UV-only から geometry を引き出す回路を学習
Stage 2 → 3   ┃  Q から LiDAR を完全除去、camera-only inference 準備完了
Stage 3 → 4   ┃  RPE 導入、視点変動を吸収、合成データへの土台
Stage 4 → 5   ┃  実 pose GT を投入 (短Δt 信頼域)
Stage 5 → 6   ┃  scale curriculum で 200m+、Σ で動的物体吸収
```

**最終形**:
- 推論時: 画像 1 枚 (LiDAR 任意) + UV クエリ → (Δu, Δv, Σ)
- 学習時: 公開DS + TSS4 (無キャリブOK) + WOVEN 全部混ぜ可
- 用途: calib / pose 推定 / 地図 / 動的検出 すべて同一 primitive
