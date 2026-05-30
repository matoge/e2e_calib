---
date: 2026-05-30T20:08+09:00
author: hfunaya
tags: ["calibnet2", "self-sup", "cross-frame", "rope", "pose-emb", "yrp", "zoom", "cfg", "monodepth", "4dgs"]
streams: ["e2e_calib", "calibnet2-design"]
status: silver
---
# cross-frame 0→1 を self-sup でどこまで持ち上げられるか — CalibNet2 戦略

**Origin:** ~/git/e2e_calib @ 6df36eb (git@github.com:matoge/e2e_calib.git)

## 文脈

CalibNet2 は cross-frame 対応の calib ネットワーク。block 1 種類で stack、KV は always own-frame、PoseEmb は type-0 (intrinsic, additive) と type-1 (Δpose R, RoPE on Q) に分離 ([[2026-05-27_calibnet2_design]] 相当)。実装は #204-#225 で骨格まで終わってる。

問題: 実戦データは pair しか出てこない (cross-frame の 2 frame、Δpose 付き) が、その GT pose に bias が乗ってる可能性が拭えない。Waymo の一部や Woven の特殊 2 seq (BA chase pose で 300m 信号追跡) を除けば、SLAM/フォトグラメトリ由来の pose は **systematic bias** を持つ。1cm 地図要件で「LLN で平均すれば真値」が成立するか保証できない。

## 戦略: self-sup で先に詰める

pair-sup を kick する前に **self-sup (= 1 frame に幾何 warp を自分でかけて解析解で当てさせる)** で visual 経路を sub-pixel まで詰める。

### なぜ self-sup を先か

- self-sup は **bias-free GT が無条件に得られる** (warp は自分で生成、解析解 exact)
- 量は無限 (どの frame でも warp かければサンプル化)
- 「ほんとうに sub-pixel まで補正できるか」の純粋検証 — pose GT の質に依存しない
- 効いたら → Woven 2 seq の BA-quality holdout で外的検証 → pair-sup 投入の判断材料
- 効かなかったら → そもそも image を sub-pixel まで詰めきれてない = aug pipeline か backbone の問題、と早期切り分け

### kamikado FULL を選ぶ理由

- kamikado: pose 無し、画像+LiDAR がきれいに揃ってる cache が既にある ([[2026-05-2x_full_tile_parity]] 相当 / project_full_tile_parity)
- pose 不要 (self-sup は 1 frame 完結)
- LiDAR clean → uv 投影の bias 小 → visual feature の "限界" を素直に測れる

## 設計: YRP + zoom の self-sup aug

### 幾何

3D YRP は image-plane で homography に閉じる (t=0 なら parallax 出ない):

$$H_{yrp} = K \cdot R(\text{yaw}, \text{pitch}, \text{roll}) \cdot K^{-1}$$

zoom は **3D camera operation じゃなく 2D image-plane similarity** として扱う。理由:

- (a) zoom 中心 = image center 固定だと退化サンプル (モデルが「中心は不動」を学んでしまう、cross-frame では任意の点が pivot)
- (b) ずらした zoom = scale + 2D translation の合成、camera は動かしてないので parallax 出ない、1 frame 解析解で安全
- (c) 画像と LiDAR uv に同じ 2D similarity をかければ整合維持、3D world は不変

$$H_{2d} = T(\text{pivot}) \cdot \text{diag}(s, s, 1) \cdot T(-\text{pivot})$$

合成: $H = H_{2d} \cdot H_{yrp}$。LiDAR uv も同じ $H$ で 2D 同次変換、$D$ 不変。

### PoseEmb 注入規則

- **YRP (R)** → type-1 RoPE (block-diag(R) を Q feature に作用)
- **zoom (s)** → identity lock (= 入れない、focal_aug=0)
- **2D pivot translation** → identity lock (= 入れない)

「YRP かつ zoom=1× の時だけ PoseEmb を入れる」のが厳密版。zoom が混ざると homography は R 単独で表現できないので hint が嘘になる → 学習に害。

### Sampling (CFG-style)

| 確率 | zoom | pivot | YRP | PoseEmb |
|---|---|---|---|---|
| 80% | 1.0× | n/a | ±5/5/2° | R_aug 注入 (hint) |
| 20% | [0.5, 2] log-uniform | 画像内任意 | ±5/5/2° | drop (R=I) |

drop 時は **PoseEmb 完全ゼロ** (R=I, intrinsic 0)。理由: drop が成立する核は条件独立性。pose 値が残ってると pose_emb 以外の経路 (KV bank の depth 分布や frame-token statistics) で pose を察知できてしまい、no-hint forward を学んでないことになる。

これは **Classifier-Free Guidance (CFG)** と構造同型。1 つの重みに:
- hint mode: $f(x, c)$
- uncond mode: $f(x, \emptyset)$

両方を学ばせる。block 1 種類で stack なので structurally に強制される。

### 重みは分裂しない、Visual は共通で吸い付く

```
image → CNN → KV bank ──────────────┐
                                     ├── cross-attn block (1 種類, shared weights)
LiDAR-Q ─ + PoseEmb (R) ─────────────┘
            ↑
       drop で I
```

KV side (画像理解) は両 variant で同じ重み・同じ feature。違うのは Q が KV のどこを引くか。

- no-hint variant の grad → **KV 抽出能力が伸びる** → hint variant も恩恵
- hint variant の grad → Q の RoPE が KV のどこを引くべきかを学ぶ → K/V projection 重みが共有なので間接的に no-hint にも効く
- drop 確率 = KV 学習圧と Q 学習圧の trade-off

p=0.2 は hint 寄り。**まず hint で exact に解けることを sanity → 徐々に上げて visual 経路を厚く** が安全な順序。

## 期待値

self-sup で kamikado に対して:

1. **hint sanity**: zoom=1 純 YRP + R_aug 注入 → UV residual ≈ 0 (RoPE が equivariant に動いてれば structural に保証)
2. **drop smoke**: drop=100% で zoom=1 純 YRP を visual のみで解ける (= 純 visual feature でも warp 解析解に到達できる)
3. **mix kick**: 80/20 で UV residual の sub-pixel 収束カーブが取れる

(2) が落ちる → visual feature が情報限界に張り付いてない (= まだ伸びる余地ある or aug pipeline がおかしい)。(2) が伸びる → cross-frame 投入 OK。

## 大数の法則ルートとの比較 (= self-sup は本筋か?)

実は pair-sup だけでも、bias は scene/時刻/方向でランダムにバラければ $1/\sqrt{N}$ で消える。Waymo (pose 良) ならこれだけで calib として十分。**self-sup は bias の "absolute zero" anchor を提供する保険** であって本筋じゃない。

ただし:
- 自前 kamikado raw / TSS4 は pose の質が読めない → systematic bias の可能性 → LLN 効かない懸念
- 1cm 地図要件で「pair の LLN 平均が真に 1cm 以下か」が事前に検証できない
- self-sup で visual 限界を見ておけば、pair-sup の bias 中央が真値かを後付けで判定できる

→ **self-sup 先行** は「保険」であり「visual 限界を測る純粋実験」でもある。

## CFG-style mix への懸念と GT pose 製造ルート

CFG-style 80/20 mix は理論的に整合してるが、**実用上は hint と no-hint の trade-off で両方の精度が下がる**可能性がある。本当に欲しいのは「正確な GT pose があって、hint exact 単独で詰める (補助輪を外した) 状態」。

GT pose は通常入手困難だが、**LiDAR の幾何的特性**を使って製造できる:

- **LiDAR は距離を正確に測る** (反射時間ベース、~40m まで非常に正確)
- 結果として **GICP は距離方向のシフトをほぼ 100% 正確に取れる**
- 残る問題は **回転** だけ (距離合わせ loss は回転に対して弱い)

つまり pose の難所は **回転の bias**。これを self-sup CalibNet2 で消せれば、GICP shift × 補正 R で **ほぼ正しい GT pose** が作れる。

### Bootstrap pipeline (P1 → P2 → P3)

```
P1: self-sup CalibNet2 (kamikado FULL, 1 frame, YRP+zoom CFG-style 80/20)
    → 1-frame visual feature を sub-pixel に詰める
    ↓
P2: GICP で距離 (~40m) を取り、その回転を P1 model で補正
    → 「回転しないように」loss をかけて pose 補正
    → 補正済み GT pose 製造
    ↓
P3: 2-frame pair-sup を P2 GT pose で学習
    → drop_p = 0 (補助輪を外す)、hint=ON 単独
    → PoseEmb lock 有/無の経路分裂が無い → visual が hint 方向に特化、精度上がるはず
```

P3 では CFG mix を捨てる。P1 = bootstrap 専用 model、P3 = production model (P1 ckpt は P3 の init)。

### P2 が成立する根拠

- P1 は 1 frame self-sup しか学んでない → 平行移動は分からない
- が、**GICP が距離 (= shift) を持ってる**ので、補正すべきは回転だけ
- 役割分担: GICP=距離、P1=回転 → 合わせて 6-DoF GT pose

### "回転しないように" loss の設計 (要詰め)

P2 で何を loss にするかは要設計。候補:

- (a) frame B を frame A の Q にして hint=GICP_R で当てに行かせ、UV residual の bias から R 補正値を出す (P1 model を inverse problem に使う)
- (b) self-sup と同じ枠組みで、A→B の変換を YRP-only homography で近似し、R_aug を学習可能パラメタにして UV residual を最小化する R を解く (per-frame-pair で R 最適化)

(b) の方が P1 model をそのまま使えて clean。ただし精度は要実証。

### 来週のゴール

**P2 まで通して "ほぼ正しい GT pose" を kamikado raw 全フレームに付ける** ところを目標にする。これが達成できれば P3 は drop_p=0 の単純な pair-sup で済む。

CFG-style mix は P1 の "1 回限りの bootstrap" として割り切る。P3 で重みを引き継いだ後は drop=0 固定、補助輪なし。

「混ぜたらどうなるか」 (= P3 で drop_p>0 を残すか) はやってみないと分からない。drop_p=0 と drop_p=0.1 の 2 本走らせて差を見る案もあるが、まず drop_p=0 から。

## 関連 stones

- monodepth は LiDAR 再現じゃなく 4DGS で真 geometry が本筋: [[2026-05-30_monodepth-は-lidar-再現じゃなく-4dgs-北極星から逆算した-pipeline]]
- GATr を frustum に入れても効かなかった話 (CRB 飽和): [[2026-05-30_gatr-を-frustum-encoder-に入れても効かなかった-情報限界-crb-の話]]

## 次

P1 (self-sup):
- #228 pandaset_full の self-sup branch に YRP+zoom homography aug 追加
- #229-#231 visualization smoke / hint sanity / drop smoke
- #232 train_cnd2_ddp に --yrp-zoom-aug 配線
- #233 kamikado-only self-sup mix kick (80/20)

P2 (GT pose 製造) — **来週のゴール**:
- GICP で kamikado raw に距離 pose 付与
- P1 model で per-frame-pair の回転補正 (loss 設計は要詰め)
- 補正済み GT pose を pair dataset に焼き込む

P3 (補助輪外しの pair-sup):
- P1 ckpt warm start, drop_p=0 固定で pair-sup kick
- 後段: Woven 2 seq holdout で外的検証

