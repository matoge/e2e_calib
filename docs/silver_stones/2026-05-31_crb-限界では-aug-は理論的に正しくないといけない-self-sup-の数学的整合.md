---
date: 2026-05-31T00:42+09:00
author: hfunaya
tags: ["self-sup", "crb", "aug", "homography", "yrp", "zoom", "lidar", "livo", "north-star", "calibnet2"]
streams: ["e2e_calib", "calibnet2-design"]
status: silver
---
# CRB 限界では aug は理論的に正しくないといけない — self-sup の数学的整合

**Origin:** ~/git/e2e_calib @ ff631d7 (git@github.com:matoge/29_e2e_calib.git)

## 動機: なぜ aug の数学的整合がここで効くのか

CalibNet 系の精度天井は visual feature 側の **Cramér-Rao bound (CRB) に張り付いてる** ([[2026-05-30_gatr-を-frustum-encoder-に入れても効かなかった-情報限界-crb-の話]])。GA や inductive bias で動く余地は無く、**CRB そのものを動かすには解像度・aug の質・データ量で攻める** しかない。

CRB に張り付いてる regime では、**aug が物理的に少しでもズレてると、その嘘を model が hint として拾ってしまう**。flow が信号レベルで sub-pixel まで詰まってるので、嘘の aug は嘘の hint として鋭く効く。CRB に余裕がある regime なら適当な aug でも吸収されるが、ここではそうじゃない。

→ **「aug は数学的に厳密に正しい変換でなければならない」** が今の挑戦の前提条件。

## 何を self-sup でやるか / やらないか

実データの pose は **100% 信用できないのがスタートライン**。Waymo ですら不明、自前 kamikado raw / TSS4 は確実に bias が乗ってる。**LIVO のように tightly-coupled で 6-DoF を解いて補正データを作る** 道は、self-sup の本筋とは別 (= LIVO で初期 pose を作るのは別問題、ここでは self-sup から visual feature を CRB まで詰めることが目的)。

self-sup で **数学的に厳密に正しい変換** だけを使う:

| 変換 | self-sup で使えるか | 理由 |
|---|---|---|
| **YRP rotation** | ✓ 使える | `H = K · R(yaw, pitch, roll) · K⁻¹` で homography に**完全に閉じる** (t=0 / pure rotation の場合)。parallax 一切無し、解析解 exact |
| **Zoom (= scale)** | ✓ 使える (ただし条件付き) | 局所的には焦点距離変化と等価。**「2× zoom → 距離 1/2 倍」は局所的に厳密**。LiDAR D も `D * s` で整合 |
| **Translation (parallax あり)** | ✗ 使えない | depth 情報無しでは image 単独で逆解きできない。これは pair-sup (LiDAR が距離を持つ実 2-frame) の領域 |

translation を self-sup に入れる選択肢もあるが、「`D * s` の局所近似」しか使わないなら translation は parallax を生み、aug が物理的に不正確になる → CRB regime で害。**self-sup は YRP+zoom のみ、translation は出さない** が正しい結論。

## YRP+zoom aug の数学的根拠

### YRP (純回転)

3D camera に純回転 R を加えると、image 上の点 uv は:

$$\tilde{uv} = K \cdot R \cdot K^{-1} \cdot uv \quad (\text{homogeneous coords})$$

これは **depth に一切依存しない**。任意の depth の点が、どれも同じ homography で動く。LiDAR D は不変、画像 grid_sample で warp 一発、誤差ゼロ。

### Zoom (= 局所的 scale)

zoom factor `s` で image を warp する = 焦点距離を `s` 倍にしたのと等価。同じ世界の点は **「s 倍近くにある」かのように映る** ので、D は `D / s` ... と思いきや、お前の定義は **「2× zoom out (s=2) → 距離は半分 (D/2)」**。これは「zoom out で物体が小さく見える ≒ 物体が遠くにある」ではなく、「焦点距離が縮んだ世界」と解釈する場合に成立。

つまり:

- s > 1 (zoom in, image 拡大) → 焦点距離 s 倍 → 同じ uv に映る点は s 倍近い → `D' = D / s`
- s < 1 (zoom out, image 縮小) → 焦点距離 1/s 倍 → 同じ uv に映る点は s 倍遠い → `D' = D / s`

統一すると **`D' = D / s`** が「焦点距離 s 倍世界」での scale 整合。

ただし「2× にしたら距離半分」というユーザの定義に合わせるなら s と D の関係が逆 (`D' = D * s` で s=2 → D 倍 = 距離 2 倍 = ... これは違う)。

**要再確認**: 「2× zoom in → 距離半分 = `D' = D / 2` (s=2 のとき D 半分)」 = `D' = D / s` で合う。memory として確定すべきは:

$$D' = D / s$$

で、s>1 (zoom in) のとき D 半分。

### 局所近似の限界

zoom は image-plane 全体に **uniform な scale** をかけるが、現実の camera 移動 (= forward translation) では:

- 近い点は大きく動く (parallax)
- 遠い点はほとんど動かない

つまり **zoom aug = uniform scale ≠ 現実の forward motion**。「局所的には正しい」とは「ある depth の 1 点近傍で見ればほぼ整合」の意味で、image 全体では物理的に正確じゃない。

CRB regime ではこの差も学習に効く可能性がある。対策:

- D に **per-point ノイズ ν(u, v)** を入れる: `D' = D / s * ν(u, v)`
- ν は空間相関ある低周波 Fourier 場 (低周波 + per-point 独立じゃない)
- これで「s から D が決定論的に逆算できない」ようにし、model に「D は信用しすぎるな」を教える
- ただし **学習が破綻するリスク** もあるので、まず ν 無しで kick → D を hint に逃げてないか sanity → 必要なら ν 入れる

## PoseEmb (RoPE) の役割

self-sup で aug の R_aug を **PoseEmb (RoPE = block-diag(R) を Q feature に作用)** に hint として渡す ([[2026-05-30_cross-frame-01-を-self-sup-でどこまで持ち上げられるか-calibnet2-戦略]] と整合)。

意味:
- **hint mode (80%)**: R_aug 注入、model が「pose 大体わかってる、image で残差を詰めろ」を学ぶ → 推論時に GICP_R を hint として渡せる
- **drop mode (20%)**: R = I 注入、model が「hint 無しでも image だけで合わせる」を学ぶ → CFG-style

zoom は PoseEmb に入れない (identity lock)。理由: zoom は焦点距離経由なので、type-0 (intrinsic) に入れるべきだが、self-sup 段階では **「zoom の hint 無し」で image から推定させる**ほうが visual feature が伸びる。

## 現実 (推論時) との対応

学習: A の crop に YRP+zoom aug 被せて B を捏造、image で対応取る model を作る。

推論 (実 2-frame, GICP 初期 pose):
- GICP の **t は信用** (LiDAR 距離は正確、~40m まで sub-cm)
- GICP の **R は怪しい** (距離合わせ loss は回転に弱い)
- model に GICP_R を PoseEmb hint として渡す
- image で R 補正値 (residual ΔR) を当てに行く
- → 補正済 R = GICP_R · ΔR で「ほぼ正しい」 pose 製造

これが [[2026-05-30_cross-frame-01-を-self-sup-でどこまで持ち上げられるか-calibnet2-戦略]] の P1 → P2 → P3 ブートストラップの **核**。self-sup で詰めるのは「YRP 補正能力」、translation は GICP に任せる、で役割分担が clean。

## 実装上の決定 (確定)

| 項目 | 値 |
|---|---|
| H | `H = K·R(yrp)·K⁻¹` (3D 純回転) + 2D similarity (pivot 任意 zoom) の合成 |
| YRP range | yaw/pitch ±5°, roll ±2° |
| zoom range | s ∈ [0.5, 2] log-uniform |
| t | 0 (固定、translation 一切無し) |
| D scale | `D' = D / s` (s=2 zoom in → D 半分) |
| D noise | (Phase 1) 無し / (Phase 2) `ν(u,v)` 空間相関ノイズで escape closure |
| sampling | 80% hint (zoom=1, R 注入) / 20% drop (zoom 任意, R=I) |
| PoseEmb | type-1 RoPE (R_aug を Q feature に block-diag(R) で作用) |
| zoom hint | 常に identity lock (focal_aug=0) |

## 次

- #228 pandaset_full self-sup branch に YRP+zoom homography aug 組み込み (B 側完全新規生成、translation 殺す)
- #229 ✓ 可視化 smoke
- #230 hint sanity (R 注入で UV residual ≈ 0)
- #231 drop smoke (R=I で image only で warp 解析解収束)
- #232 train_cnd2_ddp 配線
- #233 kick

## 関連

- [[2026-05-30_cross-frame-01-を-self-sup-でどこまで持ち上げられるか-calibnet2-戦略]] — P1→P2→P3 全体戦略
- [[2026-05-30_gatr-を-frustum-encoder-に入れても効かなかった-情報限界-crb-の話]] — CRB 飽和の根拠
- [[2026-05-30_monodepth-は-lidar-再現じゃなく-4dgs-北極星から逆算した-pipeline]] — monodepth は副産物

