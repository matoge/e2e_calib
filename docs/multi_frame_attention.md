# Multi-frame cross-attention — how to integrate N frames

PoC 設計の核心の問題を整理する。Pair (2-frame) で動いた cross-attention を
3-frame 以上に拡張すると、どこで失敗しやすいか・文献ではどう解いているかを
記録しておく。

> **Note on framing.** 文献は「自分達が解いてる問題」と「我々が解いてる
> 問題」が一致しない前提で読むこと。同じキャリブレーションでも sensor
> サイズ / 出力形式 / 教師信号 / scale すべて違う。**問題は問題から演繹的
> に解く** べきで、論文の手法はそのまま使うものではなく素材として扱う。
> このドキュメントも「論文がこうだから真似する」ではなく「我々の問題に
> 何が必要か → その必要性に対して文献はどんな例を持っているか」の順で
> 書いてある。

## 1. 問題設定

我々のタスクは **pose-conditional residual prediction**:

- 入力: query フレーム $A$ (画像 + LiDAR点) と target フレーム $B$ (画像 + LiDAR点),
  ノイジーな相対 pose hypothesis $\hat T_{AB}$
- 出力: $A$ の各点に対して, $B$ 画像内での投影位置の補正 $\Delta uv$ + 共分散 $\Sigma$
- pose は **入力条件** (output ではない)

2-frame で動いている設計:

```
q = pt_A + PoseEmb(T_AB)              # Q を pose-shift
KV = pt_B (+ img_B as deformable)
scores = Q @ K^T / √d + b_AB           # b_AB はスカラー bias
out = softmax(scores) @ V
```

3-frame 以上 ($A, M, B$) に拡張するとき、 query は $A$ のままだが
KV に $M$ と $B$ の両方が入る。 ここで:

- **どこに pose 情報をどう入れるか** が大きく性能を左右する。
- 単純に `KV = concat(pt_M, pt_B)` で 1 つの softmax にすると、 sub-pair
  $(A, M)$ の attention が 2-frame $(A, M)$ で trained したときと **異なる
  Q,KV 配置になる** ため、 学習の knowledge を再利用できない。

## 2. 失敗する naive 拡張 (現行 PoC)

```
q     = pt_A + PoseEmb(T_AB)         # ← どの KV-frame を見ようと同じ Q
KV    = concat(pt_M, pt_B)
ref   = [uv_M_hat_of_A, uv_B_hat_of_A]
bias  = [b_AM, b_AB]  (per-frame, KV 側のみ加算)
scores = Q @ K^T + bias_per_frame
softmax over [M ∪ B]
```

問題:
- Q は $A→B$ の pose しか知らない. $M$ を見るときも $A→B$ pose で
  shift された Q を使う → $A↔M$ sub-attention が 2-frame $(A, M)$ と
  別物 になる。
- pose_bias は **scalar offset** しか追加できない. Q の方向 (どこを見るか)
  には影響しない。
- Deformable attention の sampling MLP も Q だけから offset を学習する.
  Q が「$B$ を見たい」と知ってるだけだと, $M$ の image をどこからサンプル
  すべきかも誤った教師信号になる。

実証: v55 (M=GT、損失A↔Bのみ) で **+0.5px 改善** はする。
v64 (M perturbed、6 directions全 supervise) で **逆に劣化** (~3x ま val_err 悪化)。
M を auxiliary KV にとどめる "おまけ" レベルなら効くが、 きちんと統合すると
壊れる。

## 3. 正しい方法 (per-KV-frame Q)

```
for each KV frame k ∈ {M, B}:
    Q_k = pt_A + PoseEmb(T_{A→k})       # ← KV-frame ごとに違う Q
    K_k = pt_k
    scores[..., k] = Q_k @ K_k^T / √d
attn = softmax(concat scores over k)
out  = attn @ V
```

これで:
- $(A, M)$ sub-pair の Q, KV は 2-frame $(A, M)$ と完全一致 ←
  knowledge re-use できる
- $(A, B)$ も同様
- 1 つの softmax で混ぜるので, **各 frame の重要性は scores の大きさ**
  (= 学習された "どっちを信じるか") で自動調整される
- pose_bias は scalar 補助として残せる

img-side の deformable attention でも同じ理屈で, 各 KV-frame の
reference uv と Q (frame-conditional) の組み合わせで sampling を学習する。

## 4. 文献 evidence

別途 web 調査 (2024-2025 の主要 multi-view transformer 論文)。

### Reference list

| 論文 | 会議 | arXiv |
|---|---|---|
| DUSt3R (Wang et al.) | CVPR 2024 | <https://arxiv.org/abs/2312.14132> |
| MASt3R (Leroy et al.) | ECCV 2024 | <https://arxiv.org/abs/2406.09756> |
| CroCo (Weinzaepfel et al.) | NeurIPS 2022 | <https://arxiv.org/abs/2210.10716> |
| CroCo-Stereo / CroCo-Flow | ICCV 2023 | <https://arxiv.org/abs/2211.10408> |
| MVSFormer (Cao et al.) | TMLR 2023 | <https://arxiv.org/abs/2208.02541> |
| MVSTER (Wang et al.) | ECCV 2022 | <https://arxiv.org/abs/2204.07346> |
| TransMVSNet (Ding et al.) | CVPR 2022 | <https://arxiv.org/abs/2111.14600> |
| FlowFormer (Huang et al.) | ECCV 2022 | <https://arxiv.org/abs/2203.16194> |
| **GTA — Geometric Transform Attention** (Miyato et al.) | ICLR 2024 | <https://arxiv.org/abs/2310.10375> |
| Fast3R (Yang et al.) | CVPR 2025 | <https://arxiv.org/abs/2501.13928> |
| VGGT (Wang et al.) | CVPR 2025 | <https://arxiv.org/abs/2503.11651> |
| Spann3R (Wang & Agapito) | 3DV 2025 | <https://arxiv.org/abs/2408.16061> |

### (a) DUSt3R / MASt3R (Wang et al., CVPR 2024 / ECCV 2024)
**Pair-only siamese decoder**. View 1 の Q は view 2 の KV にしか
attend しない。 multi-view (>2) は出力時に pairwise を繋ぎ合わせる。
"single Q, multi-KV" を **回避する** 方針 — これは我々の (b) を
"separate forwards" で実現するのに相当。

### (b) MVS Transformers (MVSFormer, MVSTER, TransMVSNet)
**Plane-sweep / epipolar warping** で source view の特徴を ref view の
frustum に warp してから attention. 各 (ref, src_k) pair に対して
**source-specific な warped feature volume** を作る → これは Q (or KV)
が source view ごとに pose-baked になっている状態で, (b) と機能等価。

### (c) GTA — Geometric Transform Attention (Miyato et al., ICLR 2024)

![GTA mechanism](https://takerum.github.io/gta/resources/gta_mech.png)

**Q を per-KV pose rotation で multiplicatively conditioning** する設計。
論文式 (3):
$$\text{Attn}(Q, K, V) = \text{softmax}(Q P_q^k K^T / \sqrt d) V$$
ここで $P_q^k$ は (query, key) ペアごとに違う pose 変換。図の通り、 Q と
K は **token 自身に紐づく座標フレーム** (それぞれの camera frame) を持って
おり、 attention 計算の前段で **相対変換**を加えてから dot-product する。
これが「**Q の bias は KV に依存する**」を最も明示的に実装している例。

**我々の問題との framing 差**:
- GTA は **NeRF-like view synthesis** (= 与えた viewing pose に対応する
  pixel を生成する) を想定。Q は "次に出力したい pixel の pose"、 KV は
  既知 pose の image features。出力は pixel value。
- 我々は **camera-LiDAR 残差予測**。 Q は anchor frame の点、 KV は別
  view の image+pt features。出力は per-point Δuv + Σ。
- pose 入力の役割が "input condition (我々)" vs "rendering target
  (GTA)" で、用途は違うが **pose-conditional attention をどう作るか**
  という question は同じ。

そのため GTA は**手法をそのまま借りる対象ではない** — Q/K の rotation
multiplicative form は我々の枠組みでは pose embedding additive form の
方が自然 (PointMLP / image feature が learned representation で、 SE(3)
rotation を直接掛けるのは意味が薄い)。
ただし「**Q-K の組ごとに pose 変換を入れる**」という principle は流用
できる。具体的には我々の (b) — pt + PoseEmb(T_{A→k}) を per-KV-frame_k
に作る — が GTA の additive 版に相当。

### (d) Fast3R (Yang et al., CVPR 2025), VGGT (Wang et al., CVPR 2025)
**Frame-id learnable embedding** を加えて全 view tokens を 1 sequence に
concat. shared Q だが frame-id を per-token bias として加える (option c)。
特徴: pose は **モデルの output** であり, 入力条件にはしない。
我々の用途と前提が違うので そのまま流用不可。

### (e) Pose-conditional baseline (Swin RPE, T5-style)
**scalar bias of relative position**. 我々の現行 pose_bias と同種。
小さい discrete offset には有効だが, **連続 SE(3) pose では役不足** —
特に rotation が depth-dependent な pixel shift を起こす場合, scalar
bias 1 つで補正しきれない。

## 5. 結論 / Recommendation

問題から演繹で逆算すると:

1. 我々のタスクは **2-frame で動いていることが既に証明済み** (v55=2.28)。
   この時点で per-token attention path は確立されている (Q=pt+pose_emb,
   KV=他frame, 残差予測 head で Δuv+Σ)。
2. 3-frame に拡張するとき、 sub-problem としての (A↔M), (A↔B) は
   それぞれ 2-frame と **同じ問題**である必要がある。 そうでなければ
   2-frame で学んだ知識が再利用できず、 6 directions の同時最適化を
   イチから学習することになる (= v64 が失敗した理由)。
3. 「sub-problem を 2-frame と同一に保つ」とは, **(A, M) sub-attention
   の Q, K, V が 2-frame (A, M) のときと bit-identical** であること。
   Q が一つの pose しか持たないと, M を見るときも B 用の Q-shift を
   被ったまま attend する → 別問題になる。

**我々の問題が要求する形**: per-KV-frame Q (option b)。
これは GTA の principle と機能的に等価だが, additive pose embedding 版
で実装する (rotation multiplicative ではなく, 我々の token 表現は
learned MLP 出力なので additive が自然)。

他の選択肢の不採用理由:
- (a) 共有 Q + scalar bias は **sub-problem identity を壊す**ため不可
  (pose が小さい場合に偶然動くこともあるが, 大角・長 baseline で破綻)。
- (c) frame-id embedding は pose を **出力**するモデル (Fast3R/VGGT) に
  特化、我々のように pose を入力条件にする問題には情報量が足りない。

具体的書き換え:

```python
# cross_blocks の中で:
scores_blocks = []
for k, (frame_pt, pose_to_k) in enumerate(zip(pt_kv_list, poses_to_each_frame)):
    pose_emb_k = pose_mlp(pose_to_k).unsqueeze(1)
    q_k        = pt_query + pose_emb_k        # frame-conditional Q
    Q_k        = q_proj(norm_q(q_k))
    K_k        = k_proj(norm_kv(frame_pt))
    scores_blocks.append((Q_k @ K_k.transpose(-1,-2)) / sqrt(d))
scores = torch.cat(scores_blocks, dim=-1)     # softmax across all KV
attn   = softmax(scores) @ concat_V
```

これで $(A, M)$ sub-pair に注ぐ attention は完全に 2-frame と同じ
Q,K の組み合わせ → pre-train knowledge that lives in pair model
(v55 = 2.28) を multi-frame に lossless transfer できる。

## 6. 次のステップ

1. `MultiFrameCrossBlockDeform.forward` を per-KV-frame Q に書き換え
   (上記 30 行)
2. 同じデータ (PandaSet 39 scene, σ_ypr=2.0, σ_t=0.5, lidar dropout) で
   v65 として走らせ, v55 (2.28), v64 (~10) と比較
3. v65 < 2.28 なら multi-frame の本来の効果が出ている証拠
4. v65 ≈ v55 なら 3-frame でも 2-frame と同等以上は守れている (悪化しない)
   ことが確認でき, さらに 6-direction supervised (full P1) に進める
