---
date: 2026-05-30T20:10+09:00
author: hfunaya
tags: ["gatr", "calibnet", "crb", "information-limit", "equivariance", "failure"]
streams: ["e2e_calib", "failures"]
status: silver
---
# GATr を frustum encoder に入れても効かなかった — 情報限界 (CRB) の話

**Origin:** ~/git/e2e_calib @ 6df36eb (git@github.com:matoge/e2e_calib.git)

## 何をやったか

#212-#216 で GATr (Geometric Algebra Transformer) を CalibNet の frustum encoder の Q 側に投入。SO(3) equivariant な multivector 表現で LiDAR-Q を処理して、cross-attn 前段の inductive bias を強化する狙い。

- #212: GATr install + import smoke
- #213: FrustumLocalEncoderGA を実装
- #215: rotation-equivariance unit test 通過 (= 単体では equivariant に動いてる)

## 結果

**精度は変わらず。**

ablation #216 まで詰めたが、frustum を equivariant にしても calib 精度の天井は動かなかった。

## 理由: visual feature 側が CRB に張り付いてる

精度を律速してるのは Q (LiDAR) 側じゃなく **KV (画像) 側の visual feature 抽出能力**。これが既に Cramér-Rao bound (情報理論的下限) に到達してて、Q 側を綺麗にしても出力に効きようがない。

根拠:

- [[2026-05-22_session_handoff]] の ω 残差 0.45 px は理論的 sub-pixel 限界域
- [[principled_ml_phase1]] の InfoHead2x2 が σ-head に対して ~10× = Fisher 情報を読み切れてる証拠
- 既に十分なデータ量 (pandaset / kamikado / TSS4) で普通の MLP が GA の事前情報分を学習で獲得してて、inductive bias の差が消える regime

つまり GA の inductive bias で減るのは **学習効率** であって、出力精度の天井は CRB で決まってる。

## cross-attn で equivariance が壊れる懸念 (補助理由)

仮に visual 側に余地があったとしても、frustum encoder の出力が SO(3) equivariant でも cross-attn で **image plane に固定された KV** と混ぜた瞬間に equivariance は破綻する。Q が equivariant でも attn の K/V projection は SE(3) 不変じゃない。end-to-end で equivariance が保たれてないと structural benefit は出ない。

#215 の equivariance unit test は **frustum encoder 単体** で測ったもの。**cross-attn 出力で測り直したら破綻してる可能性が高い** (要確認だが優先度低、CRB 仮説のほうが effect size 大きい)。

## CRB を動かしたい時の正しい方向

GA で動かすんじゃなくて:

- **解像度を上げる** ([[resolution_hypothesis_512]] の 800×256 split は実際に効いた、ω 残差 1.79→0.45 px、これは CRB そのものを動かしてる)
- **multi-frame aggregation** で √N 改善 ([[2026-05-30_cross-frame-01-を-self-sup-でどこまで持ち上げられるか-calibnet2-戦略]] の cross-frame 路線)
- pixel SNR を上げる (露出/HDR/低光量改善 — hardware 側)

## GA の処遇

PARKED ([[tracks.md]] 参照)。設計は綺麗だが現状の bottleneck じゃない。CRB 飽和してない別タスク (例: 学習データ scarce な fine-tuning) で使う日が来るかもしれない、が今じゃない。

## 教訓

- inductive bias 系の改善は **bottleneck がそこにある時** だけ効く
- 律速箇所を確認せず inductive bias を入れに行くと「動くが効かない」になる
- 次の改善候補を選ぶ前に「**今の天井は何が決めてるか** (= CRB か、データ量か、aug か)」を先に診断する

