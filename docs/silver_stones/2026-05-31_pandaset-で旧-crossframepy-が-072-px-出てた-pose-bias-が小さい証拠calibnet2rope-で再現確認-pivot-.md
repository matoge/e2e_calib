---
date: 2026-05-31T01:44+09:00
author: hfunaya
tags: ["calibnet2", "cross-frame", "pandaset", "pose-bias", "lln", "rope", "pivot", "literature-gap", "self-sup"]
streams: ["e2e_calib", "calibnet2-design"]
status: silver
---
# PandaSet で旧 cross_frame.py が 0.72 px 出てた = pose bias が小さい証拠、CalibNet2+RoPE で再現確認 + pivot だけ修正

**Origin:** ~/git/e2e_calib @ ff631d7 (git@github.com:matoge/29_e2e_calib.git)

## 何の話か

[[2026-05-30_cross-frame-01-を-self-sup-でどこまで持ち上げられるか-calibnet2-戦略]] の self-sup 路線を組む前に、**旧 `models/cross_frame.py` が PandaSet で既に sub-pixel に到達してた事実**を再確認する。これは「PS の GT pose は cm 級で十分良い、bias が乱数的なら LLN で吸収できる」という前提を裏付ける強い実証データ。

CalibNet2 + RoPE + frame-token 構造はこれの上位互換のはずだが、**まず PS で動かして同等性を確認** してから kamikado / TSS4 に展開する。

## 旧 cross_frame.py の実績 (PandaSet)

`docs/cross_frame_blog_ja.html` 記載の数値:
- ベースライン: PandaSet val、frame baseline ±5 frame、`σ_ypr=1.0°` / `σ_t=0.2m` の HAT 摂動
- **hyp 残差**: 11.0 / 14.6 / 15.9 / 14.1 px (pose のズレで初期投影がこれだけずれる)
- **pred 後**: **0.72 / 0.62 / 1.06 / 1.88 px** (= sub-2px)
- 近 baseline サンプルでは **0.12 px**

![cross_frame v13_mix best predictions on PandaSet val (baseline ±5 frames, σ_ypr=1.0° / σ_t=0.2m)](/assets/cross_frame_hero.png)

各 panel 左: A frame (LiDAR cuboid 投影 GT)、右: B frame (色付き ★ = ネット予測, ○ = HAT, ×=GT)。hyp 残差 11–16 px が pred 後 0.6–1.9 px に収束してる、近 baseline では 0.12 px (sub-pixel)。bias が systematic にあれば pred 後にも残るはずだが、ほぼ ○ と ★ が GT 上に重なってて bias が乱数 noise レベルである強い視覚証拠。

この数字は「±5 frame baseline + ±1° / ±0.2m HAT 摂動」という決して甘くない設定で出てる。**PS の GT pose に systematic bias が乗っていれば pred 後でも残差は systematic に出るはず**で、実際 1 px 切ってる事実が「PS pose の bias が極めて小さい」 = 乱数的 noise 程度であることを示唆。

## なぜそれで動くか (LLN ルートの実証)

GT pose に **systematic bias が無く乱数 noise だけ**であれば:

- 個々の pair で hyp pose が乱数的にバラついても、scene 横断で平均すれば 0
- model は「平均的に正しい residual」に収束、つまり **bias-free な regression** を学ぶ
- 推論時に出てきた pose を真値として再学習させれば、残った noise も漸次薄まる (= **iterative refinement**)
- bias 残ってるかどうかは目視で確認できる (Woven 2 seq holdout / frame stride 増やして発散しないか / overlay 安定性)

つまり deep research 含意の「**pose-bias-aware self-sup は publish されてないが、PS のような高品質データでは自然に LLN で済んでる**」が今ここで起きてる現実。

## 旧 cross_frame.py の構造的弱点 2 つ

それでも構造的には改善余地ある:

### (1) PoseEmb が additive bias

```python
Q_A = pt_A + pose_emb_AB    # PoseMLP(6-DoF) を点トークンに足してから cross-attn
```

これは [[project_se3_token_rotation_toy]] が問題視した方式そのもの — 0.5° の rotation でも token に "1m あたり 6mm" の bias を architecture が常時注入する。**1cm 地図要件と非両立**。CalibNet2 では type-1 RoPE (block-diag(R) を Q feature に作用) に置き換え済 (#204)。

### (2) 6-DoF を 1 個の MLP に押し込んでる

```python
PoseMLP: (yaw, pitch, roll, tx, ty, tz) → D-dim
```

intrinsic 系 (focal/log_vfp) と extrinsic 系 (rotation R, translation t) を分離してない。CalibNet2 では type-0 (intrinsic, additive) と type-1 (Δpose R, RoPE) で分離 ([[2026-05-27_calibnet2_design]] 相当)。

## それでも 0.72 px

(1)(2) の構造的弱点があっても 0.72 px。理由はおそらく:
- PS の GT pose が高品質で bias がほぼ無い → architecture の bias 注入が「ほぼ恒等」regime に張り付いた
- baseline ±5 frame 範囲では 6mm/m の bias が画像 px 換算で sub-pixel に収まる (= 1 m 距離なら 0.6 cm → ~1 px、と見える)
- したがって **PS scale (= ±5 frame, 数 m 移動)** で評価する限り architecture 弱点が顕在化しない

問題は kamikado raw / TSS4 / 100m 先信号など **より遠い対応や bias 大きいデータ** で顕在化する可能性。これを CalibNet2+RoPE で構造的に潰すのが目的。

## pair builder の pivot 投影 — **既に修正済み (確認結果)**

「PS の pair builder で pivot を GT pose で reproject してるはず、これだけ直せば OK」と思っていたが、grep で確認したところ **既に POSE_HAT 化されていた**。

`datasets/pandaset_full.py:1014-1054` 該当箇所:

```python
# Pick pivot from A (preferring obj) that ALSO projects into B
# under HAT. Crop center on B = pivot's HAT projection (deploy).
...
pu_A, pv_A = float(uv_full_A[i, 0]), float(uv_full_A[i, 1])
piv_z = float(z_A[i])
...
# Crop center on B = pivot under POSE_HAT (NOT GT). Same edge-reject.
pu_B, pv_B = float(uv_Bproj_hat[i, 0]), float(uv_Bproj_hat[i, 1])
```

ポイント:
- `R_off_B = R_gt_B @ R_delta` (= POSE_HAT) で `uv_Bproj_hat` を作る (line 984)
- B crop center に **`uv_Bproj_hat` を使う**、line 1047 のコメントが `(NOT GT)` と明示
- pivot が image edge 近すぎる場合は clip じゃなく re-roll (line 1043-1044)
- これは [[2026-05-22_session_handoff_2026_05_22]] / `#184 Pair builder: crop B at POSE_HAT, not GT` で既に直したロジックと同型

つまり **train/test mismatch 問題は構造的に解決済み**。CalibNet2 + RoPE での kick 時に追加で直すべき箇所は無い。

軽微な残存依存:
- `cs` (crop size) が `piv_z` (GT depth) に依存 (line 1026-1031): 近物体には大きい crop、遠物体には標準 crop の戦略
- これは deploy 時に hint として渡す HAT depth で代替できる/学習信号には影響しない、という判断で当面 OK
- 追って検証 (= deploy 時に GT depth が無い → cs が変わる → val/test 分布ズレ) する余地はある

含意:
- 旧 cross_frame.py が PS で 0.72 px 出してた数字は **pivot 投影が POSE_HAT** の状態での結果
- CalibNet2 で kick する時は dataset 側触らずそのまま使える
- 直すべき優先は CalibNet2 アーキテクチャの差分 (RoPE 化、type-0/1 分離、frame-token、idempotent block) のみ

## 検証順序

1. **PS で旧 cross_frame.py を再現** (val 残差 0.72 px 級が出るか確認)
2. ~~pivot 修正~~ (= 既に POSE_HAT 化済み、不要)
3. **CalibNet2 + RoPE + frame-token で kick** (PS、同設定)
   - 期待: 0.72 px と同等以上 (RoPE は 0.5° → 6mm bias を構造的に消すので、近 baseline では同等、遠 baseline で改善)
4. **frame stride 漸増** (±5 → ±10 → ±20) で発散しないか観察
   - LLN 効いてれば各 stride で bias-free residual が出続ける
   - 発散したら何か系統的に効いてる、調査
5. **Iterative refinement smoke**: ckpt の出力 pose で GT を上書きして再学習、3 round くらい回して残差が薄まるか
6. (上手くいったら) kamikado / TSS4 に同じ流れを移植
7. self-sup ([[2026-05-31_crb-限界では-aug-は理論的に正しくないといけない-self-sup-の数学的整合]]) は **これと並走** で kick — bias 吸収にもう一段保険を入れる位置づけ

## CalibNet2 vs 旧 cross_frame.py で確認したい仮説

- **同等仮説**: PS の高品質 pose 下では、architecture の差 (RoPE vs additive) は ±5 frame では見えない
- **改善仮説**: frame stride 漸増 (±20 とか) では additive の bias 注入が顕在化、RoPE が 1 px を維持する
- **frame-token 仮説**: frame-token で per-frame closed な KV bank を持つ CalibNet2 は ±20 でもさらに安定
- **self-sup 補強仮説**: self-sup を混ぜると遠 stride 時の occlusion 領域 bias がさらに減る

## 関連

- [[2026-05-30_cross-frame-01-を-self-sup-でどこまで持ち上げられるか-calibnet2-戦略]]
- [[2026-05-31_crb-限界では-aug-は理論的に正しくないといけない-self-sup-の数学的整合]]
- [[2026-05-30_gatr-を-frustum-encoder-に入れても効かなかった-情報限界-crb-の話]]

## 次

- ~~pair builder で pivot 投影箇所を grep 特定、POSE_HAT 化~~ — 完了 (修正済み確認)
- `train_ps_v24_cross_frame.py` の resume / 再 kick path 確認
- まず PS で再現 → CalibNet2 で同設定 kick

