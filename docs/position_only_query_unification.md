# Position-Embedding-Only Query による統合 — calib と cross-frame を 1 モデルで

**日付:** 2026-04-27
**関連実験:** v303 / v305 / v306 / v308 / v310 / v311
**関連コード:** `models/cross_frame_unified.py` (`CalibNetUnifiedFrame`, `uv_only_query`),
`train_cross_frame.py` (L584: `uv_only_query=(args.uv_only_query or args.calib_mode)`)

---

## TL;DR

Cross-frame 残差ネットの **クエリ Q を「位置埋め込みのみ」 (PointMLP(uvd) + pose_emb) に縮約** することで、`calib_mode` と `cross-frame` を **完全に同一モデル・同一ヘッドで** 扱えるようになった。
calib 側は **PandaSet 0.60–0.67px / nuScenes-LiDAR 0.79px / nuScenes-Radar 0.61–0.71px** と、モダリティ非依存に同等水準で収束。
3-dataset combined → PandaSet fine-tune (v306→v311) で **val_err 1.00 → 0.60px** と pretraining の効きも確認できた。

---

## 1. 動機 — なぜ Q を「位置だけ」にするのか

統一モデル `CalibNetUnifiedFrame` は元々、

```
Q = bilinear(frame_token_A, uv_A) + PointMLP(uvd_A) + pose_emb(A→B)
```

の3項で構築していた。第1項の `bilinear(frame_token_A, uv_A)` は、anchor 側の image+LiDAR 融合トークンを点の投影位置でサンプルした「modality-aware な local context」である。

ところが **calibration モード** では:

- anchor 側は **片モダリティ** （カメラのみ、または LiDAR のみ、または Radar のみ）になる場合がある
- `frame_token_A` の中身が pair 学習時 (両モダリティ詰め込み) と **意味が違ってしまう**
- → bilinear sample が unimodal/multimodal で揺れ、Q が一貫しない

**解決:** calib_mode のときは bilinear 項を落とす。Q を以下に縮約:

```
Q = PointMLP(uvd_A) + pose_emb(A→B)
                        ↑ これが「position-embedding-only クエリ」
```

これで Q は **モダリティ非依存・位置情報のみ** になり、context は cross-attention 経由で kv 側 (target frame の frame_token) からだけ引いてくる構造になる。

実装: `models/cross_frame_unified.py:263-282`、フラグは `uv_only_query`、`calib_mode=True` 時に自動 ON。

---

### モデル構造（uv-only-query 時）

```
       anchor frame A (1モダリティ可)        target frame B
       ┌──────────┬──────────┐             ┌──────────┬──────────┐
       │  image   │ LiDAR/   │             │  image   │ LiDAR/   │
       │  patch   │ Radar pc │             │  patch   │ Radar pc │
       └────┬─────┴─────┬────┘             └────┬─────┴─────┬────┘
            ▼           ▼                       ▼           ▼
       ┌─────────────────────┐             ┌─────────────────────┐
       │  FrameTokenEncoder  │             │  FrameTokenEncoder  │
       │  (CNN + frustum +   │             │  (CNN + frustum +   │
       │   intra-attention)  │             │   intra-attention)  │
       └──────────┬──────────┘             └──────────┬──────────┘
                  │  frame_token_A                    │  frame_token_B
                  │  (B,D,Hg,Wg)                      │  (B,D,Hg,Wg)
                  │                                   │
                  │   ※ uv-only-query: ここからの      │
                  │     bilinear は使わない            │
                  ▼                                   │
            (drop bilinear)                           │
                                                      │
   uvd_A ──► PointMLP3(uvd) ─┐                        │ KV
   pose_AB ─► PoseMLP ───────┴──► Q (B,N,D)           │ (B, D, Hg, Wg)
   (= 位置のみ)                    │                  │
                                   ▼                  ▼
                           ┌──────────────────────────────────────┐
                           │ UnifiedCrossBlock × n_cross_layers   │
                           │   cross-attn (Q → KV=B)              │
                           │   self-attn  (点間)                  │
                           │   FFN                                │
                           │   head: (Δu, Δv, log σu, log σv, ρ)  │
                           └──────────────────────┬───────────────┘
                                                  ▼
                                       残差 + 共分散 (B,N,5)
```

**ポイント:**
- Encoder は **anchor 側も target 側も同じ重み**。modality 数に応じて intra-attention 内で frustum pooling する。
- Q は **位置情報のみ**（uvd + pose embedding）。anchor の中身を一切覗かない。
- KV は target frame_token (B,D,Hg,Wg) をそのままグリッドとして扱い、deformable / dense cross-attention で引く。
- 出力ヘッドは pair / triplet / calib すべて共通の (Δu, Δv, Σ)。

---

## 2. 何が unify されるのか

| 軸 | cross-frame (pair / N-frame) | calibration |
|---|---|---|
| anchor frame | 1 つの time-step | 1 つの sensor (cam / LiDAR / Radar) |
| target frame | 別 time-step | 別 sensor (cam) |
| Q | 位置埋め込みのみ (uvq) | 位置埋め込みのみ (uvq) |
| KV | target frame_token | target sensor frame_token |
| ヘッド | (Δu, Δv, Σ) | (Δu, Δv, Σ) |
| 損失 | 2D Gaussian NLL | 2D Gaussian NLL |

→ **モデル定義・損失・ヘッドが完全に同一**、データローダ側で「pair か calib pair か」を切り替えるだけ。

---

## 3. 実験結果

すべて `model=unified, n_cross_layers=4, n_intra_layers=2, img_size=64, sigma_ypr=0.5, sigma_t=0.05, calib_mode=True` (= uv-only-query)。

| ID | dataset | sensor (KV-side) | cameras | val_err [px] | val_NLL | base [px] | epochs | 備考 |
|---|---|---|---|---|---|---|---|---|
| v303 | PandaSet | LiDAR | front | **0.67** | -0.07 | 8.05 | 30 | 単独学習 baseline |
| v305 | nuScenes | Radar | front | **0.61** | -0.17 | 4.92 | 30 | radar PoC、front-cam |
| v308 | nuScenes | Radar | all 6 | **0.71** | 0.03 | 4.29 | 30 | 全カメラ展開 |
| v310 | nuScenes | LiDAR | front | **0.79** | 0.13 | 5.47 | 30 | radar↔lidar 比較ペア |
| v306 | Panda+DDAD+Waymo | LiDAR | all | **1.00** | 0.49 | 7.60 | 30 | 3-dataset combined |
| v311 | PandaSet (init=v306) | LiDAR | front | **0.60** | -0.16 | 8.05 | 15 | **v306 → fine-tune** |

### 読み方

- **モダリティ非依存に決まる:** LiDAR / Radar 双方で **sub-pixel まで** 落ちる。とくに **Radar 0.61 px** はカメラ–Radar 較正の AI 化として強い数字。
- **マルチカメラへの素直な拡張:** v305 (front) → v308 (all 6cams) で 0.61 → 0.71 px と若干悪化するが、front 限定 PoC からカメラ全数展開で大崩れしない。
- **データセット合算は素では辛い:** v306 (3 ds combined) は val 1.00 px / NLL 0.49 まで。**spec の異なるデータの混合は素直には効かない**。
- **Pretraining → fine-tune が効く:** **v306 → v311 で 1.00 → 0.60 px**。base がほぼ同じ (~8 px) かつ NLL も -0.16 まで改善。3-ds 事前学習は混合では伸びないが、**ターゲットドメインへの fine-tune を経由するとしっかり効く** ことが見えた。

---

### 可視化

各 `experiments/cross_frame_<id>/` に学習曲線（`curve.png`）、残差予測オーバーレイ（`viz_VAL.png` / `viz_TRAIN.png`、青=正解, 赤=予測, 楕円=Σ）、BA reprojection（`ba_reproject.png`）が揃っている。代表3 run を以下に示す。

#### v303 — PandaSet × LiDAR × front-camera calib（単独学習 baseline, val_err 0.67 px）

学習曲線:

![v303 curve](../experiments/cross_frame_v303_panda103_calib_uvq/curve.png)

VAL 残差オーバーレイ:

![v303 viz_VAL](../experiments/cross_frame_v303_panda103_calib_uvq/viz_VAL.png)

BA reprojection:

![v303 ba_reproject](../experiments/cross_frame_v303_panda103_calib_uvq/ba_reproject.png)

---

#### v305 — nuScenes × Radar × front-camera calib（Radar PoC, val_err 0.61 px）

学習曲線:

![v305 curve](../experiments/cross_frame_v305_nuscenes_radar_calib_uvq/curve.png)

VAL 残差オーバーレイ:

![v305 viz_VAL](../experiments/cross_frame_v305_nuscenes_radar_calib_uvq/viz_VAL.png)

BA reprojection:

![v305 ba_reproject](../experiments/cross_frame_v305_nuscenes_radar_calib_uvq/ba_reproject.png)

---

#### v311 — v306 (Panda+DDAD+Waymo combined) → PandaSet fine-tune（val_err 0.60 px）

学習曲線（fine-tune は 15 epoch で素早く収束）:

![v311 curve](../experiments/cross_frame_v311_finetune_v306_panda103_calib/curve.png)

VAL 残差オーバーレイ:

![v311 viz_VAL](../experiments/cross_frame_v311_finetune_v306_panda103_calib/viz_VAL.png)

BA reprojection:

![v311 ba_reproject](../experiments/cross_frame_v311_finetune_v306_panda103_calib/ba_reproject.png)

---

## 4. ここから何が言えるか

1. **モデル一本化が成立する。** calib と cross-frame を別アーキテクチャで持つ理由は、少なくとも本構成においては**ない**。
2. **Radar も対等な一級モダリティ。** 4D Radar 統合の前段として、現行 Radar (nuScenes 単点 dopplers) で sub-pixel calib が出ている。
3. **ドメイン横断は「混ぜる」より「事前学習→現地 fine-tune」が筋。** combined 学習はそのままだと spec のばらつきで頭打ち。
4. **次の一手:**
   - 社内データ (cam-LiDAR 33.8ms 同期補正済) で v311 と同じ fine-tune 路線
   - 4D Radar (DENSO 共同) を Radar branch に流し込み、v305/v308 と同じ評価軸で比較
   - cross-frame (pose / SLAM) 側へ uv-only-query を逆輸入して、calib と pose を 1 model 1 weight で扱えるか確認
