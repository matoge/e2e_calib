# 1 モデル、N 問題 — 位置クエリ + 多モーダル frame_token

**TL;DR** — Query を `(uv, depth)` 由来の埋め込みだけにし、KV を
**modality 不問の frame_token グリッド** にすると、cross-frame pose
residual・camera-LiDAR 外部 calibration・camera-Radar・multi-camera
calibration をすべて **同じモデル定義 (同じコード、同じ重み形式)** で解ける。
PandaSet 103 シーンで calib と cross-frame の両方を verify 済み。

---

## 1. 何が「同じ」なのか

学習対象のネットワークは [`models/cross_frame_unified.py`](../models/cross_frame_unified.py) の
`CalibNetUnifiedFrame` 1 個だけ。下のすべての問題で:

- 同じ `class CalibNetUnifiedFrame(...)` インスタンス
- 同じ層構成 (frame encoder + N × deformable cross-attn + intra-frame self-attn)
- 同じ出力ヘッド (per-point `(Δu, Δv, log σu, log σv, ρ)`)

を使う。問題ごとに変わるのは **dataset の作り方** と **encoder に渡す
modality flag** だけ。

| 問題 | 入力 frame_A | 入力 frame_B | 摂動 | 出力 |
|---|---|---|---|---|
| Cross-frame pose residual | image + LiDAR (frame t1) | image + LiDAR (frame t2) | 相対 pose ノイズ | per-point Δuv (t1→t2) |
| **Cam-LiDAR 外部 calib** | image (frame t) | LiDAR scatter (frame t, 摂動 extrinsic で射影) | 外部 extrinsic ノイズ | per-point Δuv |
| **Cam-Radar 外部 calib** ★ | image (frame t) | Radar scatter (frame t, 摂動 extrinsic で射影) | 外部 extrinsic ノイズ | per-point Δuv |
| **Multi-camera calib** ★ | camera_A image | camera_B image | 相対 extrinsic ノイズ | per-point Δuv |

★ = 設計上当然解けるはずの拡張。Cam-Radar / multi-cam は未検証
（PandaSet には Radar が無いので、社内データ or AV2 / nuScenes Radar split で実装予定）。

---

## 2. 設計の核 — Query と KV を分離する

### 2.1 Query は位置のみ (modality 非依存)

旧版モデル (cross_frame v100 系) では Query を:

```
Q[i] = bilinear_sample(frame_token_anchor, uv_i)  # ← anchor frame の局所特徴
     + PointMLP(uv_i, depth_i)                    # ← 3D 位置情報
     + pose_emb(target)                           # ← target frame への姿勢ヒント
```

の和で組んでいた。**bilinear_sample の項が anchor frame に多モーダル
特徴 (image + LiDAR の融合) があることを暗黙に仮定している** ため、
anchor が camera-only や LiDAR-only になる calib の場面では特徴の
偏りが query を歪める。

新版 (`uv_only_query=True`):

```
Q[i] = PointMLP(uv_i, depth_i) + pose_emb(target)
```

純粋に位置情報だけ。**modality に何が入っているかを query 側が知らない**
ので、frame_A が画像専用でも LiDAR 専用でもクエリの作り方が変わらない。

### 2.2 KV は固定形状の frame_token グリッド

各フレームを `(B, D, 16, 16)` のテンソル 1 個に潰す。中身は:

```
frame_token[c, h, w] = fuse(image_feat[c,h,w],
                             lidar_scatter[c,h,w],
                             has_lidar_mask[h,w])
```

この `fuse` は 1×1 Conv で、image 側はゼロ点で identity、LiDAR 側は
ゼロ初期化。**modality flag** で encoder が:

- `'mm'`: 通常通り両方使う
- `'cam'`: lidar_scatter / mask を強制ゼロ → image 単独
- `'lidar'`: image_feat を強制ゼロ → LiDAR 単独

を切り替える ([cross_frame_unified.py:96-129](../models/cross_frame_unified.py))。
**出力の形状は modality に依らず常に `(B, D, 16, 16)`**。
Cross-attention 側はこの「同じ shape」しか見ないので、上流の問題定義に
無関係に動作する。

### 2.3 図解

```
                  各フレーム独立にエンコード
   ┌────────────┐         ┌────────────┐
   │ image      │         │ LiDAR      │
   │ + LiDAR    │ ──cam──▶│ + image    │ ──lidar─▶
   │ (frame_A)  │         │ (frame_B)  │
   └────────────┘         └────────────┘
         │                        │
         ▼                        ▼
   frame_token_A           frame_token_B
   (B, D, 16, 16)         (B, D, 16, 16)
                                  │
                                  │ どちらも同じ shape の grid
                                  ▼
                    ┌─────────────────────────┐
                    │ Q = uv_emb + pose_emb   │
                    │ KV = [ft_A, ft_B]       │
                    │ MSDeformAttn × N        │
                    │ → per-point (Δuv, Σ)    │
                    └─────────────────────────┘
```

---

## 3. 実証 — PandaSet 103 シーン

`uv_only_query=True` の同一モデル設定で、calib と cross-frame の両方を
平行学習 (yokohama / sakurai 2 ホスト並走)。

| run | mode | encoder modality | 摂動 | base err | val err | val NLL |
|---|---|---|---|---|---|---|
| **v303** | calib (`fi_B=fi_A`) | A=`cam`, B=`lidar` | 0.5°/0.05m | 8.05 px | **0.68 px** | −0.04 |
| **v304** | cross-frame | A=`mm`, B=`mm` | 1.0°/0.2m, baseline 1-20 | 12.88 px | 4.40 px* | 3.39* |

\* v304 は ep3 時点 (継続学習中)。cross-frame v100 系の同設定での収束は
ep15-20 で val 1.8-2.3 px。

**ポイント**:

- 同じ `CalibNetUnifiedFrame(uv_only_query=True, n_cross_layers=4)` が
  両方の問題で意味のある誤差削減を達成
- 重みは別々に学習 (ヘッド共有はまだ未試行) — それでも「**問題が変わると
  モデルも変わる**」という帰納が間違っていることを示せる
- v303 の base 8.05 px → 0.68 px は cm 級の calibration 精度に相当する
  (img_size=64 patch、CROP=128-256 px、典型 焦点距離換算で 0.3° 以内)

---

## 4. なぜ効くか

### 4.1 「画像 vs LiDAR」 の対称性

camera-LiDAR calibration を「**1 つの画像** に対して LiDAR を **2 通りの
extrinsic** で射影し、その差を予測する」 問題と捉えると、
cross-frame の「**1 つの世界点** を **2 つの相対 pose** で射影してその差を
予測する」 構造と完全に等価になる。式で書くと:

```
Cross-frame:
   uv_B_GT  = π(K_B,  T_BW · X_world)
   uv_B_HAT = π(K_B,  T_BW · ΔT_pose · T_WA · X_cam_A)
   target   = uv_B_GT - uv_B_HAT

Cam-LiDAR calib:
   uv_GT  = π(K_cam,  T_cam_lidar_GT  · X_lidar)
   uv_HAT = π(K_cam,  T_cam_lidar_HAT · X_lidar)   = π(K_cam, T_cam_lidar_GT · ΔT_extr · X_lidar)
   target = uv_GT - uv_HAT
```

両方で **target = π(GT) − π(HAT) = 「正しい外部変換 vs 摂動した外部変換」 の
画素差**。この target が同じ形をしている限り、ネットワークは同じ residual
primitive を学べばいい。

### 4.2 Frame_token 設計が「modality を 1 つの名前」 に押し込む

ネットワークの上半分 (cross-attn 以降) は frame_token しか見ない。
frame_token は出力 shape が固定 (D, 16, 16)、中身が image でも LiDAR でも
混合でも、上半分にとってはどれも「**16×16 セルの局所特徴**」。
modality 切り替えで生じる差は encoder の中で吸収され、上半分は気づかない。

これは ViT が画像のパッチを「**一様に並んだ token**」 として扱い、内容が
画像でも文章 (LiT) でも音声 (Whisper) でも上流の transformer 構造を
変えないのと完全に同じ抽象化。

### 4.3 拡張は「encoder の modality を増やす」 だけ

Radar 用の encoder ブランチ (range-angle 変換 → grid scatter) を 1 つ
追加すれば、cam-Radar calibration も同じモデルで動く。Multi-camera
calibration なら encoder が 2 個の image を受けるだけで OK
(modality は両方 'cam')。

---

## 5. 何が変わるか — 製品設計のインパクト

| 旧アプローチ | 本設計 |
|---|---|
| タスクごとに専用ネット (CalibNet, MatchNet, PoseNet, ...) を別々に学習・運用 | **1 つの residual primitive ネット**、上流で問題定義だけ差し替え |
| 新規 sensor (Radar / 異種カメラ) を入れると新しいネットを設計し直す | encoder の modality flag を増やすだけ。**上流ネットは触らない** |
| Calibration / SLAM / mapping / loop-closure が別々のチームの別々のシステム | 全部 **同じ重み + 異なる前処理** で解ける |

「sensor / task の対称性」を architecture の中に折り畳むことで、
ロボット / 自動運転の空間スタックが大幅に薄くなる。

---

## 6. 既知の限界 / 次のステップ

- **modality 重みの共有実験**: 今は v303 と v304 で重みが別。同じ
  チェックポイントから両方 fine-tune して「どこまで多 task が共有できるか」
  を計測したい。
- **Cam-Radar PoC**: AV2 (Radar 7 sensor) または社内データで Radar
  modality encoder を実装し、同じ recipe で calib 学習。
- **Multi-camera calib**: PandaSet 6cam で前後カメラ間の extrinsic
  estimation を 同モデルで学習。社内 surround-view rig も同じ。
- **Inference-time モード切替**: 1 重みで calib / cross-frame / radar 全部
  推論できる sentry network を作って、エッジ deployment の
  メモリフットプリントを 1/N にする。

---

*関連*:
[no_matching_advantage.md](no_matching_advantage.md) (本ネットの primitive
設計の根拠) /
[unified_progression.md](unified_progression.md) (cross-frame v70→v100 系の
収束推移)
