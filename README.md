# e2e_calib — Cross-Attention LiDAR↔Camera Calibration

<p align="center">
  <img src="docs/images/hero.png" width="820" alt="Four validation crops: LiDAR reprojection before/after sub-pixel correction">
</p>

LiDAR 点を画像平面にマッピングする**ローカルな evidence detector**。各点ごとに
2D ガウシアン `(Δu, Δv, σu, σv, ρ)` を出力し、その共分散つき残差を Ceres ベースの
バンドルアジャスト (BA) に流して剛体補正を解きます。

> **なぜこの分解か**: フルフレーム end-to-end で剛体 6DoF を回帰するより、
> パッチ単位の局所証拠だけに学習を閉じ込めた方が (1) 小さく速く、
> (2) リグ構成に非依存で、(3) BA で不確かさ伝搬でき、(4) なぜ失敗したか debug できる。

---

## 主要な結果

### ① パッチ精度 (ps_v9_objsplit)
| 指標 | 値 |
|---|---|
| Held-out **object** reproj MSE | **0.91 px** |
| パラメータ数 | 1.62 M |
| 訓練 | PandaSet 103 scene / 33,458 crops / 200 ep / 87 min (1× GPU) |

### ② ジョイント学習 (NuScenes + PandaSet + Waymo)
| Dataset | obj MSE | bg MSE |
|---|---|---|
| PandaSet | 1.25 px | 3.16 px |
| Waymo | 1.93 px | 4.35 px |
| NuScenes | 2.32 px | 5.23 px |

→ 一つのネットワークで 3 つの異なるセンサ構成を処理。詳細は
[static/ns_ps_v2_report.html](static/ns_ps_v2_report.html) /
[static/ps_v9_report.html](static/ps_v9_report.html)。

### ③ Multi-frame BA (scene 015, 10 frames)
GT drift `ypr ‖ ‖=0.46°, t ‖ ‖=26.2 cm` を共有 6DoF で共同最適化:

| 設定 | rot_err | t_err | reproj med |
|---|---|---|---|
| pinhole | 0.035° | **2.10 cm** | 0.75 px |
| KB `k₂=+0.01` | 0.029° | **1.56 cm** | 0.72 px ← best |
| fx −0.5 % | 0.018° | 17.7 cm | 4.17 px |
| fx −1.0 % | 0.031° | 33.5 cm | 7.96 px |

k₂≈+0.01 が残差を最小化 → PandaSet のピンホールモデルに微小な pincushion がある示唆。
詳細は [experiments/all_v3_mc/ba_kb/summary.png](experiments/all_v3_mc/ba_kb/summary.png)。

### ④ Deformable cross-attn (joint NS+PS+WM, 60k/1.5k, 200 ep)
| Block | val NLL | 備考 |
|---|---|---|
| standard cross-attn | 2.074 | baseline |
| deformable SL | **1.568** | [experiments/vdef_sl/](experiments/vdef_sl/) |
| deformable ML | 1.573 | [experiments/vdef_ml/](experiments/vdef_ml/) |

bf16 native CUDA kernel でベースラインと同速 (19.3 vs 19.5 ms/iter, B=48 N=256)。

---

## アーキテクチャ

```
Image (RGB 64×64 or 128×128)
    │
    └─ ConvNeXt-mini ──→ coarse_feat (16×16)
                     └─→ fine_feat   (32×32)

LiDAR Points (N × 3, [U, V, D])
    │
    └─ PointMLP + Frustum local encoder ──→ query (N × D)
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       │  CrossAttentionBlockCov  (L1 … L4 カスケード) │
                       │   ├ (optional) image-side self-attn          │
                       │   ├ cross-attn  (pt → image)                 │
                       │   ├ self-attn   (pt → pt)                    │
                       │   └ OffsetHead → (Δu, Δv, log σu, log σv, ρ) │
                       └──────────────────────┬──────────────────────┘
                                       warp UV and recurse
                                              │
                              最終出力: (N × 5) per-point 2D ガウシアン
```

設計の肝:
- **Frustum encoder** が重要（外すと +0.81 NLL 悪化）
- **Cross-attn を先**、self-attn はその後 (self-first にすると悪化)
- **Coarse→Fine** のカスケード (`cross_coarse → cross_refine → cross_fine → cross_fine2`)
- ピクセル座標系を `[0,1]` 正規化して sinusoidal 2D PE を統一
- 出力は常に 2D 共分散つき → BA で Σ-重み付け残差として使える

---

## Quick start

```bash
# デモサーバー (インタラクティブ WebUI)
python app.py                       # http://localhost:5001

# 合成 (fast prototyping)
python train.py                     # 単一物体
python train_multi.py               # 複数物体
python train_grid_depth.py          # メイン: config_grid_depth.py で実験設定

# 実データ
python train_pandaset.py
python train_nuscenes.py
python train_waymo.py
python train_all_v2_mc.py           # ジョイント NS+PS+WM
python train_deform_sweep.py        # deformable (SL / ML sweep)

# 解析
python ba_singleframe.py            # 1 frame BA: 4px 誤差 → 0.8 px 補正
python ba_multiframe.py             # N frame シェア 6DoF BA
python ba_kb_multiframe.py          # fx / KB 感度スイープ
python icp_scan_residual.py         # スキャン残差診断
python vis_static_consistency.py    # 静止物体の一貫性チェック
```

### データ準備

```bash
# マルチクロップキャッシュ (s64 = crop ≥ 64px)
python build_all_mc_caches.py
```
出力先: `/mnt/nvme6t/e2e_calib_cache/{pandaset,nuscenes,waymo}_mc_s64_cache.pt`。
データセット別に `build_ps_full.py` / `build_ns_full.py` / `build_waymo_full.py` もあり。

---

## リポジトリ構成（flat layout）

```
.
├── app.py                           # Flask デモサーバー
├── dataset.py                       # 合成データ生成
├── dataset_{pandaset,nuscenes,waymo}.py
├── model.py                         # CalibNet
├── model_cov.py                     # CalibNetCov (共分散)
├── model_depth.py                   # CalibNetDepth (frustum + deform)
├── model_deform.py                  # deformable cross-attn ブロック
├── ops/                             # MSDeformAttn CUDA kernel (bf16 native)
├── train_*.py                       # 各種学習スクリプト
├── ba_{singleframe,multiframe,kb_multiframe,global}.py  # BA pipeline
├── vis_*.py                         # 可視化ユーティリティ
├── docs/                            # GitHub Pages 用レポート
│   ├── index.html, report.html
│   ├── images/                      # レポート挿絵
│   └── assets/                      # 生成 viz
├── static/                          # インタラクティブ UI + 技術レポート
│   ├── index.html, ns_ps_v2_report.html, ps_v9_report.html
├── experiments/                     # 実験結果 (checkpoint + log + config)
└── web_viewer/                      # Babylon.js 点群ビューワ
```

---

## 注目の実験

| Path | 何 |
|---|---|
| `experiments/ps_v9_objsplit/` | PandaSet object-split ベストモデル (0.91 px) |
| `experiments/all_v2_mc/` / `all_v3_mc/` | ジョイント NS+PS+WM マルチクロップ |
| `experiments/all_v3_mc/ba_kb/` | 12 点 fx×KB スイープ、t=1.56 cm (k₂=+0.01) |
| `experiments/vdef_{sl,ml}/` | Deformable cross-attn (val NLL 1.57) |
| `experiments/ps_mc_overfit500/` | Overfit 500 サンプルでアーキ検証 |
| `experiments/icp_residual/` | ICP スキャン残差の外部検証 |
| `experiments/turn_projection/` | 旋回時の投影ズレ解析 |

---

## 環境

- PyTorch 2.x + CUDA (bfloat16 autocast + TF32 + `torch.compile(max-autotune)`)
- RTX 5080 / 5090 でテスト
- `pyceres` (Python bindings for Ceres Solver) が BA に必要

```bash
pip install torch torchvision flask matplotlib numpy pyceres
```
