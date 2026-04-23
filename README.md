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

ルートにはアクティブなスクリプトだけを置いてる。残りは `scripts/` 配下。
`models/ datasets/ configs/` はパッケージ化済みなので import は
`from models.model_depth import CalibNetDepth` の形。

```bash
# デモサーバー (インタラクティブ WebUI)
python app.py                       # http://localhost:5001

# 合成 (fast prototyping)
python train.py                     # 単一物体
python train_multi.py               # 複数物体
python train_grid_depth.py          # メイン: configs/grid_depth.py で実験設定

# BA 解析 (active)
python ba_singleframe.py            # 1 frame BA: 4px 誤差 → 0.8 px 補正
python ba_multiframe.py             # N frame シェア 6DoF BA (scene 015 で t=2.1cm)
python ba_kb_multiframe.py          # fx / KB 感度スイープ (→ docs/ba_report.html)

# 実データ学習
python scripts/training/train_pandaset.py
python scripts/training/train_nuscenes.py
python scripts/training/train_waymo.py
python scripts/training/train_all_v2_mc.py      # ジョイント NS+PS+WM
python scripts/training/train_deform_sweep.py   # deformable (SL / ML sweep)

# 他の診断
python scripts/ba/icp_scan_residual.py              # スキャン残差
python scripts/visualization/vis_static_consistency.py
```

### データ準備

```bash
# マルチクロップキャッシュ (s64 = crop ≥ 64px)
python scripts/data_preparation/build_all_mc_caches.py
```
出力先: `/mnt/nvme6t/e2e_calib_cache/{pandaset,nuscenes,waymo}_mc_s64_cache.pt`。
データセット別に `build_ps_full.py` / `build_ns_full.py` / `build_waymo_full.py` もあり
(全部 `scripts/data_preparation/` 下)。

---

## リポジトリ構成

```
.
├── app.py                           # Flask デモサーバー
├── train{,_multi,_cov,_depth,_grid_depth}.py   # SYNTH 系 (CLAUDE.md quickstart)
├── vis{,_cov,_depth}.py                        # 対応する可視化
├── ba_{singleframe,multiframe,kb_multiframe}.py  # active BA entries
├── models/                          # PyTorch nn.Module 群 (package)
│   ├── model.py                     # CalibNet
│   ├── model_cov.py                 # CalibNetCov (共分散)
│   ├── model_depth.py               # CalibNetDepth (frustum + deform)
│   ├── model_deform.py              # deformable cross-attn ブロック
│   └── model_no_sa.py               # ablation: self-attn 無し
├── datasets/                        # データローダ (package)
│   ├── synthetic.py, sim3d.py
│   ├── pandaset.py, nuscenes.py, waymo.py
├── configs/                         # 実験 config (package)
│   └── grid_depth.py
├── ops/                             # MSDeformAttn bf16 CUDA kernel
├── scripts/
│   ├── ba/                          # less-active BA entries
│   ├── training/                    # 実データ学習スクリプト
│   ├── visualization/               # 可視化ユーティリティ
│   ├── data_preparation/            # キャッシュ/マップ構築
│   └── eval/                        # eval/verify/bench
├── docs/                            # GitHub Pages レポート群
│   ├── index.html                   # 技術概要 (01)
│   ├── report.html                  # バイリンガル実験まとめ (02)
│   ├── ba_report.html               # Multi-frame BA sweep (03)
│   ├── deform_report.html           # Deformable cross-attn (04)
│   ├── images/                      # レポート挿絵
│   └── assets/                      # 追加生成 viz
├── static/                          # WebUI + 旧技術レポート
├── experiments/                     # 実験結果 (checkpoint + log + config)
└── legacy/                          # 参考用の旧ファイル
```

### 旧 flat-layout から package 化した箇所

| before (< 2026-04-23) | after |
|---|---|
| `model_*.py` (root) | `models/model_*.py` |
| `dataset*.py` (root) | `datasets/{synthetic,sim3d,pandaset,nuscenes,waymo}.py` |
| `config_grid_depth.py` | `configs/grid_depth.py` |
| `train_pandaset.py` 等 (root) | `scripts/training/*.py` |
| `vis_*.py` (root) | `scripts/visualization/*.py` |
| `build_*.py` (root) | `scripts/data_preparation/*.py` |
| `ba_global.py`, `icp_scan_residual.py` 等 | `scripts/ba/*.py` |

import は `from models.model_depth import CalibNetDepth` のように
パッケージパスで書く。`scripts/**/*.py` は冒頭に
`sys.path.insert(0, repo_root)` の bootstrap を自動挿入済みなので
どこから実行しても import は解決する。

---

## レポート

| # | Path | 何 |
|---|---|---|
| 01 | [docs/index.html](docs/index.html) | 技術概要 (ps_v9_objsplit ベース) |
| 02 | [docs/report.html](docs/report.html) | バイリンガル実験まとめ (合成→実データ) |
| 03 | [docs/ba_report.html](docs/ba_report.html) | Multi-frame BA + fx/KB 感度スイープ |
| 04 | [docs/deform_report.html](docs/deform_report.html) | Deformable cross-attn (val_nll −0.5) |

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
