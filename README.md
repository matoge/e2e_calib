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

## Quick start — clone から PNG まで 1 分

**実測のコピペ手順** (このリポでテスト済み):

```bash
# 1) clone + LFS pull (チェックポイントは LFS でコミット済み)
git clone git@github.com:matoge/e2e_calib.git
cd e2e_calib
git lfs install
git lfs pull --include "experiments/km_wv_wm_n4_img128_ml_dense_pe_dgx2_200ep_v2/best_model.pt"
#   --include を外せば全実験の best_model.pt を落とす (大きい)

# 2) 依存
sudo apt install -y git-lfs
pip install torch torchvision flask matplotlib numpy pyceres clearml

# 3) 推論 + 可視化 ワンライナー (woven の val cache はリポにコミット済み)
PYTHONPATH=. python -m scripts.inference.infer_pipeline \
    --exp km_wv_wm_n4_img128_ml_dense_pe_dgx2_200ep_v2 \
    --cache data/woven_v3_tile \
    --split val --idxs 17,100,1000 --top-k 100 \
    --out out/quickstart_demo
```

これで以下が出ます (実測値):
```
load model: km_wv_wm_n4_img128_ml_dense_pe_dgx2_200ep_v2
build dataset: data/woven_v3_tile (val)
  len(ds)=7840, img_size=128
  idx=17:   N_valid=223  σ range 1.61-2.42px  err hyp→true=11.60  pred→true=2.33px
  idx=100:  N_valid=231  σ range 3.49-6.10px  err hyp→true=11.20  pred→true=6.40px
  idx=1000: N_valid=18   σ range 9.18-10.23px err hyp→true=17.71  pred→true=27.16px
done → out/quickstart_demo
```

`out/quickstart_demo/idx{000017,000100,001000}.png` に red→green 可視化が
出力されます。`hyp→true` が摂動入力の誤差 (px)、`pred→true` がモデル補正後の
誤差。idx=17 の **11.60 → 2.33 px** が典型的な成功シグナル。

> **ClearML から取りたいときだけ** (LFS に載ってない実験):
> ```python
> from clearml import Task
> t = Task.get_task(task_id='7e6f442a118042188609a115f139f61d')
> ckpt = t.artifacts['best_model.pt'].get_local_copy()   # → ローカル path
> ```

---

### 推論 CLI のオプション

`scripts/inference/infer_pipeline.py` が**学習と byte-identical** な前処理で
推論し、red→green 可視化 PNG を吐く**唯一の正しい入口**です。全ての
可視化・eval・BA スクリプトはこのモジュールを経由します。

```bash
# ランダム N 枚
PYTHONPATH=. python -m scripts.inference.infer_pipeline \
    --exp <NAME> --cache <CACHE_DIR> --split val --n 8 --out out/foo

# 全点描画 (タイル単位 BA debug 向け)
PYTHONPATH=. python -m scripts.inference.infer_pipeline \
    --exp <NAME> --cache <CACHE_DIR> --split val --idxs 0 --top-k -1 --out out/all
```

出力 PNG の凡例:
- `red ○` = 摂動入力 hyp_uv
- `green ○` = モデル予測 pred_uv (= hyp + Δ)
- `yellow / magenta ✗` = GT
- 黄/橙の矢印 = 補正 Δ (red → green)
- cyan/magenta 線 = 補正後の残差 (green → GT)
- lime 楕円 = per-point 2D σ

タイトル例: `idx=17 top100 of 223 valid pts σ 1.46-1.82px |pred-GT| 2.93±0.39px err b→a: 11.60→2.71px`

`b→a` が補正前→補正後の平均ピクセル誤差。

### プログラム経由で叩く

```python
import numpy as np
from scripts.inference.infer_calib    import load_calib_model
from scripts.inference.infer_pipeline import make_ds, infer_one, render_red_to_green

EXP = 'km_wv_wm_n4_img128_ml_dense_pe_dgx2_200ep_v2'
m       = load_calib_model(EXP)
ds, c   = make_ds(EXP, 'data/woven_v3_tile', split='val')
res     = infer_one(m, ds, idx=17, seed=42)
v       = res['valid']
print(f"err {np.linalg.norm(res['hyp_uv'][v]  - res['true_uv'][v], axis=1).mean():.2f}"
      f" → {np.linalg.norm(res['pred_uv'][v] - res['true_uv'][v], axis=1).mean():.2f} px")
render_red_to_green(res, 'idx17.png', top_k=100)
```

`load_calib_model` は `experiments/<exp>/config.py` の `CFG` から
`frustum_dense / use_intensity / n_layers / img_size / deform_mode / use_pose_emb`
等を全部読み取ってモデル形状を組むので、**load 時に何も指定しなくて良い**。
ckpt と config がズレてれば `size mismatch` で落ちる。

### WebUI デモサーバー

```bash
PYTHONPATH=. python -m scripts.serving.caaas_app    # http://localhost:5002
```

`caaas_app` は推論ロード周りで `infer_calib.load_calib_model` を共有し、
タイル単位の `infer_tiles(model_input_size=c['img_size'])` で sliding 推論する。

### 学習

```bash
# 実データ DDP 学習 (最近の主流。configs/<NAME>.py が直接 CFG)
PYTHONPATH=. torchrun --nproc_per_node 8 \
    scripts/training/train_ps_v3_ddp.py --cfg <NAME>

# 旧合成系 (sanity check 用)
python scripts/training/train_64.py
python scripts/training/train_sim3d.py
```

実験結果は `experiments/<name>/{best_model.pt, train.log, config.py, vis_ep*/}` に保存。

### データキャッシュを自分で作る場合

```bash
python scripts/data_preparation/build_ps_full.py     # PandaSet → v3 tiled cache
python scripts/data_preparation/build_ns_full.py     # NuScenes
python scripts/data_preparation/build_waymo_full.py  # Waymo
```

リポにコミットしてある `data/woven_v3_tile` だけで動く構成にしてあるので、
**推論を試すだけならキャッシュ構築は不要**。

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
| 05 | [docs/cross_frame_report.html](docs/cross_frame_report.html) | Cross-frame residual (dual projection, PoC 0.60 px) |

---

## 注目の実験

LFS でコミット済みなので `git lfs pull` すれば即推論できる主要モデル:

| Path | 何 |
|---|---|
| `experiments/km_wv_wm_n4_img128_ml_dense_pe_dgx2_200ep_v2/` | **メイン**: kamikado+woven+waymo joint, n=4 ML deformable + dense PE, 200ep on dgx2 |
| `experiments/km_wv_wm_tss4_n4_img128_grid16_50ep_dgx3_16gpu_warm/` | 上記から TSS4 を 80% 入れて 50ep warm-start (TSS4 fisheye キャリブ問題対策) |
| `experiments/tss4_iter1baked_n4_img128_30ep_os16_dgx2_16gpu_warm/` | TSS4 iter1 cache を baked 化 + os16 で warm-start |
| `experiments/ps_full_n4_img128_parity_dgx4_100ep/` | PandaSet 単体 parity baseline (DGX4 100ep) |
| `experiments/ps_v9_objsplit/` | (旧) PandaSet object-split ベストモデル (0.91 px) |
| `experiments/all_v3_mc/ba_kb/` | (旧) 12 点 fx×KB スイープ、t=1.56 cm (k₂=+0.01) |
| `experiments/vdef_{sl,ml}/` | (旧) Deformable cross-attn (val NLL 1.57) |

---

## 環境

- PyTorch 2.x + CUDA (fp16 autocast 必須 — bf16 だと sm_70 で fp32 emulated に
  落ちて学習時の Δuv 分布とズレる。`infer_pipeline.infer_one` は fp16 固定)
- RTX 5080 / 5090 / V100 / A100 / DGX2 でテスト
- `git-lfs` (チェックポイント取り出しに必要)
- `pyceres` (Python bindings for Ceres Solver) が BA に必要
- `clearml` は学習中の scalar / vis を見るときと、LFS に載ってない実験を
  artifact から取り出すときだけ

```bash
sudo apt install git-lfs
pip install torch torchvision flask matplotlib numpy pyceres clearml
```
