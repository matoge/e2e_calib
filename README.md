# e2e_calib — LiDAR/Camera Calibration Toy Problem

LiDAR点群とカメラ画像の対応づけ・UV補正を学習するPOC。  
Cascaded Cross-Attention + Self-Attention による端到端キャリブレーション。

## 実験結果

| 実験 | 補正前 | 補正後 | 改善率 |
|------|--------|--------|--------|
| 1物体 uniform shift | 14 px | 0.8 px | **94%** |
| 2物体 独立シフト | 14 px | 1.5 px | **89%** |
| 共分散学習 (NLL) | — | val NLL = −0.289 | — |
| Depth-aware + 不確かさ | 10 px | 1.2 px | **88%** |

### 深度対応不確かさ学習

(U, V, Depth) の3ch入力で、深度チャンネルのみから不確かさを自動推定:

```
obj1 (depth=10)  σx=0.74  σy=0.73  ← 自信あり
obj2 (depth=20)  σx=0.73  σy=0.79  ← 自信あり
bg   (depth=40)  σx=1.63  σy=1.96  ← 不確か (×2)
```

## アーキテクチャ

```
Image (1×128×128)
    │
    └─ CNNBackbone ──→ coarse_feat (D×16×16)
                   └─→ fine_feat   (D×32×32)

Points (N×2 or N×3)
    │
    └─ PointMLP ──→ query (N×D)
                      │
            ┌─────────┴──────────┐
            │  CrossAttentionBlock (Coarse)  │
            │  ├ Cross-Attn (point→image)   │
            │  ├ Self-Attn  (point→point)   │
            │  └ offset_head (D+2 → 2)      │
            └─────────┬──────────┘
                  warp UV
            ┌─────────┴──────────┐
            │  CrossAttentionBlock (Fine)    │
            └─────────┬──────────┘
                      │
               (tx, ty) × img_size   or
               (tx, ty, log_σx, log_σy, ρ)
```

**Key design choices:**
- Self-Attention が複数物体の独立シフト学習に必須（mean poolingは禁止）
- Offset head に UV position を concat: `Linear(D+2, 2)` → 空間差分の学習が可能
- 2D Sinusoidal PE を [0,1] で統一
- Coarse→Fine の2段階で精度向上

## ファイル構成

```
dataset.py        # 合成データ生成 (single / multi / depth)
model.py          # CalibNet (基本モデル)
model_cov.py      # CalibNetCov (共分散出力)
model_depth.py    # CalibNetDepth (3ch入力 + 共分散)
model_no_sa.py    # アブレーション: Self-Attention なし
train.py          # 1物体学習
train_multi.py    # 2物体学習
train_cov.py      # 共分散学習
train_depth.py    # Depth-aware学習
train_ablation.py # SA有無の比較
vis.py            # 結果可視化
vis_cov.py        # 不確かさ楕円の可視化
vis_depth.py      # Depth別グループ可視化
app.py            # Flask デモサーバー
static/index.html # インタラクティブWebUI
```

## セットアップ

```bash
pip install torch torchvision flask matplotlib numpy
```

## 学習

```bash
# 1物体
python train.py

# 2物体
python train_multi.py

# 共分散
python train_cov.py

# Depth-aware
python train_depth.py
```

## デモ

```bash
python app.py
# → http://localhost:5001
```

WebUIで seed ナビゲーション、モード切替 (1Object / 2Objects / Depth+σ)、自動再生。

## 環境

- PyTorch 2.x + CUDA
- RTX 5080 (TF32, bfloat16 AMP, torch.compile max-autotune)
