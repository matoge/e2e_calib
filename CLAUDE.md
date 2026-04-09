# CLAUDE.md

## クイックスタート

```bash
# デモサーバー起動
python app.py
# → http://localhost:5001

# 学習 (全モデル)
python train.py          # 1物体
python train_multi.py    # 2物体
python train_cov.py      # 共分散
python train_depth.py    # Depth-aware

# 可視化
python vis.py            # → result.png
python vis_cov.py        # → result_cov.png
python vis_depth.py      # → result_depth.png
```

## チェックポイント

| ファイル | モデル | val loss |
|---------|--------|---------|
| `best_model.pt` | CalibNet (1物体) | 0.18 px |
| `best_model_multi.pt` | CalibNet (2物体) | 0.41 px |
| `best_model_cov.pt` | CalibNetCov | NLL −0.289 |
| `best_model_depth.pt` | CalibNetDepth | NLL 0.997 |

## アーキテクチャ概要

- `model.py` → `CalibNet`: Coarse+Fine CrossAttention、Self-Attention付き
- `model_cov.py` → `CalibNetCov` + `gaussian2d_nll`: 共分散出力
- `model_depth.py` → `CalibNetDepth`: 3ch (U,V,D) 入力
- `dataset.py` → `make_image_and_points*`: 合成データ生成

## API エンドポイント

- `GET /api/sample?seed=N&mode=single|multi|depth`
- `GET /api/model_status`

## 注意

- `.pt` ファイルは `.gitignore` 済み（サイズ大）
- `torch.compile(max-autotune)` で初回起動が遅い（〜5分）
- CUDA必須（CPUでも動くが遅い）
